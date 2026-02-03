from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.openai_exec import (  # noqa: E402
    OpenAIExec,
    OpenAIExecConfig,
    OpenAIExecError,
    OpenAIExecResponseError,
    OpenAIExecTimeoutError,
    build_payload_for_planner,
    extract_function_call_arguments,
    write_response_artifacts,
)


class LoopRunnerError(RuntimeError):
    pass


_EXPERIMENT_LOG_PATH = _REPO_ROOT / "experiments" / "log.jsonl"
_EXPERIMENT_LOG_EXAMPLE_PATH = _REPO_ROOT / "experiments" / "log.jsonl.example"
_STATE_DIR = _REPO_ROOT / ".advisor"
_STATE_PATH = _STATE_DIR / "state.json"
_OPENAI_ARTIFACTS_DIR = _STATE_DIR / "openai"
_CODEX_ARTIFACTS_DIR = _STATE_DIR / "codex"
_MANUAL_PACKET_DIR = _REPO_ROOT / "planner_packets"
_MANUAL_PACKET_PATH = _MANUAL_PACKET_DIR / "packet.json"
_MANUAL_PROMPT_PATH = _MANUAL_PACKET_DIR / "prompt.md"
_MANUAL_DIRECTIVE_PATH = _MANUAL_PACKET_DIR / "directive.json"
_MANUAL_SCHEMA_PATH = _MANUAL_PACKET_DIR / "directive_schema.json"

_DEFAULT_ALLOWED_PATHS = ["perf_takehome.py"]
_DEFAULT_FORBIDDEN_GLOBS = ["tests/**", "problem.py"]

_GOALS = ("threshold", "best")

_STRATEGY_FAMILIES = (
    "family:schedule",
    "family:reduce_loads",
    "family:break_deps",
)
_STRATEGY_MAX_CONSECUTIVE_DEFAULT = 2
_STRATEGY_MAX_CONSECUTIVE_BONUS = 3
_STRATEGY_BONUS_MIN_IMPROVEMENT_CYCLES = 10


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(
    argv: Sequence[str],
    *,
    cwd: Path = _REPO_ROOT,
    check: bool = True,
    capture: bool = True,
    text_mode: bool = True,
) -> subprocess.CompletedProcess:
    kwargs: Dict[str, Any] = {"cwd": str(cwd), "check": False}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if text_mode:
        kwargs["text"] = True
    proc = subprocess.run(list(argv), **kwargs)  # noqa: S603,S607
    if check and proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = f"Command failed ({proc.returncode}): {' '.join(argv)}"
        details = "\n".join(x for x in (stderr, stdout) if x)
        if details:
            msg = f"{msg}\n{details}"
        raise LoopRunnerError(msg)
    return proc


def _git(*args: str, check: bool = True) -> str:
    proc = _run(["git", *args], check=check, capture=True)
    return (proc.stdout or "").strip()


def _git_show_text(ref: str, path: Path) -> str:
    rel = path
    if rel.is_absolute():
        rel = rel.relative_to(_REPO_ROOT)
    # `git show` expects POSIX paths even on Windows; normalize defensively.
    rel_str = str(rel).replace(os.sep, "/")
    proc = _run(["git", "show", f"{ref}:{rel_str}"], check=False, capture=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = "\n".join(x for x in (stderr, stdout) if x)
        msg = f"Could not read {rel_str!r} from ref {ref!r}."
        if details:
            msg = f"{msg}\n{details}"
        raise LoopRunnerError(msg)
    return proc.stdout or ""


def _ensure_clean_worktree() -> None:
    # Ignored files (e.g., experiments/log.jsonl, .advisor/) do not appear here.
    status = _git("status", "--porcelain=v1")
    if status.strip():
        raise LoopRunnerError(
            "Working tree is not clean. Commit/stash changes before starting a new iteration.\n"
            f"git status --porcelain output:\n{status}"
        )


def _slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "auto"


def _origin_remote_url() -> Optional[str]:
    url = _git("remote", "get-url", "origin", check=False).strip()
    return url or None


def _github_web_url(remote_url: str) -> Optional[str]:
    """
    Best-effort conversion of common git remotes to a GitHub https URL.
    """

    s = remote_url.strip()
    if not s:
        return None

    if s.startswith("https://github.com/"):
        web = s
    elif s.startswith("git@github.com:"):
        web = "https://github.com/" + s[len("git@github.com:") :]
    elif s.startswith("ssh://git@github.com/"):
        web = "https://github.com/" + s[len("ssh://git@github.com/") :]
    else:
        return None

    if web.endswith(".git"):
        web = web[: -len(".git")]
    return web


def _github_tree_url(web_url: str, sha: str) -> str:
    return f"{web_url.rstrip('/')}/tree/{sha}"


def _ensure_experiment_log_exists() -> None:
    if _EXPERIMENT_LOG_PATH.exists():
        return
    _EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _EXPERIMENT_LOG_EXAMPLE_PATH.exists():
        _EXPERIMENT_LOG_PATH.write_text(_EXPERIMENT_LOG_EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        _EXPERIMENT_LOG_PATH.write_text("", encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def _tail_lines(path: Path, *, n: int) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def _cdiv(a: int, b: int) -> int:
    if b <= 0:
        raise LoopRunnerError(f"Invalid divisor for cdiv: {b}")
    if a <= 0:
        return 0
    return (a + b - 1) // b


def _truncate_clean(text: str, *, max_len: int) -> str:
    s = " ".join((text or "").strip().split())
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return "…"
    return s[: max_len - 1] + "…"


def _next_iteration_id(entries: Iterable[Mapping[str, Any]]) -> int:
    max_id = 0
    for e in entries:
        try:
            iid = int(e.get("iteration_id", 0))  # type: ignore[arg-type]
        except Exception:
            continue
        max_id = max(max_id, iid)
    return max_id + 1


_ITER_BRANCH_ID_RE = re.compile(r"^iter/(\d{1,6})-")


def _next_iteration_id_from_branch_names(branches: Iterable[str]) -> int:
    max_id = 0
    for b in branches:
        m = _ITER_BRANCH_ID_RE.match(b.strip())
        if not m:
            continue
        try:
            max_id = max(max_id, int(m.group(1)))
        except ValueError:
            continue
    return max_id + 1


def _iter_slug_from_branch(branch: str) -> Optional[str]:
    """
    Extract the slug from an iter/* branch name.

    Examples:
    - iter/0007-next -> next
    - iter/12-foo-bar -> foo-bar
    """

    m = re.match(r"^iter/\d{1,6}-(.+)$", branch.strip())
    if not m:
        return None
    slug = m.group(1).strip()
    return slug or None


def _format_best_tag(*, cycles: int, slug: str, iteration_id: int) -> str:
    safe_slug = _slugify(slug)
    return f"best/{int(cycles)}-{safe_slug}-i{int(iteration_id)}"


def _next_iteration_id_from_local_branches() -> int:
    out = _git("branch", "--list", "iter/*", "--format=%(refname:short)", check=False)
    branches = [l.strip() for l in out.splitlines() if l.strip()]
    return _next_iteration_id_from_branch_names(branches)


def _best_cycles(entries: Iterable[Mapping[str, Any]]) -> Optional[int]:
    best: Optional[int] = None
    for e in entries:
        if not e.get("valid"):
            continue
        cycles = e.get("cycles")
        if cycles is None:
            continue
        try:
            c = int(cycles)  # type: ignore[arg-type]
        except Exception:
            continue
        best = c if best is None else min(best, c)
    return best


def _plateau_valid_iters_since_best(entries: Sequence[Mapping[str, Any]]) -> Optional[int]:
    best = _best_cycles(entries)
    if best is None:
        return None
    plateau = 0
    for e in reversed(entries):
        c = _valid_cycles(e)
        if c is None:
            continue
        if c == best:
            return plateau
        plateau += 1
    # Should be unreachable if best was computed from entries, but keep a safe fallback.
    return plateau


def _strategy_family_from_entry(entry: Mapping[str, Any]) -> Optional[str]:
    tags = entry.get("strategy_tags")
    if not isinstance(tags, list) or not tags:
        return None
    first = tags[0]
    if not isinstance(first, str):
        return None
    return first if first in _STRATEGY_FAMILIES else None


def _coerce_cycles(entry: Mapping[str, Any]) -> Optional[int]:
    cycles = entry.get("cycles")
    if cycles is None:
        return None
    try:
        return int(cycles)  # type: ignore[arg-type]
    except Exception:
        return None


def _valid_cycles(entry: Mapping[str, Any]) -> Optional[int]:
    if entry.get("valid") is not True:
        return None
    return _coerce_cycles(entry)


def _family_streak(entries: Sequence[Mapping[str, Any]]) -> Tuple[Optional[str], List[Mapping[str, Any]]]:
    """
    Return (family, streak_entries) where streak_entries is the contiguous suffix sharing that family.

    If the most recent entry doesn't have a recognized family tag, returns (None, []).
    """

    if not entries:
        return None, []
    family = _strategy_family_from_entry(entries[-1])
    if not family:
        return None, []
    streak: List[Mapping[str, Any]] = []
    for e in reversed(entries):
        if _strategy_family_from_entry(e) != family:
            break
        streak.append(e)
    streak.reverse()
    return family, streak


def _last_attempt_was_meaningful_win(streak_entries: Sequence[Mapping[str, Any]]) -> bool:
    """
    True iff the last attempt in the streak is valid and improves the prior streak best by >= threshold cycles.
    """

    if not streak_entries:
        return False
    prev_best: Optional[int] = None
    for e in streak_entries[:-1]:
        c = _valid_cycles(e)
        if c is None:
            continue
        prev_best = c if prev_best is None else min(prev_best, c)

    last = _valid_cycles(streak_entries[-1])
    if last is None or prev_best is None:
        return False
    improvement = prev_best - last
    return improvement >= _STRATEGY_BONUS_MIN_IMPROVEMENT_CYCLES


def _compute_strategy_family_constraints(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Compute which strategy families are allowed/blocked for the next planner directive.

    Enforcement goal: stop the loop from endlessly iterating on one strategy direction.

    Policy (documented in docs/openai-advisor-loop.md):
    - Normally allow at most 2 consecutive attempts per family.
    - One-bonus exception: if the most recent attempt was a meaningful win (>=10 cycles vs prior streak best),
      allow one extra follow-up attempt (max 3 consecutive).
    """

    family, streak = _family_streak(entries)
    streak_len = len(streak)
    last_meaningful = _last_attempt_was_meaningful_win(streak)
    max_consecutive = _STRATEGY_MAX_CONSECUTIVE_BONUS if last_meaningful else _STRATEGY_MAX_CONSECUTIVE_DEFAULT

    blocked: List[str] = []
    reason = None
    if family and streak_len >= max_consecutive:
        blocked = [family]
        reason = f"Blocked {family} due to streak_len={streak_len} >= max_consecutive={max_consecutive}."

    return {
        "allowed_families": list(_STRATEGY_FAMILIES),
        "blocked_families": blocked,
        "current_family": family,
        "current_family_streak_len": streak_len,
        "last_attempt_meaningful_win": last_meaningful,
        "max_consecutive": max_consecutive,
        "policy": {
            "max_consecutive_default": _STRATEGY_MAX_CONSECUTIVE_DEFAULT,
            "max_consecutive_bonus": _STRATEGY_MAX_CONSECUTIVE_BONUS,
            "bonus_min_improvement_cycles": _STRATEGY_BONUS_MIN_IMPROVEMENT_CYCLES,
        },
        "reason": reason,
    }


def _strategy_tags_from_directive(directive: Mapping[str, Any]) -> List[str]:
    raw = directive.get("strategy_tags")
    if isinstance(raw, list):
        tags: List[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                tags.append(item.strip())
            if len(tags) >= 4:
                break
        if tags:
            return tags

    family = directive.get("strategy_family")
    out: List[str] = []
    if isinstance(family, str) and family.strip():
        out.append(family.strip())

    mods = directive.get("strategy_modifiers")
    if isinstance(mods, list):
        for m in mods:
            if isinstance(m, str) and m.strip():
                out.append(m.strip())
            if len(out) >= 4:
                break
    return out


def _compute_min_cycles_by_engine(*, task_counts: Mapping[str, int], slot_limits: Mapping[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for engine, count in task_counts.items():
        cap = int(slot_limits.get(engine, 0))
        if cap <= 0:
            continue
        out[str(engine)] = _cdiv(int(count), cap)
    return out


def _critical_path_engine_counts(tasks: Sequence[Any]) -> Tuple[Dict[str, int], int]:
    """
    Extract one representative critical path and return (engine_counts, path_len).

    Note: This uses the `cp` field computed by KernelBuilder (unit weights).
    """

    if not tasks:
        return {}, 0

    def _cp(t: Any) -> int:
        try:
            return int(getattr(t, "cp"))
        except Exception:
            return 0

    def _succs(t: Any) -> List[int]:
        s = getattr(t, "succs", None)
        if not isinstance(s, list):
            return []
        return [int(x) for x in s]

    start = max(range(len(tasks)), key=lambda i: (_cp(tasks[i]), i))
    tid = start
    counts: Dict[str, int] = {}
    path_len = 0
    visited_guard = 0
    while True:
        visited_guard += 1
        if visited_guard > len(tasks) + 1:
            break  # safety: should never happen in a DAG

        engine = str(getattr(tasks[tid], "engine", "unknown"))
        counts[engine] = counts.get(engine, 0) + 1
        path_len += 1

        succs = _succs(tasks[tid])
        if not succs:
            break
        tid = max(succs, key=lambda s: (_cp(tasks[s]), s))
    return counts, path_len


def _compute_performance_profile_for_submission_case() -> Dict[str, Any]:
    """
    Compute a compact bottleneck profile (lower bounds + critical-path proxy) for the submission case.

    This is used to help the advisor decide when to pivot strategies (e.g., when near a bound).
    """

    try:
        from perf_takehome import KernelBuilder  # type: ignore[import-not-found]
        from problem import SLOT_LIMITS  # type: ignore[import-not-found]
    except Exception as e:
        return {"error": f"Failed to import perf_takehome/problem: {e!r}"}

    forest_height = 10
    rounds = 16
    batch_size = 256
    # Full binary tree node count; matches the submission harness for Tree.generate(height=10).
    n_nodes = (1 << (forest_height + 1)) - 1

    class _ProfileKernelBuilder(KernelBuilder):  # type: ignore[misc,valid-type]
        def __init__(self):
            super().__init__()
            self._profile_tasks: Optional[list[Any]] = None

        def _mk_task(self, tasks, last_writer, last_reader, *, engine, slot, reads=(), writes=()):  # type: ignore[override]
            if self._profile_tasks is None:
                self._profile_tasks = tasks
            return super()._mk_task(  # type: ignore[misc]
                tasks,
                last_writer,
                last_reader,
                engine=engine,
                slot=slot,
                reads=reads,
                writes=writes,
            )

    kb = _ProfileKernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    tasks = kb._profile_tasks or []

    task_counts: Dict[str, int] = {}
    cp_lb_cycles = 0
    for t in tasks:
        engine = str(getattr(t, "engine", "unknown"))
        task_counts[engine] = task_counts.get(engine, 0) + 1
        try:
            cp_lb_cycles = max(cp_lb_cycles, int(getattr(t, "cp")))
        except Exception:
            continue

    min_cycles_by_engine = _compute_min_cycles_by_engine(task_counts=task_counts, slot_limits=SLOT_LIMITS)
    resource_lb_cycles = max(min_cycles_by_engine.values()) if min_cycles_by_engine else 0
    tight_lb_cycles = max(resource_lb_cycles, cp_lb_cycles)

    schedule_cycles_estimate = len(getattr(kb, "instrs", []) or [])
    schedule_slack_cycles = schedule_cycles_estimate - tight_lb_cycles
    schedule_slack_pct = (schedule_slack_cycles / tight_lb_cycles) if tight_lb_cycles else None

    cp_engine_counts, cp_path_len = _critical_path_engine_counts(tasks)
    dominant_engine = None
    if min_cycles_by_engine:
        dominant_engine = max(min_cycles_by_engine.keys(), key=lambda k: (min_cycles_by_engine[k], k))

    dominant_engine_ops_on_cp = cp_engine_counts.get(dominant_engine, 0) if dominant_engine else 0
    dominant_engine_cp_fraction = (
        (dominant_engine_ops_on_cp / cp_path_len) if (dominant_engine and cp_path_len) else None
    )

    return {
        "profile_case": {
            "forest_height": forest_height,
            "n_nodes": n_nodes,
            "batch_size": batch_size,
            "rounds": rounds,
        },
        "task_counts_by_engine": task_counts,
        "min_cycles_by_engine": min_cycles_by_engine,
        "resource_lb_cycles": resource_lb_cycles,
        "cp_lb_cycles": cp_lb_cycles,
        "tight_lb_cycles": tight_lb_cycles,
        "critical_path_engine_counts": cp_engine_counts,
        "critical_path_len_cycles": cp_path_len,
        "dominant_engine": dominant_engine,
        "dominant_engine_ops_on_cp": dominant_engine_ops_on_cp,
        "dominant_engine_cp_fraction": dominant_engine_cp_fraction,
        "schedule_cycles_estimate": schedule_cycles_estimate,
        "schedule_slack_cycles": schedule_slack_cycles,
        "schedule_slack_pct": schedule_slack_pct,
    }


def _extract_kernelbuilder_source() -> str:
    path = _REPO_ROOT / "perf_takehome.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    start = None
    end = None
    for i, line in enumerate(lines):
        if start is None and line.startswith("class KernelBuilder"):
            start = i
            continue
        if start is not None and line.startswith("BASELINE"):
            end = i
            break
    if start is None:
        raise LoopRunnerError("Could not find `class KernelBuilder` in perf_takehome.py")
    if end is None:
        end = len(lines)
    return "\n".join(lines[start:end]).rstrip()


_CYCLES_RE = re.compile(r"\bCYCLES:\s*(\d+)\b")


def parse_cycles_from_submission_tests(output: str) -> Optional[int]:
    matches = _CYCLES_RE.findall(output)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


_RAN_TESTS_RE = re.compile(r"^Ran\s+(\d+)\s+tests?\s+in\s+", re.MULTILINE)
_CORRECTNESS_FAIL_RE = re.compile(r"^(FAIL|ERROR):\s*test_kernel_correctness\b.*\bCorrectnessTests\b", re.MULTILINE)


def parse_correctness_from_submission_tests(output: str) -> Optional[bool]:
    """
    Return True/False if we can confidently infer correctness status from submission_tests.py output,
    else None.

    Note: submission_tests.py mixes correctness + performance-threshold tests. Failing a speed threshold
    should not mark the attempt as incorrect.
    """
    if not _RAN_TESTS_RE.search(output):
        return None
    if "Incorrect output values" in output:
        return False
    if _CORRECTNESS_FAIL_RE.search(output):
        return False
    return True


def _load_dotenv_api_key(path: Path) -> None:
    """
    Load ONLY OPENAI_API_KEY from `.env` (if present).

    This intentionally does *not* load other knobs like OPENAI_BACKGROUND_POLL_INTERVAL, so
    "real" planner calls poll at the default 60s cadence unless explicitly exported in the shell.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key != "OPENAI_API_KEY":
            continue
        if os.environ.get(key):
            return
        os.environ[key] = value.strip().strip('"').strip("'")
        return


def _enforce_default_poll_cadence() -> None:
    # Make sure "real" (non-test) planner calls use defaults (60s), even if the user previously
    # exported fast polling for smoke tests in their shell.
    for key in ("OPENAI_BACKGROUND_POLL_INTERVAL", "OPENAI_BACKGROUND_PROGRESS_EVERY"):
        if key in os.environ:
            os.environ.pop(key, None)
    # Guardrail: default background poll timeout is intentionally finite so the loop doesn't hang
    # forever on stuck OpenAI jobs. Users can override by exporting OPENAI_BACKGROUND_POLL_TIMEOUT.
    os.environ.setdefault("OPENAI_BACKGROUND_POLL_TIMEOUT", "14400")


@dataclass(frozen=True)
class IterationState:
    iteration_id: int
    branch: str
    base_branch: str
    base_sha: str
    threshold_target: Optional[int]
    packet: Dict[str, Any]
    directive: Dict[str, Any]
    advisor_response_id: Optional[str]


def _write_state(state: IterationState) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration_id": state.iteration_id,
        "branch": state.branch,
        "base_branch": state.base_branch,
        "base_sha": state.base_sha,
        "threshold_target": state.threshold_target,
        "packet": state.packet,
        "directive": state.directive,
        "advisor_response_id": state.advisor_response_id,
        "created_at_utc": _utc_now_iso(),
    }
    _STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_state() -> IterationState:
    if not _STATE_PATH.exists():
        raise LoopRunnerError("No advisor state found. Run `python3 tools/loop_runner.py plan` first.")
    data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    return IterationState(
        iteration_id=int(data["iteration_id"]),
        branch=str(data["branch"]),
        base_branch=str(data.get("base_branch") or "main"),
        base_sha=str(data["base_sha"]),
        threshold_target=(int(data["threshold_target"]) if data.get("threshold_target") is not None else None),
        packet=dict(data.get("packet") or {}),
        directive=dict(data.get("directive") or {}),
        advisor_response_id=(str(data["advisor_response_id"]) if data.get("advisor_response_id") else None),
    )


def _build_planner_prompt(packet: Mapping[str, Any]) -> str:
    packet_json = json.dumps(packet, indent=2, sort_keys=True)
    goal = packet.get("goal") if isinstance(packet.get("goal"), str) else "threshold"
    best_cycles = packet.get("best_cycles")
    threshold_target = packet.get("threshold_target")
    aspiration_cycles = packet.get("aspiration_cycles")
    plateau = packet.get("plateau_valid_iters_since_best")
    objective_lines = []
    if goal == "best":
        objective_lines = [
            "Goal: find a NEW BEST (cycles strictly less than best_cycles). There is no fixed stop threshold.",
            f"- best_cycles: {best_cycles}",
            f"- aspiration_cycles (soft target; not a stop condition): {aspiration_cycles}",
            f"- plateau_valid_iters_since_best: {plateau}",
        ]
    else:
        objective_lines = [
            "Goal: meet the target threshold_target (stop condition) while also improving best_cycles when possible.",
            f"- threshold_target: {threshold_target}",
            f"- best_cycles: {best_cycles}",
        ]
    objective_block = "\n".join(objective_lines)
    return textwrap.dedent(
        f"""
        You are the optimization advisor for this repository.

        Objective:
        {objective_block}

        Hard rules:
        - Never suggest modifying `tests/` or any test harness code.
        - Prefer changes limited to `perf_takehome.py` unless explicitly justified.
        - Do not propose patches/diffs; output plan-only.
        - You may use the `web_search` tool if needed.
        - You MUST return the final answer by calling the `emit_optimization_directive` function tool.

        Strategy family requirements (enforced by the runner):
        - `strategy_family` MUST be one of: {', '.join(_STRATEGY_FAMILIES)}
        - If `packet.strategy_family_constraints.blocked_families` is non-empty, you MUST choose a different `strategy_family`.
        - `strategy_modifiers` should be 0–3 short strings describing the concrete tactic (e.g., gather/hash/addressing).

        First, perform a novelty check against `experiment_log_tail`:
        - Avoid repeating the same (family + modifiers) combination unless you explain what is different and why it might work now.

        Use `performance_profile` to reason about lower bounds and when to pivot:
        - `resource_lb_cycles = max(min_cycles_by_engine.values())` is a throughput bound (ignores dependencies).
        - `cp_lb_cycles` is a critical-path bound (ignores engine capacity).
        - `tight_lb_cycles = max(resource_lb_cycles, cp_lb_cycles)` is a better sanity lower bound.
        - If `threshold_target <= tight_lb_cycles`, scheduling tweaks alone cannot meet the target; the next plan MUST reduce the bound
          (typically fewer tasks on the dominant engine and/or breaking dependency chains). This is only relevant when goal=threshold.
        - For goal=best: if you are plateaued and near the bound (e.g., `schedule_slack_pct <= 0.04`), prioritize plans that reduce the
          bound (dominant-engine task count) over micro-scheduling.

        Here is the current IterationPacket (JSON):
        ```json
        {packet_json}
        ```
        """
    ).strip()


def _build_manual_planner_prompt(packet: Mapping[str, Any], *, directive_schema: Mapping[str, Any]) -> str:
    packet_json = json.dumps(packet, indent=2, sort_keys=True)
    schema_json = json.dumps(directive_schema, indent=2, sort_keys=True)

    iteration_id = packet.get("iteration_id")
    branch = packet.get("branch")
    base_branch = packet.get("base_branch")
    goal = packet.get("goal") if isinstance(packet.get("goal"), str) else "threshold"
    best_cycles = packet.get("best_cycles")
    threshold_target = packet.get("threshold_target")
    aspiration_cycles = packet.get("aspiration_cycles")
    plateau = packet.get("plateau_valid_iters_since_best")

    sfc = packet.get("strategy_family_constraints")
    blocked_families: List[str] = []
    if isinstance(sfc, dict):
        raw = sfc.get("blocked_families")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    blocked_families.append(item.strip())

    repo = packet.get("repo")
    repo_lines: List[str] = []
    if isinstance(repo, dict):
        origin_url = repo.get("origin_url")
        github_web = repo.get("github_web_url")
        base_sha = repo.get("base_sha")
        worktree_path = repo.get("worktree_path")
        if isinstance(origin_url, str) and origin_url:
            repo_lines.append(f"- git remote origin: {origin_url}")
        if isinstance(github_web, str) and github_web:
            repo_lines.append(f"- GitHub repo: {github_web}")
        if isinstance(base_sha, str) and base_sha:
            repo_lines.append(f"- Base commit (authoritative): {base_sha}")
        if isinstance(github_web, str) and github_web and isinstance(base_sha, str) and base_sha:
            repo_lines.append(f"- Permalink for code lookup: {_github_tree_url(github_web, base_sha)}")
        if isinstance(worktree_path, str) and worktree_path:
            repo_lines.append(f"- Executor worktree (local only): {worktree_path}")
    repo_block = "\n".join(repo_lines) if repo_lines else "- (not provided)"

    return textwrap.dedent(
        f"""
        You are the optimization advisor for this repository.

        Iteration context:
        - iteration_id: {iteration_id}
        - plan branch (local): {branch}
        - base branch: {base_branch}
        - goal: {goal}
        - best_cycles: {best_cycles}
        - threshold_target: {threshold_target}
        - aspiration_cycles: {aspiration_cycles}
        - plateau_valid_iters_since_best: {plateau}

        Repository context (for correct GitHub lookups):
        {repo_block}

        If you choose to look at GitHub source, use the *permalink commit* above (not `main` / HEAD),
        because `main` may advance while this iteration is running.

        Hard rules:
        - Never suggest modifying `tests/` or any test harness code.
        - Prefer changes limited to `perf_takehome.py` unless explicitly justified.
        - Do not propose patches/diffs; output plan-only.
        - You may use `web_search` (if available) to research patterns/techniques, but keep suggestions implementable.

        Strategy family requirements (enforced by the runner):
        - `strategy_family` MUST be one of: {', '.join(_STRATEGY_FAMILIES)}
        - Blocked families for this iteration: {blocked_families if blocked_families else '(none)'}
        - If blocked families is non-empty, you MUST choose a different `strategy_family`.
        - `strategy_modifiers` should be 0–3 short strings describing the concrete tactic (e.g., gather/hash/addressing).

        Novelty check:
        - Before proposing a plan, read `experiment_log_tail` and avoid repeating the same strategy family/tactic
          unless you explain what is different and why it might work now.

        Use `performance_profile` to reason about lower bounds and when to pivot:
        - `resource_lb_cycles = max(min_cycles_by_engine.values())` is a throughput bound (ignores dependencies).
        - `cp_lb_cycles` is a critical-path bound (ignores engine capacity).
        - `tight_lb_cycles = max(resource_lb_cycles, cp_lb_cycles)` is a better sanity lower bound.
        - If goal=threshold and `threshold_target <= tight_lb_cycles`, scheduling tweaks alone cannot meet the target; the next plan MUST
          reduce the bound (typically fewer tasks on the dominant engine and/or breaking dependency chains).
        - If goal=best and you are plateaued and near the bound (e.g., `schedule_slack_pct <= 0.04`), prioritize plans that reduce the
          bound (dominant-engine task count) over micro-scheduling.

        Code context note:
        - `code_context` may be empty. If you need more code, request it via `next_packet_requests` (e.g., specific
          function bodies or the full `perf_takehome.py`).

        Output contract:
        - Return ONE JSON object and nothing else (no markdown, no code fences).
        - The object MUST match this JSON Schema (including all required keys):
        ```json
        {schema_json}
        ```

        Here is the current IterationPacket (JSON):
        ```json
        {packet_json}
        ```
        """
    ).strip()


def _build_codex_planner_prompt(packet: Mapping[str, Any], *, directive_schema: Mapping[str, Any]) -> str:
    packet_json = json.dumps(packet, indent=2, sort_keys=True)
    schema_json = json.dumps(directive_schema, indent=2, sort_keys=True)

    iteration_id = packet.get("iteration_id")
    branch = packet.get("branch")
    base_branch = packet.get("base_branch")
    goal = packet.get("goal") if isinstance(packet.get("goal"), str) else "threshold"
    best_cycles = packet.get("best_cycles")
    threshold_target = packet.get("threshold_target")
    aspiration_cycles = packet.get("aspiration_cycles")
    plateau = packet.get("plateau_valid_iters_since_best")

    sfc = packet.get("strategy_family_constraints")
    blocked_families: List[str] = []
    if isinstance(sfc, dict):
        raw = sfc.get("blocked_families")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    blocked_families.append(item.strip())

    return textwrap.dedent(
        f"""
        You are the optimization advisor for this repository.

        Iteration context:
        - iteration_id: {iteration_id}
        - iteration branch (local): {branch}
        - base branch: {base_branch}
        - goal: {goal}
        - best_cycles: {best_cycles}
        - threshold_target: {threshold_target}
        - aspiration_cycles: {aspiration_cycles}
        - plateau_valid_iters_since_best: {plateau}

        Hard rules:
        - Never suggest modifying `tests/` or any test harness code.
        - Prefer changes limited to `perf_takehome.py` unless explicitly justified.
        - Do not propose patches/diffs; output plan-only.

        Strategy family requirements (enforced by the runner):
        - `strategy_family` MUST be one of: {', '.join(_STRATEGY_FAMILIES)}
        - Blocked families for this iteration: {blocked_families if blocked_families else '(none)'}
        - If blocked families is non-empty, you MUST choose a different `strategy_family`.
        - `strategy_modifiers` should be 0–3 short strings describing the concrete tactic (e.g., gather/hash/addressing).

        Novelty check:
        - Before proposing a plan, read `experiment_log_tail` and avoid repeating the same strategy family/tactic
          unless you explain what is different and why it might work now.

        Use `performance_profile` to reason about lower bounds and when to pivot:
        - `resource_lb_cycles = max(min_cycles_by_engine.values())` is a throughput bound (ignores dependencies).
        - `cp_lb_cycles` is a critical-path bound (ignores engine capacity).
        - `tight_lb_cycles = max(resource_lb_cycles, cp_lb_cycles)` is a better sanity lower bound.
        - If goal=threshold and `threshold_target <= tight_lb_cycles`, scheduling tweaks alone cannot meet the target; the next plan MUST
          reduce the bound (typically fewer tasks on the dominant engine and/or breaking dependency chains).
        - If goal=best and you are plateaued and near the bound (e.g., `schedule_slack_pct <= 0.04`), prioritize plans that reduce the
          bound (dominant-engine task count) over micro-scheduling.

        Output contract:
        - Return ONE JSON object and nothing else (no markdown, no code fences).
        - The object MUST match this JSON Schema (including all required keys):
        ```json
        {schema_json}
        ```

        Here is the current IterationPacket (JSON):
        ```json
        {packet_json}
        ```
        """
    ).strip()


def _planner_directive_schema(*, blocked_families: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    blocked = {str(x) for x in (blocked_families or []) if isinstance(x, str)}
    allowed_families = [f for f in _STRATEGY_FAMILIES if f not in blocked]
    if not allowed_families:
        allowed_families = list(_STRATEGY_FAMILIES)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "objective": {"type": "string"},
            "primary_hypothesis": {"type": "string"},
            "strategy_family": {"type": "string", "enum": allowed_families},
            "strategy_modifiers": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 3,
            },
            "risk": {"type": "string", "enum": ["Low", "Medium", "High"]},
            "expected_effect_cycles": {"type": "integer"},
            "change_summary": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
            },
            "step_plan": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 12,
            },
            "validation": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 6,
                    },
                    "pass_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 6,
                    },
                },
                "required": ["commands", "pass_criteria"],
            },
            "next_packet_requests": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 6,
            },
            "did_web_search": {"type": "boolean"},
        },
        "required": [
            "objective",
            "primary_hypothesis",
            "strategy_family",
            "strategy_modifiers",
            "risk",
            "expected_effect_cycles",
            "change_summary",
            "step_plan",
            "validation",
            "next_packet_requests",
            "did_web_search",
        ],
    }


def _branch_exists(branch: str) -> bool:
    out = _git("branch", "--list", branch, check=False)
    return bool(out.strip())


def _remote_branch_exists(branch: str, *, remote: str = "origin") -> bool:
    # NB: Requires an up-to-date fetch; callers should `git fetch --prune` first.
    out = _git("branch", "-r", "--list", f"{remote}/{branch}", check=False)
    return bool(out.strip())


def _branch_exists_any(branch: str) -> bool:
    return _branch_exists(branch) or _remote_branch_exists(branch)


def _create_plan_branch(*, iteration_id: int, slug: str, base_branch: str, no_pull: bool) -> Tuple[str, str, int]:
    """
    Create a plan/* branch used to prepare a manual planner packet (no API call).

    Returns (branch_name, base_sha, chosen_iteration_id).
    """

    _ensure_clean_worktree()

    _git("fetch", "--prune", "origin")
    _git("checkout", base_branch)
    if not no_pull:
        _git("pull", "--ff-only", "origin", base_branch)

    base_sha = _git("rev-parse", "HEAD")
    chosen = int(iteration_id)
    while True:
        branch = f"plan/{chosen:04d}-{_slugify(slug)}"
        if not _branch_exists_any(branch):
            break
        chosen += 1
    _git("checkout", "-b", branch)
    return branch, base_sha, chosen


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _parse_json_relaxed(text: str) -> Any:
    """
    Parse JSON, tolerating common ChatGPT wrappers like ```json fences.
    """

    stripped = _CODE_FENCE_RE.sub("", text).strip()
    if not stripped:
        raise LoopRunnerError("Empty JSON content.")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        raise LoopRunnerError(f"Invalid JSON: {e}") from e


def _validate_directive(directive: Mapping[str, Any], *, schema: Mapping[str, Any]) -> None:
    props = schema.get("properties")
    required = schema.get("required")
    if not isinstance(props, dict) or not isinstance(required, list):
        raise LoopRunnerError("Internal error: directive schema is malformed.")

    missing = [k for k in required if k not in directive]
    if missing:
        raise LoopRunnerError(f"Directive is missing required keys: {missing}")

    extras = [k for k in directive.keys() if k not in props]
    if extras:
        raise LoopRunnerError(f"Directive has unexpected keys (not allowed): {extras}")

    # Basic type checks for the fields we rely on in logs and execution.
    def _expect_type(key: str, typ: type) -> None:
        v = directive.get(key)
        if not isinstance(v, typ):
            raise LoopRunnerError(f"Directive.{key} must be {typ.__name__}, got {type(v).__name__}.")

    _expect_type("objective", str)
    _expect_type("primary_hypothesis", str)
    _expect_type("risk", str)
    _expect_type("expected_effect_cycles", int)
    _expect_type("did_web_search", bool)

    _expect_type("strategy_family", str)
    strategy_family = (str(directive.get("strategy_family") or "")).strip()
    if not strategy_family:
        raise LoopRunnerError("Directive.strategy_family must be a non-empty string.")
    allowed_families = None
    try:
        raw = schema["properties"]["strategy_family"]["enum"]
        if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
            allowed_families = [str(x) for x in raw]
    except Exception:
        allowed_families = None
    if not allowed_families:
        allowed_families = list(_STRATEGY_FAMILIES)
    if strategy_family not in set(allowed_families):
        raise LoopRunnerError(
            "Directive.strategy_family must be one of: " + ", ".join([repr(x) for x in allowed_families])
        )

    strategy_modifiers = directive.get("strategy_modifiers")
    if not isinstance(strategy_modifiers, list) or not all(isinstance(x, str) for x in strategy_modifiers):
        raise LoopRunnerError("Directive.strategy_modifiers must be an array of strings.")
    if len(strategy_modifiers) > 3:
        raise LoopRunnerError("Directive.strategy_modifiers must have 0..3 items.")
    if any(not (s or "").strip() for s in strategy_modifiers):
        raise LoopRunnerError("Directive.strategy_modifiers must not contain empty strings.")

    change_summary = directive.get("change_summary")
    if not isinstance(change_summary, list) or not all(isinstance(x, str) for x in change_summary):
        raise LoopRunnerError("Directive.change_summary must be an array of strings.")
    if not (1 <= len(change_summary) <= 5):
        raise LoopRunnerError("Directive.change_summary must have 1..5 items.")

    step_plan = directive.get("step_plan")
    if not isinstance(step_plan, list) or not all(isinstance(x, str) for x in step_plan):
        raise LoopRunnerError("Directive.step_plan must be an array of strings.")
    if not (3 <= len(step_plan) <= 12):
        raise LoopRunnerError("Directive.step_plan must have 3..12 items.")

    validation = directive.get("validation")
    if not isinstance(validation, dict):
        raise LoopRunnerError("Directive.validation must be an object.")
    commands = validation.get("commands")
    pass_criteria = validation.get("pass_criteria")
    if not isinstance(commands, list) or not all(isinstance(x, str) for x in commands):
        raise LoopRunnerError("Directive.validation.commands must be an array of strings.")
    if not isinstance(pass_criteria, list) or not all(isinstance(x, str) for x in pass_criteria):
        raise LoopRunnerError("Directive.validation.pass_criteria must be an array of strings.")

    next_packet_requests = directive.get("next_packet_requests")
    if not isinstance(next_packet_requests, list) or not all(isinstance(x, str) for x in next_packet_requests):
        raise LoopRunnerError("Directive.next_packet_requests must be an array of strings.")
    if len(next_packet_requests) > 6:
        raise LoopRunnerError("Directive.next_packet_requests must have 0..6 items.")

    risk = str(directive.get("risk") or "")
    if risk not in ("Low", "Medium", "High"):
        raise LoopRunnerError("Directive.risk must be one of: Low, Medium, High.")


def _create_iteration_branch(
    *, iteration_id: int, slug: str, base_branch: str, no_pull: bool
) -> Tuple[str, str, int]:
    _ensure_clean_worktree()

    _git("fetch", "--prune", "origin")
    _git("checkout", base_branch)
    if not no_pull:
        _git("pull", "--ff-only", "origin", base_branch)

    base_sha = _git("rev-parse", "HEAD")
    chosen = int(iteration_id)
    while True:
        branch = f"iter/{chosen:04d}-{_slugify(slug)}"
        if not _branch_exists_any(branch):
            break
        chosen += 1
    _git("checkout", "-b", branch)
    return branch, base_sha, chosen


def _directive_looks_complete(directive: Mapping[str, Any]) -> bool:
    if not directive:
        return False
    step_plan = directive.get("step_plan")
    return isinstance(step_plan, list) and len(step_plan) >= 1


def _build_codex_exec_env() -> Dict[str, str]:
    env = dict(os.environ)
    if not env.get("CODEX_HOME"):
        candidate = _REPO_ROOT / ".codex_home"
        if candidate.exists():
            env["CODEX_HOME"] = str(candidate)
    return env


def _run_codex_exec_for_planner(
    *,
    prompt: str,
    output_schema_path: Path,
    output_last_message_path: Path,
    model: Optional[str],
) -> subprocess.CompletedProcess:
    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise LoopRunnerError("Could not find `codex` on PATH; required for codex-plan mode.")

    argv = [
        codex_bin,
        "exec",
        "-C",
        str(_REPO_ROOT),
        "-s",
        "read-only",
        "-c",
        'approval_policy="never"',
        "--color",
        "never",
        "--output-schema",
        str(output_schema_path),
        "--output-last-message",
        str(output_last_message_path),
    ]
    if model:
        argv.extend(["-m", model])
    # Read prompt from stdin.
    argv.append("-")

    return subprocess.run(  # noqa: S603,S607
        argv,
        input=prompt,
        cwd=str(_REPO_ROOT),
        env=_build_codex_exec_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _checkout_or_create_branch(branch: str, *, base_sha: str) -> None:
    current = _git("branch", "--show-current", check=False)
    if current and current == branch:
        return

    exists = bool(_git("rev-parse", "--verify", branch, check=False))
    if exists:
        _git("checkout", branch)
        return

    _git("checkout", "-b", branch, base_sha)


def _resume_planner_response(*, state: IterationState, model: str) -> Dict[str, Any]:
    if not state.advisor_response_id:
        raise LoopRunnerError("No advisor_response_id in state; cannot resume.")

    _ensure_clean_worktree()
    _git("fetch", "origin", check=False)
    _checkout_or_create_branch(state.branch, base_sha=state.base_sha)

    if not os.environ.get("OPENAI_API_KEY"):
        _load_dotenv_api_key(_REPO_ROOT / ".env")
    _enforce_default_poll_cadence()

    cfg = OpenAIExecConfig.from_env()
    client = OpenAIExec(cfg)

    response_id = state.advisor_response_id
    head = client.retrieve_response(response_id=response_id)
    status = head.get("status") if isinstance(head.get("status"), str) else None
    if status in ("queued", "in_progress"):
        response_json = client.poll_response(response_id=response_id, initial_status=status)
    else:
        response_json = head

    tool_name = "emit_optimization_directive"
    blocked_families = None
    sfc = state.packet.get("strategy_family_constraints")
    if isinstance(sfc, dict) and isinstance(sfc.get("blocked_families"), list):
        blocked_families = sfc.get("blocked_families")
    payload = build_payload_for_planner(
        model=model,
        prompt=_build_planner_prompt(state.packet),
        directive_schema=_planner_directive_schema(blocked_families=blocked_families),
        directive_tool_name=tool_name,
    )
    stem = f"iter_{state.iteration_id:04d}" + (f"_{response_id}" if response_id else "")
    req_path, resp_path = write_response_artifacts(
        dir_path=_OPENAI_ARTIFACTS_DIR,
        stem=stem,
        request_payload=payload,
        response_json=response_json,
    )
    print(
        f"[loop_runner] saved OpenAI artifacts: {req_path.relative_to(_REPO_ROOT)} "
        f"{resp_path.relative_to(_REPO_ROOT)}",
        file=sys.stderr,
    )

    directive = extract_function_call_arguments(response_json, function_name=tool_name)
    _write_state(
        IterationState(
            iteration_id=state.iteration_id,
            branch=state.branch,
            base_branch=state.base_branch,
            base_sha=state.base_sha,
            threshold_target=state.threshold_target,
            packet=state.packet,
            directive=directive,
            advisor_response_id=response_id,
        )
    )
    return directive


def cmd_plan(args: argparse.Namespace) -> int:
    if _STATE_PATH.exists():
        try:
            existing = _read_state()
        except Exception:
            existing = None
        if (
            existing
            and existing.advisor_response_id
            and not _directive_looks_complete(existing.directive)
            and not args.offline
        ):
            print(
                f"[loop_runner] resuming in-progress planner response id={existing.advisor_response_id} "
                f"for {existing.branch!r}",
                file=sys.stderr,
            )
            directive = _resume_planner_response(state=existing, model=str(existing.packet.get("advisor_model") or args.model))
            print(json.dumps(directive, indent=2, sort_keys=True))
            return 0

    _ensure_experiment_log_exists()
    entries = _read_jsonl(_EXPERIMENT_LOG_PATH)
    iteration_id = max(_next_iteration_id(entries), _next_iteration_id_from_local_branches())
    best = _best_cycles(entries)
    plateau = _plateau_valid_iters_since_best(entries)
    family_constraints = _compute_strategy_family_constraints(entries)
    goal = str(args.goal or "threshold")
    if goal not in _GOALS:
        goal = "threshold"
    threshold_target: Optional[int] = None
    if goal == "threshold" and args.threshold is not None:
        threshold_target = int(args.threshold)
    aspiration_cycles: Optional[int] = None
    if goal == "best" and best is not None:
        aspiration_cycles = int(best) - 1

    if args.no_branch:
        branch = _git("branch", "--show-current", check=False) or "detached"
        base_sha = _git("rev-parse", "HEAD", check=False)
        if not base_sha:
            raise LoopRunnerError("Could not determine HEAD SHA.")
    else:
        branch, base_sha, chosen_iteration_id = _create_iteration_branch(
            iteration_id=iteration_id,
            slug=args.slug,
            base_branch=args.base_branch,
            no_pull=bool(args.no_pull),
        )
        iteration_id = chosen_iteration_id

    code_context: Dict[str, str] = {}
    if args.code_context == "kernelbuilder":
        code_context["perf_takehome.py#KernelBuilder"] = _extract_kernelbuilder_source()
    elif args.code_context == "full":
        code_context["perf_takehome.py"] = (_REPO_ROOT / "perf_takehome.py").read_text(
            encoding="utf-8", errors="replace"
        )

    tail = _tail_lines(_EXPERIMENT_LOG_PATH, n=int(args.experiment_log_tail_lines))

    origin_url = _origin_remote_url()
    github_web = _github_web_url(origin_url) if isinstance(origin_url, str) else None

    packet: Dict[str, Any] = {
        "iteration_id": iteration_id,
        "timestamp_utc": _utc_now_iso(),
        "branch": branch,
        "base_branch": args.base_branch,
        "base_sha": base_sha,
        "repo": {
            "origin_url": origin_url,
            "github_web_url": github_web,
            "base_sha": base_sha,
            "worktree_path": str(_REPO_ROOT),
        },
        "goal": goal,
        "threshold_target": threshold_target,
        "best_cycles": best,
        "aspiration_cycles": aspiration_cycles,
        "plateau_valid_iters_since_best": plateau,
        "constraints": {
            "allowed_paths": list(_DEFAULT_ALLOWED_PATHS),
            "forbidden_globs": list(_DEFAULT_FORBIDDEN_GLOBS),
            "notes": [
                "Never modify tests/ or benchmark semantics.",
                "Prefer changes in perf_takehome.py only.",
            ],
        },
        "strategy_family_constraints": family_constraints,
        "experiment_log_tail": tail,
        "code_context": code_context,
        "advisor_model": args.model,
    }
    packet["performance_profile"] = _compute_performance_profile_for_submission_case()

    prompt = _build_planner_prompt(packet)

    if not os.environ.get("OPENAI_API_KEY"):
        _load_dotenv_api_key(_REPO_ROOT / ".env")
    _enforce_default_poll_cadence()

    tool_name = "emit_optimization_directive"
    response_id: Optional[str] = None
    if args.offline:
        directive = {
            "objective": "Offline directive (hermetic test)",
            "primary_hypothesis": "This mode validates the loop runner without making OpenAI API calls.",
            "strategy_family": "family:schedule",
            "strategy_modifiers": ["offline"],
            "risk": "Low",
            "expected_effect_cycles": 0,
            "change_summary": ["No-op; offline directive"],
            "step_plan": [
                "Confirm loop runner can create state without network calls",
                "Make a small change in perf_takehome.py",
                "Run record to benchmark and append to experiments/log.jsonl",
            ],
            "validation": {
                "commands": ["git diff origin/main tests/", "python3 -B tests/submission_tests.py"],
                "pass_criteria": ["tests diff is empty", "submission tests complete"],
            },
            "next_packet_requests": [],
            "did_web_search": False,
        }
    else:
        cfg = OpenAIExecConfig.from_env()
        client = OpenAIExec(cfg)

        payload = build_payload_for_planner(
            model=args.model,
            prompt=prompt,
            directive_schema=_planner_directive_schema(blocked_families=family_constraints.get("blocked_families")),
            directive_tool_name=tool_name,
        )

        max_attempts = int(os.environ.get("OPENAI_PLANNER_MAX_ATTEMPTS", "1"))
        last_err: Optional[BaseException] = None
        response_json: Dict[str, Any]
        for attempt in range(1, max_attempts + 1):
            response_id = None
            try:
                post_json = client.start_response(**payload)
                response_id = post_json.get("id") if isinstance(post_json.get("id"), str) else None
                if not response_id:
                    raise OpenAIExecResponseError("Planner POST missing id.", response_id=None)

                stem = f"iter_{iteration_id:04d}" + (f"_{response_id}" if response_id else "")
                write_response_artifacts(
                    dir_path=_OPENAI_ARTIFACTS_DIR,
                    stem=stem,
                    request_payload=payload,
                    response_json=post_json,
                )
                _write_state(
                    IterationState(
                        iteration_id=iteration_id,
                        branch=branch,
                        base_branch=args.base_branch,
                        base_sha=base_sha,
                        threshold_target=threshold_target,
                        packet=packet,
                        directive={},
                        advisor_response_id=response_id,
                    )
                )

                response_json = client.poll_response(
                    response_id=response_id,
                    initial_status=(post_json.get("status") if isinstance(post_json.get("status"), str) else None),
                )
                req_path, resp_path = write_response_artifacts(
                    dir_path=_OPENAI_ARTIFACTS_DIR,
                    stem=stem,
                    request_payload=payload,
                    response_json=response_json,
                )
                print(
                    f"[loop_runner] saved OpenAI artifacts: {req_path.relative_to(_REPO_ROOT)} "
                    f"{resp_path.relative_to(_REPO_ROOT)}",
                    file=sys.stderr,
                )

                directive = extract_function_call_arguments(response_json, function_name=tool_name)
                break
            except OpenAIExecTimeoutError as e:
                last_err = e
                if response_id:
                    print(
                        f"[loop_runner] planner response still running (id={response_id}). "
                        "Run `python3 tools/loop_runner.py resume` to continue polling.",
                        file=sys.stderr,
                    )
                raise
            except (OpenAIExecError, OpenAIExecResponseError) as e:
                last_err = e
                if response_id:
                    try:
                        failure_json = client.retrieve_response(
                            response_id=response_id,
                            raise_on_terminal=False,
                        )
                        stem = f"iter_{iteration_id:04d}" + (f"_{response_id}" if response_id else "")
                        req_path, resp_path = write_response_artifacts(
                            dir_path=_OPENAI_ARTIFACTS_DIR,
                            stem=stem,
                            request_payload=payload,
                            response_json=failure_json,
                        )
                        print(
                            f"[loop_runner] saved OpenAI failure artifacts: {req_path.relative_to(_REPO_ROOT)} "
                            f"{resp_path.relative_to(_REPO_ROOT)}",
                            file=sys.stderr,
                        )
                    except Exception as artifact_err:
                        print(
                            f"[loop_runner] warning: failed to fetch/persist OpenAI failure response id={response_id}: "
                            f"{artifact_err}",
                            file=sys.stderr,
                        )
                if attempt >= max_attempts:
                    raise
                print(
                    f"[loop_runner] planner call failed (attempt {attempt}/{max_attempts}); retrying: {e}",
                    file=sys.stderr,
                )
                time.sleep(min(60.0, 5.0 * attempt))
        else:
            raise LoopRunnerError(f"Planner call failed after {max_attempts} attempts: {last_err}")

    _write_state(
        IterationState(
            iteration_id=iteration_id,
            branch=branch,
            base_branch=args.base_branch,
            base_sha=base_sha,
            threshold_target=threshold_target,
            packet=packet,
            directive=directive,
            advisor_response_id=response_id,
        )
    )

    print(json.dumps(directive, indent=2, sort_keys=True))
    return 0


def cmd_codex_plan(args: argparse.Namespace) -> int:
    """
    Create an iter/* branch and request a plan from Codex CLI (no direct OpenAI API calls).

    This mode spawns `codex exec` in a read-only sandbox and expects a JSON object matching the
    OptimizationDirective schema.
    """

    _ensure_experiment_log_exists()
    entries = _read_jsonl(_EXPERIMENT_LOG_PATH)
    iteration_id = max(_next_iteration_id(entries), _next_iteration_id_from_local_branches())
    best = _best_cycles(entries)
    plateau = _plateau_valid_iters_since_best(entries)
    family_constraints = _compute_strategy_family_constraints(entries)
    goal = str(args.goal or "threshold")
    if goal not in _GOALS:
        goal = "threshold"
    threshold_target: Optional[int] = None
    if goal == "threshold" and args.threshold is not None:
        threshold_target = int(args.threshold)
    aspiration_cycles: Optional[int] = None
    if goal == "best" and best is not None:
        aspiration_cycles = int(best) - 1

    if args.no_branch:
        branch = _git("branch", "--show-current", check=False) or "detached"
        base_sha = _git("rev-parse", "HEAD", check=False)
        if not base_sha:
            raise LoopRunnerError("Could not determine HEAD SHA.")
    else:
        branch, base_sha, chosen_iteration_id = _create_iteration_branch(
            iteration_id=iteration_id,
            slug=args.slug,
            base_branch=args.base_branch,
            no_pull=bool(args.no_pull),
        )
        iteration_id = chosen_iteration_id

    code_context: Dict[str, str] = {}
    if args.code_context == "kernelbuilder":
        code_context["perf_takehome.py#KernelBuilder"] = _extract_kernelbuilder_source()
    elif args.code_context == "full":
        code_context["perf_takehome.py"] = (_REPO_ROOT / "perf_takehome.py").read_text(encoding="utf-8", errors="replace")

    tail = _tail_lines(_EXPERIMENT_LOG_PATH, n=int(args.experiment_log_tail_lines))

    origin_url = _origin_remote_url()
    github_web = _github_web_url(origin_url) if isinstance(origin_url, str) else None

    advisor_model = args.model or "codex-default"
    packet: Dict[str, Any] = {
        "iteration_id": iteration_id,
        "timestamp_utc": _utc_now_iso(),
        "branch": branch,
        "base_branch": args.base_branch,
        "base_sha": base_sha,
        "repo": {
            "origin_url": origin_url,
            "github_web_url": github_web,
            "base_sha": base_sha,
            "worktree_path": str(_REPO_ROOT),
        },
        "goal": goal,
        "threshold_target": threshold_target,
        "best_cycles": best,
        "aspiration_cycles": aspiration_cycles,
        "plateau_valid_iters_since_best": plateau,
        "constraints": {
            "allowed_paths": list(_DEFAULT_ALLOWED_PATHS),
            "forbidden_globs": list(_DEFAULT_FORBIDDEN_GLOBS),
            "notes": [
                "Never modify tests/ or benchmark semantics.",
                "Prefer changes in perf_takehome.py only.",
            ],
        },
        "strategy_family_constraints": family_constraints,
        "experiment_log_tail": tail,
        "code_context": code_context,
        "advisor_model": advisor_model,
    }
    packet["performance_profile"] = _compute_performance_profile_for_submission_case()

    blocked_families = None
    if isinstance(family_constraints, dict):
        raw = family_constraints.get("blocked_families")
        if isinstance(raw, list):
            blocked_families = [x for x in raw if isinstance(x, str)]
    schema = _planner_directive_schema(blocked_families=blocked_families)
    prompt = _build_codex_planner_prompt(packet, directive_schema=schema)

    _CODEX_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"iter_{iteration_id:04d}"
    packet_path = _CODEX_ARTIFACTS_DIR / f"{stem}_packet.json"
    schema_path = _CODEX_ARTIFACTS_DIR / f"{stem}_schema.json"
    prompt_path = _CODEX_ARTIFACTS_DIR / f"{stem}_prompt.md"
    stdout_path = _CODEX_ARTIFACTS_DIR / f"{stem}_stdout.txt"
    stderr_path = _CODEX_ARTIFACTS_DIR / f"{stem}_stderr.txt"
    cmd_path = _CODEX_ARTIFACTS_DIR / f"{stem}_cmd.txt"
    output_last_message_path = _CODEX_ARTIFACTS_DIR / f"{stem}_last_message.txt"

    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    proc = _run_codex_exec_for_planner(
        prompt=prompt,
        output_schema_path=schema_path,
        output_last_message_path=output_last_message_path,
        model=(str(args.model) if args.model else None),
    )
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    cmd_path.write_text(" ".join(proc.args) + "\n", encoding="utf-8")  # type: ignore[arg-type]

    if proc.returncode != 0:
        raise LoopRunnerError(
            "codex exec failed.\n"
            f"- cmd: {cmd_path.relative_to(_REPO_ROOT)}\n"
            f"- stdout: {stdout_path.relative_to(_REPO_ROOT)}\n"
            f"- stderr: {stderr_path.relative_to(_REPO_ROOT)}"
        )

    if not output_last_message_path.exists():
        raise LoopRunnerError(
            "codex exec completed but did not write --output-last-message.\n"
            f"- expected: {output_last_message_path.relative_to(_REPO_ROOT)}"
        )

    raw_last = output_last_message_path.read_text(encoding="utf-8", errors="replace")
    directive_obj = _parse_json_relaxed(raw_last)
    if not isinstance(directive_obj, dict):
        raise LoopRunnerError("Codex planner output must be a JSON object.")
    _validate_directive(directive_obj, schema=schema)

    _write_state(
        IterationState(
            iteration_id=iteration_id,
            branch=branch,
            base_branch=args.base_branch,
            base_sha=base_sha,
            threshold_target=threshold_target,
            packet=packet,
            directive=dict(directive_obj),
            advisor_response_id=None,
        )
    )

    print(json.dumps(directive_obj, indent=2, sort_keys=True))
    return 0


_PLAN_BRANCH_RE = re.compile(r"^plan/(\d+)-(.+)$")


def _parse_plan_branch(branch: str) -> Optional[Tuple[int, str]]:
    m = _PLAN_BRANCH_RE.match(branch.strip())
    if not m:
        return None
    try:
        iid = int(m.group(1))
    except ValueError:
        return None
    slug = m.group(2)
    return iid, slug


def cmd_manual_pack(args: argparse.Namespace) -> int:
    """
    Create a plan/* branch and write a ChatGPT-ready packet + prompt.

    This avoids OpenAI API calls entirely: you paste the prompt into ChatGPT (gpt-5.2-pro),
    then paste the JSON directive into planner_packets/directive.json.
    """

    _ensure_experiment_log_exists()
    entries = _read_jsonl(_EXPERIMENT_LOG_PATH)
    iteration_id = max(_next_iteration_id(entries), _next_iteration_id_from_local_branches())
    best = _best_cycles(entries)
    plateau = _plateau_valid_iters_since_best(entries)
    family_constraints = _compute_strategy_family_constraints(entries)
    goal = str(args.goal or "threshold")
    if goal not in _GOALS:
        goal = "threshold"
    threshold_target: Optional[int] = None
    if goal == "threshold" and args.threshold is not None:
        threshold_target = int(args.threshold)
    aspiration_cycles: Optional[int] = None
    if goal == "best" and best is not None:
        aspiration_cycles = int(best) - 1

    branch, base_sha, chosen_iteration_id = _create_plan_branch(
        iteration_id=iteration_id,
        slug=args.slug,
        base_branch=args.base_branch,
        no_pull=bool(args.no_pull),
    )

    code_context: Dict[str, str] = {}
    if args.code_context == "kernelbuilder":
        code_context["perf_takehome.py#KernelBuilder"] = _extract_kernelbuilder_source()
    elif args.code_context == "full":
        code_context["perf_takehome.py"] = (_REPO_ROOT / "perf_takehome.py").read_text(
            encoding="utf-8", errors="replace"
        )

    tail = _tail_lines(_EXPERIMENT_LOG_PATH, n=int(args.experiment_log_tail_lines))

    origin_url = _origin_remote_url()
    github_web = _github_web_url(origin_url) if isinstance(origin_url, str) else None

    packet: Dict[str, Any] = {
        "iteration_id": chosen_iteration_id,
        "timestamp_utc": _utc_now_iso(),
        "branch": branch,
        "base_branch": args.base_branch,
        "base_sha": base_sha,
        "repo": {
            "origin_url": origin_url,
            "github_web_url": github_web,
            "base_sha": base_sha,
            "worktree_path": str(_REPO_ROOT),
        },
        "goal": goal,
        "threshold_target": threshold_target,
        "best_cycles": best,
        "aspiration_cycles": aspiration_cycles,
        "plateau_valid_iters_since_best": plateau,
        "constraints": {
            "allowed_paths": list(_DEFAULT_ALLOWED_PATHS),
            "forbidden_globs": list(_DEFAULT_FORBIDDEN_GLOBS),
            "notes": [
                "Never modify tests/ or benchmark semantics.",
                "Prefer changes in perf_takehome.py only.",
            ],
        },
        "strategy_family_constraints": family_constraints,
        "experiment_log_tail": tail,
        "code_context": code_context,
        "advisor_model": "gpt-5.2-pro",
    }
    packet["performance_profile"] = _compute_performance_profile_for_submission_case()

    blocked_families = None
    if isinstance(family_constraints, dict):
        raw = family_constraints.get("blocked_families")
        if isinstance(raw, list):
            blocked_families = [x for x in raw if isinstance(x, str)]
    schema = _planner_directive_schema(blocked_families=blocked_families)
    prompt = _build_manual_planner_prompt(packet, directive_schema=schema)

    _MANUAL_PACKET_DIR.mkdir(parents=True, exist_ok=True)
    _MANUAL_PACKET_PATH.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _MANUAL_SCHEMA_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _MANUAL_PROMPT_PATH.write_text(prompt + "\n", encoding="utf-8")
    if not _MANUAL_DIRECTIVE_PATH.exists():
        _MANUAL_DIRECTIVE_PATH.write_text("{}\n", encoding="utf-8")

    _git("add", str(_MANUAL_PACKET_DIR))
    _git("commit", "-m", "chore: prepare manual planner packet")

    print(f"[loop_runner] manual planner packet written to {str(_MANUAL_PACKET_DIR.relative_to(_REPO_ROOT))}/")
    print(f"[loop_runner] paste {str(_MANUAL_PROMPT_PATH.relative_to(_REPO_ROOT))} into ChatGPT")
    print(
        f"[loop_runner] paste ChatGPT JSON into {str(_MANUAL_DIRECTIVE_PATH.relative_to(_REPO_ROOT))} and commit it"
    )
    print("[loop_runner] then run: python3 tools/loop_runner.py manual-apply")
    return 0


def cmd_manual_apply(args: argparse.Namespace) -> int:
    """
    Read planner_packets/directive.json and create an iter/* branch + .advisor/state.json.
    """

    if args.from_ref:
        packet_text = _git_show_text(args.from_ref, _MANUAL_PACKET_PATH)
        prompt_text = _git_show_text(args.from_ref, _MANUAL_PROMPT_PATH)
        raw_directive = _git_show_text(args.from_ref, _MANUAL_DIRECTIVE_PATH)
    else:
        if not _MANUAL_PACKET_PATH.exists() or not _MANUAL_PROMPT_PATH.exists() or not _MANUAL_DIRECTIVE_PATH.exists():
            raise LoopRunnerError(
                "Missing manual planner files. Run `python3 tools/loop_runner.py manual-pack` first and commit the packet."
            )
        packet_text = _MANUAL_PACKET_PATH.read_text(encoding="utf-8", errors="replace")
        prompt_text = _MANUAL_PROMPT_PATH.read_text(encoding="utf-8", errors="replace")
        raw_directive = _MANUAL_DIRECTIVE_PATH.read_text(encoding="utf-8", errors="replace")

    packet = _parse_json_relaxed(packet_text)
    if not isinstance(packet, dict):
        raise LoopRunnerError("planner_packets/packet.json must be a JSON object.")

    directive_obj = _parse_json_relaxed(raw_directive)
    if not isinstance(directive_obj, dict):
        raise LoopRunnerError("planner_packets/directive.json must be a JSON object.")

    sfc = packet.get("strategy_family_constraints")
    blocked_families = None
    if isinstance(sfc, dict):
        raw = sfc.get("blocked_families")
        if isinstance(raw, list):
            blocked_families = [x for x in raw if isinstance(x, str)]
    schema = _planner_directive_schema(blocked_families=blocked_families)
    _validate_directive(directive_obj, schema=schema)

    base_branch = str(packet.get("base_branch") or args.base_branch or "main")
    base_sha_expected = str(packet.get("base_sha") or "")
    if not base_sha_expected:
        raise LoopRunnerError("planner_packets/packet.json is missing base_sha.")

    _git("fetch", "--prune", "origin")
    base_sha_origin = _git("rev-parse", f"origin/{base_branch}", check=False)
    if base_sha_origin and base_sha_origin != base_sha_expected and not bool(args.allow_stale_base):
        raise LoopRunnerError(
            "Base branch advanced since manual-pack.\n"
            f"- base_branch: {base_branch}\n"
            f"- packet base_sha: {base_sha_expected}\n"
            f"- origin/{base_branch} now: {base_sha_origin}\n"
            "Rerun manual-pack to regenerate the packet, or pass --allow-stale-base to proceed anyway."
        )

    branchish_for_slug = args.from_ref or (_git("branch", "--show-current", check=False) or "")
    # Allow parsing plan slugs from refs like origin/plan/0001-next or refs/heads/plan/0001-next.
    branchish_for_slug = re.sub(r"^(?:refs/(?:heads|remotes)/)", "", branchish_for_slug.strip())
    if branchish_for_slug.startswith("origin/"):
        branchish_for_slug = branchish_for_slug[len("origin/") :]
    parsed = _parse_plan_branch(branchish_for_slug)

    try:
        iteration_id = int(packet.get("iteration_id", 0))  # type: ignore[arg-type]
    except Exception:
        raise LoopRunnerError("planner_packets/packet.json is missing a valid iteration_id.")
    if iteration_id <= 0:
        raise LoopRunnerError("planner_packets/packet.json has invalid iteration_id.")

    slug = args.slug if args.slug is not None else (parsed[1] if parsed else "manual")

    iter_branch, base_sha, chosen_iteration_id = _create_iteration_branch(
        iteration_id=iteration_id,
        slug=slug,
        base_branch=base_branch,
        no_pull=bool(args.no_pull),
    )
    iteration_id = chosen_iteration_id

    packet = dict(packet)
    packet["iteration_id"] = iteration_id
    packet["timestamp_utc"] = _utc_now_iso()
    packet["branch"] = iter_branch
    packet["base_branch"] = base_branch
    packet["base_sha"] = base_sha

    # Preserve a local archive that survives branch switching/deletion.
    archive_dir = _STATE_DIR / "manual_archive" / f"iter_{iteration_id:04d}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "packet.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (archive_dir / "directive.json").write_text(
        json.dumps(directive_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (archive_dir / "prompt.md").write_text(prompt_text, encoding="utf-8")
    if args.from_ref:
        (archive_dir / "from_ref.txt").write_text(args.from_ref.strip() + "\n", encoding="utf-8")

    _write_state(
        IterationState(
            iteration_id=iteration_id,
            branch=iter_branch,
            base_branch=base_branch,
            base_sha=base_sha,
            threshold_target=(
                int(packet.get("threshold_target")) if packet.get("threshold_target") is not None else None
            ),
            packet=packet,
            directive=dict(directive_obj),
            advisor_response_id=None,
        )
    )

    print(json.dumps(directive_obj, indent=2, sort_keys=True))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    state = _read_state()
    if _directive_looks_complete(state.directive):
        print(json.dumps(state.directive, indent=2, sort_keys=True))
        return 0

    directive = _resume_planner_response(
        state=state,
        model=str(state.packet.get("advisor_model") or "gpt-5.2-pro"),
    )
    print(json.dumps(directive, indent=2, sort_keys=True))
    return 0


def _git_diff_tests_is_empty() -> Tuple[bool, str]:
    _git("fetch", "origin", check=False)
    diff = _git("diff", "origin/main", "tests/", check=False)
    return (diff.strip() == "", diff)


def _compute_changed_files(base_sha: str) -> List[str]:
    names: List[str] = []

    proc = _run(["git", "diff", "--name-only", f"{base_sha}..HEAD"], check=False, capture=True)
    names.extend([l.strip() for l in (proc.stdout or "").splitlines() if l.strip()])

    proc = _run(["git", "diff", "--name-only"], check=False, capture=True)
    names.extend([l.strip() for l in (proc.stdout or "").splitlines() if l.strip()])

    proc = _run(["git", "ls-files", "--others", "--exclude-standard"], check=False, capture=True)
    names.extend([l.strip() for l in (proc.stdout or "").splitlines() if l.strip()])

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def cmd_record(args: argparse.Namespace) -> int:
    _ensure_experiment_log_exists()
    state = _read_state()

    entries = _read_jsonl(_EXPERIMENT_LOG_PATH)
    best_before = _best_cycles(entries)

    current_branch = _git("branch", "--show-current", check=False)
    if current_branch and current_branch != state.branch:
        print(
            f"[loop_runner] warning: current branch {current_branch!r} != state branch {state.branch!r}",
            file=sys.stderr,
        )

    tests_diff_empty, tests_diff = _git_diff_tests_is_empty()
    if not tests_diff_empty:
        print("[loop_runner] ERROR: tests/ changed vs origin/main. This is forbidden.", file=sys.stderr)
        print(tests_diff, file=sys.stderr)

    proc = _run(["python3", "-B", "tests/submission_tests.py"], check=False, capture=True)
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if args.print_test_output:
        print(combined.rstrip())

    cycles = parse_cycles_from_submission_tests(combined)
    correctness_pass = parse_correctness_from_submission_tests(combined)
    valid = (correctness_pass is True) and tests_diff_empty

    head_sha = _git("rev-parse", "HEAD", check=False) or "(unknown)"
    dirty = bool(_git("status", "--porcelain=v1", check=False).strip())
    if dirty:
        # Keep head_sha machine-friendly; note dirtiness separately.
        dirty_note = "working tree dirty (uncommitted changes); head_sha refers to last commit"
    else:
        dirty_note = ""

    files_changed = _compute_changed_files(state.base_sha)

    constraints = state.packet.get("constraints")
    allowed_paths = list(_DEFAULT_ALLOWED_PATHS)
    forbidden_globs = list(_DEFAULT_FORBIDDEN_GLOBS)
    if isinstance(constraints, dict):
        raw_allowed = constraints.get("allowed_paths")
        if isinstance(raw_allowed, list) and raw_allowed:
            allowed_paths = [str(x) for x in raw_allowed if isinstance(x, str) and x.strip()]
        raw_forbidden = constraints.get("forbidden_globs")
        if isinstance(raw_forbidden, list) and raw_forbidden:
            forbidden_globs = [str(x) for x in raw_forbidden if isinstance(x, str) and x.strip()]

    forbidden_files: List[str] = []
    for path in files_changed:
        p = PurePosixPath(path)
        if any(p.match(glob) for glob in forbidden_globs):
            forbidden_files.append(path)

    disallowed_files = [p for p in files_changed if p not in allowed_paths]
    scope_ok = not forbidden_files and not disallowed_files
    if not scope_ok:
        if forbidden_files:
            print(
                "[loop_runner] ERROR: forbidden files changed (scope violation):\n"
                + "\n".join(f"- {p}" for p in forbidden_files),
                file=sys.stderr,
            )
        if disallowed_files:
            print(
                "[loop_runner] ERROR: files changed outside allowlist (scope violation):\n"
                + "\n".join(f"- {p}" for p in disallowed_files),
                file=sys.stderr,
            )
        print(f"[loop_runner] allowed_paths: {allowed_paths}", file=sys.stderr)
        print(f"[loop_runner] forbidden_globs: {forbidden_globs}", file=sys.stderr)

    delta_vs_best: Optional[int] = None
    if cycles is not None and best_before is not None:
        delta_vs_best = cycles - best_before

    directive = state.directive
    strategy_tags = _strategy_tags_from_directive(directive)
    hypothesis = str(directive.get("primary_hypothesis") or "")
    change_summary = list(directive.get("change_summary") or [])
    objective = str(directive.get("objective") or "")

    if correctness_pass is True and cycles is not None:
        if state.threshold_target is not None and cycles > state.threshold_target:
            result_summary = f"PASS correctness; cycles={cycles} (above target {state.threshold_target})"
        else:
            result_summary = f"PASS correctness; cycles={cycles}"
    elif correctness_pass is False and cycles is not None:
        result_summary = f"FAIL correctness; cycles={cycles}"
    elif correctness_pass is False:
        result_summary = "FAIL correctness; cycles unavailable"
    elif cycles is not None:
        result_summary = f"UNKNOWN correctness; cycles={cycles}"
    else:
        result_summary = "UNKNOWN correctness; cycles unavailable"

    notes_parts = [x for x in [dirty_note, objective] if x]
    notes = " | ".join(notes_parts) if notes_parts else None

    valid = bool(valid and scope_ok)

    entry: Dict[str, Any] = {
        "iteration_id": state.iteration_id,
        "timestamp_utc": _utc_now_iso(),
        "branch": state.branch,
        "base_branch": state.base_branch,
        "base_sha": state.base_sha,
        "head_sha": head_sha,
        "files_changed": files_changed,
        "scope_ok": scope_ok,
        "scope_disallowed_files": disallowed_files,
        "scope_forbidden_files": forbidden_files,
        "tests_diff_empty": tests_diff_empty,
        "correctness_pass": correctness_pass,
        "valid": valid,
        "cycles": cycles,
        "delta_vs_best": delta_vs_best,
        "strategy_tags": strategy_tags,
        "hypothesis": hypothesis,
        "change_summary": change_summary,
        "result_summary": result_summary,
        "merged_to_main": False,
    }
    if notes:
        entry["notes"] = notes

    with _EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")

    # Human-readable recap.
    print(json.dumps(entry, indent=2, sort_keys=True))

    if valid and cycles is not None and best_before is not None and cycles < best_before:
        print(f"[loop_runner] NEW BEST: {cycles} cycles (prev best {best_before})")
    if state.threshold_target is not None and valid and cycles is not None and cycles <= state.threshold_target:
        print(f"[loop_runner] THRESHOLD MET: cycles={cycles} <= target={state.threshold_target}")

    return 0 if valid else 1


def _latest_entry_for_iteration(entries: Sequence[Mapping[str, Any]], *, iteration_id: int) -> Optional[Mapping[str, Any]]:
    for e in reversed(entries):
        try:
            if int(e.get("iteration_id", 0)) == int(iteration_id):  # type: ignore[arg-type]
                return e
        except Exception:
            continue
    return None


def cmd_status(args: argparse.Namespace) -> int:
    """
    Print a concise per-iteration status update for humans running long loops.

    Intended usage:
    - After planning (to see what we're attempting)
    - After record (to see the outcome)
    """

    _ensure_experiment_log_exists()
    state = _read_state()
    packet = state.packet or {}
    directive = state.directive or {}

    goal = str(packet.get("goal") or ("best" if state.threshold_target is None else "threshold"))
    best_cycles = packet.get("best_cycles")
    aspiration_cycles = packet.get("aspiration_cycles")
    plateau = packet.get("plateau_valid_iters_since_best")

    strategy_family = str(directive.get("strategy_family") or "")
    mods = directive.get("strategy_modifiers")
    if isinstance(mods, list):
        modifiers = [str(x).strip() for x in mods if isinstance(x, str) and x.strip()]
    else:
        modifiers = []

    risk = str(directive.get("risk") or "")
    expected = directive.get("expected_effect_cycles")
    hypothesis = _truncate_clean(str(directive.get("primary_hypothesis") or ""), max_len=140)
    summary_items = directive.get("change_summary")
    if isinstance(summary_items, list):
        changes = "; ".join(_truncate_clean(str(x), max_len=80) for x in summary_items if isinstance(x, str) and x.strip())
    else:
        changes = ""
    changes = _truncate_clean(changes, max_len=160)

    print(
        "[loop_runner] ATTEMPT: "
        f"iter={state.iteration_id:04d} "
        f"branch={state.branch} "
        f"goal={goal} "
        f"best_cycles={best_cycles} "
        f"threshold_target={state.threshold_target} "
        f"aspiration_cycles={aspiration_cycles} "
        f"plateau={plateau} "
        f"family={strategy_family} "
        f"modifiers={','.join(modifiers) if modifiers else '(none)'} "
        f"risk={risk or '(unknown)'} "
        f"expected_effect_cycles={expected}"
    )
    if hypothesis:
        print(f"[loop_runner] HYPOTHESIS: {hypothesis}")
    if changes:
        print(f"[loop_runner] CHANGE_SUMMARY: {changes}")

    entries = _read_jsonl(_EXPERIMENT_LOG_PATH)
    entry = _latest_entry_for_iteration(entries, iteration_id=state.iteration_id)
    if not entry:
        print("[loop_runner] OUTCOME: (not recorded yet)")
        return 0

    cycles = entry.get("cycles")
    valid = entry.get("valid")
    scope_ok = entry.get("scope_ok")
    delta = entry.get("delta_vs_best")
    tests_diff_empty = entry.get("tests_diff_empty")
    result_summary = _truncate_clean(str(entry.get("result_summary") or ""), max_len=200)
    files_changed = entry.get("files_changed")
    n_files = len(files_changed) if isinstance(files_changed, list) else None

    print(
        "[loop_runner] OUTCOME: "
        f"valid={valid} scope_ok={scope_ok} cycles={cycles} delta_vs_best={delta} tests_diff_empty={tests_diff_empty} "
        f"files_changed={n_files} summary={result_summary}"
    )
    return 0


def cmd_ensure_best_base(args: argparse.Namespace) -> int:
    """
    Ensure a rolling "best so far" base branch exists on the remote and is fast-forward up to date locally.

    Default behavior:
    - best_branch: opt/best
    - source_branch: main (used only to seed the best branch if missing)
    - remote: origin
    """

    _ensure_clean_worktree()

    remote = str(args.remote or "origin")
    best_branch = str(args.best_branch or "opt/best").strip()
    source_branch = str(args.source_branch or "main").strip()
    if not best_branch:
        raise LoopRunnerError("--best-branch must be non-empty.")
    if not source_branch:
        raise LoopRunnerError("--source-branch must be non-empty.")

    current_branch = _git("branch", "--show-current", check=False)
    current_sha = _git("rev-parse", "HEAD", check=False)
    return_ref = current_branch or current_sha
    if not return_ref:
        raise LoopRunnerError("Could not determine current git ref.")

    _git("fetch", "--prune", remote)

    if _remote_branch_exists(best_branch, remote=remote):
        # Keep local best_branch in sync.
        if not _branch_exists(best_branch):
            _git("checkout", "--track", "-b", best_branch, f"{remote}/{best_branch}")
        else:
            _git("checkout", best_branch)
            _git("pull", "--ff-only", remote, best_branch)
    else:
        # Seed the remote best branch from the remote source branch.
        if not _remote_branch_exists(source_branch, remote=remote):
            raise LoopRunnerError(f"Remote missing {remote}/{source_branch}; cannot seed {best_branch!r}.")

        _git("checkout", source_branch)
        _git("pull", "--ff-only", remote, source_branch)

        # Create origin/<best_branch> at the current source HEAD without mutating local branches.
        _run(["git", "push", remote, f"HEAD:refs/heads/{best_branch}"], check=True)
        _git("fetch", "--prune", remote)

        remote_sha = _git("rev-parse", f"{remote}/{best_branch}", check=False)
        if not remote_sha:
            raise LoopRunnerError(f"Failed to create remote branch {remote}/{best_branch}.")

        if _branch_exists(best_branch):
            local_sha = _git("rev-parse", best_branch, check=False)
            if local_sha != remote_sha:
                raise LoopRunnerError(
                    "Local best branch exists but differs from remote-seeded branch.\n"
                    f"- local {best_branch}: {local_sha}\n"
                    f"- remote {remote}/{best_branch}: {remote_sha}\n"
                    "Resolve manually (rename/delete local branch) and rerun."
                )
            _git("branch", "--set-upstream-to", f"{remote}/{best_branch}", best_branch)
        else:
            _git("checkout", "--track", "-b", best_branch, f"{remote}/{best_branch}")

    # Return to whatever the user had checked out.
    _git("checkout", return_ref)
    print(f"[loop_runner] ensured base branch: {best_branch} (remote={remote}, seed={source_branch})")
    return 0


def _latest_valid_cycles_for_head(
    entries: Sequence[Mapping[str, Any]], *, branch: str, head_sha: str
) -> Optional[int]:
    for e in reversed(entries):
        if e.get("valid") is not True:
            continue
        if str(e.get("branch") or "") != branch:
            continue
        if str(e.get("head_sha") or "") != head_sha:
            continue
        cycles = _coerce_cycles(e)
        if cycles is not None:
            return cycles
    return None


def _latest_valid_cycles_for_iteration(
    entries: Sequence[Mapping[str, Any]], *, iteration_id: int, branch: Optional[str] = None
) -> Optional[int]:
    for e in reversed(entries):
        if e.get("valid") is not True:
            continue
        try:
            if int(e.get("iteration_id", 0)) != int(iteration_id):  # type: ignore[arg-type]
                continue
        except Exception:
            continue
        if branch is not None and str(e.get("branch") or "") != branch:
            continue
        cycles = _coerce_cycles(e)
        if cycles is not None:
            return cycles
    return None


def cmd_tag_best(args: argparse.Namespace) -> int:
    """
    Create an annotated best/* tag for the current HEAD.

    Notes:
    - This is intentionally conservative: it requires a clean worktree so tags point at a real commit.
    - If --cycles is omitted, it infers cycles from the most recent valid experiments/log.jsonl entry
      matching (branch, head_sha).
    """

    _ensure_experiment_log_exists()
    _ensure_clean_worktree()

    state = _read_state()
    branch = _git("branch", "--show-current", check=False) or state.branch
    head_sha = _git("rev-parse", "HEAD", check=False)
    if not head_sha:
        raise LoopRunnerError("Could not determine HEAD sha for tagging.")

    entries = _read_jsonl(_EXPERIMENT_LOG_PATH)
    cycles = int(args.cycles) if args.cycles is not None else None
    if cycles is None:
        cycles = _latest_valid_cycles_for_head(entries, branch=branch, head_sha=head_sha)
    if cycles is None:
        cycles = _latest_valid_cycles_for_iteration(entries, iteration_id=state.iteration_id, branch=branch)
        if cycles is not None:
            print(
                "[loop_runner] warning: inferring cycles for tag from iteration_id (no matching head_sha entry). "
                "This usually means `record` ran with a dirty worktree before the commit was created.",
                file=sys.stderr,
            )
    if cycles is None:
        raise LoopRunnerError(
            "Could not infer cycles for current HEAD.\n"
            "Either pass --cycles explicitly, or run `python3 tools/loop_runner.py record` and ensure the\n"
            "latest valid entry matches the current (branch, head_sha)."
        )

    slug = str(args.slug) if args.slug else (_iter_slug_from_branch(branch) or "auto")
    tag = _format_best_tag(cycles=cycles, slug=slug, iteration_id=state.iteration_id)

    existing = _git("tag", "--list", tag, check=False).strip()
    if existing:
        existing_sha = _git("rev-list", "-n", "1", tag, check=False).strip()
        if existing_sha == head_sha:
            print(f"[loop_runner] best tag already exists: {tag} -> {head_sha}")
        else:
            raise LoopRunnerError(
                f"Refusing to overwrite existing tag {tag!r} (points to {existing_sha}, current HEAD is {head_sha})."
            )
    else:
        _run(["git", "tag", "-a", tag, "-m", f"NEW BEST: {cycles} cycles"], check=True)
        print(f"[loop_runner] created best tag: {tag} -> {head_sha}")

    if args.push:
        remote = str(args.remote or "origin")
        _run(["git", "push", remote, tag], check=True)
        print(f"[loop_runner] pushed tag to {remote}: {tag}")

    return 0


def cmd_push_best_tags(args: argparse.Namespace) -> int:
    tags = [t.strip() for t in _git("tag", "-l", "best/*", check=False).splitlines() if t.strip()]
    if not tags:
        print("[loop_runner] no best/* tags found.")
        return 0

    remote = str(args.remote or "origin")
    if args.dry_run:
        print(json.dumps({"remote": remote, "tags": tags}, indent=2, sort_keys=True))
        return 0

    _run(["git", "push", remote, *tags], check=True)
    print(f"[loop_runner] pushed {len(tags)} tag(s) to {remote}.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Codex↔gpt-5.2-pro advisor loop helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Create iteration branch and request a plan from the advisor.")
    p_plan.add_argument("--model", default="gpt-5.2-pro")
    p_plan.add_argument("--base-branch", default="main")
    p_plan.add_argument("--no-pull", action="store_true", help="Do not `git pull` the base branch.")
    p_plan.add_argument(
        "--goal",
        choices=_GOALS,
        default="threshold",
        help="Optimization objective: meet a fixed threshold (stop condition) or search for best possible cycles.",
    )
    p_plan.add_argument("--threshold", type=int, default=1363, help="Target cycles (only used for --goal threshold).")
    p_plan.add_argument("--slug", default="auto", help="Short branch slug (used in iter/NNNN-<slug>).")
    p_plan.add_argument(
        "--no-branch",
        action="store_true",
        help="Do not create/check out an iter/* branch (useful while developing the runner).",
    )
    p_plan.add_argument(
        "--offline",
        action="store_true",
        help="Do not call OpenAI; write state using a stub directive (hermetic test).",
    )
    p_plan.add_argument(
        "--code-context",
        choices=["kernelbuilder", "full", "none"],
        default="kernelbuilder",
        help="How much code to include in the advisor packet.",
    )
    p_plan.add_argument("--experiment-log-tail-lines", type=int, default=20)
    p_plan.set_defaults(func=cmd_plan)

    p_cplan = sub.add_parser(
        "codex-plan",
        help="Create iteration branch and request a plan from Codex CLI (no OpenAI API calls).",
    )
    p_cplan.add_argument(
        "--model",
        default=None,
        help="Codex model override for the planner (defaults to the codex config).",
    )
    p_cplan.add_argument("--base-branch", default="main")
    p_cplan.add_argument("--no-pull", action="store_true", help="Do not `git pull` the base branch.")
    p_cplan.add_argument(
        "--goal",
        choices=_GOALS,
        default="threshold",
        help="Optimization objective: meet a fixed threshold (stop condition) or search for best possible cycles.",
    )
    p_cplan.add_argument("--threshold", type=int, default=1363, help="Target cycles (only used for --goal threshold).")
    p_cplan.add_argument("--slug", default="auto", help="Short branch slug (used in iter/NNNN-<slug>).")
    p_cplan.add_argument(
        "--no-branch",
        action="store_true",
        help="Do not create/check out an iter/* branch (useful while developing the runner).",
    )
    p_cplan.add_argument(
        "--code-context",
        choices=["kernelbuilder", "full", "none"],
        default="kernelbuilder",
        help="How much code to include in the advisor packet.",
    )
    p_cplan.add_argument("--experiment-log-tail-lines", type=int, default=20)
    p_cplan.set_defaults(func=cmd_codex_plan)

    p_mpack = sub.add_parser(
        "manual-pack",
        help="Create a plan/* branch and write planner_packets/* for manual ChatGPT planning (no API calls).",
    )
    p_mpack.add_argument("--base-branch", default="main")
    p_mpack.add_argument("--no-pull", action="store_true", help="Do not `git pull` the base branch.")
    p_mpack.add_argument(
        "--goal",
        choices=_GOALS,
        default="threshold",
        help="Optimization objective: meet a fixed threshold (stop condition) or search for best possible cycles.",
    )
    p_mpack.add_argument("--threshold", type=int, default=1363, help="Target cycles (only used for --goal threshold).")
    p_mpack.add_argument("--slug", default="auto", help="Short branch slug (used in plan/NNNN-<slug>).")
    p_mpack.add_argument(
        "--code-context",
        choices=["kernelbuilder", "full", "none"],
        default="none",
        help="How much code to include in the manual packet.",
    )
    p_mpack.add_argument("--experiment-log-tail-lines", type=int, default=20)
    p_mpack.set_defaults(func=cmd_manual_pack)

    p_mapply = sub.add_parser(
        "manual-apply",
        help="Create an iter/* branch + .advisor/state.json from planner_packets/directive.json.",
    )
    p_mapply.add_argument(
        "--from-ref",
        default=None,
        help="Read planner_packets/* from this git ref (e.g. plan/0001-next) instead of the current worktree.",
    )
    p_mapply.add_argument("--base-branch", default="main")
    p_mapply.add_argument("--no-pull", action="store_true", help="Do not `git pull` the base branch.")
    p_mapply.add_argument(
        "--allow-stale-base",
        action="store_true",
        help="Proceed even if origin/<base-branch> advanced since manual-pack (not recommended).",
    )
    p_mapply.add_argument(
        "--slug",
        default=None,
        help="Override the iter/* branch slug (defaults to the plan/* branch slug when available).",
    )
    p_mapply.set_defaults(func=cmd_manual_apply)
    p_resume = sub.add_parser("resume", help="Resume polling a prior in-progress advisor plan (from .advisor/state.json).")
    p_resume.set_defaults(func=cmd_resume)

    p_record = sub.add_parser("record", help="Run submission tests and append an entry to experiments/log.jsonl.")
    p_record.add_argument(
        "--print-test-output",
        action="store_true",
        help="Print full submission_tests.py output before the JSON log entry.",
    )
    p_record.set_defaults(func=cmd_record)

    p_status = sub.add_parser("status", help="Print a concise summary of the current attempt + outcome (if recorded).")
    p_status.set_defaults(func=cmd_status)

    p_ensure_best = sub.add_parser(
        "ensure-best-base",
        help="Ensure the rolling best base branch exists on the remote (default opt/best) and is up to date locally.",
    )
    p_ensure_best.add_argument("--best-branch", default="opt/best")
    p_ensure_best.add_argument("--source-branch", default="main")
    p_ensure_best.add_argument("--remote", default="origin")
    p_ensure_best.set_defaults(func=cmd_ensure_best_base)

    p_tag_best = sub.add_parser("tag-best", help="Create an annotated best/* tag for the current HEAD.")
    p_tag_best.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Cycles to encode in the tag (defaults to the last valid entry for current HEAD).",
    )
    p_tag_best.add_argument(
        "--slug",
        default=None,
        help="Slug for the tag (defaults to the iter/* branch slug when available).",
    )
    p_tag_best.add_argument("--push", action="store_true", help="Push the created tag to the remote.")
    p_tag_best.add_argument("--remote", default="origin")
    p_tag_best.set_defaults(func=cmd_tag_best)

    p_push_best = sub.add_parser("push-best-tags", help="Push all local best/* tags to the remote.")
    p_push_best.add_argument("--remote", default="origin")
    p_push_best.add_argument("--dry-run", action="store_true")
    p_push_best.set_defaults(func=cmd_push_best_tags)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
