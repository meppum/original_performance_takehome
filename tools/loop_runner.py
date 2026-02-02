from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
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

_DEFAULT_ALLOWED_PATHS = ["perf_takehome.py"]
_DEFAULT_FORBIDDEN_GLOBS = ["tests/**"]

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
    return textwrap.dedent(
        f"""
        You are the optimization advisor for this repository.

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
          (typically fewer tasks on the dominant engine and/or breaking dependency chains).

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


def _create_iteration_branch(*, iteration_id: int, slug: str, base_branch: str, no_pull: bool) -> Tuple[str, str]:
    _ensure_clean_worktree()

    _git("fetch", "origin")
    _git("checkout", base_branch)
    if not no_pull:
        _git("pull", "--ff-only", "origin", base_branch)

    base_sha = _git("rev-parse", "HEAD")
    branch = f"iter/{iteration_id:04d}-{_slugify(slug)}"
    _git("checkout", "-b", branch)
    return branch, base_sha


def _directive_looks_complete(directive: Mapping[str, Any]) -> bool:
    if not directive:
        return False
    step_plan = directive.get("step_plan")
    return isinstance(step_plan, list) and len(step_plan) >= 1


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
    family_constraints = _compute_strategy_family_constraints(entries)

    if args.no_branch:
        branch = _git("branch", "--show-current", check=False) or "detached"
        base_sha = _git("rev-parse", "HEAD", check=False)
        if not base_sha:
            raise LoopRunnerError("Could not determine HEAD SHA.")
    else:
        branch, base_sha = _create_iteration_branch(
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

    packet: Dict[str, Any] = {
        "iteration_id": iteration_id,
        "timestamp_utc": _utc_now_iso(),
        "branch": branch,
        "base_branch": args.base_branch,
        "base_sha": base_sha,
        "threshold_target": (int(args.threshold) if args.threshold is not None else None),
        "best_cycles": best,
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
                        threshold_target=(int(args.threshold) if args.threshold is not None else None),
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
            threshold_target=(int(args.threshold) if args.threshold is not None else None),
            packet=packet,
            directive=directive,
            advisor_response_id=response_id,
        )
    )

    print(json.dumps(directive, indent=2, sort_keys=True))
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

    entry: Dict[str, Any] = {
        "iteration_id": state.iteration_id,
        "timestamp_utc": _utc_now_iso(),
        "branch": state.branch,
        "base_branch": state.base_branch,
        "base_sha": state.base_sha,
        "head_sha": head_sha,
        "files_changed": files_changed,
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Codex↔gpt-5.2-pro advisor loop helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Create iteration branch and request a plan from the advisor.")
    p_plan.add_argument("--model", default="gpt-5.2-pro")
    p_plan.add_argument("--base-branch", default="main")
    p_plan.add_argument("--no-pull", action="store_true", help="Do not `git pull` the base branch.")
    p_plan.add_argument("--threshold", type=int, default=1363)
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

    p_resume = sub.add_parser("resume", help="Resume polling a prior in-progress advisor plan (from .advisor/state.json).")
    p_resume.set_defaults(func=cmd_resume)

    p_record = sub.add_parser("record", help="Run submission tests and append an entry to experiments/log.jsonl.")
    p_record.add_argument(
        "--print-test-output",
        action="store_true",
        help="Print full submission_tests.py output before the JSON log entry.",
    )
    p_record.set_defaults(func=cmd_record)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
