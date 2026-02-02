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


def _next_iteration_id(entries: Iterable[Mapping[str, Any]]) -> int:
    max_id = 0
    for e in entries:
        try:
            iid = int(e.get("iteration_id", 0))  # type: ignore[arg-type]
        except Exception:
            continue
        max_id = max(max_id, iid)
    return max_id + 1


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

        First, perform a novelty check against `experiment_log_tail`:
        - Avoid repeating the same `strategy_tags` combination unless you explain what is different and why it might work now.

        Here is the current IterationPacket (JSON):
        ```json
        {packet_json}
        ```
        """
    ).strip()


def _planner_directive_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "objective": {"type": "string"},
            "primary_hypothesis": {"type": "string"},
            "strategy_tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
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
            "strategy_tags",
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


def cmd_plan(args: argparse.Namespace) -> int:
    _ensure_experiment_log_exists()
    entries = _read_jsonl(_EXPERIMENT_LOG_PATH)
    iteration_id = _next_iteration_id(entries)
    best = _best_cycles(entries)

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
        "experiment_log_tail": tail,
        "code_context": code_context,
    }

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
            "strategy_tags": ["offline"],
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
            directive_schema=_planner_directive_schema(),
            directive_tool_name=tool_name,
        )

        response_json = client.create_response(**payload)
        response_id = response_json.get("id") if isinstance(response_json.get("id"), str) else None
        stem = f"iter_{iteration_id:04d}" + (f"_{response_id}" if response_id else "")
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
    strategy_tags = list(directive.get("strategy_tags") or [])
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
