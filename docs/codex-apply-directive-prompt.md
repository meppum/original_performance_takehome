# Codex Prompt: Apply Existing Directive (Copy/Paste)

Use this when a planner directive has already been materialized in `.advisor/state.json` (via `offline-plan`, `codex-plan`, `codex-api-plan`, or `manual-apply`).

You are Codex CLI (executor). A planner directive is already available locally in `.advisor/state.json`.

Hard rules (non-negotiable):
- Never modify anything under `tests/`.
- Before and after, prove `git diff origin/main tests/` is empty.
- Use `python3 -B tests/submission_tests.py` as the source of truth for cycles and correctness.
- Do not run any planner command (`offline-plan` / `codex-plan` / `codex-api-plan`) (planner step is already done).

1) Preflight
- `git status --porcelain=v1` (must be empty)
- `git diff origin/main tests/` (must be empty)

2) Implement the directive
- Read `.advisor/state.json` and implement `directive.step_plan`.
- Prefer changes limited to `perf_takehome.py` unless explicitly justified.

3) Stop here (executor-only)
- Do not run `record`, do not commit, and do not push.
- Show what changed so the driver can make the commit reproducible:
  - `git diff --stat`
  - `git diff origin/main tests/`  (must be empty)
- Exit with the worktree potentially dirty (expected).
