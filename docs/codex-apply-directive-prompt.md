# Codex Prompt: Apply Existing Directive (Copy/Paste)

Use this when a planner directive has already been materialized in `.advisor/state.json` (via `python3 tools/loop_runner.py plan`, `python3 tools/loop_runner.py codex-plan`, or `python3 tools/loop_runner.py manual-apply`).

You are Codex CLI (executor). A planner directive is already available locally in `.advisor/state.json`.

Hard rules (non-negotiable):
- Never modify anything under `tests/`.
- Before and after, prove `git diff origin/main tests/` is empty.
- Use `python3 -B tests/submission_tests.py` as the source of truth for cycles and correctness.
- Do not run `python3 tools/loop_runner.py plan` (planner step is already done).

1) Preflight
- `git status --porcelain=v1` (must be empty)
- `git diff origin/main tests/` (must be empty)

2) Implement the directive
- Read `.advisor/state.json` and implement `directive.step_plan`.
- Prefer changes limited to `perf_takehome.py` unless explicitly justified.

3) Benchmark and record
- `python3 tools/loop_runner.py record`

4) Merge only if it’s a new best
- If the `record` output prints `NEW BEST:`, then:
  - `git add -A`
  - `git diff origin/main tests/`  (must be empty)
  - `git commit -m "feat: iter/<id>-<slug>"`
  - `python3 tools/loop_runner.py tag-best --push`
  - `git push -u origin HEAD`
  - `gh pr create --fill --base main --head "$(git branch --show-current)"`
  - `printf 'y\n' | gh pr merge --squash --delete-branch`
- If it is NOT a new best or correctness fails:
  - Do not merge into `main`.
  - Restore a clean worktree so an outer driver (e.g., a `while true` shell loop) can run the next iteration:
    - `git restore --staged --worktree -- .`
    - `git status --porcelain=v1` (must be empty)
    - `git diff origin/main tests/` (must be empty)
  - Exit.
