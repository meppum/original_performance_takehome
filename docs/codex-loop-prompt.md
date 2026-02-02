# Codex Loop Prompt (Copy/Paste)

## Launch Codex for this repo

This repo’s Codex agent instructions are tracked in `.codex_home/AGENTS.md`. To use them (and avoid accidentally layering in global instructions from `~/.codex`), launch Codex with a project-scoped home:

```bash
CODEX_HOME="$PWD/.codex_home" codex --cd "$PWD"
```

Use the prompt below as your first message to Codex CLI after launching it.

## Prompt

You are Codex CLI (executor). Optimize this repository’s performance metric (minimize cycles) while preserving correctness.

Hard rules (non-negotiable):
- Never modify anything under `tests/`.
- Before and after every iteration, prove `git diff origin/main tests/` is empty.
- Use `python3 -B tests/submission_tests.py` as the source of truth for cycles and correctness.

Loop until `cycles <= 1363` or I say “stop”:

1) Sync baseline
- `git checkout main`
- `git pull --ff-only origin main`

2) Create a new iteration branch and request a planner directive
- `python3 tools/loop_runner.py plan --threshold 1363 --slug next`

If Codex is interrupted while the planner call is still running, rerun the same command:
- `python3 tools/loop_runner.py plan --threshold 1363 --slug next`

It will resume the in-progress OpenAI response using `.advisor/state.json` (no duplicate planner request). You can also run:
- `python3 tools/loop_runner.py resume`

3) Implement the plan
- Read `.advisor/state.json` and implement `directive.step_plan`.
- Prefer changes limited to `perf_takehome.py` unless explicitly justified.
- Do not touch `tests/` (ever).

4) Benchmark and record the attempt
- `python3 tools/loop_runner.py record`

5) Merge only if it’s a new best
- If the `record` output prints `NEW BEST:`, then:
  - `git add -A`
  - `git diff origin/main tests/`  (must be empty)
  - `git commit -m "feat: iter/<id>-<slug>"`
  - `git push -u origin HEAD`
  - `gh pr create --fill --base main --head "$(git branch --show-current)"`
  - `gh pr merge --squash --delete-branch`  (if prompted, confirm interactively)
- If it is NOT a new best or correctness fails:
  - Do not merge into `main`.
  - Discard the iteration branch and return to step (1).
