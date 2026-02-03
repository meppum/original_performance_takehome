# Codex Loop Prompt (Copy/Paste)

## End-to-end runbook (fresh terminal → iterative loop)

The “automatic loop” is driven by **Codex CLI**, not by `tools/loop_runner.py` alone:

- `python3 tools/loop_runner.py plan` syncs `main`, creates an `iter/*` branch, calls the OpenAI advisor, and writes `.advisor/state.json`.
- `python3 tools/loop_runner.py codex-plan` is an alternative planner step that spawns `codex exec` to produce a directive (no `OPENAI_API_KEY`).
- Convenience: `tools/codex_planner_exec.sh` runs one `codex-plan` + apply-directive Codex exec.
- Codex reads `.advisor/state.json` and implements `directive.step_plan` (usually in `perf_takehome.py`).
- `python3 tools/loop_runner.py record` runs `python3 -B tests/submission_tests.py` and appends to `experiments/log.jsonl`.
- If it’s a **new best**, Codex runs hermetic tooling tests before opening a PR: `python3 -m unittest discover -s tools/tests`.

### One-time prerequisites (as needed)

```bash
cd /home/ubuntu/development/original_performance_takehome

# sanity: confirm origin
git remote -v

# provide your key (only needed for `plan`; `codex-plan` does not use `OPENAI_API_KEY`)
ls -la .env .env.example

# GitHub CLI auth (needed for PR create/merge)
gh auth status
```

### Start the loop (recommended)

1) Launch Codex with project-scoped instructions:

```bash
CODEX_HOME="$PWD/.codex_home" codex --cd "$PWD"
```

2) Paste the prompt in the next section (“Prompt”) into the Codex chat.

3) Let Codex run until it hits the target (threshold goal) or you say `stop` (best goal).

### Long runs (avoid “model compaction”): one iteration per `codex exec`

If you want a long-running search without relying on a single giant chat session, run exactly **one iteration per**
`codex exec` invocation:

```bash
while true; do
  # Best-possible search (no fixed threshold; stop with Ctrl-C)
  tools/codex_planner_exec.sh --goal best --slug next
done
```

### If Codex is interrupted mid-planner call

To resume polling without creating a new paid planner request:

```bash
python3 tools/loop_runner.py resume
```

Notes:

- Interrupting the Codex run (Esc) only stops the *local polling*; it does **not** cancel the background planner job.
- Do **not** try to “unstick” the planner by cancelling it or by deleting/editing `.advisor/state.json`.
  - That pattern tends to create duplicate paid requests.

Then restart Codex and tell it to continue implementing the directive in `.advisor/state.json`.

### Stop the loop

Type `stop` in the Codex chat.

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

Planner safety rules (cost + correctness):
- While `python3 tools/loop_runner.py plan ...` is polling, **do not** run other commands, and **do not** re-run `plan`. Just wait for it to complete.
- If the planner poll is interrupted (Esc, network drop, Codex restart), resume with `python3 tools/loop_runner.py resume` (no new paid request).
- If the planner fails (`OpenAI response status=failed`) or times out, **stop and ask me** before starting a fresh planner request (retries are paid).
- Never cancel planner jobs and never delete/edit `.advisor/state.json`.

Loop until either (a) goal=threshold and `cycles <= threshold_target` (stored in `.advisor/state.json`), or (b) I say “stop”.

`python3 tools/loop_runner.py record` will print `THRESHOLD MET:` when the current result meets the target.

1) Sync baseline
- `git checkout main`
- `git pull --ff-only origin main`

2) Create a new iteration branch and request a planner directive
- Choose ONE:
  - OpenAI advisor (threshold goal): `python3 tools/loop_runner.py plan --threshold <n> --slug next`
  - OpenAI advisor (best goal): `python3 tools/loop_runner.py plan --goal best --slug next`
  - Codex advisor (threshold goal): `python3 tools/loop_runner.py codex-plan --threshold <n> --slug next`
  - Codex advisor (best goal): `python3 tools/loop_runner.py codex-plan --goal best --slug next`

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
  - `python3 tools/loop_runner.py tag-best --push`
  - `git push -u origin HEAD`
  - `gh pr create --fill --base main --head "$(git branch --show-current)"`
  - `printf 'y\n' | gh pr merge --squash --delete-branch`
- If it is NOT a new best or correctness fails:
  - Do not merge into `main`.
  - Discard the iteration branch and return to step (1).
