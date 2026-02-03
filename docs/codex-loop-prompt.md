# Codex Loop Prompt (Copy/Paste)

## End-to-end runbook (fresh terminal → iterative loop)

The “automatic loop” is driven by **Codex CLI**, not by `tools/loop_runner.py` alone:

- `python3 tools/loop_runner.py codex-plan` creates an `iter/*` branch and spawns `codex exec` (read-only) to produce a directive (no OpenAI API calls).
- Convenience: `tools/codex_planner_exec.sh` runs **one full iteration**:
  - ensure `opt/best` exists (rolling best base branch)
  - `codex-plan` → apply directive via `codex exec` → commit → `record`
  - on **NEW BEST**: push a `best/*` tag and fast-forward `opt/best` on origin
- Codex reads `.advisor/state.json` and implements `directive.step_plan` (usually in `perf_takehome.py`).
- `python3 tools/loop_runner.py record` runs `python3 -B tests/submission_tests.py` and appends to `experiments/log.jsonl`.

## Loop Diagram (Codex-only, Rolling Best)

```mermaid
flowchart TD
  A[Start iteration\n(shell while loop)] --> B[tools/codex_planner_exec.sh]
  B --> C[ensure-best-base\n(create/update origin/opt/best)]
  C --> D[codex-plan\n(create iter/* from opt/best\n+ write .advisor/state.json)]
  D --> E[codex exec (apply)\n(read directive.step_plan\nedit perf_takehome.py)]
  E --> F[git add -A\ncommit reproducible snapshot]
  F --> G[record\nrun tests/submission_tests.py\nappend experiments/log.jsonl]

  G --> H{record valid?\n(correct + tests unchanged\n+ scope_ok)}
  H -- no --> X[STOP (outer loop breaks)]

  H -- yes --> I{NEW BEST?}
  I -- yes --> J[tag-best --push\n(best/* tag to origin)]
  J --> K[ff opt/best to iter commit\npush origin/opt/best]
  K --> A

  I -- no --> L[reset --hard HEAD~1\n(drop temp commit)]
  L --> A

  subgraph Scope Enforcement (in record)
    S1[Allowed: perf_takehome.py]
    S2[Forbidden: tests/**, problem.py]
  end
  G -. checks .-> S1
  G -. checks .-> S2
```

### One-time prerequisites (as needed)

```bash
cd /home/ubuntu/development/original_performance_takehome

# sanity: confirm origin
git remote -v

# verify you can push best/* tags + opt/best branch
git ls-remote --heads origin
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
  tools/codex_planner_exec.sh --goal best --slug next || break
done
```

### If Codex is interrupted

If you are using the OpenAI API planner (`python3 tools/loop_runner.py plan`), you can resume polling without creating a new paid request:

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

1) Ensure the rolling best base exists
- `python3 tools/loop_runner.py ensure-best-base`  (creates/updates `opt/best` on origin)

2) Create a new iteration branch and request a planner directive (no OpenAI API calls)
- Choose ONE:
  - Best-possible search: `python3 tools/loop_runner.py codex-plan --goal best --slug next --base-branch opt/best`
  - Threshold search: `python3 tools/loop_runner.py codex-plan --threshold <n> --slug next --base-branch opt/best`

3) Implement the plan (executor)
- Read `.advisor/state.json` and implement `directive.step_plan`.
- Prefer changes limited to `perf_takehome.py` (record enforces an allowlist).
- Do not touch `tests/` or `problem.py` (record enforces this).

4) Make the benchmark reproducible, then record
- `git add -A`
- `git commit -m "perf: $(git branch --show-current)"`
- `python3 tools/loop_runner.py record`

5) Preserve only if it’s a NEW BEST
- If `record` prints `NEW BEST:`, then:
  - `python3 tools/loop_runner.py tag-best --push`
  - Fast-forward the rolling base so future iterations start from the best:
    - `git checkout opt/best`
    - `git merge --ff-only <iter-branch>`
    - `git push origin opt/best`
- If it is NOT a new best (but is valid), discard the temp commit so the next iteration starts clean:
  - `git reset --hard HEAD~1`
