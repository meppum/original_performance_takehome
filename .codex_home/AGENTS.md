# Agent Instructions (original_performance_takehome)

This repo is a **performance optimization harness**. The primary goal is to **minimize simulated CPU cycles** while preserving correctness.

## Success Metric (Source of Truth)

- Run: `python3 tests/submission_tests.py`
- Use the **cycle count it prints** as the only performance metric.

## Mandatory Loop Design Gate (Long-Running Optimization Loops)

Before starting a long-running automated loop (multi-hour and/or many-iteration), do all of the following once up front.
This is **not** a cap; it’s a preflight to avoid wasting time/tokens on impossible targets or unsafe workflows:

1) **Define the objective**
- Either: “hit a target threshold” (e.g., `cycles <= 1363`) OR “find the best cycles possible”.
- If “best possible”, either explicitly declare it unbounded (manual stop) or define budgets/stop conditions (time, iterations, token/cost, plateau rules).

2) **Feasibility / lower-bound check**
- Compute (or approximate) a conservative lower bound on cycles from repo-local telemetry.
- If a chosen threshold is below that bound: stop and say so; the next work must be “reduce the bound”, not micro-scheduling.

3) **Best preservation (never lose the best)**
- On every `NEW BEST:`: commit + create an **annotated tag** `best/<cycles>-<slug>-i<k>`.
- Push those `best/*` tags to `origin` regularly (helper: `python3 tools/loop_runner.py push-best-tags`).
- Never rewrite history or delete tags that represent a best.

4) **No surprise constraints**
- Do not invent extra stop conditions (LOC caps, “no refactors”, etc.) without asking first.
- Any new constraint must include (a) why it exists and (b) an alternative with trade-offs.

5) **Allow multi-step changes safely**
- It’s OK for intermediate commits to be non-improving if they are on a scratch/feature branch.
- Only “bless” (tag/preserve/merge) commits that are a verified NEW BEST on `submission_tests.py`.

## What Counts As “Cheating” vs Legit Optimization

- Legit: changing `perf_takehome.py` to emit a faster instruction stream **with identical outputs** on the benchmark inputs.
  - This can include emitting fewer simulated `load`/`alu`/`valu` operations if semantics are preserved.
- Cheating: modifying `tests/`, changing benchmark semantics (e.g., `problem.py`), hardcoding benchmark data, or manufacturing speedups.

## Non-Negotiables (Anti-Cheating Guardrails)

1) **Never modify `tests/`**

- Before and after every iteration, run:
  - `git diff origin/main tests/`
- It must be empty.

2) **Do not change benchmark semantics**

- Treat edits to `problem.py` as **high-risk** and do not touch it unless explicitly requested.
- Default policy: `problem.py` is **forbidden** for automated loops (enforced by `python3 tools/loop_runner.py record`).
- Do not “enable” multicore or change core-count logic to manufacture speedups.

3) **No secrets / PII**

- Never print or persist API keys/tokens. Redact if they appear in logs.
- Use `OPENAI_API_KEY` from the environment (e.g., load from a local `.env` file). This repo ignores `.env` and provides `.env.example`.

## Default Scope

Unless explicitly expanded, changes should be limited to:

- `perf_takehome.py`

## Advisor Loop Policy (gpt-5.2-pro)

- Advisor output is **plan-only**: it returns a step-by-step plan; **Codex writes code**.
- Use the interaction contract in `docs/openai-advisor-loop.md`.
- Keep optimization memory in `experiments/log.jsonl` (schema described in `docs/experiment-log.md`) so the loop doesn’t repeat itself.
- `experiments/log.jsonl` is intentionally **gitignored** (like `.env`) so it persists across branch switches even when iteration branches are deleted. Use `experiments/log.jsonl.example` as the schema/sample.
- Use background-mode Responses API calls for planner requests to avoid long-lived request timeouts; see `tools/openai_exec.py`.
- Enable optional planner research via web search (`tools: [{\"type\":\"web_search\"}, ...]`).
- For planner output, prefer strict function calling (tool parameters JSON Schema) over response-format structured outputs (not supported by `gpt-5.2-pro`).
- Use `python3 tools/loop_runner.py plan` / `python3 tools/loop_runner.py record` to manage iteration branches and append local experiment log entries.

Manual planner mode (no OpenAI API calls):

- Use `python3 tools/loop_runner.py manual-pack` to create a `plan/*` branch and generate `planner_packets/prompt.md`.
- After the user pastes ChatGPT output into `planner_packets/directive.json` and commits it, run `python3 tools/loop_runner.py manual-apply` to create the real `iter/*` branch + `.advisor/state.json`.

Codex planner mode (no OpenAI API calls; no copy/paste):

- Use `python3 tools/loop_runner.py codex-plan --goal best --slug next` to spawn `codex exec` in read-only mode and write the directive to `.advisor/state.json` (no fixed stop threshold).
- Or, for a fixed stop condition, use `python3 tools/loop_runner.py codex-plan --threshold <n> --slug next` (writes `threshold_target=<n>` to `.advisor/state.json`).
- Recommended driver (one full iteration; accumulates improvements on `opt/best` and pushes best tags):
  - `tools/codex_planner_exec.sh --goal best --slug next`
- Notes:
  - `record` enforces the default file-scope allowlist (`perf_takehome.py`) and forbids `tests/**` and `problem.py`.
  - Use `python3 tools/loop_runner.py ensure-best-base` to create/update `opt/best` manually if needed.

### Iteration File-Scope Guardrail (Do Not Self-Modify The Loop)

During **performance iterations** (`iter/*` branches), treat the loop tooling as immutable:

- Allowed to edit (default): `perf_takehome.py`
- Do **not** edit: `tools/loop_runner.py`, `tools/openai_exec.py`, `tools/tests/**`, `docs/**`, `.codex_home/AGENTS.md`
- Do **not** use escape hatches (`--no-branch`, `--offline`) unless explicitly instructed by the user.

If the iteration fails because the tooling is wrong (traceback in `loop_runner.py plan/record`, OpenAI schema errors, polling bugs, etc.):

1) **Stop the iteration** (do not patch tooling inside `iter/*`).
2) Create a dedicated fix branch off `main` (e.g., `fix/loop-runner-*`), apply the minimal fix, run `python3 -m unittest discover -s tools/tests`, then merge to `main`.
3) Restart the iteration from step (1) on a fresh `iter/*` branch.

### OpenAI Spend Guardrails (No Cancel / No Ad-hoc OpenAI Calls)

When the loop is running, it is extremely easy to waste time/money by accidentally creating duplicate planner requests.

Hard rules:

- **Do not call OpenAI directly** via ad-hoc scripts (e.g., `python3 - <<'PY'` with `requests`, `curl`, etc.).
  - The only allowed OpenAI interactions are `python3 tools/loop_runner.py plan` and `python3 tools/loop_runner.py resume`.
- **Do not cancel planner requests.**
  - Do not implement a `cancel` command and do not hit the `.../cancel` endpoint.
  - In Codex, the only safe “stop” mechanism is interrupting the local poll (Esc) and then deciding what to do next.
- **Do not delete or hand-edit `.advisor/state.json`** (or `.advisor/openai/*` artifacts).
  - If `.advisor/state.json` exists and the planner is `queued`/`in_progress`, prefer `python3 tools/loop_runner.py resume` to continue polling (no duplicate paid request).
  - If the planner fails/times out repeatedly, stop and ask the user before attempting anything that would start a fresh paid request.

### Collab (Sub-agents)

Use sub-agents to reduce wall-clock time, but keep them **analysis-only**:

- Spawn at most 2 sub-agents per iteration while waiting on (a) background planner polling and/or (b) `submission_tests.py`.
- Sub-agents must not edit files or run destructive commands; the main agent is the only writer/merger.
- Recommended roles:
  - **explorer**: locate hotspots/invariants and propose 2–3 concrete next-step candidates (with file/function anchors).
  - **reviewer**: sanity-check the proposed diff for correctness + anti-cheat guardrails before merging into `main`.

### Context Strategy (for strong advisor suggestions)

Do not send full repo diffs by default. Instead, send:

- `git_sha` of the code being evaluated
- last cycle count + best cycle count so far
- a short summary of what changed
- targeted code context: full bodies of the functions touched + small surrounding context

At the start of a session (or when changing baselines), it can be worth sending the full `perf_takehome.py` once; after that, keep packets incremental.

If the advisor requests more, escalate by providing exactly what it asked for (e.g., one full file).

#### Diff Policy (Recommended)

- Always include `git diff --stat`.
- Include the full unified diff for allowlisted files (typically `perf_takehome.py`) when it’s small (rule of thumb: a few hundred lines).
- If the diff is large, prefer sending:
  - full bodies of the changed/hot functions (with a little surrounding context), plus
  - only the diff hunks that touch those functions.

Rationale: diffs are explicit but often omit the unchanged context needed for correct optimization advice.

#### Advisor Packet Requirements (Codex → Advisor)

Make every iteration packet consistent and small enough that the advisor can still reason:

- Always include: `iteration_id`, `branch`, `git_sha`, `best_cycles`, `current_cycles`, `threshold_target`
- Always include guardrail proof: `git diff origin/main tests/` output (must be empty)
- Always include: `git diff --stat` and an allowlist of files touched
- Include either:
  - the full diff for allowlisted files (when small), or
  - the full bodies of the functions touched + the specific hunks for those functions
- Include failures verbatim (traceback top frames + assertion message) when tests fail
- Always include optimization memory:
  - `experiment_log_tail`: last 10–20 lines of `experiments/log.jsonl`
  - The advisor must start with a “novelty check”: avoid repeating the same `strategy_tags` combination unless it explains what is different and why it might work now

Token hygiene (hard guidance):

- Keep advisor packets ≤ ~20k characters by default.
- If you need more context, have the advisor request it explicitly and send only what it asked for.

## PR Workflow: One Branch Per Turn (Merge on New Best)

Default branch is `main`. Do not commit directly to `main`.

Each iteration (“turn”) starts from a fresh branch off `main`. If (and only if) the iteration produces a **new all-time best** cycle count (and is correct), it is merged back into `main` via PR.

Branch naming:

- Iterations: `iter/NNNN-short-desc`

Per turn:

```bash
git fetch origin
git checkout main
git pull --ff-only origin main

git checkout -b iter/0007-short-desc

# implement changes (prefer only perf_takehome.py)
git diff origin/main tests/
python3 tests/submission_tests.py
```

If correctness holds and cycles are **strictly better** than the best so far:

```bash
python3 -m unittest discover -s tools/tests

git commit -am "feat: iter/0007-short-desc"
git push -u origin iter/0007-short-desc

# open PR (or update an existing PR)
gh pr create --fill --base main --head iter/0007-short-desc

# merge PR (prefer squash + delete branch)
printf 'y\n' | gh pr merge --squash --delete-branch
```

If it doesn’t improve or fails correctness: **do not merge** into `main` (close the PR and delete the branch).

Experiment memory:

- Always record the attempt in `experiments/log.jsonl` (see `docs/experiment-log.md`).
- Because it is gitignored, it won’t be lost when you create/delete `iter/*` branches.

## Validation Checklist (Every Iteration)

- `git diff origin/main tests/` is empty
- `python3 tests/submission_tests.py` completes successfully
- Cycle count is **strictly better** than the previous best before merging
- Scope constraints respected (default: only `perf_takehome.py`)
- Before opening a PR: `python3 -m unittest discover -s tools/tests` (hermetic; must pass)

## Test-case requests (skill routing)

If the user asks for “test cases” / “test-case ideas” / “what should we test”, default to producing a scenario list (not writing test code) unless they explicitly ask you to implement tests.

If the environment has the relevant skills available, pick the most specific one and say which you used:

- Unit scope (single function/method/class): `identify-unit-test-cases`
- Cross-component boundary (API↔DB, service↔queue, etc.): `identify-integration-test-cases`
- User journey (hermetic, no live externals): `identify-hermetic-e2e-test-cases`
- Live external-service canary (explicitly requested): `identify-live-e2e-test-cases`
- QA plan/test strategy (Given/When/Then + risk-based coverage): `qa-engineer`

Guardrail reminder: do not modify `tests/` (see “Non-Negotiables”). If the user wants implemented tests, stop and ask where tests should live and whether the guardrail is being explicitly overridden.
