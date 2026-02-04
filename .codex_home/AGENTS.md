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

Default scope for automated iterations is intentionally narrow (enforced by `python3 tools/loop_runner.py record`):

- `perf_takehome.py`

If you believe a change outside this allowlist is necessary:

1) Stop and explain why it’s needed.
2) Ask for explicit approval to expand scope (do not silently broaden scope).

## Planner / Loop Policy

- Planner output is **plan-only**: it returns a step-by-step plan (`OptimizationDirective`); **Codex writes code**.
- Available planner modes + wrappers: `docs/planning-modes.md`.
- Keep optimization memory in `experiments/log.jsonl` (schema described in `docs/experiment-log.md`) so the loop doesn’t repeat itself.
- `experiments/log.jsonl` is intentionally **gitignored** (like `.env`) so it persists across branch switches even when iteration branches are deleted. Use `experiments/log.jsonl.example` as the schema/sample.

Manual planner mode (ChatGPT UI; copy/paste):

- `python3 tools/loop_runner.py manual-pack` → paste `planner_packets/prompt.md` into ChatGPT → commit `planner_packets/directive.json`
- `python3 tools/loop_runner.py manual-apply` writes `.advisor/state.json` on a real `iter/*` branch
- Optional apply helper: `tools/manual_planner_exec.sh`

Codex planner modes (no copy/paste):

- ChatGPT-login planning: `python3 tools/loop_runner.py codex-plan ...`
- API-key planning (default `gpt-5.2-pro`): `python3 tools/loop_runner.py codex-api-plan ...`
- Recommended unattended drivers (one full iteration; accumulates on `opt/best` and pushes best tags):
  - `tools/codex_planner_exec.sh --goal best --slug next`
  - `tools/codex_api_planner_exec.sh --goal best --slug next`

Offline planner mode (hermetic):

- `python3 tools/loop_runner.py offline-plan ...` writes a stub directive (tests/sanity checks).

### Iteration File-Scope Guardrail (Do Not Self-Modify The Loop)

During **performance iterations** (`iter/*` branches), treat the loop tooling as immutable:

- Allowed to edit (default): `perf_takehome.py`
- Do **not** edit: `tools/loop_runner.py`, `tools/tests/**`, `docs/**`, `.codex_home/AGENTS.md`
- Do **not** use escape hatches (`--no-branch`, `offline-plan`) unless explicitly instructed by the user.

If the iteration fails because the tooling is wrong (traceback in `loop_runner.py *plan/record`, directive schema errors, etc.):

1) **Stop the iteration** (do not patch tooling inside `iter/*`).
2) Create a dedicated fix branch off `main` (e.g., `fix/loop-runner-*`), apply the minimal fix, run `python3 -m unittest discover -s tools/tests`, then merge to `main`.
3) Restart the iteration from step (1) on a fresh `iter/*` branch.

### Planner Cost Guardrails (No Ad-hoc OpenAI Calls)

Hard rules:

- **Do not call OpenAI directly** via ad-hoc scripts (e.g., `python3 - <<'PY'` with `requests`, `curl`, etc.).
- **Do not delete or hand-edit `.advisor/state.json`**.
  - If planning fails, inspect `.advisor/codex*/` artifacts and rerun the planner explicitly (it creates a new iteration).

### Collab (Sub-agents)

Use sub-agents to reduce wall-clock time, but keep them **analysis-only**:

- Spawn at most 2 sub-agents per iteration while waiting on (a) background planner polling and/or (b) `submission_tests.py`.
- Sub-agents must not edit files or run destructive commands; the main agent is the only writer/merger.
- Recommended roles:
  - **explorer**: locate hotspots/invariants and propose 2–3 concrete next-step candidates (with file/function anchors).
  - **reviewer**: sanity-check the proposed diff for correctness + anti-cheat guardrails before merging into `main`.

### Guidance (Keep It Flexible)

- Examples and recipes in this repo are defaults, not hard constraints. If you have a better idea, use it.
- Optimize for iteration throughput:
  - implement `directive.step_plan`,
  - keep `tests/` unchanged,
  - and hand control back to the driver (`tools/*_planner_exec.sh`) for commit/benchmark/tag.
- If a plan seems under-specified, request *specific* missing context (a function body, a trace snippet, etc.) rather than asking for “all the code”.

## Test-case requests (skill routing)

If the user asks for “test cases” / “test-case ideas” / “what should we test”, default to producing a scenario list (not writing test code) unless they explicitly ask you to implement tests.

If the environment has the relevant skills available, pick the most specific one and say which you used:

- Unit scope (single function/method/class): `identify-unit-test-cases`
- Cross-component boundary (API↔DB, service↔queue, etc.): `identify-integration-test-cases`
- User journey (hermetic, no live externals): `identify-hermetic-e2e-test-cases`
- Live external-service canary (explicitly requested): `identify-live-e2e-test-cases`
- QA plan/test strategy (Given/When/Then + risk-based coverage): `qa-engineer`

Guardrail reminder: do not modify `tests/` (see “Non-Negotiables”). If the user wants implemented tests, stop and ask where tests should live and whether the guardrail is being explicitly overridden.
