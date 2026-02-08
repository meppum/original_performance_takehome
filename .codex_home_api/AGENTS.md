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
  - If pushing is unavailable (no credentials/network), still create the local annotated tags and push later; do not block the loop.
- Never rewrite history or delete tags that represent a best.

4) **No surprise constraints**
- Do not invent extra stop conditions (LOC caps, “no refactors”, etc.) without asking first.
- Any new constraint must include (a) why it exists and (b) an alternative with trade-offs.

5) **Allow multi-step changes safely**
- It’s OK for intermediate commits to be non-improving if they are on a scratch/feature branch.
- Only “bless” (tag/preserve/merge) commits that are a verified NEW BEST on `submission_tests.py`.

## What Counts As “Cheating” vs Legit Optimization

This section is **illustrative, not exhaustive**. Any approach is acceptable if it preserves
semantics and follows the non-negotiable guardrails below.

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

- Never print, echo, or commit API keys/tokens. Redact if they appear in logs.
- It is OK for Codex CLI to store credentials inside `CODEX_HOME` (e.g., `auth.json`), since these dirs are gitignored.
- For env-based auth, prefer `CODEX_API_KEY` (`OPENAI_API_KEY` is accepted as an alias).

## Default Scope

Default scope for automated iterations is intentionally narrow (enforced by `python3 tools/loop_runner.py record`):

- `perf_takehome.py`

If you believe a change outside this allowlist is necessary:

1) Explain why it’s needed and what it would enable.
2) Interactive mode: stop and ask for explicit approval before expanding scope (do not silently broaden scope).
3) Unattended mode: do **not** expand scope. Record the out-of-scope recommendation (e.g., in the directive `objective`) and proceed with the best in-scope Plan B.

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

### Roles (Planner vs Executor vs Driver)

This optimization loop has three distinct roles; do not mix responsibilities:

- **Planner** (advisor): outputs plan-only (`OptimizationDirective`). Must not edit code, run tests, commit/tag/push, or run `record`.
- **Executor** (apply): implements `directive.step_plan` in the worktree and then exits. Must not commit/tag/push or run `record`.
  - Avoid repeated local test loops; rely on the driver for benchmarking. At most run one quick sanity check if needed to prevent an obviously broken attempt.
- **Driver** (loop runner / `tools/*_planner_exec.sh`): creates branches, runs plan+apply, commits for reproducibility, runs `python3 tools/loop_runner.py record`,
  tags/pushes on NEW BEST, and decides when to stop.

Assume the loop is **unattended** when running via `tools/*_planner_exec.sh`. In unattended mode, never stall waiting for approval or user answers; always produce an in-scope Plan B.

### Progress, Plateau, and Pivot Protocol (Required)

This protocol exists to prevent “busy looping” where iterations run but no meaningful improvement occurs.
The advisor must (1) detect stagnation, (2) diagnose the limiting factor using telemetry, and (3) pivot mechanisms without human prompting.

#### Constants (define once; reuse consistently)

Default values (adjust only with evidence):

- `MEANINGFUL_WIN_CYCLES = 10`
- `HARD_PLATEAU_NO_NEW_BEST_VALID = 6`
- `SOFT_PLATEAU_NO_MEANINGFUL_WIN_VALID = 8`
- `RECENT_SIGNATURE_WINDOW_VALID = 10`
- `INVALID_STREAK_LIMIT = 3`
- `MAX_TELEMETRY_ONLY_ITERS = 1` (per plateau episode)

#### Definitions

- **Valid attempt:** an iteration whose result is valid (`valid=true`; correctness passes and scope/tests guardrails are satisfied).
- **New best:** `new_best=true` (cycles strictly lower than `best_before` as recorded by `record`).
- **Meaningful win:** a New best that improves `best_before` by at least `MEANINGFUL_WIN_CYCLES`.
  - Small New bests still count as progress; “meaningful” is used for escalation, not for declaring hard plateau.

#### Plateau detection (soft vs hard)

- **Hard plateau (pivot REQUIRED):** no New best in the last `HARD_PLATEAU_NO_NEW_BEST_VALID` valid attempts.
  - Prefer `packet.plateau_valid_iters_since_best` when present; treat `plateau_valid_iters_since_best >= HARD_PLATEAU_NO_NEW_BEST_VALID` as hard plateau.
  - Otherwise compute from `packet.experiment_summary.recent_attempts` / `experiment_log_tail`.
- **Soft plateau (exploration REQUIRED):** not hard plateau, AND no Meaningful win in the last `SOFT_PLATEAU_NO_MEANINGFUL_WIN_VALID` valid attempts.

#### Invalid attempt recovery mode (correctness before optimization)

If `INVALID_STREAK_LIMIT` invalid attempts occur consecutively:

- Stop performance experimentation and enter recovery mode.
- Propose the smallest change likely to restore validity (scope correctness, revert suspect changes, remove accidental outputs).
- Do not propose high-risk structural optimizations until validity is restored.

#### Mandatory per-iteration diagnosis (must precede strategy selection)

Before choosing an approach, read telemetry from `packet.performance_profile` (and `packet.best_cycles`):

- Bounds: `resource_lb_cycles`, `cp_lb_cycles`, `tight_lb_cycles`
- Slack: `schedule_slack_pct`, `schedule_slack_cycles`, `schedule_cycles_estimate`
- Resource drivers: `dominant_engine`, `task_counts_by_engine`, `min_cycles_by_engine`

Classify the current regime (rules of thumb; be consistent):

- **Slack-limited:** `schedule_slack_pct > 0.04`
- **Bound-limited:** `schedule_slack_pct <= 0.04` OR `(best_cycles - tight_lb_cycles) <= 40`
- **Critical-path-limited:** `cp_lb_cycles >= resource_lb_cycles - 10`

If required telemetry is missing or `performance_profile` contains an error:

- Use `next_packet_requests` to request the missing fields/code.
- You may emit at most `MAX_TELEMETRY_ONLY_ITERS` “telemetry-only” directives per plateau episode; otherwise propose a conservative in-scope change with explicit low confidence.

The plan must target the diagnosed limiter (do not keep proposing slack-only fixes when bound-limited, etc.).

#### Pivot rules (mechanism shift requirement)

- If **hard plateau**: mechanism shift is REQUIRED on the next valid attempt.
- If **soft plateau**: mechanism shift is REQUIRED within the next 2 valid attempts (you may take one low-risk exploit attempt first, but must explore next).

Mechanism shift requirements:

- Not sufficient: minor parameter tuning, retuning priorities, trivial local reorderings, “same idea with slightly different constants”.
- Required: a different mechanism that plausibly changes the diagnosed limiter (instruction shape, dataflow, reuse/caching pattern, dependency structure, representation).

#### Novelty contract (enforced)

Maintain a short **strategy signature** per attempt: `strategy_family` + up to 3 `strategy_modifiers`.

- Use `packet.experiment_summary.recent_strategy_combos` and/or `experiment_log_tail` to avoid repeating signatures within the last `RECENT_SIGNATURE_WINDOW_VALID` valid attempts.
- If you repeat a recent signature, explicitly state:
  (a) what mechanism is new,
  (b) where you will change code (anchors),
  (c) why the prior attempt failed and why this differs.

#### Expected-effect contract (must be justified)

Each plan must state an expected effect tied to the diagnosed limiter:

- Slack-limited: expected slack reduction / better overlap.
- Bound-limited: expected reduction in the dominant bound (e.g., lower `min_cycles_by_engine` for the dominant engine) and why.
- Critical-path-limited: expected reduction in serial dependency length.

If claiming a throughput-bound reduction, include a plausibility check using `task_counts_by_engine` and slot limits (from `problem.SLOT_LIMITS`, or request via `next_packet_requests`):

- `lb_e = ceil(tasks_e / slots_e)`
- To reach target `T`: `tasks_e <= slots_e * T` (rough check)

#### Escalation ladder

If plateau persists for a long window (rule of thumb: `plateau_valid_iters_since_best >= 12` valid attempts):

- Increase novelty/risk (still within guardrails) OR request specific missing context,
  but also include an in-scope fallback plan to avoid a no-op iteration.

#### Output discipline

- Every directive should propose at least one concrete code change likely to affect the diagnosed limiter.
- Telemetry-only directives are allowed only when they are the fastest unblocker and only up to `MAX_TELEMETRY_ONLY_ITERS` per plateau episode.
- Avoid random churn: if confidence is low, prefer a conservative, well-justified change rather than arbitrary edits.
- Avoid debug output that could be mistaken for the submission harness’s cycle-count marker (the loop runner parses it to extract cycles).

### Helper Artifacts (Gitignored Only)

- Allowed: `.advisor/**` (gitignored) and append-only `experiments/log.jsonl` (gitignored).
- Not allowed during performance iterations: creating new tracked helper files (including under `experiments/`) without explicit approval.

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

- Examples and recipes in this repo are defaults and **non-exhaustive** (not a whitelist). If you have a better idea, use it.
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
