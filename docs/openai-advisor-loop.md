# Codex CLI ↔ OpenAI Advisor Loop (gpt-5.2-pro)

This document describes a safe, repeatable interaction pattern for using:

- **Codex CLI (gpt-5.2 xhigh)** as the *executor* that edits code and runs checks locally.
- **OpenAI API (gpt-5.2-pro)** as an *advisor* that proposes the next optimization iteration.

The core design goal is: **minimize cycle count while preserving correctness**, with strong guardrails to prevent “LLM cheating” (e.g., modifying tests).

## Roles

### Codex CLI (Executor)

Responsibilities:

- Apply changes in the repo (preferably only `perf_takehome.py` unless explicitly expanding scope).
- Run the canonical validation commands.
- Extract a compact run summary and send it to the advisor for the next iteration.
- Enforce hard guardrails (especially “do not touch `tests/`”).

### gpt-5.2-pro (Advisor)

Responsibilities:

- Read the current iteration summary (cycle count, diffs, failures, constraints).
- Propose the *next* optimization attempt as an explicit, checkable plan (plan-only; Codex writes code).
- Be conservative about changes that could compromise validity (e.g., modifying benchmark harness code).

## Guardrails (Hard Rules)

These are enforced by Codex CLI before/after each iteration:

1) **Never modify `tests/`**

- Preflight and postflight check:
  - `git diff origin/main tests/`
  - Must be empty.

2) **Always measure cycles via the repo’s benchmark**

- `python3 tests/submission_tests.py` (use `python` only if your environment provides it)
- Treat its cycle count as the source of truth.

3) **Keep changes narrowly scoped unless explicitly expanding**

Default allowlist:

- Allowed: `perf_takehome.py`
- Forbidden unless explicitly approved: `tests/**`
- High-risk (should require explicit approval): `problem.py` (can change benchmarking semantics)

4) **No secrets / PII**

- Never include API keys, tokens, or local secrets in advisor packets.
- If logs contain secrets, redact before sending.

## OpenAI API Configuration

### API Key

Use the `OPENAI_API_KEY` environment variable. For local development, a common pattern is:

- Create a local `.env` file (untracked; do not commit).
- Load it into your shell before running your loop.
- This repo includes `.env.example` and ignores `.env` via `.gitignore`.

### Reasoning Effort (gpt-5.2-pro)

Set the advisor’s reasoning effort explicitly for reproducibility:

- Hardcode for the planner role: `reasoning: { "effort": "xhigh" }`
- Available options for `gpt-5.2-pro`: `medium`, `high`, `xhigh`

### Web Search (Optional)

If you want the planner to be able to research when needed, enable the web search tool in the Responses API request:

- `tools: [{ "type": "web_search" }, ...]`

The helper in `tools/openai_exec.py` enables `web_search` in `build_payload_for_planner(...)`.

### Reference Implementation (Python)

This repo includes a small helper that implements:

- background mode + polling
- periodic poll-status output (defaults to every minute)
- retry/backoff for 429/5xx and transient network errors
- strict function-calling “structured outputs” (JSON Schema tool parameters)
- output text extraction

See `tools/openai_exec.py`.

### Retry Budget (Avoid Duplicate Paid Planner Requests)

There are two layers of retry behavior:

1) **HTTP retries** inside `tools/openai_exec.py` (for 429/5xx and transient network issues).
2) **Planner POST retries** inside `python3 tools/loop_runner.py plan` when the OpenAI job fails.

Planner POST retries are expensive because they create a new request. To reduce cost and avoid duplicate work:

- Default behavior is **one** planner attempt (`OPENAI_PLANNER_MAX_ATTEMPTS=1`).
- If you want automatic re-attempts on transient OpenAI-side failures, export a larger value in your shell, e.g.:
  - `export OPENAI_PLANNER_MAX_ATTEMPTS=2`

If a planner job fails, `loop_runner.py` will save the failure response JSON under `.advisor/openai/` when possible.

Raw OpenAI artifacts:

- Planner calls (`python3 tools/loop_runner.py plan`) save the request and full response JSON under `.advisor/openai/` (gitignored).
- If a planner call is interrupted mid-poll (network drop, Ctrl-C, Codex restart), rerun `python3 tools/loop_runner.py plan` to resume the in-progress response using `.advisor/state.json` (no duplicate planner request). You can also run `python3 tools/loop_runner.py resume`.
- Smoke tests (`python3 tools/live_smoke_test_planner.py`) save artifacts under `.advisor/openai_smoke/` (gitignored).

Live smoke test (real API call; tiny payload):

```bash
python3 tools/live_smoke_test_planner.py
```

To prove web search works end-to-end:

```bash
python3 tools/live_smoke_test_planner.py --require-web-search
```

## Recommended Interaction Contract (Default)

Use the OpenAI **Responses API** with **function calling** and `strict: true` so the advisor returns deterministic, schema-shaped output that Codex can parse.

Note: `gpt-5.2-pro` supports function calling but does **not** support response-format (“text.format”) structured outputs, so the contract below uses a function tool for the `OptimizationDirective`.

Important OpenAI constraint: when `strict: true` is enabled for a function tool, OpenAI requires the schema’s `required` array to include **every** key in `properties` (no optional properties). If a field is “optional” in spirit, include it anyway but allow an empty value (e.g., `[]` / `""`) or allow `null`.

Design principle: **Codex sends a small `IterationPacket`; advisor returns an `OptimizationDirective`.**

### Project Policy: Plan-only Advisor Output

For this project, the advisor output is **plan-only**:

- Advisor returns a concrete step-by-step plan.
- Codex writes and applies the code changes.
- Rationale: lower risk of forbidden edits (especially `tests/`) and easier enforcement of repo-local conventions.

Patch-generating advisor output (e.g., unified diffs) is intentionally **not used** here.

This repo’s README explicitly warns about modifying `tests/`; plan-only keeps that guardrail easier to enforce.

## Data Model

### `IterationPacket` (Codex → Advisor)

Send only what’s needed to decide the next iteration; keep it compact.

Minimum fields:

```json
{
  "iteration_id": 7,
  "git_sha": "abc1234",
  "threshold_target": 1500,
  "best_cycles": 1620,
  "current_cycles": 1604,
  "tests_unchanged": true,
  "tests_diff": "",
  "submission_tests_summary": "PASS some thresholds; cycles=1604",
  "recent_changes": {
    "files_changed": ["perf_takehome.py"],
    "summary": [
      "Refactor hot loop to reduce memory traffic",
      "Precompute constant masks"
    ]
  },
  "errors": [],
  "constraints": {
    "allowed_paths": ["perf_takehome.py"],
    "forbidden_globs": ["tests/**"],
    "notes": [
      "Do not change benchmark semantics",
      "Do not rely on multicore"
    ]
  }
}
```

If something failed, include concise failure details:

- Assertion messages / traceback top frames
- The exact line that prints cycle count (if available)

Avoid sending full diffs unless explicitly requested.

### Plateau/Pivot Protocol (Make the Advisor Creative)

Once the loop approaches a hard bottleneck (e.g., `min_cycles_by_engine["load"]` is close to the observed cycle count),
small instruction reshuffles tend to plateau. To force the advisor to “zoom out” and invent new directions safely,
Codex should provide *bottleneck telemetry* and the advisor prompt should include explicit pivot requirements.

Important: `threshold_target` is user-chosen and can be arbitrarily low. Treat it as:

- a **stop condition** (when you meet it), and
- a **feasibility check** (if it is below the current lower bounds, you must change the bounds),

but **do not** use it to decide how long to “force” a given strategy family. Pivot timing should be strategy-based.

#### Lower-bound gating (roofline-style)

For each iteration, compute a conservative lower bound and ask the advisor to reason against it.

At minimum, compute the **resource lower bound**:

- `resource_lb_cycles = max(min_cycles_by_engine.values())`

This is the “roofline” implied by per-engine throughput. The real optimum can be **higher** due to cross-engine
dependencies and critical-path constraints, so treat this as a *sanity bound*, not a guarantee.

Also compute a **critical-path lower bound** (dependency bound):

- `cp_lb_cycles = longest_dependency_chain_length` (unit weights; one task per cycle minimum along a chain)

Then define a better “sanity bound bundle”:

- `tight_lb_cycles = max(resource_lb_cycles, cp_lb_cycles)`

Use this bound in the advisor prompt as a feasibility check:

- If `threshold_target <= tight_lb_cycles`, then the current approach cannot meet the target via scheduling alone.
  The next plan MUST reduce the bound itself (usually by reducing the task count of the dominant engine and/or breaking a dependency chain).

Use it as a pivot trigger to avoid wasting iterations on diminishing returns:

- Define `headroom_cycles = best_cycles - tight_lb_cycles`
- Define `headroom_pct = headroom_cycles / tight_lb_cycles`
- If `headroom_pct <= 0.04` (within **~4%**) *and* you have plateaued for `N>=3` iterations (no new best),
  require a pivot to a new `strategy_family` that targets reducing the dominant bound, not reshuffling instructions.

Recommended default: **4%** (reasonable starting point). If you find you are pivoting too early, relax to ~5%; if you
find you are thrashing on micro-tweaks near the bound, tighten to ~3%.

#### Strategy families + pivot timing (avoid dead ends)

Define a small set of **strategy families** and encode them explicitly in the advisor directive (`strategy_family` +
`strategy_modifiers`) so the loop can avoid getting stuck and can enforce pivot timing deterministically.

**Family contract (required):**

- The advisor MUST set `strategy_family` to exactly one of:
  - `family:schedule` (packing/overlap; reduce `schedule_slack_pct` without materially changing bounds)
  - `family:reduce_loads` (lower `resource_lb_cycles`, typically fewer `load` tasks or wider loads)
  - `family:break_deps` (lower `cp_lb_cycles` by breaking dependency chains / changing dataflow)
- The advisor SHOULD set `strategy_modifiers` to 0–3 short strings, e.g.:
  - `gather`, `hash`, `addressing`, `scratch-layout`, `scheduler-heuristic`

`tools/loop_runner.py record` writes `strategy_tags = [strategy_family] + strategy_modifiers` into the experiment log so
the family remains stable even as the specific tactic changes.

Use this pivot policy:

- **Max 2 consecutive attempts per family.** After two tries, require a different family.
- **Two-strikes rule:** if you have **2 consecutive non-improving** attempts in the same family, pivot immediately.
- **One-bonus exception:** if the most recent attempt in a family produced a meaningful win (rule of thumb: ≥10 cycles improvement),
  allow **one extra** follow-up attempt in the same family before pivoting (i.e., max 3 consecutive).

Definitions (so this is enforceable from `experiments/log.jsonl`):

- An **attempt** is one appended entry in `experiments/log.jsonl` (usually produced by `python3 tools/loop_runner.py record`).
- A **family streak** is the contiguous suffix of the log where `strategy_tags[0]` is the same `family:*` value.
- The **streak best** is the minimum `cycles` observed so far in that streak among entries with `valid=true`.
- An attempt is **improving** iff `valid=true` and it sets a new streak best.
- An attempt is **non-improving** otherwise (including `valid!=true`).

Rationale: using global-best as the only comparator can prematurely kill promising families that start above the global
best but trend downward; streak-based improvement measures whether the family itself is still yielding signal.

Rationale: LLM implementations are noisy (a single regression may be an implementation issue), but iteration cost is
high—so you want fast exploration without thrashing.

#### Add a `performance_profile` section to every packet

Include a compact summary the advisor can reason from:

- `gap_to_target`: `best_cycles - threshold_target` (or `current_cycles - threshold_target`)
- `task_counts_by_engine`: total tasks per engine (real tasks only; excludes “idle filler”)
- `min_cycles_by_engine`: per-engine throughput bound (`ceil(task_count / SLOT_LIMITS[engine])`)
- `resource_lb_cycles`: `max(min_cycles_by_engine.values())`
- `cp_lb_cycles`: longest dependency-chain lower bound (unit weights)
- `tight_lb_cycles`: `max(resource_lb_cycles, cp_lb_cycles)`
- `critical_path_engine_counts`: counts of engines along one representative critical path
  - `dominant_engine_ops_on_cp` and `dominant_engine_cp_fraction` are useful proxies for “is the dominant engine structurally on the critical path?”
- `schedule_cycles_estimate`: `len(kb.instrs)` (the current scheduler’s predicted makespan)
- `schedule_slack_pct`: `(schedule_cycles_estimate - tight_lb_cycles) / tight_lb_cycles`
- `plateau_stats`: `iters_since_new_best`, `best_iteration_id`, `regressions_last_5`
- `top_cycle_limits`: the 2–3 engines closest to the observed cycles

This helps the advisor distinguish:

- “Still schedulable” improvements (reduce `schedule_slack_pct` by better overlap/packing), vs
- “Must change the algorithm” improvements (reduce `resource_lb_cycles` and/or `cp_lb_cycles`).

Implementation note: `python3 tools/loop_runner.py plan` includes a `performance_profile` for the submission case by default.

#### Force a pivot when stuck

Add hard requirements to the advisor prompt:

- **Bottleneck math first:** use `tight_lb_cycles` to decide whether scheduling-only work is plausible vs whether the plan
  must reduce bounds (e.g., fewer loads, fewer load-dependent stages, better reuse; or breaking serial chains).
- **Plateau rule:** if `iters_since_new_best >= N` (recommend `N=3`), the plan MUST use a new `strategy_family`
  (no overlap with the last N iterations), and explicitly explain what new mechanism it exploits.

#### Require a strategy portfolio (but execute one plan)

Creativity improves when the advisor must compare alternatives. Require:

- Propose **3 orthogonal approaches** (e.g., reduce load count, reduce dependency depth, reshape schedule/overlap).
- Pick **one** as the `step_plan`, and record the 2 rejected approaches briefly (e.g., in `change_summary` or a dedicated
  `alternatives_considered` list if you extend the schema).

#### Include one “wild card” idea (guardrail-safe)

Require exactly one high-risk but guardrail-compliant idea per plan:

- Must not touch `tests/` or benchmark semantics.
- If not executed now, specify what evidence/measurement would justify trying it.

#### Treat missing info as a valid outcome

If the advisor cannot propose a credible next step, it should request specific missing artifacts via
`next_packet_requests` (e.g., “send full perf_takehome.py”, “send per-round load task counts”, “send top task types by
critical path”).

#### Use web search strategically (optional)

If plateaued, allow the advisor to use `web_search` to find relevant mechanisms/prior art. The output should remain
plan-only and must stay within the repo’s anti-cheat guardrails.

### Optimization Memory: Experiment Log

To avoid repeating strategies, maintain an append-only experiment log:

- Canonical file (local, gitignored): `experiments/log.jsonl`
- Schema + guidance: `docs/experiment-log.md`

Each iteration packet should include:

- `experiment_log_tail`: last 10–20 lines of `experiments/log.jsonl`

The advisor should begin each plan with a **novelty check** against this tail: avoid proposing the same (family + modifiers)
combination unless it explains what is different and why it might work now.

### Getting “Deep Code Context” Without Full Diffs

If the advisor needs to reason deeply about the code, prefer **targeted, structured context** over full-file dumps:

0) **Bootstrap once, then go incremental**

- On iteration 0 (or whenever you switch baselines), it can be worth sending the full `perf_takehome.py` (or the full bodies of the key kernel-building functions) once so the advisor can form a correct mental model.
- After that, stick to targeted excerpts and changed-region context.

1) **Always pin the exact code version**

- Include `git_sha` and the exact branch name being evaluated.
- If you include code excerpts, include their file path and a stable anchor (function name and/or line numbers).

2) **Send the smallest complete unit**

Best default is “full function bodies” for the functions touched in the last iteration (plus ~10 lines of surrounding context).

3) **Include changed-region context**

Instead of a full diff, include either:

- `changed_regions`: a list of `{file, anchor, before_excerpt, after_excerpt}` (short excerpts), or
- `diff_hunks`: only the hunks for allowlisted files, capped to a small line budget.

4) **Escalate by request**

If the advisor can’t confidently propose a next step, it should request more context via `next_packet_requests` (e.g., “send full `perf_takehome.py`” or “send the body of `KernelBuilder.build_kernel` and its callers”).

### Recommended “Diff vs Context” Policy (Plan-only Advisor)

Diffs are useful, but they often omit the unchanged context needed to reason correctly. Use this policy to keep packets both **high-signal** and **small**:

1) **Always include a tiny summary**

- `git_sha`
- cycle count (current + best)
- `git diff --stat` (or an equivalent file/line summary)

2) **Send full diffs only when small (allowlisted files only)**

- If the diff for allowlisted files (usually just `perf_takehome.py`) is “small enough” (rule of thumb: a few hundred lines), include the full unified diff.
- Never include diffs for forbidden paths (`tests/**`) other than the empty `git diff origin/main tests/` output proving it’s unchanged.

3) **When diffs get large, switch to code context**

If the diff becomes large or touches many distant regions:

- Include full bodies of the functions you modified (plus a little surrounding context), and
- Include only the specific diff hunks that correspond to those functions.

4) **If the advisor asks for it, send the full allowlisted file**

When the advisor explicitly requests it (or you’re bootstrapping a new baseline), include the full `perf_takehome.py` once, then return to incremental packets.

### `OptimizationDirective` (Advisor → Codex)

Emit the directive as a **function tool call**, with arguments matching the schema below (plan-only; no patches).

Minimum fields:

```json
{
  "objective": "Reduce cycles by lowering memory pressure in the hot path",
  "primary_hypothesis": "We can avoid repeated loads by caching frequently-used values in registers",
  "expected_effect_cycles": -40,
  "risk": "Medium: may accidentally change semantics at boundaries",
  "step_plan": [
    "Identify the hottest loop in perf_takehome.py (based on trace/inspection)",
    "Rewrite the loop to reduce redundant loads/stores",
    "Add/adjust micro-asserts locally (not in tests/) to validate invariants during dev",
    "Run validation commands; if cycles improve and correctness holds, keep change"
  ],
  "validation": {
    "commands": [
      "git diff origin/main tests/",
      "python3 tests/submission_tests.py"
    ],
    "pass_criteria": [
      "tests diff is empty",
      "submission tests run completes",
      "cycle count is <= previous best"
    ]
  },
  "rollback_policy": {
    "revert_if": [
      "any correctness failure",
      "cycle count worsens by > 5",
      "touches forbidden paths"
    ]
  },
  "next_packet_requests": [
    "Include the specific loop/function name you changed",
    "Include the exact cycle count line from submission_tests.py output"
  ]
}
```

## Loop Control (Codex-side)

Suggested loop skeleton:

1) **Preflight**
   - Ensure working tree is clean *except* the intended changes.
   - Ensure the directive only touches allowlisted files.

2) **Implement**
   - Apply changes and keep them small per iteration.

3) **Validate**
   - Run:
     - `git diff origin/main tests/`
     - `python3 tests/submission_tests.py`
   - Parse cycle count.

4) **Record**
   - Update `best_cycles` if improved.
   - Summarize change (files + short bullet summary).
   - Append an entry to `experiments/log.jsonl` (wins and losses) so the advisor doesn’t repeat itself.

5) **Stop conditions**
   - Stop when `best_cycles <= threshold_target`, or when `max_iters` reached, or after `no_improvement_iters` consecutive regressions/no-gain.

### Helper Script: `tools/loop_runner.py`

This repo includes a small helper to make the loop more repeatable:

1) Create a fresh `iter/*` branch off `main` and ask the advisor for the next plan:

```bash
python3 tools/loop_runner.py plan --threshold 1363
```

2) After you implement the plan locally, benchmark and append a JSONL entry to your local experiment log:

```bash
python3 tools/loop_runner.py record --print-test-output
```

Notes:
- “Real” planner calls should poll at the default **60s cadence**. `tools/loop_runner.py` loads only `OPENAI_API_KEY` from `.env` and unsets any fast-poll env vars so smoke-test settings don’t leak into real runs.
- Smoke tests can still force fast polling by exporting `OPENAI_BACKGROUND_POLL_INTERVAL` / `OPENAI_BACKGROUND_PROGRESS_EVERY` in the shell when running `tools/live_smoke_test_planner.py`.

## Prompting Guidance (Advisor)

Keep the advisor focused on *next-step decisions*, not long essays.

Advisor instructions to include (as system/developer content in your API call):

- “You are optimizing cycles while preserving correctness.”
- “Never suggest modifying `tests/`.”
- “Prefer changes in `perf_takehome.py`.”
- “You may use web search if needed, but you must return the final directive by calling the `emit_optimization_directive` function tool (no plain-text answer).”
- “If information is missing, request it via `next_packet_requests` rather than guessing.”

## What To Send (Token Hygiene)

Recommended: send only:

- cycle count and pass/fail status
- short change summary (not full diff)
- the relevant hot function excerpt *only if needed* (≤ ~80 lines)
- any error/traceback snippet that explains failures

Avoid sending entire files or large traces unless the advisor explicitly requests them.

## Git Workflow: Branch-per-Turn PRs (Merge on New Best)

Default branch is `main`. Do not commit directly to `main`.

Each iteration (“turn”) starts from a fresh branch off `main`. If (and only if) the iteration produces a **new all-time best** cycle count and correctness holds, merge it back into `main` via PR.

Suggested flow:

1) **Start a new iteration branch**

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
git checkout -b iter/0007-short-desc
```

2) **Implement + validate**

```bash
git diff origin/main tests/
python3 tests/submission_tests.py
```

3) **If (and only if) it’s a new best, open a PR and merge**

```bash
git commit -am "feat: iter/0007-short-desc"
git push -u origin iter/0007-short-desc
gh pr create --fill --base main --head iter/0007-short-desc
gh pr merge --squash --delete-branch --yes
```

If the iteration did **not** improve cycles or failed correctness, do **not** merge it into `main`.

4) **Record the attempt**

Record every attempt in `experiments/log.jsonl` (schema: `docs/experiment-log.md`) and include a tail of the log in each planner packet so the advisor avoids repeating strategies.
