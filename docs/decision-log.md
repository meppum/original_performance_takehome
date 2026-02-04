# Decision Log

This file records **durable, non-trivial decisions** about how the Codex↔advisor loop operates. It is not for per-iteration tactics; those belong in `experiments/log.jsonl`.

## 2026-02-01 — Plan-Only Planner Directive

- Decision: planner output is **plan-only** (an `OptimizationDirective`); Codex CLI is the sole code editor.
- Decision: validate planner output against a JSON Schema before writing `.advisor/state.json`.

References:
- `docs/planning-modes.md`
- `tools/loop_runner.py`

## 2026-02-03 — Goal Modes: Threshold vs Best

- Decision: support two explicit optimization objectives across all planner modes (`offline-plan`, `manual-pack`, `codex-plan`, `codex-api-plan`):
  - `--goal threshold` (default): a fixed stop condition via `threshold_target`.
  - `--goal best`: search for a **NEW BEST** (no fixed stop threshold).
- Decision: when `goal=best`, `threshold_target` is intentionally unset (`null`) and the packet includes:
  - `aspiration_cycles = best_cycles - 1` as a soft target (not a stop condition).
  - `plateau_valid_iters_since_best` to encourage pivots when progress stalls.

References:
- `tools/loop_runner.py`

## 2026-02-03 — Codex Planner Modes

- Decision: add `python3 tools/loop_runner.py codex-plan` to spawn `codex exec` (read-only) and produce an `OptimizationDirective`.
- Decision: add `python3 tools/loop_runner.py codex-api-plan` to do the same but require `CODEX_API_KEY` (default planner model: `gpt-5.2-pro`).
- Decision: run the Codex planner in a **read-only** sandbox and validate its output against the same directive schema.
- Decision: persist Codex planner artifacts under:
  - `.advisor/codex/` (ChatGPT-login planner)
  - `.advisor/codex_api/` (API-key planner)

References:
- `docs/planning-modes.md`
- `tools/loop_runner.py`

## 2026-02-03 — Rolling Best Base (`opt/best`) + Reproducible Records

- Decision: use `opt/best` as a rolling base branch so improvements accumulate without merging to `main`.
  - On **NEW BEST**, fast-forward `opt/best` to the new best commit and push to origin.
  - Initialize/update with: `python3 tools/loop_runner.py ensure-best-base`.
- Decision: make experiment records reproducible by committing before benchmarking.
  - This ensures `experiments/log.jsonl.head_sha` matches the tested code.

References:
- `tools/codex_planner_exec.sh`
- `tools/loop_runner.py`

## 2026-02-03 — File-Scope Enforcement in `record`

- Decision: enforce anti-cheat scope in code (not just prompts) during `python3 tools/loop_runner.py record`.
  - Allowed (default): `perf_takehome.py`
  - Forbidden (default): `tests/**`, `problem.py`

References:
- `tools/loop_runner.py`

## 2026-02-04 — Global Best from Durable `best/*` Tags

- Decision: derive `best_cycles` as the minimum across:
  - durable `best/*` tags (parsed from the tag name), and
  - the local experiment log (`experiments/log.jsonl`).
- Decision: treat the local experiment log as **optimization memory**, not the source of truth for the global best.

Rationale:
- `experiments/log.jsonl` is gitignored and can be missing/stale on fresh clones or multi-machine runs.
- Using tags prevents accidentally fast-forwarding `opt/best` to a worse commit due to missing local history.

References:
- `tools/loop_runner.py`

## 2026-02-04 — `record` Emits Machine-Readable Outcome Fields

- Decision: `python3 tools/loop_runner.py record` prints a single JSON object to stdout and includes:
  - `new_best` (boolean)
  - `threshold_met` (boolean)
  - `best_before` (cycles used as the comparison baseline)
- Decision: wrappers should key off these JSON fields (not grepping human text).

References:
- `tools/loop_runner.py`
- `tools/codex_planner_exec.sh`

## 2026-02-02 — Local Experiment Log (Avoid Losing Memory)

- Decision: keep `experiments/log.jsonl` as a **local, gitignored** JSONL file so it persists across frequent `iter/*` branch creation/deletion.
- Decision: keep `experiments/log.jsonl.example` tracked as a schema seed and sample.

Rationale:
- The “branch-per-turn, merge-only-on-new-best” workflow would otherwise drop non-merged experiment history, increasing the chance of repeating failed strategies.

## 2026-02-02 — Plateau/Pivot Protocol (Creativity Forcing)

- Decision: include a small `performance_profile` section in each advisor packet (bottleneck telemetry like `task_counts_by_engine`, `min_cycles_by_engine`, `cp_lb_cycles`, and plateau stats).
- Decision: require a lower-bound feasibility check each iteration using `tight_lb_cycles = max(resource_lb_cycles, cp_lb_cycles)`.
  - If the target is below this bound, the next plan must reduce the bound (not just reschedule).
- Decision: treat “near bound” as a pivot signal (recommended default: within ~4% of the current tight lower bound, plus a plateau window) to avoid wasting iterations on micro-tweaks.
- Decision: do not use `threshold_target` to decide pivot timing (it is user-chosen and can be arbitrarily low); use it only as a stop condition and feasibility check.
- Decision: pivot based on strategy families to avoid dead ends:
  - Family tag contract: `strategy_tags[0]` is required and must be one of `family:schedule`, `family:reduce_loads`, `family:break_deps`.
  - Max 2 consecutive attempts per `strategy_tags` family.
  - Two-strikes rule: pivot after 2 consecutive non-improving attempts within the same family.
    - “Non-improving” is measured against the best `cycles` seen so far in the current family streak (contiguous segment of the experiment log with the same family tag), considering only `valid=true` entries.
    - Attempts with `valid!=true` count as non-improving for strike purposes.
  - One-bonus exception: after a meaningful win (≈≥10 cycles), allow one extra follow-up attempt in the same family.
- Decision: when the loop is plateaued (e.g., no new best in N iterations), require the advisor to pivot to a new `strategy_tags` family and explain the new mechanism.
- Decision: require a portfolio of 3 orthogonal approaches (reduce load count, reduce dependency depth, reshape overlap) and then pick one to execute.
- Decision: include exactly one “wild card” idea per plan (high risk but guardrail-safe), with evidence criteria for when to try it.
- Decision: treat missing info as a valid outcome; the advisor should request specific artifacts via `next_packet_requests` instead of guessing.
- Decision: allow optional `web_search` during plateau periods to discover mechanisms/prior art, while keeping output plan-only.

References:
- `docs/planning-modes.md`
