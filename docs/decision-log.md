# Decision Log

This file records **durable, non-trivial decisions** about how the Codex↔advisor loop operates. It is not for per-iteration tactics; those belong in `experiments/log.jsonl`.

## 2026-02-01 — Plan-Only Advisor + Background Mode

- Decision: `gpt-5.2-pro` is an **advisor** that outputs a step-by-step plan; Codex CLI is the sole code editor.
- Decision: hardcode planner reasoning effort to `xhigh`.
- Decision: enable background mode automatically for `xhigh` requests and **poll** until completion, printing a heartbeat at least every 60s.
- Decision: enable optional planner research via the Responses API `web_search` tool.
- Decision: use **strict function calling** (tool parameters JSON Schema) for structured planner output.
  - Rationale: `gpt-5.2-pro` supports function calling; response-format structured outputs (`text.format`) are not supported reliably.

References:
- `docs/openai-advisor-loop.md`
- `tools/openai_exec.py`

## 2026-02-02 — Local Experiment Log (Avoid Losing Memory)

- Decision: keep `experiments/log.jsonl` as a **local, gitignored** JSONL file so it persists across frequent `iter/*` branch creation/deletion.
- Decision: keep `experiments/log.jsonl.example` tracked as a schema seed and sample.

Rationale:
- The “branch-per-turn, merge-only-on-new-best” workflow would otherwise drop non-merged experiment history, increasing the chance of repeating failed strategies.

## 2026-02-02 — Plateau/Pivot Protocol (Creativity Forcing)

- Decision: include a small `performance_profile` section in each advisor packet (bottleneck telemetry like `min_cycles_by_engine`, task counts, and plateau stats).
- Decision: require a lower-bound feasibility check each iteration using `resource_lb_cycles = max(min_cycles_by_engine.values())`; if the target is below this bound, the next plan must reduce the bound (not just reschedule).
- Decision: treat “near bound” as a pivot signal (recommended default: within ~4% of the current resource lower bound, plus a plateau window) to avoid wasting iterations on micro-tweaks.
- Decision: when the loop is plateaued (e.g., no new best in N iterations), require the advisor to pivot to a new `strategy_tags` family and explain the new mechanism.
- Decision: require a portfolio of 3 orthogonal approaches (reduce load count, reduce dependency depth, reshape overlap) and then pick one to execute.
- Decision: include exactly one “wild card” idea per plan (high risk but guardrail-safe), with evidence criteria for when to try it.
- Decision: treat missing info as a valid outcome; the advisor should request specific artifacts via `next_packet_requests` instead of guessing.
- Decision: allow optional `web_search` during plateau periods to discover mechanisms/prior art, while keeping output plan-only.

References:
- `docs/openai-advisor-loop.md`
