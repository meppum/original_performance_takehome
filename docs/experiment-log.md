# Experiment Log (Optimization Memory)

The goal of this log is to prevent the Codex↔advisor loop from repeatedly trying the same ideas. It captures:

- what was tried (strategy tags + change summary)
- what happened (correctness + cycles)
- what we learned (why it helped/hurt)

It is designed to be both:

- **machine-friendly** (JSONL so Codex can reliably extract a recent digest), and
- **planner-friendly** (short, structured entries).

## Files

- Canonical log (local, gitignored): `experiments/log.jsonl` (append-only; one JSON object per line)
- Sample/schema seed (tracked): `experiments/log.jsonl.example`

## Current Status (This Repo)

As of 2026-02-01:

- Best known cycle count (submission harness): **1443 cycles**
  - Where: `main` @ `4ba7986` (merged via PR #1 from `perf/optimize-kernel`)
  - Validation: `python3 -B tests/submission_tests.py`
  - Status: correctness passes; still fails the strictest threshold (`test_opus45_improved_harness`, `<1363`)

### What We Did So Far (High-Level)

Starting from the initial optimization work (before adopting the gpt-5.2-pro “advisor loop” contract):

1) Built an optimized kernel generator in `perf_takehome.py`:
   - Task-graph + list scheduling into VLIW bundles
   - Keep batch state in scratch (single `vload`/`vstore`)
   - Fuse 3 hash stages with `valu.multiply_add`
   - Special-case early-depth rounds to reduce gathers
   - Result: **1878 cycles** (correct, but not enough to pass `<1790`)

2) Iterated on scheduler heuristics:
   - Key discovery: scheduling *short* critical-path tasks first (`cp` ascending) dramatically improves overlap.
   - Result: **1443 cycles** (current best; merged to `main`)

3) Notable attempted-but-reverted tweaks:
   - Dropping WAR/anti-dependencies in the scheduler broke correctness (used stale scratch values inside bundles).
   - Replacing `flow.vselect` with mask-based selects in `valu` increased cycles (1913) despite removing `flow`.

## When to Write an Entry

Write exactly one entry per iteration branch once you have results:

- after `python3 tests/submission_tests.py` finishes (pass or fail)
- after you’ve confirmed `git diff origin/main tests/` is empty

Tip: `python3 tools/loop_runner.py record` appends an entry after running the submission tests.

Even if an iteration does not beat the best cycle count, **record it** so the advisor can avoid repeating it.

Practical note: because failed `iter/*` branches are usually not merged (and often deleted), the log is kept **out of git** (gitignored) so it persists across branch switches.

## Minimal Schema (One Line Per Iteration)

Each line in `experiments/log.jsonl` should be a single JSON object with these fields:

- `iteration_id` (integer, monotonically increasing)
- `timestamp_utc` (ISO 8601 string)
- `branch` (e.g., `iter/0007-short-desc`)
- `base_branch` (e.g., `main`)
- `base_sha` (git SHA the iteration started from)
- `head_sha` (git SHA that was benchmarked)
- `files_changed` (array of paths)
- `tests_diff_empty` (boolean)
- `valid` (boolean: submission tests ran and correctness checks passed)
- `cycles` (number or null if not available)
- `delta_vs_best` (number or null; negative is better)
- `strategy_tags` (array of short strings; see tag guidance below)
- `hypothesis` (short string)
- `change_summary` (array of 1–5 short bullets)
- `result_summary` (short string: what happened + any thresholds passed)
- `merged_to_main` (boolean)
- `notes` (optional string; “what we learned / next lead”)

Example entry:

```json
{"iteration_id":7,"timestamp_utc":"2026-02-01T03:10:00Z","branch":"iter/0007-cache-path","base_branch":"main","base_sha":"abc1234","head_sha":"def5678","files_changed":["perf_takehome.py"],"tests_diff_empty":true,"valid":true,"cycles":1604,"delta_vs_best":-16,"strategy_tags":["family:reduce_loads","gather","addressing"],"hypothesis":"Cache hot values to avoid redundant loads","change_summary":["Hoist repeated loads out of inner loop","Replace branch with mask/sel"],"result_summary":"PASS correctness; cycles=1604 (improved)","merged_to_main":true,"notes":"Helped; next try unrolling by 2 with same cache layout."}
```

## Strategy Tags (Keep Them Small and Consistent)

Tags are the main mechanism for “don’t repeat yourself.” Prefer 1–4 tags per iteration.

Good tags are short and specific, like:

- `family:schedule`, `family:reduce_loads`, `family:break_deps` (required first tag; see below)
- `gather`, `hash`, `addressing`
- `strength-reduce`, `const-fold`
- `branch-elim`, `cmov-mask`
- `unroll`, `software-pipeline`
- `task-fusion`, `dep-break`
- `layout-change`, `prefetch`

Avoid tags that are too vague:

- bad: `optimize`, `speedup`, `refactor`

### Family Tag Contract (Required)

To make it possible to enforce pivot timing and avoid dead ends, `strategy_tags` must encode a stable “strategy family”.

Contract:

- `strategy_tags[0]` MUST be exactly one of:
  - `family:schedule` (packing/overlap; reduce scheduler slack)
  - `family:reduce_loads` (lower load/resource bound by issuing fewer loads or wider loads)
  - `family:break_deps` (lower dependency bound by breaking serial chains / changing dataflow)
- `strategy_tags[1:]` are optional modifiers (up to 3) describing the concrete tactic (e.g., `gather`, `hash`, `addressing`).

Why: tactics change iteration-to-iteration; families should remain stable so we can measure whether a line of attack is
still producing improvements.

## How the Advisor Should Use This Log

Every advisor plan should begin with a **novelty check**:

1) Look at the last ~10 entries and their `strategy_tags`.
2) Avoid repeating the same tag combination unless it explains what is different.
3) Prefer exploring a new axis when stuck (e.g., if three `unroll` attempts fail, try `reduce-loads` or `dep-break`).

## What to Send in Each Iteration Packet

To keep prompts small, don’t send the entire log. Send:

- `experiment_log_tail`: last 10–20 lines of `experiments/log.jsonl`
- optionally `experiment_log_digest`: a tiny derived summary (e.g., “best delta per tag”)
