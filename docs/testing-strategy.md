# Test Strategy (Loop Tooling + Performance Kernel)

This repo has two distinct risk surfaces:

1) **Kernel correctness + performance** (`perf_takehome.py`)
2) **Loop tooling** (`tools/loop_runner.py`)

The goal is to keep iteration velocity high while preventing the two failure modes that waste the most time/money:

- **Incorrect results** (kernel breaks correctness)
- **Loop “thrash”** (repeatedly trying the same strategy family, or re-posting expensive planner requests)

## Test Pyramid (Target Split)

Aim for roughly:

- **70–80% unit** (fast, deterministic, isolated)
- **15–25% integration** (real modules + small problem sizes)
- **5–10% E2E** (hermetic by default; live is gated)

## Layers and What They Cover

### Unit tests (fast, hermetic)

Run:

```bash
python3 -m unittest discover -s tools/tests
```

Primary targets:

- `tools/loop_runner.py`
  - Parsing: `CYCLES:` extraction, correctness parsing (treat speed-threshold failures as correct).
  - Bottleneck math: throughput lower bounds, critical-path proxy extraction.
  - Guardrails: directive schema shape + tests/ mutation detection + allowlist enforcement.
  - Pivot policy: strategy-family streak accounting and blocking rules.

Acceptance criteria examples (Given/When/Then):

- **Given** `offline-plan`, **when** planning an iteration, **then** no network calls are made and `.advisor/state.json` is written.
- **Given** a `submission_tests.py` output that only fails the speed threshold, **when** parsing correctness, **then** it is treated as correctness-pass (so we still record useful iteration data).
- **Given** 2 consecutive attempts in `family:schedule` without a meaningful win, **when** planning the next iteration, **then** `family:schedule` is blocked.

### Integration tests (small, real code paths)

Goal: catch “it imports but doesn’t actually work” regressions without running the full (slow) submission harness.

Examples:

- Build a small kernel via `KernelBuilder.build_kernel(...)` and assert basic invariants:
  - instr stream is non-empty
  - scratch usage is within bounds
  - no exceptions during build

Current coverage:

- `tools/tests/test_perf_takehome_invariants.py`

### Hermetic E2E tests (no network, no mutation of your working repo)

Goal: validate the “plan → implement → record” mechanics and git guardrails without calling OpenAI.

Approach:

- Spin up a throwaway temp git repo with a local bare `origin`.
- Include only the minimal files needed for `tools/loop_runner.py`.
- Run:
  - `python3 tools/loop_runner.py offline-plan ...`
  - `python3 tools/loop_runner.py record`
- Assert:
  - `.advisor/state.json` is created
  - an entry is appended to `experiments/log.jsonl`
  - `strategy_tags` are derived consistently (`[strategy_family] + strategy_modifiers`)

Current coverage:

- `tools/tests/test_e2e_loop_runner_hermetic.py`

## Non-Functional Checkpoints (High Value Here)

- **Determinism:** hermetic tests must not depend on network or wall clock; avoid randomness unless seeded.
- **Security:** never write secrets into artifacts; `.env` and `.advisor/` must stay gitignored.
- **Reliability:** transient OpenAI job failures should be persisted (failure artifacts) and should not wedge the loop indefinitely.

## What This Suite Is *Not*

- It does not replace `python3 -B tests/submission_tests.py` as the source of truth for cycles/correctness.
- It intentionally avoids modifying anything under `tests/`.

## Appendix: Identified Test Cases (Prioritized Backlog)

This appendix is a concrete “what to test” list (per layer). It’s intentionally redundant with the suite: it serves as a
living checklist for gaps and for future refactors.

### Unit (tools/loop_runner.py)

1) **Cycles parsing picks the last `CYCLES:` line**
2) **Correctness parsing treats speed-threshold-only failures as correct**
3) **Bound math is stable**
   - `min_cycles_by_engine`, `resource_lb_cycles`, and the critical-path proxy behave as expected on small graphs.
4) **Strategy family pivot enforcement**
   - After 2 consecutive attempts in the same family without a meaningful win, block that family for the next directive.
   - Allow a one-attempt bonus when the last attempt improved the prior streak best by ≥10 cycles.
5) **Schema constraint**
   - When a family is blocked, `_planner_directive_schema(...)` should not allow it in `strategy_family`.

### Integration (perf_takehome.py ↔ problem.py)

1) **Kernel build smoke (small sizes)**
   - Assert: instruction stream is non-empty; scratch usage stays within bounds; build does not crash.

### Hermetic E2E (no OpenAI; throwaway temp repo)

1) **Offline plan → record**
   - Assert: `.advisor/state.json` created; `experiments/log.jsonl` appended; `strategy_tags` derived from family/modifiers.
2) **Guardrail: tests/ mutation makes iteration invalid**
   - Assert: `record` marks `tests_diff_empty=false` and `valid=false` (and exits non-zero).
3) **Pivot metadata is surfaced**
   - Seed the experiment log with a “streak” and assert the packet includes `strategy_family_constraints.blocked_families`.
