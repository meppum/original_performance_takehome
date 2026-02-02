# Test Strategy (Loop Tooling + Performance Kernel)

This repo has two distinct risk surfaces:

1) **Kernel correctness + performance** (`perf_takehome.py`)
2) **Codex↔advisor loop tooling** (`tools/loop_runner.py`, `tools/openai_exec.py`)

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

- `tools/openai_exec.py`
  - Payload construction: background auto-enable for `xhigh`, strict function tool, optional `web_search`.
  - Resilience: HTTP retry/backoff behavior and error surfacing.
  - Output parsing: function-call argument extraction and output-text extraction.
  - Artifact safety: stem/schema sanitization.
- `tools/loop_runner.py`
  - Parsing: `CYCLES:` extraction, correctness parsing (treat speed-threshold failures as correct).
  - Bottleneck math: throughput lower bounds, critical-path proxy extraction.
  - Guardrails: default polling cadence enforcement, directive schema shape.
  - Pivot policy: strategy-family streak accounting and blocking rules.

Acceptance criteria examples (Given/When/Then):

- **Given** `reasoning_effort="xhigh"`, **when** making a planner request, **then** the request runs in background mode and can be polled safely.
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
  - `python3 tools/loop_runner.py plan --offline ...`
  - `python3 tools/loop_runner.py record`
- Assert:
  - `.advisor/state.json` is created
  - an entry is appended to `experiments/log.jsonl`
  - `strategy_tags` are derived consistently (`[strategy_family] + strategy_modifiers`)

Current coverage:

- `tools/tests/test_e2e_loop_runner_hermetic.py`

### Live E2E (optional, explicitly gated)

Live tests are useful as “canary” checks for:

- auth/env wiring (`OPENAI_API_KEY`, endpoint overrides)
- response schema drift / API changes
- background polling behavior in real conditions

They are also **expensive and flaky by nature**, so they should not run by default.

Current tooling:

- `python3 tools/live_smoke_test_planner.py`

Suggested gating rule (if we automate live tests later):

- Only run when both are set:
  - `RUN_LIVE_TESTS=1`
  - `OPENAI_API_KEY=...`

## Non-Functional Checkpoints (High Value Here)

- **Cost control:** planner POST retry budget defaults to 1; retries should prefer resuming/polling rather than re-POSTing.
- **Determinism:** hermetic tests must not depend on network or wall clock; avoid randomness unless seeded.
- **Security:** never write secrets into artifacts; `.env` and `.advisor/` must stay gitignored.
- **Reliability:** transient OpenAI job failures should be persisted (failure artifacts) and should not wedge the loop indefinitely.

## What This Suite Is *Not*

- It does not replace `python3 -B tests/submission_tests.py` as the source of truth for cycles/correctness.
- It intentionally avoids modifying anything under `tests/`.

