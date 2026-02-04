# Search Strategies for the Optimization Loop (Bandits, Frontier Search)

This document captures **ideas for making the loop smarter and faster** by improving *which kind of change we try next*,
without relaxing correctness/scope guardrails and without any OpenAI API calls.

Nothing in this document is implemented yet; treat it as a design scratchpad.

## Context: what the loop can control

The current workflow:

- Starts each attempt from a rolling best base branch (`opt/best`).
- Uses a planner (Codex, read-only) to produce a structured directive (`OptimizationDirective`).
- Uses an executor (Codex) to implement the directive.
- Uses `python3 tools/loop_runner.py record` as the only source of truth:
  - correctness (`correctness_pass`)
  - performance (`cycles`)
  - scope enforcement (`scope_ok`, `tests_diff_empty`)

So the “search” problem is: **given the history in `experiments/log.jsonl`, choose what to try next** (e.g., what
strategy family to pursue), then ask the planner to propose the concrete code edit.

## Multi-Armed Bandits (MAB)

### What are the “arms”?

In this loop, the cleanest arms are:

- `strategy_family` (best default): `family:schedule`, `family:reduce_loads`, `family:break_deps`

Other possible arm definitions (higher variance / more complex):

- `(strategy_family, strategy_modifiers)` combinations
- “macro tactics” like “reduce LB”, “improve slack”, “break critical path depth”, if we formalize them

### What is the “reward”?

The reward should be computable from one `record` entry.

Good options:

- **Bernoulli (success/fail):** `success = (valid == true && NEW_BEST)`
  - Pros: robust to noise; best-aligned with “find best possible”
  - Cons: discards magnitude information
- **Magnitude when successful:** `improvement = best_before - cycles` when `success`, else `0`
  - Pros: prefers bigger wins
  - Cons: noisier; depends on best_before which changes over time

Notes:

- If `valid != true`, reward should be `0` (or “invalid”) and should count against the responsible arm.
- The system is **non-stationary**: as `opt/best` improves, what works changes. Any bandit needs recency awareness.

### Recommended default: hierarchical Thompson sampling with decay

Use a bandit over `strategy_family` to choose which family to try next:

1. For each family, track `(successes, failures)` over a **recent window** or with **exponential decay**.
2. Sample a probability of success per family from `Beta(successes+1, failures+1)`.
3. Pick the family with the highest sampled probability.
4. Provide this as a **soft prior** to the planner:
   - include a `family_prior` list in the planner packet, e.g. `["family:reduce_loads", "family:break_deps", ...]`
   - planner can override, but must justify when it does

Why Thompson sampling:

- Simple, robust, and naturally balances exploration vs exploitation.
- Works well with sparse rewards like “NEW BEST happened”.

Where decay/windowing fits:

- Sliding window: only count the last N valid attempts, or last N attempts overall
- Exponential decay: weight attempts by `w = exp(-age / tau)` for a configurable `tau` (in iterations)

### Sharp edges / failure modes

- **Non-stationarity:** without decay, the bandit will “lock in” to what used to work.
- **Cold start:** force at least one attempt per family per “era” (e.g., per 20 iterations).
- **Credit assignment:** “family” is coarse; big within-family variance can hide useful distinctions.
- **Over-exploitation:** cap max consecutive picks, or inject epsilon exploration (e.g., 10% random family).
- **Bound-limited regimes:** if `tight_lb_cycles` is near current best, wins are structurally rare; the bandit can only
  reduce wasted repetition, not guarantee new bests.

## Frontier / Population Search (alternative)

Instead of always exploring from `opt/best`, maintain a small **frontier** of diverse high-quality candidates:

- Keep top-K branches/commits (e.g., best + a few near-best but “different” attempts).
- Explore from multiple parents to escape local minima (best-first / beam search / evolutionary flavor).

Trade-offs:

- Higher bookkeeping cost (multiple bases, more tags/branches).
- Must preserve the invariant: **only fast-forward `opt/best` on true NEW BEST**.

This is complementary to bandits:

- Bandit chooses *what kind* of change.
- Frontier search chooses *where* to branch from.

## Next design questions (before implementing)

1. Arm definition: `strategy_family` only, or `(family, modifiers)`?
2. Reward: Bernoulli NEW BEST only, or include improvement magnitude?
3. Determinism: seed selection by `iteration_id` for reproducibility, or allow true randomness?
4. Non-stationarity: sliding window size vs exponential decay time constant?
5. Guardrails: how strongly should the planner be allowed to override the bandit suggestion?

