# Agent Instructions (original_performance_takehome)

This repo is a **performance optimization harness**. The primary goal is to **minimize simulated CPU cycles** while preserving correctness.

## Success Metric (Source of Truth)

- Run: `python3 tests/submission_tests.py`
- Use the **cycle count it prints** as the only performance metric.

## Non-Negotiables (Anti-Cheating Guardrails)

1) **Never modify `tests/`**

- Before and after every iteration, run:
  - `git diff origin/main tests/`
- It must be empty.

2) **Do not change benchmark semantics**

- Treat edits to `problem.py` as **high-risk** and do not touch it unless explicitly requested.
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
git commit -am "feat: iter/0007-short-desc"
git push -u origin iter/0007-short-desc

# open PR (or update an existing PR)
gh pr create --fill --base main --head iter/0007-short-desc

# merge PR (prefer squash + delete branch)
gh pr merge --squash --delete-branch --yes
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
