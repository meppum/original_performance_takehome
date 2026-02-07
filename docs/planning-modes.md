# Planning Modes (Loop Runner)

This repo supports multiple ways to produce an `OptimizationDirective` (planner output) and then apply it with Codex CLI (implementation).

## Shared Concepts

- Planner output is always written to `.advisor/state.json`.
- Implementation reads `.advisor/state.json` and executes `directive.step_plan`.
- `python3 tools/loop_runner.py record` is the source of truth for correctness + cycles and appends to `experiments/log.jsonl` (gitignored).
- Guardrail: never modify `tests/` (record enforces this).

## Modes

### 1) Codex Planner (ChatGPT login → plan, ChatGPT login → implement)

- Planner: `python3 tools/loop_runner.py codex-plan ...`
- One-iteration wrapper: `tools/codex_planner_exec.sh ...`
- Auth: requires `CODEX_HOME` ChatGPT login (`codex login --device-auth`).

### 2) Codex API Planner (API key → plan, ChatGPT login → implement)

- Planner: `python3 tools/loop_runner.py codex-api-plan ...`
- One-iteration wrapper: `tools/codex_api_planner_exec.sh ...`
- Auth:
  - Planning requires either:
    - `CODEX_API_KEY` (env or local `.env`). `OPENAI_API_KEY` is accepted as an alias.
    - Or a stored API-key login in `CODEX_HOME` (`codex login --with-api-key`).
  - Implementation requires a ChatGPT login in `CODEX_HOME` (`codex login --device-auth`).
- Default planner model: `gpt-5.2-pro` (override with `--model`).

### 3) Manual Planner (ChatGPT UI copy/paste → plan, ChatGPT login → implement)

- Generate packet/prompt on a `plan/*` branch: `python3 tools/loop_runner.py manual-pack ...`
- Paste ChatGPT JSON into `planner_packets/directive.json` and commit.
- Materialize `iter/*` + `.advisor/state.json`: `python3 tools/loop_runner.py manual-apply ...`
- One-shot apply wrapper: `tools/manual_planner_exec.sh ...`

### 4) Offline Planner (hermetic; no network)

- `python3 tools/loop_runner.py offline-plan ...`
- Writes a stub directive (useful for tests and runner sanity checks).

## Recommended Long-Run Loops

### Codex planner loop (all ChatGPT login)

```bash
CODEX_HOME="$PWD/.codex_home" codex login --device-auth

while true; do
  tools/codex_planner_exec.sh --goal best --slug next || break
done
```

### Hybrid loop (API-key planning, ChatGPT-login implementation)

```bash
CODEX_HOME="$PWD/.codex_home" codex login --device-auth

# One-time: login for planning using a separate Codex home (recommended).
mkdir -p .codex_home_api
printenv OPENAI_API_KEY | CODEX_HOME="$PWD/.codex_home_api" codex login --with-api-key
# Verify: CODEX_HOME="$PWD/.codex_home_api" codex login status  # should say "Logged in using an API key"

# The wrapper automatically uses .codex_home_api for planning if it exists.
# (Override with: CODEX_HOME_PLANNER=... tools/codex_api_planner_exec.sh ...)

while true; do
  tools/codex_api_planner_exec.sh --goal best --slug next || break
done
```

## Implementation Overrides (All Wrappers)

All `tools/*_planner_exec.sh` wrappers support:

- `--impl-model <model>` → passed to the implementation `codex exec -m ...`
- `--impl-reasoning-effort <effort>` → passed as `-c model_reasoning_effort="..."`

Example:

```bash
tools/codex_api_planner_exec.sh --goal best --slug next \
  --impl-model gpt-5.2 \
  --impl-reasoning-effort high
```
