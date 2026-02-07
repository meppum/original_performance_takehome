#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Always default Codex to repo-scoped home dirs. This keeps the loop self-contained and ensures
# instructions/config are pulled from this working tree instead of the global `~/.codex`.
#
# - Implementation (apply) uses ChatGPT login via CODEX_HOME (default: .codex_home)
# - Planning (codex-api-plan) prefers a separate API-key home if present (default: .codex_home_api)
codex_home_impl="${CODEX_HOME:-$repo_root/.codex_home}"
codex_home_planner="${CODEX_HOME_PLANNER:-$repo_root/.codex_home_api}"
if [[ -z "${CODEX_HOME_PLANNER:-}" ]] && [[ ! -d "$codex_home_planner" ]]; then
  codex_home_planner="$codex_home_impl"
fi

# Fast-fail with a clear message if this repo-scoped Codex home isn't authenticated yet.
#
# IMPORTANT: We intentionally check ChatGPT login WITHOUT any API key env vars present. This mode uses
# CODEX_API_KEY (or OPENAI_API_KEY as an alias) for *planning*, but uses ChatGPT login for *implementation*.
if ! CODEX_HOME="$codex_home_impl" env -u CODEX_API_KEY -u OPENAI_API_KEY codex login status >/dev/null 2>&1; then
  echo "[codex_loop] Codex is not logged in for implementation (ChatGPT login)."
  echo "[codex_loop] Run: CODEX_HOME=\"$codex_home_impl\" codex login --device-auth"
  echo "[codex_loop] Then re-run this loop."
  exit 2
fi

# Convenience wrapper for one "plan + apply + record" iteration using Codex as the planner
# (authenticated via CODEX_API_KEY / OPENAI_API_KEY) and Codex as the implementer (authenticated via ChatGPT login).
#
# Usage:
#   tools/codex_api_planner_exec.sh --goal best --slug next
#   tools/codex_api_planner_exec.sh --threshold 1363 --slug next
#
# This runs:
#   1) python3 tools/loop_runner.py codex-api-plan ...
#   2) codex exec (apply directive) with API key env vars unset
#   3) commit -> record -> tag/push on NEW BEST -> advance opt/best

ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[codex_loop] START $ts args: $*"
echo "[codex_loop] CODEX_HOME (impl): $codex_home_impl"
echo "[codex_loop] CODEX_HOME (plan): $codex_home_planner"

plan_args=()
impl_model=""
impl_reasoning_effort=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --impl-model)
      impl_model="${2:-}"
      shift 2
      ;;
    --impl-model=*)
      impl_model="${1#--impl-model=}"
      shift
      ;;
    --impl-reasoning-effort)
      impl_reasoning_effort="${2:-}"
      shift 2
      ;;
    --impl-reasoning-effort=*)
      impl_reasoning_effort="${1#--impl-reasoning-effort=}"
      shift
      ;;
    *)
      plan_args+=("$1")
      shift
      ;;
  esac
done

args=("${plan_args[@]}")

base_branch=""
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    --base-branch)
      base_branch="${args[$((i+1))]:-}"
      ;;
    --base-branch=*)
      base_branch="${args[$i]#--base-branch=}"
      ;;
  esac
done

# Default to the rolling best branch so improvements accumulate automatically.
if [[ -z "${base_branch}" ]]; then
  base_branch="opt/best"
  args=(--base-branch "$base_branch" "${args[@]}")
fi

if [[ "$base_branch" == "opt/best" ]]; then
  # Seed/advance `opt/best` from the tooling branch so iter/* branches keep the loop runner + scripts.
  python3 tools/loop_runner.py ensure-best-base --best-branch opt/best --source-branch dev/codex-planner-mode
fi

CODEX_HOME="$codex_home_planner" python3 tools/loop_runner.py codex-api-plan "${args[@]}"
python3 tools/loop_runner.py status

apply_cmd=(
  codex exec
  -C "$repo_root"
  -s workspace-write
  -c 'approval_policy="never"'
  --color never
)
if [[ -n "${impl_model}" ]]; then
  apply_cmd+=(-m "$impl_model")
fi
if [[ -n "${impl_reasoning_effort}" ]]; then
  apply_cmd+=(-c "model_reasoning_effort=\"${impl_reasoning_effort}\"")
fi
apply_cmd+=(-)

CODEX_HOME="$codex_home_impl" env -u CODEX_API_KEY -u OPENAI_API_KEY "${apply_cmd[@]}" < docs/codex-apply-directive-prompt.md

python3 tools/loop_runner.py status

tmp="$(mktemp)"
cleanup() {
  rm -f "$tmp" || true
}
trap cleanup EXIT

# Reproducibility: commit the exact code we benchmark.
git add -A
did_commit=0
if git diff --cached --quiet; then
  echo "[codex_loop] no code changes to commit"
else
  branch="$(git branch --show-current)"
  git commit -m "perf: ${branch}"
  did_commit=1
fi

# Benchmark + record (prints one JSON object; includes `new_best`/`threshold_met` booleans).
set +e
python3 tools/loop_runner.py record | tee "$tmp"
record_rc="${PIPESTATUS[0]}"
set -e
python3 tools/loop_runner.py status

if [[ "$record_rc" != "0" ]]; then
  echo "[codex_loop] record failed (rc=$record_rc); stopping"
  exit "$record_rc"
fi

if python3 - "$tmp" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    entry = json.load(f)

sys.exit(0 if entry.get("new_best") else 1)
PY
then
  python3 tools/loop_runner.py tag-best --push

  if [[ "$base_branch" == "opt/best" ]]; then
    iter_branch="$(git branch --show-current)"
    git checkout "$base_branch"
    git merge --ff-only "$iter_branch"
    git push origin "$base_branch"
    git checkout "$iter_branch"
    echo "[codex_loop] advanced opt/best and pushed"
  fi
else
  # Valid but not best: discard the temp commit so the next iteration starts clean.
  if [[ "$did_commit" == "1" ]]; then
    git reset --hard HEAD~1
  fi
  git status --porcelain=v1
  echo "[codex_loop] not a new best; cleaned"
fi
