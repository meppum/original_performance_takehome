#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Always default Codex to the repo-scoped home dir so the loop is self-contained.
export CODEX_HOME="${CODEX_HOME:-$repo_root/.codex_home}"

# Usage:
#   1) python3 tools/loop_runner.py manual-pack --threshold 1363 --slug next
#   2) paste planner_packets/prompt.md into ChatGPT (gpt-5.2-pro)
#   3) paste ChatGPT JSON into planner_packets/directive.json and commit it
#   4) tools/manual_planner_exec.sh
#
# If the manual packet lives on a separate `plan/*` branch/worktree:
#   tools/manual_planner_exec.sh --from-ref plan/0001-next

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

python3 tools/loop_runner.py manual-apply "${plan_args[@]}"

if ! env -u CODEX_API_KEY -u OPENAI_API_KEY codex login status >/dev/null 2>&1; then
  echo "[codex_loop] Codex is not logged in for CODEX_HOME=$CODEX_HOME"
  echo "[codex_loop] Run: CODEX_HOME=\"$CODEX_HOME\" codex login --device-auth"
  echo "[codex_loop] Then re-run this loop."
  exit 2
fi

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

env -u CODEX_API_KEY -u OPENAI_API_KEY "${apply_cmd[@]}" < docs/codex-apply-directive-prompt.md
