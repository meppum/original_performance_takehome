#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Convenience wrapper for one "plan + apply + record" iteration using Codex as the planner.
#
# Usage:
#   # Best-possible search (no fixed threshold):
#   tools/codex_planner_exec.sh --goal best --slug next
#
#   # Threshold search (stop condition is `cycles <= threshold_target`):
#   tools/codex_planner_exec.sh --threshold 1363 --slug next
#
# This runs:
#   1) python3 tools/loop_runner.py codex-plan ...
#   2) codex exec (apply directive + record)

ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[codex_loop] START $ts args: $*"

python3 tools/loop_runner.py codex-plan "$@"
python3 tools/loop_runner.py status

CODEX_HOME="${CODEX_HOME:-$repo_root/.codex_home}" \
  codex exec -C "$repo_root" - < docs/codex-apply-directive-prompt.md

python3 tools/loop_runner.py status
