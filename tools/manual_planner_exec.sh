#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Usage:
#   1) python3 tools/loop_runner.py manual-pack --threshold 1363 --slug next
#   2) paste planner_packets/prompt.md into ChatGPT (gpt-5.2-pro)
#   3) paste ChatGPT JSON into planner_packets/directive.json and commit it
#   4) tools/manual_planner_exec.sh

python3 tools/loop_runner.py manual-apply "$@"

CODEX_HOME="${CODEX_HOME:-$repo_root/.codex_home}" \
  codex exec -C "$repo_root" - < docs/codex-apply-directive-prompt.md
