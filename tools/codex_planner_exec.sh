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
#   2) codex exec (apply directive)
#   3) commit -> record -> tag/push on NEW BEST -> advance opt/best

ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[codex_loop] START $ts args: $*"

args=("$@")

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
  python3 tools/loop_runner.py ensure-best-base --best-branch opt/best --source-branch main
fi

python3 tools/loop_runner.py codex-plan "${args[@]}"
python3 tools/loop_runner.py status

CODEX_HOME="${CODEX_HOME:-$repo_root/.codex_home}" \
  codex exec -C "$repo_root" - < docs/codex-apply-directive-prompt.md

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

# Benchmark + record (prints JSON + NEW BEST/THRESHOLD MET markers).
set +e
python3 tools/loop_runner.py record | tee "$tmp"
record_rc="${PIPESTATUS[0]}"
set -e
python3 tools/loop_runner.py status

if [[ "$record_rc" != "0" ]]; then
  echo "[codex_loop] record failed (rc=$record_rc); stopping"
  exit "$record_rc"
fi

if grep -q "\\[loop_runner\\] NEW BEST:" "$tmp"; then
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
