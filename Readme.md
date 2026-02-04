# Anthropic's Original Performance Take-Home

This repo contains a version of Anthropic's original performance take-home, before Claude Opus 4.5 started doing better than humans given only 2 hours.

The original take-home was a 4-hour one that starts close to the contents of this repo, after Claude Opus 4 beat most humans at that, it was updated to a 2-hour one which started with code which achieved 18532 cycles (7.97x faster than this repo starts you). This repo is based on the newer take-home which has a few more instructions and comes with better debugging tools, but has the starter code reverted to the slowest baseline. After Claude Opus 4.5 we started using a different base for our time-limited take-homes.

Now you can try to beat Claude Opus 4.5 given unlimited time!

## Performance benchmarks 

Measured in clock cycles from the simulated machine. All of these numbers are for models doing the 2 hour version which started at 18532 cycles:

- **2164 cycles**: Claude Opus 4 after many hours in the test-time compute harness
- **1790 cycles**: Claude Opus 4.5 in a casual Claude Code session, approximately matching the best human performance in 2 hours
- **1579 cycles**: Claude Opus 4.5 after 2 hours in our test-time compute harness
- **1548 cycles**: Claude Sonnet 4.5 after many more than 2 hours of test-time compute
- **1487 cycles**: Claude Opus 4.5 after 11.5 hours in the harness
- **1363 cycles**: Claude Opus 4.5 in an improved test time compute harness
- **??? cycles**: Best human performance ever is substantially better than the above, but we won't say how much.

While it's no longer a good time-limited test, you can still use this test to get us excited about hiring you! If you optimize below 1487 cycles, beating Claude Opus 4.5's best performance at launch, email us at performance-recruiting@anthropic.com with your code (and ideally a resume) so we can be appropriately impressed, especially if you get near the best solution we've seen. New model releases may change what threshold impresses us though, and no guarantees that we keep this readme updated with the latest on that.

Run `python tests/submission_tests.py` to see which thresholds you pass.

## Warning: LLMs can cheat

None of the solutions we received on the first day post-release below 1300 cycles were valid solutions. In each case, a language model modified the tests to make the problem easier.

If you use an AI agent, we recommend instructing it not to change the `tests/` folder and to use `tests/submission_tests.py` for verification.

If you are running an agentic optimization loop in this fork, see:

- `docs/openai-advisor-loop.md` (Codex CLI ↔ OpenAI planner contract + background/polling guidance)
- `docs/testing-strategy.md` (risk-based unit/integration/E2E test coverage for the loop tooling)
- `docs/experiment-log.md` (how to track attempted strategies to avoid repeated loops)

## Repo Tooling Tests (Loop Runner / OpenAI Helper)

This repo includes a small internal test suite for the Codex↔advisor loop tooling (hermetic; no network calls by default):

```bash
python3 -m unittest discover -s tools/tests
```

### Codex CLI instructions

This repo keeps Codex agent instructions in `.codex_home/AGENTS.md` so you can opt into them as a project-scoped home:

```bash
CODEX_HOME="$PWD/.codex_home" codex --cd "$PWD"
```

See `docs/codex-loop-prompt.md` for a copy/paste starter prompt.

### Codex-only optimization loop (no OpenAI API calls)

On `dev/codex-planner-mode`, you can run an unattended “best-chasing” loop that only preserves improvements by pushing
`best/*` tags and fast-forwarding `opt/best` on origin:

```bash
git checkout dev/codex-planner-mode
git pull --ff-only origin dev/codex-planner-mode

while true; do
  tools/codex_planner_exec.sh --goal best --slug next || break
done
```

If `codex exec` fails with `401 Unauthorized`, check `codex login status` and authenticate (or export `OPENAI_API_KEY`).

Please run the following commands to validate your submission, and mention that you did so when submitting:
```
# This should be empty, the tests folder must be unchanged
git diff origin/main tests/
# You should pass some of these tests and use the cycle count this prints
python tests/submission_tests.py
```

An example of this kind of hack is a model noticing that `problem.py` has multicore support, implementing multicore as an optimization, noticing there's no speedup and "debugging" that `N_CORES = 1` and "fixing" the core count so they get a speedup. Multicore is disabled intentionally in this version.
