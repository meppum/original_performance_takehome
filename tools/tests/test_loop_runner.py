import unittest


class ParseCyclesTests(unittest.TestCase):
    def test_parse_cycles_picks_last(self):
        from tools.loop_runner import parse_cycles_from_submission_tests

        output = "\n".join(
            [
                "Testing forest_height=10, rounds=16, batch_size=256",
                "CYCLES:  1500",
                "some other line",
                "CYCLES:  1490",
            ]
        )
        self.assertEqual(parse_cycles_from_submission_tests(output), 1490)

    def test_parse_cycles_missing(self):
        from tools.loop_runner import parse_cycles_from_submission_tests

        self.assertIsNone(parse_cycles_from_submission_tests("no cycles here"))


class ParseCorrectnessTests(unittest.TestCase):
    def test_correctness_true_when_only_speed_threshold_fails(self):
        from tools.loop_runner import parse_correctness_from_submission_tests

        output = "\n".join(
            [
                "Testing forest_height=10, rounds=16, batch_size=256",
                "CYCLES:  1443",
                "....F....",
                "FAIL: test_opus45_improved_harness (SpeedTests)",
                "AssertionError",
                "Ran 9 tests in 1.23s",
                "FAILED (failures=1)",
            ]
        )
        self.assertIs(parse_correctness_from_submission_tests(output), True)

    def test_correctness_false_on_incorrect_output_values(self):
        from tools.loop_runner import parse_correctness_from_submission_tests

        output = "\n".join(
            [
                "ERROR: test_kernel_correctness (CorrectnessTests)",
                "AssertionError: Incorrect output values",
                "Ran 9 tests in 1.23s",
                "FAILED (errors=1)",
            ]
        )
        self.assertIs(parse_correctness_from_submission_tests(output), False)

    def test_correctness_unknown_when_no_ran_line(self):
        from tools.loop_runner import parse_correctness_from_submission_tests

        self.assertIsNone(parse_correctness_from_submission_tests("Traceback..."))


class SlugifyTests(unittest.TestCase):
    def test_slugify_basic(self):
        from tools.loop_runner import _slugify

        self.assertEqual(_slugify("Hello, World!"), "hello-world")

    def test_slugify_empty(self):
        from tools.loop_runner import _slugify

        self.assertEqual(_slugify("   "), "auto")


class IterationIdTests(unittest.TestCase):
    def test_next_iteration_id_from_branch_names(self):
        from tools.loop_runner import _next_iteration_id_from_branch_names

        self.assertEqual(
            _next_iteration_id_from_branch_names(["main", "iter/0007-next", "iter/0011-foo", "topic/123"]),
            12,
        )
        self.assertEqual(_next_iteration_id_from_branch_names(["main", "topic/123"]), 1)


class BestTagTests(unittest.TestCase):
    def test_iter_slug_from_branch(self):
        from tools.loop_runner import _iter_slug_from_branch

        self.assertEqual(_iter_slug_from_branch("iter/0007-next"), "next")
        self.assertEqual(_iter_slug_from_branch("iter/12-foo-bar"), "foo-bar")
        self.assertIsNone(_iter_slug_from_branch("main"))

    def test_format_best_tag_slugified(self):
        from tools.loop_runner import _format_best_tag

        self.assertEqual(_format_best_tag(cycles=1436, slug="Next Run!", iteration_id=7), "best/1436-next-run-i7")

    def test_latest_valid_cycles_for_head(self):
        from tools.loop_runner import _latest_valid_cycles_for_head

        entries = [
            {"valid": True, "branch": "iter/0001-next", "head_sha": "aaa", "cycles": 1500},
            {"valid": False, "branch": "iter/0001-next", "head_sha": "bbb", "cycles": 1490},
            {"valid": True, "branch": "iter/0001-next", "head_sha": "bbb", "cycles": 1480},
        ]
        self.assertEqual(_latest_valid_cycles_for_head(entries, branch="iter/0001-next", head_sha="bbb"), 1480)
        self.assertIsNone(_latest_valid_cycles_for_head(entries, branch="iter/9999-x", head_sha="bbb"))

    def test_latest_valid_cycles_for_iteration(self):
        from tools.loop_runner import _latest_valid_cycles_for_iteration

        entries = [
            {"iteration_id": 1, "valid": True, "branch": "iter/0001-next", "head_sha": "aaa", "cycles": 1500},
            {"iteration_id": 2, "valid": True, "branch": "iter/0002-next", "head_sha": "bbb", "cycles": 1490},
            {"iteration_id": 2, "valid": False, "branch": "iter/0002-next", "head_sha": "ccc", "cycles": 1480},
            {"iteration_id": 2, "valid": True, "branch": "iter/0002-next", "head_sha": "ddd", "cycles": 1470},
        ]
        self.assertEqual(_latest_valid_cycles_for_iteration(entries, iteration_id=2), 1470)
        self.assertEqual(_latest_valid_cycles_for_iteration(entries, iteration_id=2, branch="iter/0002-next"), 1470)
        self.assertIsNone(_latest_valid_cycles_for_iteration(entries, iteration_id=2, branch="iter/9999-x"))


class BoundBundleTests(unittest.TestCase):
    def test_compute_min_cycles_by_engine(self):
        from tools.loop_runner import _compute_min_cycles_by_engine

        counts = {"load": 5, "alu": 12, "valu": 0}
        slot_limits = {"load": 2, "alu": 12, "valu": 6}
        out = _compute_min_cycles_by_engine(task_counts=counts, slot_limits=slot_limits)
        self.assertEqual(out["load"], 3)  # ceil(5/2)
        self.assertEqual(out["alu"], 1)  # ceil(12/12)

    def test_critical_path_engine_counts(self):
        from dataclasses import dataclass

        from tools.loop_runner import _critical_path_engine_counts

        @dataclass
        class T:
            engine: str
            succs: list[int]
            cp: int

        # 0 -> 1 -> 2 is the critical path (len 3).
        tasks = [
            T("load", [1, 3], 3),
            T("alu", [2], 2),
            T("valu", [], 1),
            T("flow", [], 1),
        ]
        counts, path_len = _critical_path_engine_counts(tasks)
        self.assertEqual(path_len, 3)
        self.assertEqual(counts["load"], 1)
        self.assertEqual(counts["alu"], 1)
        self.assertEqual(counts["valu"], 1)


class RepoUrlTests(unittest.TestCase):
    def test_github_web_url_from_https(self):
        from tools.loop_runner import _github_web_url

        self.assertEqual(
            _github_web_url("https://github.com/meppum/original_performance_takehome.git"),
            "https://github.com/meppum/original_performance_takehome",
        )

    def test_github_web_url_from_ssh(self):
        from tools.loop_runner import _github_web_url

        self.assertEqual(
            _github_web_url("git@github.com:meppum/original_performance_takehome.git"),
            "https://github.com/meppum/original_performance_takehome",
        )


class CodeContextTests(unittest.TestCase):
    def test_extract_kernelbuilder_source_contains_expected_anchors(self):
        from tools.loop_runner import _extract_kernelbuilder_source

        src = _extract_kernelbuilder_source()
        self.assertIn("class KernelBuilder", src)
        self.assertIn("def build_kernel", src)


class PlannerSchemaTests(unittest.TestCase):
    def test_planner_directive_schema_requires_all_properties(self):
        from tools.loop_runner import _planner_directive_schema

        schema = _planner_directive_schema()
        props = schema.get("properties")
        required = schema.get("required")

        self.assertIsInstance(props, dict)
        self.assertIsInstance(required, list)
        self.assertEqual(set(required), set(props.keys()))

    def test_planner_schema_excludes_blocked_families(self):
        from tools.loop_runner import _planner_directive_schema

        schema = _planner_directive_schema(blocked_families=["family:schedule"])
        fam = schema["properties"]["strategy_family"]["enum"]
        self.assertNotIn("family:schedule", fam)


class StrategyFamilyConstraintsTests(unittest.TestCase):
    def test_blocks_family_after_two_attempts_without_meaningful_win(self):
        from tools.loop_runner import _compute_strategy_family_constraints

        entries = [
            {"strategy_tags": ["family:schedule", "hash"], "valid": True, "cycles": 1500},
            {"strategy_tags": ["family:schedule", "hash"], "valid": True, "cycles": 1495},  # +5 (not meaningful)
        ]
        c = _compute_strategy_family_constraints(entries)
        self.assertEqual(c["current_family"], "family:schedule")
        self.assertEqual(c["current_family_streak_len"], 2)
        self.assertEqual(c["blocked_families"], ["family:schedule"])

    def test_allows_one_bonus_attempt_after_meaningful_win(self):
        from tools.loop_runner import _compute_strategy_family_constraints

        entries = [
            {"strategy_tags": ["family:reduce_loads", "gather"], "valid": True, "cycles": 1500},
            {"strategy_tags": ["family:reduce_loads", "gather"], "valid": True, "cycles": 1485},  # +15 meaningful
        ]
        c = _compute_strategy_family_constraints(entries)
        self.assertEqual(c["current_family"], "family:reduce_loads")
        self.assertEqual(c["current_family_streak_len"], 2)
        self.assertEqual(c["blocked_families"], [])


class ManualDirectiveParsingTests(unittest.TestCase):
    def test_parse_json_relaxed_strips_code_fences(self):
        from tools.loop_runner import _parse_json_relaxed

        obj = _parse_json_relaxed("```json\n{\"ok\": true}\n```")
        self.assertEqual(obj, {"ok": True})


class ManualPlannerPromptTests(unittest.TestCase):
    def test_manual_prompt_mentions_permalink_and_blocked_families(self):
        from tools.loop_runner import _build_manual_planner_prompt

        packet = {
            "iteration_id": 1,
            "branch": "plan/0001-e2e",
            "base_branch": "main",
            "repo": {
                "origin_url": "https://github.com/meppum/original_performance_takehome.git",
                "github_web_url": "https://github.com/meppum/original_performance_takehome",
                "base_sha": "deadbeef",
                "worktree_path": "/tmp/worktree",
            },
            "strategy_family_constraints": {"blocked_families": ["family:schedule"]},
        }
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

        prompt = _build_manual_planner_prompt(packet, directive_schema=schema)
        self.assertIn("Permalink for code lookup", prompt)
        self.assertIn("deadbeef", prompt)
        self.assertIn("Blocked families for this iteration", prompt)
        self.assertIn("family:schedule", prompt)


class CodexPlannerPromptTests(unittest.TestCase):
    def test_codex_prompt_mentions_branch_and_blocked_families(self):
        from tools.loop_runner import _build_codex_planner_prompt

        packet = {
            "iteration_id": 1,
            "branch": "iter/0001-e2e",
            "base_branch": "main",
            "threshold_target": 1363,
            "strategy_family_constraints": {"blocked_families": ["family:schedule"]},
        }
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

        prompt = _build_codex_planner_prompt(packet, directive_schema=schema)
        self.assertIn("iteration branch (local)", prompt)
        self.assertIn("iter/0001-e2e", prompt)
        self.assertIn("threshold_target: 1363", prompt)
        self.assertIn("Blocked families for this iteration", prompt)
        self.assertIn("family:schedule", prompt)


class DirectiveValidationTests(unittest.TestCase):
    def test_validate_directive_accepts_strategy_family_schema(self):
        from tools.loop_runner import _planner_directive_schema, _validate_directive

        schema = _planner_directive_schema()
        directive = {
            "objective": "reduce cycles",
            "primary_hypothesis": "fewer ops reduces cycles",
            "strategy_family": "family:schedule",
            "strategy_modifiers": ["manual"],
            "risk": "Low",
            "expected_effect_cycles": 10,
            "change_summary": ["small refactor"],
            "step_plan": ["step 1", "step 2", "step 3"],
            "validation": {"commands": ["python3 -B tests/submission_tests.py"], "pass_criteria": ["CYCLES decreases"]},
            "next_packet_requests": [],
            "did_web_search": False,
        }
        _validate_directive(directive, schema=schema)

    def test_validate_directive_rejects_blocked_strategy_family(self):
        from tools.loop_runner import LoopRunnerError, _planner_directive_schema, _validate_directive

        schema = _planner_directive_schema(blocked_families=["family:schedule"])
        directive = {
            "objective": "reduce cycles",
            "primary_hypothesis": "fewer ops reduces cycles",
            "strategy_family": "family:schedule",
            "strategy_modifiers": [],
            "risk": "Low",
            "expected_effect_cycles": 10,
            "change_summary": ["small refactor"],
            "step_plan": ["step 1", "step 2", "step 3"],
            "validation": {"commands": ["python3 -B tests/submission_tests.py"], "pass_criteria": ["CYCLES decreases"]},
            "next_packet_requests": [],
            "did_web_search": False,
        }
        with self.assertRaises(LoopRunnerError):
            _validate_directive(directive, schema=schema)

    def test_validate_directive_rejects_empty_strategy_modifier(self):
        from tools.loop_runner import LoopRunnerError, _planner_directive_schema, _validate_directive

        schema = _planner_directive_schema()
        directive = {
            "objective": "reduce cycles",
            "primary_hypothesis": "fewer ops reduces cycles",
            "strategy_family": "family:schedule",
            "strategy_modifiers": [""],
            "risk": "Low",
            "expected_effect_cycles": 10,
            "change_summary": ["small refactor"],
            "step_plan": ["step 1", "step 2", "step 3"],
            "validation": {"commands": ["python3 -B tests/submission_tests.py"], "pass_criteria": ["CYCLES decreases"]},
            "next_packet_requests": [],
            "did_web_search": False,
        }
        with self.assertRaises(LoopRunnerError):
            _validate_directive(directive, schema=schema)


class PollCadenceTests(unittest.TestCase):
    def test_real_planner_calls_default_to_60s(self):
        import os

        from tools.loop_runner import _enforce_default_poll_cadence
        from tools.openai_exec import OpenAIExecConfig

        old_env = dict(os.environ)
        try:
            os.environ["OPENAI_API_KEY"] = "test"
            os.environ["OPENAI_BACKGROUND_POLL_INTERVAL"] = "5"
            os.environ["OPENAI_BACKGROUND_PROGRESS_EVERY"] = "5"

            _enforce_default_poll_cadence()
            self.assertNotIn("OPENAI_BACKGROUND_POLL_INTERVAL", os.environ)
            self.assertNotIn("OPENAI_BACKGROUND_PROGRESS_EVERY", os.environ)

            cfg = OpenAIExecConfig.from_env()
            self.assertEqual(cfg.background_poll_interval_s, 60.0)
            self.assertEqual(cfg.background_progress_every_s, 60.0)
        finally:
            os.environ.clear()
            os.environ.update(old_env)


class GitShowTextTests(unittest.TestCase):
    def test_git_show_text_raises_with_ref_and_path(self):
        from pathlib import Path
        from unittest.mock import patch

        from tools.loop_runner import LoopRunnerError, _git_show_text

        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "fatal: bad object"

        with patch("tools.loop_runner._run", return_value=_Proc()):
            with self.assertRaises(LoopRunnerError) as ctx:
                _git_show_text("badref", Path("planner_packets/packet.json"))

        msg = str(ctx.exception)
        self.assertIn("badref", msg)
        self.assertIn("planner_packets/packet.json", msg)


class StatusCommandTests(unittest.TestCase):
    def test_status_reports_not_recorded_when_missing_log_entry(self):
        import argparse
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from unittest.mock import patch

        import tools.loop_runner as lr

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            state_path = td_path / "state.json"
            log_path = td_path / "log.jsonl"

            state_path.write_text(
                json.dumps(
                    {
                        "iteration_id": 2,
                        "branch": "iter/0002-next",
                        "base_branch": "main",
                        "base_sha": "deadbeef",
                        "threshold_target": None,
                        "packet": {"goal": "best", "best_cycles": 1436, "aspiration_cycles": 1435, "plateau_valid_iters_since_best": 3},
                        "directive": {
                            "strategy_family": "family:reduce_loads",
                            "strategy_modifiers": ["gather"],
                            "risk": "Medium",
                            "expected_effect_cycles": 10,
                            "primary_hypothesis": "Fewer loads reduces cycles.",
                            "change_summary": ["Reduce redundant loads."],
                        },
                        "advisor_response_id": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            # Only iteration_id=1 is present; state is iteration_id=2.
            log_path.write_text(json.dumps({"iteration_id": 1, "valid": True, "cycles": 1500}) + "\n", encoding="utf-8")

            with patch.object(lr, "_STATE_PATH", state_path), patch.object(lr, "_EXPERIMENT_LOG_PATH", log_path):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    lr.cmd_status(argparse.Namespace())

            out = buf.getvalue()
            self.assertIn("[loop_runner] ATTEMPT:", out)
            self.assertIn("[loop_runner] OUTCOME: (not recorded yet)", out)

    def test_status_reports_outcome_when_log_entry_exists(self):
        import argparse
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from unittest.mock import patch

        import tools.loop_runner as lr

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            state_path = td_path / "state.json"
            log_path = td_path / "log.jsonl"

            state_path.write_text(
                json.dumps(
                    {
                        "iteration_id": 1,
                        "branch": "iter/0001-next",
                        "base_branch": "main",
                        "base_sha": "deadbeef",
                        "threshold_target": None,
                        "packet": {"goal": "best", "best_cycles": 1436, "aspiration_cycles": 1435, "plateau_valid_iters_since_best": 0},
                        "directive": {
                            "strategy_family": "family:break_deps",
                            "strategy_modifiers": [],
                            "risk": "Low",
                            "expected_effect_cycles": 5,
                            "primary_hypothesis": "Break deps to overlap work.",
                            "change_summary": ["Break a dependency chain."],
                        },
                        "advisor_response_id": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            log_path.write_text(
                json.dumps(
                    {
                        "iteration_id": 1,
                        "valid": True,
                        "cycles": 1400,
                        "delta_vs_best": -36,
                        "tests_diff_empty": True,
                        "files_changed": ["perf_takehome.py"],
                        "result_summary": "PASS correctness; cycles=1400",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(lr, "_STATE_PATH", state_path), patch.object(lr, "_EXPERIMENT_LOG_PATH", log_path):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    lr.cmd_status(argparse.Namespace())

            out = buf.getvalue()
            self.assertIn("[loop_runner] ATTEMPT:", out)
            self.assertIn("family=family:break_deps", out)
            self.assertIn("[loop_runner] OUTCOME:", out)
            self.assertIn("cycles=1400", out)


if __name__ == "__main__":
    unittest.main()
