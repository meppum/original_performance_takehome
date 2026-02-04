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


class RecordFailFastTests(unittest.TestCase):
    def test_record_skips_benchmark_when_tests_changed(self):
        import argparse
        import io
        import json
        import tempfile
        from contextlib import redirect_stderr, redirect_stdout
        from pathlib import Path
        from unittest.mock import patch

        import tools.loop_runner as lr

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "log.jsonl"
            log_path.write_text(json.dumps({"iteration_id": 1, "valid": True, "cycles": 1500}) + "\n", encoding="utf-8")

            state = lr.IterationState(
                iteration_id=2,
                branch="iter/0002-next",
                base_branch="main",
                base_sha="deadbeef",
                threshold_target=None,
                packet={},
                directive={},
                advisor_response_id=None,
            )

            def fake_git(*args: str, check: bool = True) -> str:
                if args == ("branch", "--show-current"):
                    return state.branch
                if args == ("rev-parse", "HEAD"):
                    return "aaa"
                if args == ("status", "--porcelain=v1"):
                    return ""
                return ""

            def should_not_run(*_args, **_kwargs):
                raise AssertionError("tests/submission_tests.py should not run when tests have changed")

            with (
                patch.object(lr, "_EXPERIMENT_LOG_PATH", log_path),
                patch.object(lr, "_read_state", return_value=state),
                patch.object(lr, "_git", side_effect=fake_git),
                patch.object(lr, "_git_diff_tests_is_empty", return_value=(False, "diff...")),
                patch.object(lr, "_compute_changed_files", return_value=["perf_takehome.py"]),
                patch.object(lr, "_run", side_effect=should_not_run),
            ):
                buf_out = io.StringIO()
                buf_err = io.StringIO()
                with redirect_stdout(buf_out), redirect_stderr(buf_err):
                    rc = lr.cmd_record(argparse.Namespace(print_test_output=False))
                self.assertEqual(rc, 1)

            entries = lr._read_jsonl(log_path)
            self.assertGreaterEqual(len(entries), 2)
            last = entries[-1]
            self.assertEqual(last["iteration_id"], 2)
            self.assertEqual(last["branch"], state.branch)
            self.assertIs(last["valid"], False)
            self.assertIs(last["tests_diff_empty"], False)
            self.assertIsNone(last["cycles"])
            self.assertIsNone(last["correctness_pass"])
            self.assertIn("SKIP benchmark", last["result_summary"])
            self.assertIn("tests changed", last["result_summary"])

    def test_record_skips_benchmark_on_scope_violation(self):
        import argparse
        import io
        import json
        import tempfile
        from contextlib import redirect_stderr, redirect_stdout
        from pathlib import Path
        from unittest.mock import patch

        import tools.loop_runner as lr

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "log.jsonl"
            log_path.write_text(json.dumps({"iteration_id": 1, "valid": True, "cycles": 1500}) + "\n", encoding="utf-8")

            state = lr.IterationState(
                iteration_id=2,
                branch="iter/0002-next",
                base_branch="main",
                base_sha="deadbeef",
                threshold_target=None,
                packet={},
                directive={},
                advisor_response_id=None,
            )

            def fake_git(*args: str, check: bool = True) -> str:
                if args == ("branch", "--show-current"):
                    return state.branch
                if args == ("rev-parse", "HEAD"):
                    return "aaa"
                if args == ("status", "--porcelain=v1"):
                    return ""
                return ""

            def should_not_run(*_args, **_kwargs):
                raise AssertionError("tests/submission_tests.py should not run on scope violation")

            with (
                patch.object(lr, "_EXPERIMENT_LOG_PATH", log_path),
                patch.object(lr, "_read_state", return_value=state),
                patch.object(lr, "_git", side_effect=fake_git),
                patch.object(lr, "_git_diff_tests_is_empty", return_value=(True, "")),
                patch.object(lr, "_compute_changed_files", return_value=["problem.py"]),
                patch.object(lr, "_run", side_effect=should_not_run),
            ):
                buf_out = io.StringIO()
                buf_err = io.StringIO()
                with redirect_stdout(buf_out), redirect_stderr(buf_err):
                    rc = lr.cmd_record(argparse.Namespace(print_test_output=False))
                self.assertEqual(rc, 1)

            entries = lr._read_jsonl(log_path)
            self.assertGreaterEqual(len(entries), 2)
            last = entries[-1]
            self.assertIs(last["valid"], False)
            self.assertIs(last["tests_diff_empty"], True)
            self.assertIs(last["scope_ok"], False)
            self.assertIn("problem.py", last["scope_forbidden_files"])
            self.assertIsNone(last["cycles"])
            self.assertIsNone(last["correctness_pass"])
            self.assertIn("SKIP benchmark", last["result_summary"])
            self.assertIn("scope violation", last["result_summary"])

    def test_record_runs_benchmark_when_scope_and_tests_ok(self):
        import argparse
        import io
        import json
        import tempfile
        from contextlib import redirect_stderr, redirect_stdout
        from pathlib import Path
        from unittest.mock import patch

        import tools.loop_runner as lr

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "log.jsonl"
            log_path.write_text(json.dumps({"iteration_id": 1, "valid": True, "cycles": 1500}) + "\n", encoding="utf-8")

            state = lr.IterationState(
                iteration_id=2,
                branch="iter/0002-next",
                base_branch="main",
                base_sha="deadbeef",
                threshold_target=None,
                packet={},
                directive={},
                advisor_response_id=None,
            )

            def fake_git(*args: str, check: bool = True) -> str:
                if args == ("branch", "--show-current"):
                    return state.branch
                if args == ("rev-parse", "HEAD"):
                    return "aaa"
                if args == ("status", "--porcelain=v1"):
                    return ""
                return ""

            class Proc:
                returncode = 1
                stdout = "\n".join(
                    [
                        "Testing forest_height=10, rounds=16, batch_size=256",
                        "CYCLES:  1400",
                        "....F....",
                        "FAIL: test_opus45_improved_harness (SpeedTests)",
                        "AssertionError",
                        "Ran 9 tests in 1.23s",
                        "FAILED (failures=1)",
                    ]
                )
                stderr = ""

            def fake_run(cmd, *, check, capture):
                self.assertEqual(cmd, ["python3", "-B", "tests/submission_tests.py"])
                self.assertIs(check, False)
                self.assertIs(capture, True)
                return Proc()

            with (
                patch.object(lr, "_EXPERIMENT_LOG_PATH", log_path),
                patch.object(lr, "_read_state", return_value=state),
                patch.object(lr, "_git", side_effect=fake_git),
                patch.object(lr, "_git_diff_tests_is_empty", return_value=(True, "")),
                patch.object(lr, "_compute_changed_files", return_value=["perf_takehome.py"]),
                patch.object(lr, "_run", side_effect=fake_run),
            ):
                buf_out = io.StringIO()
                buf_err = io.StringIO()
                with redirect_stdout(buf_out), redirect_stderr(buf_err):
                    rc = lr.cmd_record(argparse.Namespace(print_test_output=False))
                self.assertEqual(rc, 0)

            entries = lr._read_jsonl(log_path)
            self.assertGreaterEqual(len(entries), 2)
            last = entries[-1]
            self.assertIs(last["valid"], True)
            self.assertEqual(last["cycles"], 1400)
            self.assertIs(last["correctness_pass"], True)
            self.assertEqual(last["delta_vs_best"], -100)
            self.assertIn("PASS correctness", last["result_summary"])


class PlannerMemorySummaryTests(unittest.TestCase):
    def test_entry_reason_reports_expected_causes(self):
        from tools.loop_runner import _entry_reason

        self.assertEqual(_entry_reason({"valid": False, "tests_diff_empty": False}), "tests changed")
        self.assertEqual(_entry_reason({"valid": False, "scope_ok": False}), "scope violation")
        self.assertEqual(_entry_reason({"valid": False, "correctness_pass": False}), "incorrect output")
        self.assertEqual(
            _entry_reason({"valid": False, "tests_diff_empty": False, "scope_ok": False}),
            "tests changed, scope violation",
        )
        self.assertEqual(_entry_reason({"valid": False}), "invalid (unknown reason)")
        self.assertIsNone(_entry_reason({"valid": True}))

    def test_compute_experiment_summary_is_compact_and_consistent(self):
        from unittest.mock import patch

        import tools.loop_runner as lr

        families = ["family:schedule", "family:reduce_loads", "family:break_deps"]
        entries = []
        for i in range(1, 21):
            fam = families[i % len(families)]
            tags = [fam, f"m{i}"]
            if i % 2 == 0:
                entries.append(
                    {
                        "iteration_id": i,
                        "branch": f"iter/{i:04d}-next",
                        "head_sha": f"sha{i}",
                        "valid": True,
                        "cycles": 1500 - i,
                        "delta_vs_best": None,
                        "strategy_tags": tags,
                        "result_summary": "PASS correctness",
                        "tests_diff_empty": True,
                        "scope_ok": True,
                        "correctness_pass": True,
                    }
                )
            else:
                entries.append(
                    {
                        "iteration_id": i,
                        "branch": f"iter/{i:04d}-next",
                        "head_sha": f"sha{i}",
                        "valid": False,
                        "cycles": None,
                        "delta_vs_best": None,
                        "strategy_tags": tags,
                        "result_summary": "FAIL correctness",
                        "tests_diff_empty": False if i % 3 == 0 else True,
                        "scope_ok": False if i % 5 == 0 else True,
                        "correctness_pass": False,
                    }
                )

        # Make this unit test hermetic: ignore any real best/* tags in the working repo.
        with patch.object(lr, "_best_cycles_from_best_tags", return_value=None):
            summary = lr._compute_experiment_summary(entries)

        self.assertEqual(summary["best_cycles"], 1480)  # 1500 - 20

        # Per-family best should match the best among valid entries.
        expected_best_by_family = {}
        for e in entries:
            if e.get("valid") is not True:
                continue
            fam = e["strategy_tags"][0]
            expected_best_by_family[fam] = min(expected_best_by_family.get(fam, 10**9), e["cycles"])
        self.assertEqual(summary["best_cycles_by_family"], expected_best_by_family)

        self.assertEqual(summary["last_attempt"]["iteration_id"], 20)
        self.assertEqual(summary["last_valid_attempt"]["iteration_id"], 20)

        self.assertEqual(len(summary["recent_attempts"]), 8)
        self.assertEqual(summary["recent_attempts"][0]["iteration_id"], 13)
        self.assertEqual(summary["recent_attempts"][-1]["iteration_id"], 20)

        self.assertEqual(len(summary["recent_strategy_combos"]), 15)
        self.assertEqual(summary["recent_strategy_combos"][0]["iteration_id"], 6)
        self.assertEqual(summary["recent_strategy_combos"][-1]["iteration_id"], 20)

        invalid = summary["invalid_counts"]
        self.assertEqual(invalid["total"], 10)
        self.assertEqual(invalid["correctness"], 10)
        self.assertEqual(invalid["tests"], 3)  # i in {3,9,15}
        self.assertEqual(invalid["scope"], 2)  # i in {5,15}


class ManualPackPacketTests(unittest.TestCase):
    def test_manual_pack_includes_experiment_summary(self):
        import argparse
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from unittest.mock import patch

        import tools.loop_runner as lr

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manual_dir = root / "planner_packets"
            manual_packet_path = manual_dir / "packet.json"
            manual_prompt_path = manual_dir / "prompt.md"
            manual_directive_path = manual_dir / "directive.json"
            manual_schema_path = manual_dir / "directive_schema.json"
            log_path = root / "log.jsonl"
            log_path.write_text(
                json.dumps(
                    {
                        "iteration_id": 1,
                        "valid": True,
                        "cycles": 1500,
                        "strategy_tags": ["family:schedule"],
                        "tests_diff_empty": True,
                        "scope_ok": True,
                        "correctness_pass": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_git(*_args: str, check: bool = True) -> str:
                return ""

            def fake_create_plan_branch(*, iteration_id: int, slug: str, base_branch: str, no_pull: bool):
                return (f"plan/{iteration_id:04d}-{slug}", "deadbeef", iteration_id)

            with (
                patch.object(lr, "_REPO_ROOT", root),
                patch.object(lr, "_EXPERIMENT_LOG_PATH", log_path),
                patch.object(lr, "_MANUAL_PACKET_DIR", manual_dir),
                patch.object(lr, "_MANUAL_PACKET_PATH", manual_packet_path),
                patch.object(lr, "_MANUAL_PROMPT_PATH", manual_prompt_path),
                patch.object(lr, "_MANUAL_DIRECTIVE_PATH", manual_directive_path),
                patch.object(lr, "_MANUAL_SCHEMA_PATH", manual_schema_path),
                patch.object(lr, "_git", side_effect=fake_git),
                patch.object(lr, "_origin_remote_url", return_value=None),
                patch.object(lr, "_next_iteration_id_from_local_branches", return_value=1),
                patch.object(lr, "_create_plan_branch", side_effect=fake_create_plan_branch),
                patch.object(lr, "_compute_performance_profile_for_submission_case", return_value={"ok": True}),
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = lr.cmd_manual_pack(
                        argparse.Namespace(
                            base_branch="main",
                            no_pull=True,
                            goal="best",
                            threshold=1363,
                            slug="next",
                            code_context="none",
                            experiment_log_tail_lines=5,
                        )
                    )
                self.assertEqual(rc, 0)

            packet = json.loads(manual_packet_path.read_text(encoding="utf-8"))
            self.assertIn("experiment_summary", packet)
            self.assertEqual(packet["experiment_summary"]["best_cycles"], 1500)


class GlobalBestFromTagsTests(unittest.TestCase):
    def test_best_cycles_from_best_tags_parses_min(self):
        from unittest.mock import patch

        import tools.loop_runner as lr

        def fake_git(*args: str, check: bool = True) -> str:
            if args == ("tag", "-l", "best/*"):
                return "\n".join(
                    [
                        "best/1436-next-i7",
                        "best/1441-next-i4",
                        "best/not-a-number",
                        "best/12oops",
                        "",
                    ]
                )
            return ""

        with patch.object(lr, "_git", side_effect=fake_git):
            self.assertEqual(lr._best_cycles_from_best_tags(fetch=False), 1436)

    def test_best_cycles_overall_prefers_tags_min(self):
        from unittest.mock import patch

        import tools.loop_runner as lr

        entries = [{"valid": True, "cycles": 1443}]
        with patch.object(lr, "_best_cycles_from_best_tags", return_value=1400):
            self.assertEqual(lr._best_cycles_overall(entries), 1400)


class GlobalBestRecordInteractionTests(unittest.TestCase):
    def test_record_does_not_emit_new_best_when_worse_than_tag_best(self):
        import argparse
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from unittest.mock import patch

        import tools.loop_runner as lr

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "log.jsonl"
            log_path.write_text(
                json.dumps({"iteration_id": 1, "valid": True, "cycles": 1443}) + "\n",
                encoding="utf-8",
            )

            state = lr.IterationState(
                iteration_id=2,
                branch="iter/0002-next",
                base_branch="main",
                base_sha="deadbeef",
                threshold_target=None,
                packet={},
                directive={},
                advisor_response_id=None,
            )

            def fake_git(*args: str, check: bool = True) -> str:
                if args == ("branch", "--show-current"):
                    return state.branch
                if args == ("rev-parse", "HEAD"):
                    return "aaa"
                if args == ("status", "--porcelain=v1"):
                    return ""
                return ""

            class Proc:
                returncode = 1
                stdout = "\n".join(
                    [
                        "Testing forest_height=10, rounds=16, batch_size=256",
                        "CYCLES:  1420",
                        "Ran 9 tests in 1.23s",
                        "FAILED (failures=1)",
                        "FAIL: test_opus45_improved_harness (SpeedTests)",
                    ]
                )
                stderr = ""

            def fake_run(cmd, *, check, capture):
                self.assertEqual(cmd, ["python3", "-B", "tests/submission_tests.py"])
                return Proc()

            with (
                patch.object(lr, "_EXPERIMENT_LOG_PATH", log_path),
                patch.object(lr, "_read_state", return_value=state),
                patch.object(lr, "_git", side_effect=fake_git),
                patch.object(lr, "_git_diff_tests_is_empty", return_value=(True, "")),
                patch.object(lr, "_compute_changed_files", return_value=["perf_takehome.py"]),
                patch.object(lr, "_best_cycles_from_best_tags", return_value=1400),
                patch.object(lr, "_run", side_effect=fake_run),
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = lr.cmd_record(argparse.Namespace(print_test_output=False))
                out = buf.getvalue()

            self.assertEqual(rc, 0)
            entry = json.loads(out)
            self.assertIs(entry["new_best"], False)

    def test_record_emits_new_best_when_beating_tag_best(self):
        import argparse
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from unittest.mock import patch

        import tools.loop_runner as lr

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "log.jsonl"
            log_path.write_text(
                json.dumps({"iteration_id": 1, "valid": True, "cycles": 1443}) + "\n",
                encoding="utf-8",
            )

            state = lr.IterationState(
                iteration_id=2,
                branch="iter/0002-next",
                base_branch="main",
                base_sha="deadbeef",
                threshold_target=None,
                packet={},
                directive={},
                advisor_response_id=None,
            )

            def fake_git(*args: str, check: bool = True) -> str:
                if args == ("branch", "--show-current"):
                    return state.branch
                if args == ("rev-parse", "HEAD"):
                    return "aaa"
                if args == ("status", "--porcelain=v1"):
                    return ""
                return ""

            class Proc:
                returncode = 1
                stdout = "\n".join(
                    [
                        "Testing forest_height=10, rounds=16, batch_size=256",
                        "CYCLES:  1390",
                        "Ran 9 tests in 1.23s",
                        "FAILED (failures=1)",
                        "FAIL: test_opus45_improved_harness (SpeedTests)",
                    ]
                )
                stderr = ""

            def fake_run(cmd, *, check, capture):
                self.assertEqual(cmd, ["python3", "-B", "tests/submission_tests.py"])
                return Proc()

            with (
                patch.object(lr, "_EXPERIMENT_LOG_PATH", log_path),
                patch.object(lr, "_read_state", return_value=state),
                patch.object(lr, "_git", side_effect=fake_git),
                patch.object(lr, "_git_diff_tests_is_empty", return_value=(True, "")),
                patch.object(lr, "_compute_changed_files", return_value=["perf_takehome.py"]),
                patch.object(lr, "_best_cycles_from_best_tags", return_value=1400),
                patch.object(lr, "_run", side_effect=fake_run),
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = lr.cmd_record(argparse.Namespace(print_test_output=False))
                out = buf.getvalue()

            self.assertEqual(rc, 0)
            entry = json.loads(out)
            self.assertIs(entry["new_best"], True)


class TestsDiffGuardrailTests(unittest.TestCase):
    def test_git_diff_tests_is_empty_fails_closed_on_fetch_error(self):
        from unittest.mock import patch

        import tools.loop_runner as lr

        class Proc:
            def __init__(self, rc: int, out: str = "", err: str = ""):
                self.returncode = rc
                self.stdout = out
                self.stderr = err

        def fake_run(argv, **_kwargs):
            self.assertEqual(argv, ["git", "fetch", "--prune", "--tags", "origin"])
            return Proc(1, err="fatal: no such remote")

        with patch.object(lr, "_run", side_effect=fake_run):
            ok, diff = lr._git_diff_tests_is_empty()

        self.assertIs(ok, False)
        self.assertIn("git fetch origin", diff)


if __name__ == "__main__":
    unittest.main()
