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


class ManualDirectiveParsingTests(unittest.TestCase):
    def test_parse_json_relaxed_strips_code_fences(self):
        from tools.loop_runner import _parse_json_relaxed

        obj = _parse_json_relaxed("```json\n{\"ok\": true}\n```")
        self.assertEqual(obj, {"ok": True})


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


if __name__ == "__main__":
    unittest.main()
