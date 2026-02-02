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


class CodeContextTests(unittest.TestCase):
    def test_extract_kernelbuilder_source_contains_expected_anchors(self):
        from tools.loop_runner import _extract_kernelbuilder_source

        src = _extract_kernelbuilder_source()
        self.assertIn("class KernelBuilder", src)
        self.assertIn("def build_kernel", src)


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


if __name__ == "__main__":
    unittest.main()
