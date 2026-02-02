import json
import os
import subprocess
import unittest
from pathlib import Path


class LivePlannerSmokeTests(unittest.TestCase):
    def _run_smoke(self, extra_args: list[str]) -> dict:
        repo_root = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        env.setdefault("OPENAI_BACKGROUND_POLL_INTERVAL", "5")
        env.setdefault("OPENAI_BACKGROUND_PROGRESS_EVERY", "5")
        env.setdefault("OPENAI_BACKGROUND_POLL_TIMEOUT", "1800")

        proc = subprocess.run(
            ["python3", "tools/live_smoke_test_planner.py", *extra_args],
            cwd=str(repo_root),
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if "OPENAI_API_KEY is not set" in combined:
                raise RuntimeError("missing_api_key")
            raise AssertionError(f"live smoke test failed:\n{combined}")

        return json.loads(proc.stdout)

    def test_live_smoke_planner_basic(self):
        if os.environ.get("RUN_LIVE_TESTS") != "1":
            self.skipTest("Set RUN_LIVE_TESTS=1 to enable live OpenAI smoke tests.")

        try:
            data = self._run_smoke([])
        except RuntimeError as e:
            if str(e) == "missing_api_key":
                self.skipTest("OPENAI_API_KEY not configured for live tests.")
            raise

        self.assertEqual(data.get("status"), "completed")
        directive = data.get("directive") or {}
        self.assertIsInstance(directive, dict)
        self.assertIn("objective", directive)
        self.assertIn("step_plan", directive)
        self.assertIn("did_web_search", directive)

    def test_live_smoke_planner_with_web_search(self):
        if os.environ.get("RUN_LIVE_TESTS") != "1":
            self.skipTest("Set RUN_LIVE_TESTS=1 to enable live OpenAI smoke tests.")
        if os.environ.get("RUN_LIVE_TESTS_WEB_SEARCH") != "1":
            self.skipTest("Set RUN_LIVE_TESTS_WEB_SEARCH=1 to enable live web_search smoke test.")

        try:
            data = self._run_smoke(["--require-web-search"])
        except RuntimeError as e:
            if str(e) == "missing_api_key":
                self.skipTest("OPENAI_API_KEY not configured for live tests.")
            raise

        self.assertEqual(data.get("status"), "completed")
        self.assertIn("web_search_call", data.get("output_types") or [])
        directive = data.get("directive") or {}
        self.assertIs(directive.get("did_web_search"), True)


if __name__ == "__main__":
    unittest.main()

