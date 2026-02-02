import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class LoopRunnerHermeticE2ETests(unittest.TestCase):
    def test_offline_plan_and_record_in_temp_repo(self):
        # This test creates a throwaway git repo (with a local bare origin) and runs:
        # - `python3 tools/loop_runner.py plan --offline`
        # - `python3 tools/loop_runner.py record`
        #
        # It is intentionally hermetic: no network calls and no mutation of this working repo.
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            work = tmp_root / "work"
            origin_bare = tmp_root / "origin.git"
            work.mkdir(parents=True, exist_ok=True)

            # Minimal repo layout for loop_runner.py.
            (work / "tools").mkdir()
            (work / "tests").mkdir()
            (work / "experiments").mkdir()

            shutil.copyfile(repo_root / ".gitignore", work / ".gitignore")
            shutil.copyfile(repo_root / "tools" / "__init__.py", work / "tools" / "__init__.py")
            shutil.copyfile(repo_root / "tools" / "openai_exec.py", work / "tools" / "openai_exec.py")
            shutil.copyfile(repo_root / "tools" / "loop_runner.py", work / "tools" / "loop_runner.py")

            (work / "problem.py").write_text(
                textwrap.dedent(
                    """
                    SLOT_LIMITS = {"load": 2, "store": 1, "flow": 1, "valu": 4, "alu": 4}
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (work / "perf_takehome.py").write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass


                    class KernelBuilder:
                        @dataclass
                        class _Task:
                            engine: str
                            succs: list[int]
                            cp: int

                        def __init__(self):
                            self.instrs = []

                        def _mk_task(self, tasks, last_writer, last_reader, *, engine, slot, reads=(), writes=()):
                            t = KernelBuilder._Task(engine=str(engine), succs=[], cp=len(tasks) + 1)
                            tasks.append(t)
                            if len(tasks) >= 2:
                                tasks[-2].succs.append(len(tasks) - 1)
                            return t

                        def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
                            tasks = []
                            last_writer = {}
                            last_reader = {}
                            self._mk_task(tasks, last_writer, last_reader, engine="load", slot=("const", 0, 0))
                            self._mk_task(tasks, last_writer, last_reader, engine="alu", slot=("noop",))
                            self._mk_task(tasks, last_writer, last_reader, engine="valu", slot=("noop",))
                            self.instrs = [{"alu": [("noop",)]}] * 10
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (work / "tests" / "submission_tests.py").write_text(
                textwrap.dedent(
                    """
                    import sys

                    print("Testing forest_height=10, rounds=16, batch_size=256")
                    print("CYCLES:  1500")
                    print("......F..")
                    print("FAIL: test_opus45_improved_harness (__main__.SpeedTests.test_opus45_improved_harness)")
                    print("AssertionError")
                    print("Ran 9 tests in 0.01s")
                    print("FAILED (failures=1)")
                    sys.exit(1)
                    """
                ).lstrip(),
                encoding="utf-8",
            )

            # Initialize git + local origin.
            subprocess.run(["git", "init"], cwd=str(work), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "checkout", "-b", "main"], cwd=str(work), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=str(work),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test Runner"],
                cwd=str(work),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(["git", "add", "-A"], cwd=str(work), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(
                ["git", "commit", "-m", "init"],
                cwd=str(work),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            subprocess.run(
                ["git", "init", "--bare", str(origin_bare)],
                cwd=str(tmp_root),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "remote", "add", "origin", str(origin_bare)],
                cwd=str(work),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "push", "-u", "origin", "main"],
                cwd=str(work),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            env = dict(os.environ)
            env.pop("OPENAI_API_KEY", None)  # force offline path to remain offline

            plan = subprocess.run(
                [
                    "python3",
                    "tools/loop_runner.py",
                    "plan",
                    "--offline",
                    "--threshold",
                    "1363",
                    "--slug",
                    "e2e",
                    "--code-context",
                    "none",
                    "--experiment-log-tail-lines",
                    "1",
                ],
                cwd=str(work),
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            directive = json.loads(plan.stdout)
            self.assertEqual(directive["strategy_family"], "family:schedule")
            self.assertIn("offline", directive["strategy_modifiers"])

            record = subprocess.run(
                ["python3", "tools/loop_runner.py", "record"],
                cwd=str(work),
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            entry = json.loads(record.stdout)
            self.assertEqual(entry["cycles"], 1500)
            self.assertTrue(entry["valid"])
            self.assertEqual(entry["strategy_tags"][:1], ["family:schedule"])


if __name__ == "__main__":
    unittest.main()
