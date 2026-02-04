import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class LoopRunnerHermeticE2ETests(unittest.TestCase):
    def _init_temp_repo(self, *, repo_root: Path, tmp_root: Path) -> Path:
        work = tmp_root / "work"
        origin_bare = tmp_root / "origin.git"
        work.mkdir(parents=True, exist_ok=True)

        # Minimal repo layout for loop_runner.py.
        (work / "tools").mkdir()
        (work / "tests").mkdir()
        (work / "experiments").mkdir()

        shutil.copyfile(repo_root / ".gitignore", work / ".gitignore")
        shutil.copyfile(repo_root / "tools" / "__init__.py", work / "tools" / "__init__.py")
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
        subprocess.run(
            ["git", "checkout", "-b", "main"],
            cwd=str(work),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
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
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(work), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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

        return work

    def _offline_env(self) -> dict:
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)  # force offline path to remain offline
        env.pop("CODEX_API_KEY", None)  # force offline path to remain offline
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def test_offline_plan_and_record_in_temp_repo(self):
        # This test creates a throwaway git repo (with a local bare origin) and runs:
        # - `python3 tools/loop_runner.py offline-plan`
        # - `python3 tools/loop_runner.py record`
        #
        # It is intentionally hermetic: no network calls and no mutation of this working repo.
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            work = self._init_temp_repo(repo_root=repo_root, tmp_root=tmp_root)
            env = self._offline_env()

            plan = subprocess.run(
                [
                    "python3",
                    "tools/loop_runner.py",
                    "offline-plan",
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

    def test_manual_pack_apply_and_record_in_temp_repo(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            work = self._init_temp_repo(repo_root=repo_root, tmp_root=tmp_root)
            env = self._offline_env()

            subprocess.run(
                [
                    "python3",
                    "tools/loop_runner.py",
                    "manual-pack",
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

            packet = json.loads((work / "planner_packets" / "packet.json").read_text(encoding="utf-8"))
            self.assertIn("strategy_family_constraints", packet)

            directive = {
                "objective": "reduce cycles",
                "primary_hypothesis": "exercise manual planner flow",
                "strategy_family": "family:schedule",
                "strategy_modifiers": ["manual"],
                "risk": "Low",
                "expected_effect_cycles": 1,
                "change_summary": ["no-op directive for hermetic test"],
                "step_plan": ["step 1", "step 2", "step 3"],
                "validation": {
                    "commands": ["python3 -B tests/submission_tests.py"],
                    "pass_criteria": ["Directive parses and record completes"],
                },
                "next_packet_requests": [],
                "did_web_search": False,
            }
            (work / "planner_packets" / "directive.json").write_text(
                json.dumps(directive, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "add", "planner_packets/directive.json"],
                cwd=str(work),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "commit", "-m", "add directive"],
                cwd=str(work),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            subprocess.run(
                ["python3", "tools/loop_runner.py", "manual-apply", "--no-pull"],
                cwd=str(work),
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            state = json.loads((work / ".advisor" / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["directive"]["strategy_family"], "family:schedule")

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
            self.assertTrue(entry["valid"])
            self.assertEqual(entry["strategy_tags"][:2], ["family:schedule", "manual"])

    def test_tests_mutation_marks_iteration_invalid(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            work = self._init_temp_repo(repo_root=repo_root, tmp_root=tmp_root)
            env = self._offline_env()

            subprocess.run(
                [
                    "python3",
                    "tools/loop_runner.py",
                    "offline-plan",
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

            # Intentionally violate the guardrail.
            (work / "tests" / "submission_tests.py").write_text(
                (work / "tests" / "submission_tests.py").read_text(encoding="utf-8") + "\n# mutated\n",
                encoding="utf-8",
            )

            record = subprocess.run(
                ["python3", "tools/loop_runner.py", "record"],
                cwd=str(work),
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertNotEqual(record.returncode, 0)
            entry = json.loads(record.stdout)
            self.assertFalse(entry["tests_diff_empty"])
            self.assertFalse(entry["valid"])

    def test_plan_packet_exposes_blocked_family_when_streak_exhausted(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            work = self._init_temp_repo(repo_root=repo_root, tmp_root=tmp_root)
            env = self._offline_env()

            # Seed an exhausted streak for family:schedule (2 attempts, no meaningful win).
            (work / "experiments" / "log.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"iteration_id": 1, "valid": True, "cycles": 1500, "strategy_tags": ["family:schedule"]}
                        ),
                        json.dumps(
                            {"iteration_id": 2, "valid": True, "cycles": 1495, "strategy_tags": ["family:schedule"]}
                        ),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    "python3",
                    "tools/loop_runner.py",
                    "offline-plan",
                    "--threshold",
                    "1363",
                    "--slug",
                    "e2e",
                    "--code-context",
                    "none",
                    "--experiment-log-tail-lines",
                    "2",
                ],
                cwd=str(work),
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            state = json.loads((work / ".advisor" / "state.json").read_text(encoding="utf-8"))
            blocked = state["packet"]["strategy_family_constraints"]["blocked_families"]
            self.assertEqual(blocked, ["family:schedule"])


if __name__ == "__main__":
    unittest.main()
