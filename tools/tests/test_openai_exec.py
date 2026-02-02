import unittest
from unittest import mock

from tools import openai_exec


class FakeResponse:
    def __init__(self, status_code, json_body=None, *, text="", headers=None):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json_body


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def request(self, *, method, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers or {},
                "json": json,
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise AssertionError("FakeSession: no more scripted responses.")
        return self._responses.pop(0)


class SanitizeSchemaNameTests(unittest.TestCase):
    def test_sanitize_schema_name_basic(self):
        self.assertEqual(openai_exec.sanitize_schema_name(" My Schema! "), "My_Schema")

    def test_sanitize_schema_name_empty_falls_back(self):
        self.assertEqual(openai_exec.sanitize_schema_name("!!!"), "schema")

    def test_sanitize_schema_name_truncates(self):
        name = "a" * 200
        out = openai_exec.sanitize_schema_name(name)
        self.assertLessEqual(len(out), 64)


class SanitizeArtifactStemTests(unittest.TestCase):
    def test_sanitize_artifact_stem_removes_path_separators(self):
        self.assertEqual(openai_exec.sanitize_artifact_stem(" iter/0001-next "), "iter_0001-next")

    def test_sanitize_artifact_stem_empty_falls_back(self):
        self.assertEqual(openai_exec.sanitize_artifact_stem("///"), "artifact")


class WriteResponseArtifactsTests(unittest.TestCase):
    def test_write_response_artifacts_writes_request_and_response_json(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            dir_path = Path(td)
            req, resp = openai_exec.write_response_artifacts(
                dir_path=dir_path,
                stem="smoke/resp_123",
                request_payload={"model": "gpt-5.2-pro", "prompt": "hi"},
                response_json={"id": "resp_123", "status": "completed"},
            )
            self.assertTrue(req.exists())
            self.assertTrue(resp.exists())
            self.assertTrue(req.name.endswith(".request.json"))
            self.assertTrue(resp.name.endswith(".response.json"))
            self.assertEqual(json.loads(req.read_text(encoding="utf-8"))["model"], "gpt-5.2-pro")
            self.assertEqual(json.loads(resp.read_text(encoding="utf-8"))["id"], "resp_123")


class ExtractOutputTextTests(unittest.TestCase):
    def test_prefers_output_text(self):
        self.assertEqual(openai_exec._extract_output_text({"output_text": "hi"}), "hi")

    def test_joins_output_text_chunks(self):
        body = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "hi"},
                        {"type": "output_text", "text": " there"},
                    ]
                }
            ]
        }
        self.assertEqual(openai_exec._extract_output_text(body), "hi there")

    def test_raises_on_refusal(self):
        body = {"output": [{"content": [{"type": "refusal", "text": "no"}]}]}
        with self.assertRaises(openai_exec.OpenAIExecResponseError):
            openai_exec._extract_output_text(body)


class CreateResponseTests(unittest.TestCase):
    def _client_with(self, session):
        cfg = openai_exec.OpenAIExecConfig(
            api_key="test-key",
            responses_endpoint="https://example.invalid/v1/responses",
            # Make polling tight and quiet in tests.
            background_poll_interval_s=0.0,
            background_poll_timeout_s=5.0,
            background_progress_every_s=0.0,
            http_max_attempts=1,
        )
        return openai_exec.OpenAIExec(cfg, session=session)

    def test_start_response_does_not_poll(self):
        session = FakeSession([FakeResponse(200, {"id": "r1", "status": "queued"})])
        client = self._client_with(session)

        resp = client.start_response(model="gpt-5.2-pro", prompt="ping", reasoning_effort="xhigh")
        self.assertEqual(resp["id"], "r1")
        self.assertEqual(resp["status"], "queued")

        self.assertEqual(len(session.calls), 1)
        post = session.calls[0]
        self.assertEqual(post["method"], "POST")
        self.assertTrue(post["json"]["background"])
        self.assertTrue(post["json"]["store"])

    @mock.patch.object(openai_exec.time, "sleep", autospec=True)
    def test_background_auto_enabled_for_xhigh(self, sleep_mock):
        session = FakeSession(
            [
                FakeResponse(200, {"id": "r1", "status": "queued"}),
                FakeResponse(200, {"id": "r1", "status": "completed", "output_text": "ok"}),
            ]
        )
        client = self._client_with(session)

        text = client.create_response_text(model="gpt-5.2-pro", prompt="ping", reasoning_effort="xhigh")
        self.assertEqual(text, "ok")

        self.assertEqual(len(session.calls), 2)
        post = session.calls[0]
        self.assertEqual(post["method"], "POST")
        self.assertTrue(post["json"]["background"])
        self.assertTrue(post["json"]["store"])
        self.assertEqual(post["json"]["model"], "gpt-5.2-pro")
        self.assertEqual(post["json"]["input"], "ping")
        self.assertEqual(post["json"]["reasoning"], {"effort": "xhigh"})

        get = session.calls[1]
        self.assertEqual(get["method"], "GET")
        self.assertTrue(get["url"].endswith("/r1"))

    @mock.patch.object(openai_exec.time, "sleep", autospec=True)
    def test_function_tool_payload_with_web_search(self, sleep_mock):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "name": "emit_plan",
                                "arguments": "{\"ok\": true}",
                            }
                        ],
                    },
                )
            ]
        )
        client = self._client_with(session)

        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        tools = [
            {"type": "web_search"},
            openai_exec.build_function_tool(name="emit_plan", description="Emit plan.", parameters=schema, strict=True),
        ]
        out = client.create_response_function_call_arguments(
            function_name="emit_plan",
            model="gpt-5.2-pro",
            prompt="Return a plan.",
            reasoning_effort="medium",
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=False,
        )
        self.assertEqual(out, {"ok": True})

        self.assertEqual(len(session.calls), 1)
        payload = session.calls[0]["json"]
        self.assertEqual(payload["tools"][0]["type"], "web_search")
        self.assertEqual(payload["tools"][1]["type"], "function")
        self.assertEqual(payload["tools"][1]["name"], "emit_plan")
        self.assertTrue(payload["tools"][1]["strict"])
        self.assertEqual(payload["tools"][1]["parameters"], schema)
        self.assertEqual(payload["tool_choice"], "required")
        self.assertFalse(payload["parallel_tool_calls"])


class ExtractFunctionCallArgumentsTests(unittest.TestCase):
    def test_extracts_function_call_arguments_json_string(self):
        resp = {
            "output": [
                {"type": "web_search_call", "status": "completed"},
                {"type": "function_call", "name": "emit", "arguments": "{\"x\": 1}"},
            ]
        }
        self.assertEqual(openai_exec.extract_function_call_arguments(resp, function_name="emit"), {"x": 1})

    def test_raises_when_missing(self):
        resp = {"output": [{"type": "output_text", "text": "hi"}]}
        with self.assertRaises(openai_exec.OpenAIExecResponseError):
            openai_exec.extract_function_call_arguments(resp, function_name="emit")


class RetryBackoffTests(unittest.TestCase):
    @mock.patch.object(openai_exec.random, "uniform", autospec=True, return_value=0.0)
    @mock.patch.object(openai_exec.time, "sleep", autospec=True)
    def test_429_honors_retry_after_header(self, sleep_mock, uniform_mock):
        session = FakeSession(
            [
                FakeResponse(429, text="rate limited", headers={"Retry-After": "2"}),
                FakeResponse(200, {"status": "completed", "output_text": "ok"}),
            ]
        )
        cfg = openai_exec.OpenAIExecConfig(
            api_key="test-key",
            responses_endpoint="https://example.invalid/v1/responses",
            http_max_attempts=2,
            http_backoff_base_s=0.0,
            http_backoff_max_s=0.0,
            background_progress_every_s=0.0,
        )
        client = openai_exec.OpenAIExec(cfg, session=session)

        out = client.create_response_text(model="gpt-5.2-pro", prompt="ping", reasoning_effort="medium")
        self.assertEqual(out, "ok")
        # With base/cap at 0, Retry-After should dominate.
        sleep_mock.assert_called()
        slept = float(sleep_mock.call_args[0][0])
        self.assertEqual(slept, 2.0)

    @mock.patch.object(openai_exec.random, "uniform", autospec=True, return_value=0.0)
    @mock.patch.object(openai_exec.time, "sleep", autospec=True)
    def test_retries_on_500_then_succeeds(self, sleep_mock, uniform_mock):
        session = FakeSession(
            [
                FakeResponse(500, text="server error", headers={"Retry-After": "0"}),
                FakeResponse(200, {"status": "completed", "output_text": "ok"}),
            ]
        )
        cfg = openai_exec.OpenAIExecConfig(
            api_key="test-key",
            responses_endpoint="https://example.invalid/v1/responses",
            http_max_attempts=2,
            http_backoff_base_s=0.0,
            http_backoff_max_s=0.0,
            background_progress_every_s=0.0,
        )
        client = openai_exec.OpenAIExec(cfg, session=session)

        out = client.create_response_text(model="gpt-5.2-pro", prompt="ping", reasoning_effort="medium")
        self.assertEqual(out, "ok")
        self.assertEqual(len(session.calls), 2)
        self.assertGreaterEqual(sleep_mock.call_count, 1)

    def test_400_is_not_retried(self):
        session = FakeSession([FakeResponse(400, text="bad request")])
        cfg = openai_exec.OpenAIExecConfig(
            api_key="test-key",
            responses_endpoint="https://example.invalid/v1/responses",
            http_max_attempts=3,
            http_backoff_base_s=0.0,
            http_backoff_max_s=0.0,
            background_progress_every_s=0.0,
        )
        client = openai_exec.OpenAIExec(cfg, session=session)
        with self.assertRaises(openai_exec.OpenAIExecResponseError):
            client.create_response_text(model="gpt-5.2-pro", prompt="ping", reasoning_effort="medium")
        self.assertEqual(len(session.calls), 1)


class PollingTests(unittest.TestCase):
    def test_poll_response_times_out(self):
        class Clock:
            def __init__(self):
                self.now = 0.0

            def monotonic(self):
                return self.now

            def sleep(self, s):
                self.now += float(s)

        clock = Clock()
        session = FakeSession(
            [
                FakeResponse(200, {"id": "r1", "status": "queued"}),
                FakeResponse(200, {"id": "r1", "status": "queued"}),
            ]
        )
        cfg = openai_exec.OpenAIExecConfig(
            api_key="test-key",
            responses_endpoint="https://example.invalid/v1/responses",
            background_poll_interval_s=0.6,
            background_poll_timeout_s=1.0,
            background_progress_every_s=0.0,
            http_max_attempts=1,
        )
        client = openai_exec.OpenAIExec(cfg, session=session)

        with (
            mock.patch.object(openai_exec.time, "monotonic", autospec=True, side_effect=clock.monotonic),
            mock.patch.object(openai_exec.time, "sleep", autospec=True, side_effect=clock.sleep),
        ):
            with self.assertRaises(openai_exec.OpenAIExecTimeoutError):
                client.poll_response(response_id="r1", initial_status="queued")

    def test_poll_response_emits_progress_heartbeat(self):
        import contextlib
        import io

        class Clock:
            def __init__(self):
                self.now = 0.0

            def monotonic(self):
                return self.now

            def sleep(self, s):
                self.now += float(s)

        clock = Clock()
        session = FakeSession(
            [
                FakeResponse(200, {"id": "r1", "status": "in_progress"}),
                FakeResponse(200, {"id": "r1", "status": "completed", "output_text": "ok"}),
            ]
        )
        cfg = openai_exec.OpenAIExecConfig(
            api_key="test-key",
            responses_endpoint="https://example.invalid/v1/responses",
            background_poll_interval_s=60.0,
            background_poll_timeout_s=180.0,
            background_progress_every_s=60.0,
            http_max_attempts=1,
        )
        client = openai_exec.OpenAIExec(cfg, session=session)

        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            mock.patch.object(openai_exec.time, "monotonic", autospec=True, side_effect=clock.monotonic),
            mock.patch.object(openai_exec.time, "sleep", autospec=True, side_effect=clock.sleep),
        ):
            resp = client.poll_response(response_id="r1", initial_status="queued")
        self.assertEqual(resp["status"], "completed")
        out = stderr.getvalue()
        self.assertIn("[openai_exec] polling id=r1", out)
        self.assertIn("[openai_exec] polled id=r1", out)
        self.assertIn("elapsed=1m", out)


class RetrieveResponseTests(unittest.TestCase):
    def test_retrieve_response_can_return_terminal_failure_without_raising(self):
        session = FakeSession(
            [
                FakeResponse(200, {"id": "r1", "status": "failed", "error": {"message": "boom"}}),
                FakeResponse(200, {"id": "r2", "status": "failed", "error": {"message": "boom"}}),
            ]
        )
        cfg = openai_exec.OpenAIExecConfig(
            api_key="test-key",
            responses_endpoint="https://example.invalid/v1/responses",
            http_max_attempts=1,
            background_progress_every_s=0.0,
        )
        client = openai_exec.OpenAIExec(cfg, session=session)

        out = client.retrieve_response(response_id="r1", raise_on_terminal=False)
        self.assertEqual(out["status"], "failed")
        with self.assertRaises(openai_exec.OpenAIExecResponseError):
            client.retrieve_response(response_id="r2", raise_on_terminal=True)


if __name__ == "__main__":
    unittest.main()
