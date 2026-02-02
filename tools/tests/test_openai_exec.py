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


if __name__ == "__main__":
    unittest.main()
