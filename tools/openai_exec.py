from __future__ import annotations

import dataclasses
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import requests


class OpenAIExecError(RuntimeError):
    pass


class OpenAIExecTimeoutError(OpenAIExecError):
    pass


class OpenAIExecResponseError(OpenAIExecError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, response_id: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_id = response_id


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise OpenAIExecError(f"Invalid {name}={raw!r}; expected number") from e


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise OpenAIExecError(f"Invalid {name}={raw!r}; expected integer") from e


def _parse_retry_after_seconds(headers: Mapping[str, str]) -> Optional[float]:
    # OpenAI and proxies typically send Retry-After in seconds.
    ra = None
    for key in ("Retry-After", "retry-after"):
        if key in headers:
            ra = headers[key]
            break
    if not ra:
        return None
    try:
        seconds = float(ra)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return seconds


def _backoff_seconds(
    attempt: int, *, base: float, cap: float, retry_after_s: Optional[float] = None
) -> float:
    exp = base * (2 ** max(0, attempt - 1))
    delay = min(cap, exp)
    if retry_after_s is not None:
        delay = max(delay, retry_after_s)
    # jitter up to 25% to avoid thundering herd
    jitter = random.uniform(0.0, delay * 0.25)
    return delay + jitter


_SCHEMA_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


def sanitize_schema_name(name: str) -> str:
    s = name.strip()
    s = _SCHEMA_NAME_RE.sub("_", s)
    s = s.strip("_")
    if not s:
        return "schema"
    # Keep short-ish for compatibility.
    return s[:64]


_ARTIFACT_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_artifact_stem(stem: str) -> str:
    """
    Produce a filesystem-safe artifact stem.

    Note: This intentionally does not allow path separators.
    """

    s = stem.strip()
    s = _ARTIFACT_STEM_RE.sub("_", s).strip("._-")
    if not s:
        return "artifact"
    return s[:160]


def write_response_artifacts(
    *,
    dir_path: Path,
    stem: str,
    request_payload: Mapping[str, Any],
    response_json: Mapping[str, Any],
) -> Tuple[Path, Path]:
    """
    Write the OpenAI request/response pair to disk as JSON.

    Intended for debugging/auditing and for sending richer context back to the advisor.
    Callers must ensure secrets are not present in request_payload.
    """

    safe_stem = sanitize_artifact_stem(stem)
    dir_path.mkdir(parents=True, exist_ok=True)

    req_path = dir_path / f"{safe_stem}.request.json"
    resp_path = dir_path / f"{safe_stem}.response.json"

    req_path.write_text(
        json.dumps(dict(request_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resp_path.write_text(
        json.dumps(dict(response_json), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return req_path, resp_path


def build_function_tool(
    *,
    name: str,
    parameters: Mapping[str, Any],
    description: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    tool: Dict[str, Any] = {
        "type": "function",
        "name": name,
        "parameters": dict(parameters),
        "strict": bool(strict),
    }
    if description:
        tool["description"] = description
    return tool


def extract_function_call_arguments(response_json: Mapping[str, Any], *, function_name: str) -> Dict[str, Any]:
    output = response_json.get("output")
    if not isinstance(output, list):
        raise OpenAIExecResponseError("No tool output (missing output[]).")

    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        if item.get("name") != function_name:
            continue

        args = item.get("arguments")
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError as e:
                raise OpenAIExecResponseError(
                    f"Function call arguments are not valid JSON for {function_name!r}: {args[:500]!r}"
                ) from e
            if not isinstance(parsed, dict):
                raise OpenAIExecResponseError(
                    f"Function call arguments must be a JSON object for {function_name!r}."
                )
            return parsed

        raise OpenAIExecResponseError(f"Unexpected function_call.arguments type for {function_name!r}.")

    raise OpenAIExecResponseError(f"No function_call output found for {function_name!r}.")


def _extract_output_text(response_json: Mapping[str, Any]) -> str:
    if isinstance(response_json.get("output_text"), str) and response_json["output_text"]:
        return response_json["output_text"]

    output = response_json.get("output")
    if not isinstance(output, list):
        raise OpenAIExecResponseError("No output text (missing output/output_text).")

    chunks: List[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in ("refusal", "output_refusal"):
                raise OpenAIExecResponseError("Model refusal (no output_text).")
            if part_type == "output_text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])

    text = "".join(chunks).strip()
    if not text:
        raise OpenAIExecResponseError("No output text (empty output_text chunks).")
    return text


def _raise_if_terminal_error(response_json: Mapping[str, Any], *, response_id: Optional[str] = None) -> None:
    status = response_json.get("status")
    if status in (None, "completed"):
        return

    if status in ("failed", "cancelled", "incomplete"):
        details = []
        error = response_json.get("error")
        if isinstance(error, dict):
            msg = error.get("message")
            if isinstance(msg, str) and msg:
                details.append(msg)
        incomplete = response_json.get("incomplete_details")
        if isinstance(incomplete, dict):
            reason = incomplete.get("reason")
            if isinstance(reason, str) and reason:
                details.append(f"incomplete_reason={reason}")
        detail_str = "; ".join(details) if details else "No additional details."
        raise OpenAIExecResponseError(
            f"OpenAI response status={status}. {detail_str}", response_id=response_id
        )


@dataclasses.dataclass(frozen=True)
class OpenAIExecConfig:
    api_key: str
    responses_endpoint: str = "https://api.openai.com/v1/responses"
    open_timeout_s: float = 30.0
    read_timeout_s: float = 600.0
    http_max_attempts: int = 5
    http_backoff_base_s: float = 1.0
    http_backoff_max_s: float = 30.0
    background_poll_interval_s: float = 60.0
    background_poll_timeout_s: float = 14400.0
    background_progress_every_s: float = 60.0
    organization: Optional[str] = None
    project: Optional[str] = None

    @classmethod
    def from_env(cls) -> "OpenAIExecConfig":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise OpenAIExecError("OPENAI_API_KEY is not set.")

        endpoint = os.environ.get("OPENAI_RESPONSES_ENDPOINT", cls.responses_endpoint).strip()
        if not endpoint:
            endpoint = cls.responses_endpoint

        return cls(
            api_key=api_key,
            responses_endpoint=endpoint,
            open_timeout_s=_env_float("OPENAI_HTTP_OPEN_TIMEOUT", cls.open_timeout_s),
            read_timeout_s=_env_float("OPENAI_HTTP_READ_TIMEOUT", cls.read_timeout_s),
            http_max_attempts=_env_int("OPENAI_HTTP_MAX_ATTEMPTS", cls.http_max_attempts),
            http_backoff_base_s=_env_float("OPENAI_HTTP_BACKOFF_BASE", cls.http_backoff_base_s),
            http_backoff_max_s=_env_float("OPENAI_HTTP_BACKOFF_MAX", cls.http_backoff_max_s),
            background_poll_interval_s=_env_float(
                "OPENAI_BACKGROUND_POLL_INTERVAL", cls.background_poll_interval_s
            ),
            background_poll_timeout_s=_env_float(
                "OPENAI_BACKGROUND_POLL_TIMEOUT", cls.background_poll_timeout_s
            ),
            background_progress_every_s=_env_float(
                "OPENAI_BACKGROUND_PROGRESS_EVERY", cls.background_progress_every_s
            ),
            organization=os.environ.get("OPENAI_ORG") or None,
            project=os.environ.get("OPENAI_PROJECT") or None,
        )


class OpenAIExec:
    def __init__(self, config: OpenAIExecConfig, *, session: Optional[requests.Session] = None):
        self._config = config
        self._session = session or requests.Session()

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._config.organization:
            headers["OpenAI-Organization"] = self._config.organization
        if self._config.project:
            headers["OpenAI-Project"] = self._config.project
        return headers

    def _request_json(self, method: str, url: str, *, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        last_error: Optional[BaseException] = None

        for attempt in range(1, self._config.http_max_attempts + 1):
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    headers=self._headers(),
                    json=payload,
                    timeout=(self._config.open_timeout_s, self._config.read_timeout_s),
                )
                if resp.status_code >= 200 and resp.status_code < 300:
                    return resp.json()

                retry_after_s = _parse_retry_after_seconds(resp.headers)
                if resp.status_code == 429 or 500 <= resp.status_code <= 599:
                    delay = _backoff_seconds(
                        attempt,
                        base=self._config.http_backoff_base_s,
                        cap=self._config.http_backoff_max_s,
                        retry_after_s=retry_after_s,
                    )
                    last_error = OpenAIExecResponseError(
                        f"HTTP {resp.status_code} from OpenAI (retrying).",
                        status_code=resp.status_code,
                    )
                    time.sleep(delay)
                    continue

                # Non-retryable HTTP error.
                raise OpenAIExecResponseError(
                    f"HTTP {resp.status_code} from OpenAI: {resp.text[:1000]}",
                    status_code=resp.status_code,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                delay = _backoff_seconds(
                    attempt,
                    base=self._config.http_backoff_base_s,
                    cap=self._config.http_backoff_max_s,
                )
                time.sleep(delay)
                continue
            except requests.RequestException as e:
                # Treat as retryable network error by default.
                last_error = e
                delay = _backoff_seconds(
                    attempt,
                    base=self._config.http_backoff_base_s,
                    cap=self._config.http_backoff_max_s,
                )
                time.sleep(delay)
                continue

        raise OpenAIExecError(f"OpenAI request failed after {self._config.http_max_attempts} attempts: {last_error}")

    def create_response(
        self,
        *,
        model: str,
        prompt: str,
        store: bool = False,
        background: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        tools: Optional[List[Mapping[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
        json_schema: Optional[Mapping[str, Any]] = None,
        json_schema_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "input": prompt,
            "store": bool(store),
        }

        if reasoning_effort is not None:
            if reasoning_effort not in ("low", "medium", "high", "xhigh"):
                raise OpenAIExecError(f"Invalid reasoning_effort={reasoning_effort!r}")
            payload["reasoning"] = {"effort": reasoning_effort}

        # Background mode "auto" rule: enable when reasoning is xhigh unless explicitly overridden.
        if background is None and reasoning_effort == "xhigh":
            background = True

        if background:
            payload["background"] = True
            payload["store"] = True

        if tools is not None:
            payload["tools"] = [dict(t) for t in tools]

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = bool(parallel_tool_calls)

        if json_schema is not None:
            schema_name = sanitize_schema_name(json_schema_name or "schema")
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": schema_name,
                    "schema": json_schema,
                }
            }

        post_json = self._request_json("POST", self._config.responses_endpoint, payload=payload)
        response_id = post_json.get("id")
        status = post_json.get("status")

        if background:
            if not isinstance(response_id, str) or not response_id:
                raise OpenAIExecResponseError("Background response missing id.", response_id=None)
            return self._poll_response(response_id=response_id, initial_status=status)

        _raise_if_terminal_error(post_json, response_id=response_id if isinstance(response_id, str) else None)
        return post_json

    def _poll_response(self, *, response_id: str, initial_status: Optional[str] = None) -> Dict[str, Any]:
        start = time.monotonic()
        deadline = start + self._config.background_poll_timeout_s
        next_progress_at = start + self._config.background_progress_every_s

        status = initial_status or "queued"
        if self._config.background_progress_every_s > 0:
            print(f"[openai_exec] polling id={response_id} status={status} elapsed=0s", file=sys.stderr)
        while status in ("queued", "in_progress"):
            now = time.monotonic()
            if now >= deadline:
                raise OpenAIExecTimeoutError(
                    f"OpenAI background response polling timed out after {self._config.background_poll_timeout_s}s "
                    f"(id={response_id})."
                )

            sleep_s = min(self._config.background_poll_interval_s, max(0.0, deadline - now))
            time.sleep(sleep_s)

            url = f"{self._config.responses_endpoint.rstrip('/')}/{response_id}"
            resp_json = self._request_json("GET", url)
            status = resp_json.get("status") or status

            _raise_if_terminal_error(resp_json, response_id=response_id)
            now = time.monotonic()
            if self._config.background_progress_every_s > 0 and now >= next_progress_at:
                elapsed = int(now - start)
                print(f"[openai_exec] polled id={response_id} status={status} elapsed={elapsed}s", file=sys.stderr)
                next_progress_at = now + self._config.background_progress_every_s
            if status == "completed":
                return resp_json

        # Unexpected status: treat as terminal error with best available details.
        url = f"{self._config.responses_endpoint.rstrip('/')}/{response_id}"
        resp_json = self._request_json("GET", url)
        _raise_if_terminal_error(resp_json, response_id=response_id)
        return resp_json

    def create_response_text(self, **kwargs: Any) -> str:
        response_json = self.create_response(**kwargs)
        response_id = response_json.get("id")
        if isinstance(response_id, str):
            _raise_if_terminal_error(response_json, response_id=response_id)
        return _extract_output_text(response_json)

    def create_response_function_call_arguments(self, *, function_name: str, **kwargs: Any) -> Dict[str, Any]:
        response_json = self.create_response(**kwargs)
        response_id = response_json.get("id")
        if isinstance(response_id, str):
            _raise_if_terminal_error(response_json, response_id=response_id)
        return extract_function_call_arguments(response_json, function_name=function_name)


def build_payload_for_planner(
    *,
    model: str,
    prompt: str,
    directive_schema: Mapping[str, Any],
    directive_tool_name: str = "emit_optimization_directive",
) -> Dict[str, Any]:
    # Convenience helper to make it hard to accidentally drift the planner policy.
    return {
        "model": model,
        "prompt": prompt,
        "store": False,
        "background": None,  # auto-enabled because reasoning_effort is xhigh
        "reasoning_effort": "xhigh",
        "tools": [
            {"type": "web_search"},
            build_function_tool(
                name=directive_tool_name,
                description="Emit the optimization plan as a structured object.",
                parameters=dict(directive_schema),
                strict=True,
            ),
        ],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }


if __name__ == "__main__":
    # Minimal manual smoke test / example:
    #   OPENAI_API_KEY=... python3 tools/openai_exec.py "Say hello"
    if len(sys.argv) < 2:
        print("Usage: python3 tools/openai_exec.py <prompt>", file=sys.stderr)
        raise SystemExit(2)

    cfg = OpenAIExecConfig.from_env()
    client = OpenAIExec(cfg)
    text = client.create_response_text(model="gpt-5.2-pro", prompt=sys.argv[1], reasoning_effort="xhigh")
    print(text)
