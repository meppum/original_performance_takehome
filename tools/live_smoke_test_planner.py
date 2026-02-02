from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.openai_exec import (  # noqa: E402
    OpenAIExec,
    OpenAIExecConfig,
    build_payload_for_planner,
    extract_function_call_arguments,
    write_response_artifacts,
)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def _repo_root() -> Path:
    return _REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Live smoke test for gpt-5.2-pro planner calls.")
    parser.add_argument("--model", default="gpt-5.2-pro")
    parser.add_argument(
        "--require-web-search",
        action="store_true",
        help="Ask the model to use web_search before emitting the directive (still small).",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        _load_dotenv(_repo_root() / ".env")

    cfg = OpenAIExecConfig.from_env()
    client = OpenAIExec(cfg)

    tool_name = "emit_optimization_directive"
    directive_schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "objective": {"type": "string"},
            "step_plan": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 3,
            },
            "did_web_search": {"type": "boolean"},
        },
        "required": ["objective", "step_plan", "did_web_search"],
    }

    if args.require_web_search:
        prompt = (
            "This is a tiny live smoke test.\n"
            "Use the web_search tool exactly once for the query: 'OpenAI Responses API web_search tool'.\n"
            f"Then call {tool_name} with did_web_search=true.\n"
            "Keep objective and step_plan short.\n"
        )
    else:
        prompt = (
            "This is a tiny live smoke test.\n"
            "Do not use web_search.\n"
            f"Call {tool_name} with did_web_search=false.\n"
            "Keep objective and step_plan short.\n"
        )

    payload = build_payload_for_planner(
        model=args.model,
        prompt=prompt,
        directive_schema=directive_schema,
        directive_tool_name=tool_name,
    )

    # Use create_response() so we can also report tool usage in output[].
    response_json = client.create_response(**payload)
    response_id = response_json.get("id") if isinstance(response_json.get("id"), str) else None
    stem = "smoke" + (f"_{response_id}" if response_id else "")
    req_path, resp_path = write_response_artifacts(
        dir_path=_repo_root() / ".advisor" / "openai_smoke",
        stem=stem,
        request_payload=payload,
        response_json=response_json,
    )
    print(
        f"[live_smoke_test_planner] saved OpenAI artifacts: {req_path.relative_to(_repo_root())} "
        f"{resp_path.relative_to(_repo_root())}",
        file=sys.stderr,
    )

    output = response_json.get("output")
    output_types = []
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and isinstance(item.get("type"), str):
                output_types.append(item["type"])

    directive = extract_function_call_arguments(response_json, function_name=tool_name)

    status = response_json.get("status")
    print(
        json.dumps(
            {
                "id": response_id,
                "status": status,
                "output_types": output_types,
                "directive": directive,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
