#!/usr/bin/env python3
"""Run live, dependency-free checks against a running mlx-flash server.

Example: python scripts/verify_openai_api.py --base-url http://127.0.0.1:8080
The JSON report is written to stdout and can be saved as release evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


class CheckFailure(RuntimeError):
    pass


def request(base_url: str, path: str, *, payload: dict | None = None, timeout: float) -> tuple[int, dict, float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url.rstrip("/") + path, data=body, headers=headers)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status, raw = response.status, response.read(2 * 1024 * 1024)
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read(64 * 1024)
    elapsed = round((time.monotonic() - started) * 1000, 1)
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CheckFailure(f"{path} returned non-JSON response (HTTP {status}).") from None
    if not isinstance(decoded, dict):
        raise CheckFailure(f"{path} returned a non-object JSON response (HTTP {status}).")
    return status, decoded, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=180.0, help="per-request timeout; generation can be slow on first run")
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 8192 or args.timeout <= 0:
        parser.error("--max-tokens must be 1..8192 and --timeout must be positive")

    checks: list[dict] = []

    def check(name, path, *, payload=None, expected_status=200, verify=None):
        status, result, elapsed = request(args.base_url, path, payload=payload, timeout=args.timeout)
        if status != expected_status:
            raise CheckFailure(f"{name}: expected HTTP {expected_status}, got {status}: {result}")
        if verify:
            verify(result)
        checks.append({"name": name, "status": "passed", "http_status": status, "elapsed_ms": elapsed})
        return result

    def verify_health(value):
        if value.get("status") != "ok" or value.get("model_loaded") is not True:
            raise CheckFailure("server is not healthy or has no loaded model")

    try:
        health = check("health", "/health", verify=verify_health)
        capabilities = check("capabilities", "/v1/capabilities")
        if not capabilities.get("endpoints", {}).get("chat_completions", {}).get("supported"):
            raise CheckFailure("capability inventory does not advertise chat completions")
        if capabilities.get("vision", {}).get("supported") is not False:
            raise CheckFailure("vision capability is unexpectedly reported; verify implementation before release")
        checks[-1]["assertions"] = ["chat completions advertised", "vision reported unavailable"]
        models = check("models", "/v1/models")
        if not models.get("data") or not isinstance(models["data"][0].get("id"), str):
            raise CheckFailure("models response has no model id")

        chat = check("chat_completions", "/v1/chat/completions", payload={
            "model": "local", "messages": [{"role": "user", "content": "Reply with one short greeting."}],
            "max_tokens": args.max_tokens, "temperature": 0,
        })
        if not chat.get("choices") or not isinstance(chat["choices"][0].get("message", {}).get("content"), str):
            raise CheckFailure("chat completion response has no assistant text")

        completion = check("completions", "/v1/completions", payload={
            "model": "local", "prompt": "Write one short greeting.", "max_tokens": args.max_tokens, "temperature": 0,
        })
        if not completion.get("choices") or not isinstance(completion["choices"][0].get("text"), str):
            raise CheckFailure("completion response has no text")

        response = check("responses", "/v1/responses", payload={
            "model": "local", "input": "Write one short greeting.", "max_output_tokens": args.max_tokens, "temperature": 0,
        })
        if response.get("status") != "completed" or not response.get("output"):
            raise CheckFailure("responses endpoint did not return a completed output")

        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
        structured = check("structured_output", "/v1/chat/completions", payload={
            "model": "local", "messages": [{"role": "user", "content": 'Return exactly {"ok":true}.'}],
            "max_tokens": args.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "smoke_check", "strict": True, "schema": schema}},
        })
        try:
            structured_value = json.loads(structured["choices"][0]["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError):
            raise CheckFailure("structured output was not valid JSON") from None
        if not isinstance(structured_value, dict) or set(structured_value) != {"ok"} or not isinstance(structured_value["ok"], bool):
            raise CheckFailure("structured output did not satisfy the smoke-test schema")

        check("invalid_schema_rejected", "/v1/chat/completions", payload={
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "bad", "schema": {"type": "unknown"}}},
        }, expected_status=400)
        check("image_input_rejected", "/v1/chat/completions", payload={
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}],
        }, expected_status=400)

        report = {"result": "passed", "checked_at": datetime.now(timezone.utc).isoformat(), "base_url": args.base_url,
                  "model": health.get("model"), "checks": checks}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (CheckFailure, OSError, TimeoutError, urllib.error.URLError) as error:
        report = {"result": "failed", "checked_at": datetime.now(timezone.utc).isoformat(), "base_url": args.base_url,
                  "checks": checks, "error": str(error)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
