"""Regression tests for live proof gates, with only the HTTP boundary replaced."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

from scripts import verify_openai_api as verifier


class LiveVerifierGateTests(unittest.TestCase):
    def run_verifier(self, *, greeting="Hello", record=None):
        record = record if record is not None else {"name": "Alex", "code": "X7", "quantity": 2, "note": None}

        def request(base, path, *, payload=None, timeout):
            if path == "/health":
                return 200, {"status": "ok", "model_loaded": True, "model": "fixture"}, 1
            if path == "/v1/capabilities":
                return 200, {"endpoints": {"chat_completions": {"supported": True}}, "vision": {"supported": False}}, 1
            if path == "/v1/models":
                return 200, {"data": [{"id": "local"}]}, 1
            if path == "/v1/completions":
                return 200, {"choices": [{"text": greeting}]}, 1
            if path == "/v1/responses":
                return 200, {"status": "completed", "output": [{"content": [{"text": greeting}]}]}, 1
            if payload.get("response_format", {}).get("json_schema", {}).get("name") == "bad":
                return 400, {"error": {}}, 1
            if isinstance(payload["messages"][0]["content"], list):
                return 400, {"error": {}}, 1
            name = payload.get("response_format", {}).get("json_schema", {}).get("name")
            text = (
                json.dumps(record)
                if name == "source_record"
                else json.dumps({"ok": True})
                if name == "smoke_check"
                else greeting
            )
            return 200, {"choices": [{"message": {"content": text}}]}, 1

        output = io.StringIO()
        with (
            patch.object(verifier, "request", side_effect=request),
            patch("sys.argv", ["verify"]),
            contextlib.redirect_stdout(output),
        ):
            status = verifier.main()
        return status, json.loads(output.getvalue())

    def test_valid_source_record_is_accepted(self):
        status, report = self.run_verifier()
        self.assertEqual(status, 0)
        self.assertEqual(report["result"], "passed")
        self.assertEqual(len(report["checks"]), 10)

    def test_control_tokens_and_empty_generation_never_pass_live_gate(self):
        for text in ("", " ", "<|im_start|>" * 20, "Hi<|im_end|>", "<|startoftext|>", "<|endoftext|>"):
            with self.subTest(text=text):
                status, report = self.run_verifier(greeting=text)
                self.assertEqual(status, 1)
                self.assertEqual(report["result"], "failed")

    def test_valid_json_with_wrong_source_fact_or_invented_value_fails(self):
        for quantity, note in ((3, None), (2, "invented")):
            with self.subTest(quantity=quantity, note=note):
                status, report = self.run_verifier(
                    record={"name": "Alex", "code": "X7", "quantity": quantity, "note": note}
                )
                self.assertEqual(status, 1)
                self.assertEqual(report["result"], "failed")
