"""Tests for the server's OpenAI-compatible request contract."""

import unittest

from mlx_flash_compress.openai_compat import APIRequestError, api_capabilities, loads_json, parse_response_format, validate_chat_request, validate_structured_output


class StructuredOutputContractTests(unittest.TestCase):
    def test_chat_request_validation_rejects_ignored_or_unsafe_inputs(self):
        base = {"messages": [{"role": "user", "content": "hello"}]}
        validate_chat_request({**base, "reasoning_effort": "none", "max_completion_tokens": 128})
        bad_requests = [
            {**base, "messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
            {**base, "unknown_parameter": 1},
            {**base, "max_tokens": True},
            {**base, "temperature": float("nan")},
            {**base, "reasoning_effort": "high"},
        ]
        for request in bad_requests:
            with self.subTest(request=request), self.assertRaises(APIRequestError):
                validate_chat_request(request)

    def test_request_json_rejects_duplicates_non_finite_and_deep_input(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b"[" * 1200 + b"0" + b"]" * 1200):
            with self.subTest(raw=raw), self.assertRaises(APIRequestError):
                loads_json(raw)

    def test_capability_inventory_does_not_claim_vision_or_unimplemented_endpoints(self):
        capabilities = api_capabilities()
        self.assertTrue(capabilities["structured_outputs"]["supported"])
        self.assertEqual(capabilities["structured_outputs"]["enforcement"], "validated_before_success")
        self.assertFalse(capabilities["vision"]["supported"])
        self.assertFalse(capabilities["image_operations"]["supported"])
        self.assertTrue(capabilities["endpoints"]["responses"]["supported"])

    def test_accepts_a_valid_strict_json_schema_format(self):
        response_format = parse_response_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "proposal",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            }
        )

        self.assertEqual(response_format.name, "proposal")
        self.assertIs(response_format.strict, True)
        self.assertEqual(validate_structured_output('{"ok":true}', response_format), {"ok": True})

    def test_rejects_malformed_or_unsupported_response_formats(self):
        cases = [
            {"type": "json_schema", "json_schema": {"schema": {"type": "invalid-test-schema"}}},
            {"type": "json_schema", "json_schema": {"schema": {"type": "object", "required": "ok"}}},
            {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "string", "pattern": "(a+)+$"}}},
            {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "string", "pattern": "a*a*a*a*a*a*a*a*a*b"}}},
            {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "unevaluatedProperties": False}}},
            {"type": "yaml_object"},
        ]
        for response_format in cases:
            with self.subTest(response_format=response_format), self.assertRaises(APIRequestError):
                parse_response_format(response_format)

    def test_strict_output_must_match_schema_and_be_json(self):
        response_format = parse_response_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "proposal",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            }
        )

        for content in ('not json', '{"ok":"yes"}', '{"ok":true,"extra":1}'):
            with self.subTest(content=content), self.assertRaises(APIRequestError):
                validate_structured_output(content, response_format)

    def test_rejects_duplicate_keys_and_non_finite_json(self):
        response_format = parse_response_format({"type": "json_object"})
        for content in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(content=content), self.assertRaises(APIRequestError):
                validate_structured_output(content, response_format)

    def test_supports_application_schema_patterns_unions_and_constraints(self):
        response_format = parse_response_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "owner_proposal",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "price": {"anyOf": [{"type": "string", "pattern": "^[0-9]+(\\.[0-9]{1,2})?$"}, {"type": "null"}]},
                            "tags": {"type": "array", "maxItems": 2, "items": {"type": "string", "enum": ["known", "new"]}},
                        },
                        "required": ["price", "tags"],
                        "additionalProperties": False,
                    },
                },
            }
        )
        self.assertEqual(validate_structured_output('{"price":"80.50","tags":["known"]}', response_format), {"price": "80.50", "tags": ["known"]})


if __name__ == "__main__":
    unittest.main()
