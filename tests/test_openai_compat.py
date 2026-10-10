"""Tests for the server's OpenAI-compatible request contract."""

import json
import random
import unittest

from mlx_flash_compress.openai_compat import (
    APIRequestError,
    api_capabilities,
    loads_json,
    parse_response_format,
    validate_chat_request,
    validate_structured_output,
)


class StructuredOutputContractTests(unittest.TestCase):
    def test_seeded_schema_mutations_fail_closed(self):
        rng = random.Random(20261010)
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1, "maximum": 100},
                "tags": {"type": "array", "maxItems": 3, "items": {"type": "string", "enum": ["a", "b"]}},
                "optional": {"type": ["string", "null"]},
            },
            "required": ["count", "tags", "optional"],
            "additionalProperties": False,
        }
        fmt = parse_response_format({"type": "json_schema", "json_schema": {"name": "mutations", "schema": schema}})
        for index in range(250):
            good = {
                "count": rng.randint(1, 100),
                "tags": [rng.choice(["a", "b"]) for _ in range(rng.randrange(4))],
                "optional": rng.choice([None, "Zażółć 🌸"]),
            }
            self.assertEqual(validate_structured_output(json.dumps(good), fmt), good)
            bad = dict(good)
            mutation = index % 5
            if mutation == 0:
                bad["count"] = rng.choice([True, False, 0, 101, "2", None])
            elif mutation == 1:
                bad["tags"] = ["a"] * 4
            elif mutation == 2:
                bad["optional"] = rng.choice([1, [], {}])
            elif mutation == 3:
                bad.pop(rng.choice(list(good)))
            else:
                bad["unexpected"] = "not allowed"
            with self.subTest(index=index, mutation=mutation), self.assertRaises(APIRequestError):
                validate_structured_output(json.dumps(bad), fmt)

    def test_output_limits_unicode_and_nesting_do_not_escape_validation(self):
        fmt = parse_response_format({"type": "json_object"})
        for raw in ('{"x":"\\ud800"}', '{"x":' + "[" * 65 + "0" + "]" * 65 + "}", '{"x":"' + "x" * (256 * 1024) + '"}'):
            # Escaped Unicode surrogates are JSON data; raw surrogates are
            # invalid UTF-8 for an API response and must be rejected.
            raw = raw.replace("\\ud800", "\ud800")
            with self.subTest(length=len(raw)), self.assertRaises(APIRequestError):
                validate_structured_output(raw, fmt)

    def test_schema_failure_identifies_constraints_without_echoing_model_values(self):
        response_format = parse_response_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "record",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "amount": {"anyOf": [{"type": "string", "pattern": "^[0-9]+$"}, {"type": "null"}]}
                        },
                        "required": ["amount"],
                        "additionalProperties": False,
                    },
                },
            }
        )
        with self.assertRaises(APIRequestError) as caught:
            validate_structured_output(
                '{"amount":"private-model-value","secret":"private-extra-value"}', response_format
            )
        self.assertIn("amount", str(caught.exception))
        self.assertIn("pattern", str(caught.exception))
        self.assertNotIn("private-model-value", str(caught.exception))
        self.assertNotIn("private-extra-value", str(caught.exception))
        self.assertEqual(caught.exception.status, 502)

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
            {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {"type": "string", "pattern": "a*a*a*a*a*a*a*a*a*b"}},
            },
            {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {"type": "object", "unevaluatedProperties": False}},
            },
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

        for content in ("not json", '{"ok":"yes"}', '{"ok":true,"extra":1}'):
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
                            "price": {
                                "anyOf": [{"type": "string", "pattern": "^[0-9]+(\\.[0-9]{1,2})?$"}, {"type": "null"}]
                            },
                            "tags": {
                                "type": "array",
                                "maxItems": 2,
                                "items": {"type": "string", "enum": ["known", "new"]},
                            },
                        },
                        "required": ["price", "tags"],
                        "additionalProperties": False,
                    },
                },
            }
        )
        self.assertEqual(
            validate_structured_output('{"price":"80.50","tags":["known"]}', response_format),
            {"price": "80.50", "tags": ["known"]},
        )


if __name__ == "__main__":
    unittest.main()
