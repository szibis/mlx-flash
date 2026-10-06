"""Model-free guards for native API proof interpretation."""

import copy
import json
import unittest
from unittest.mock import patch

import native_api_proofs as proofs


def response():
    return {
        "id": "proof",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "synthetic"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        "mlx_flash_compress": {
            "native_generation_metadata": True,
            "usage_source": "exact_mlx_lm_generation",
            "cached_prompt_tokens": 9,
            "processed_prompt_tokens": 1,
        },
    }


def stream():
    result = response()
    first = {
        "id": "proof",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "synthetic"}, "finish_reason": None}],
    }
    last = {
        "id": "proof",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        "usage": result["usage"],
        "mlx_flash_compress": result["mlx_flash_compress"],
    }
    return "".join("data: " + json.dumps(item) + "\n\n" for item in (first, last)) + "data: [DONE]\n\n"


class NativeProofTests(unittest.TestCase):
    def test_failed_native_check_is_retained_without_response_content(self):
        state = {
            "prompt_cache": {
                "enabled": True,
                "entries": 0,
                "max_entries": 2,
                "memory_bytes": 0,
                "max_bytes": 32,
                "reused_tokens": 0,
                "processed_tokens": 0,
            },
            "stats": {"requests": 0, "tokens_generated": 0},
        }
        checks = []
        with patch.object(proofs, "exchange", return_value=(200, "{}", "application/json")):
            with self.assertRaisesRegex(RuntimeError, "malformed completion"):
                proofs.run_native_proofs(
                    None, lambda _: state, {"model_family": "lfm2_moe"}, "lfm2_moe", (None,), checks, "small"
                )
        self.assertEqual(checks, [{"check": "native.cold", "model_size": "small", "passed": False}])

    def test_capabilities_must_match_cached_model_before_api_calls(self):
        with self.assertRaisesRegex(RuntimeError, "cached family"):
            proofs.run_native_proofs(None, None, {"model_family": "qwen3_5"}, "gemma4", (False, True), [], "small")

    def test_budget_one_accepts_native_eos_or_length_and_exact_accounting(self):
        for finish in ("stop", "length"):
            data = response()
            data["choices"][0]["finish_reason"] = finish
            self.assertEqual(proofs.validate_completion(data, 1)["text"], "synthetic")

    def test_malformed_usage_cache_or_finish_cannot_pass(self):
        mutations = [
            ("usage", "total_tokens", 12),
            ("usage", "completion_tokens", True),
            ("usage", "prompt_tokens", 0),
            ("mlx_flash_compress", "processed_prompt_tokens", 0),
            ("mlx_flash_compress", "cached_prompt_tokens", -1),
            ("mlx_flash_compress", "native_generation_metadata", False),
        ]
        for section, key, value in mutations:
            data = response()
            data[section][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                proofs.validate_completion(data, 1)
        for malformed in ({}, {"choices": []}, dict(response(), choices=[{"finish_reason": None}])):
            with self.assertRaises(RuntimeError):
                proofs.validate_completion(malformed, 1)

    def test_sse_requires_done_finish_usage_and_well_formed_chunks(self):
        self.assertEqual(proofs.parse_sse(stream(), 1)["text"], "synthetic")
        for malformed in (
            stream().replace("data: [DONE]\n\n", ""),
            stream() + "data: {}\n\n",
            stream().replace('"finish_reason": "length"', '"finish_reason": null'),
            stream().replace("data: ", "event: ", 1),
            "data: not-json\n\ndata: [DONE]\n\n",
        ):
            with self.subTest(malformed=malformed), self.assertRaises(RuntimeError):
                proofs.parse_sse(malformed, 1)

    def test_sse_requires_nonempty_identity_across_chunks(self):
        for malformed in (
            stream().replace('"id": "proof"', '"id": ""'),
            stream().replace('"id": "proof"', '"id": "changed"', 1),
        ):
            with self.subTest(malformed=malformed), self.assertRaises(RuntimeError):
                proofs.parse_sse(malformed, 1)

    def test_sse_requires_terminated_final_frame(self):
        for malformed in (stream().rstrip(), stream()[:-1]):
            with self.subTest(malformed=malformed), self.assertRaises(RuntimeError):
                proofs.parse_sse(malformed, 1)

    def test_completion_requires_identity_and_object(self):
        for key, value in (("id", ""), ("id", None), ("object", "chat.completion.chunk")):
            data = response()
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                proofs.validate_completion(data, 1)
        for key in ("id", "object"):
            data = response()
            del data[key]
            with self.subTest(missing=key), self.assertRaises(RuntimeError):
                proofs.validate_completion(data, 1)

    def test_status_rejects_unbounded_cache_and_counts_only_completed(self):
        state = {
            "prompt_cache": {
                "enabled": True,
                "entries": 1,
                "max_entries": 2,
                "memory_bytes": 16,
                "max_bytes": 32,
                "reused_tokens": 9,
                "processed_tokens": 1,
            },
            "stats": {"requests": 1, "tokens_generated": 1},
        }
        proofs.validate_status(state)
        after = copy.deepcopy(state)
        after["stats"]["requests"] += 1
        with self.assertRaises(RuntimeError):
            proofs.assert_rejected_unchanged(state, after)
        for key, value in (("entries", 3), ("memory_bytes", 33), ("memory_bytes", -1)):
            after = copy.deepcopy(state)
            after["prompt_cache"][key] = value
            with self.assertRaises(RuntimeError):
                proofs.validate_status(after)

    def test_scopes_and_profiles_are_distinct_and_artifacts_hash_content(self):
        false = proofs.proof_payload(False, "same")
        true = proofs.proof_payload(True, "same")
        fresh = proofs.proof_payload(False, "fresh")
        self.assertNotEqual(false, true)
        self.assertNotEqual(false["cache_scope"], fresh["cache_scope"])
        summary = proofs.proof_record("synthetic", false, proofs.validate_completion(response(), 1))
        self.assertNotIn("synthetic", json.dumps(summary).replace("synthetic", "", 1))
        self.assertEqual(len(summary["input_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
