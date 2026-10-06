"""Model-free native API contract tests; no Metal generation is performed."""

import importlib.util
import io
import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Import an isolated source module with native dependencies replaced before
# import. This also works in headless environments without initializing Metal.
spec = importlib.util.spec_from_file_location(
    "measurement_serve", Path(__file__).parents[1] / "mlx_flash_compress/serve.py"
)
serve = importlib.util.module_from_spec(spec)
with patch.dict(
    sys.modules,
    {"mlx": MagicMock(), "mlx.core": MagicMock(), "mlx_lm": MagicMock(), "mlx_lm.sample_utils": MagicMock(), "mlx_lm.models": MagicMock(), "mlx_lm.models.cache": MagicMock()},
):
    spec.loader.exec_module(serve)
batch_spec = importlib.util.spec_from_file_location(
    "measurement_batching", Path(__file__).parents[1] / "mlx_flash_compress/continuous_batching.py"
)
batching = importlib.util.module_from_spec(batch_spec)
with patch.dict(sys.modules, {"mlx": MagicMock(), "mlx.core": MagicMock(), "measurement_batching": batching}):
    batch_spec.loader.exec_module(batching)


def state():
    value = object.__new__(serve.InferenceState)
    value.model = object()
    value.tokenizer = MagicMock()
    value.tokenizer.bos_token = None
    value.tokenizer.encode.return_value = list(range(17))  # Input encoding, never output re-tokenization.
    value.model_name = "fixture"
    value.kv_bits = 0
    value.prompt_reuse = serve.PromptReuse()
    value.spec_engine = None
    value.speculative = "none"
    value.batching = False
    value.engine = None
    value.mem_mgr = MagicMock()
    value.total_requests = value.total_tokens = 0
    value.chat_lock = threading.RLock()
    value._format_messages = lambda messages: "formatted prompt"
    return value


def responses(reason="length"):
    return iter(
        [
            SimpleNamespace(
                text="Hel",
                prompt_tokens=17,
                generation_tokens=1,
                generation_tps=12.0,
                prompt_tps=85.0,
                peak_memory=3.0,
                finish_reason=None,
            ),
            SimpleNamespace(
                text="lo",
                prompt_tokens=17,
                generation_tokens=2,
                generation_tps=10.0,
                prompt_tps=85.0,
                peak_memory=3.1,
                finish_reason=reason,
            ),
        ]
    )


def handler(value, payload):
    obj = object.__new__(serve.ChatHandler)
    body = json.dumps(payload).encode()
    obj.headers = {"Content-Length": str(len(body))}
    obj.rfile = io.BytesIO(body)
    obj.wfile = io.BytesIO()
    obj.server_state = value
    obj.results = []
    obj._send_json = lambda data, status=200: obj.results.append((status, data))
    obj.send_response = MagicMock()
    obj.send_header = MagicMock()
    obj.end_headers = MagicMock()
    return obj


class GenerationMeasurementsTests(unittest.TestCase):
    def setUp(self):
        self.mem = SimpleNamespace(pressure_level="normal", available_gb=8)
        self.native = patch.object(
            serve, "stream_generate", create=True, side_effect=lambda *args, **kwargs: responses()
        ).start()
        self.sampler = patch.object(serve, "make_sampler", create=True, return_value="sampler").start()
        patch.object(serve, "generate", return_value="Hello").start()
        patch.object(serve, "get_memory_state", return_value=self.mem).start()
        patch.object(serve, "mx").start()
        self.addCleanup(patch.stopall)

    def test_actual_native_counts_stop_and_sampling_reach_result(self):
        value = state()
        result = value.generate([{"role": "user", "content": "hello"}], 2, 0.1)
        self.assertEqual(result.get("prompt_tokens"), 17)
        self.assertEqual(result.get("tokens"), 2)
        self.assertEqual(result.get("finish_reason"), "length")
        self.assertEqual(result["output"], "Hello")
        self.assertEqual(result.get("generation_tps"), 10.0)
        self.assertEqual(value.total_tokens, 2)
        self.sampler.assert_called_once_with(temp=0.1, top_p=1.0, top_k=0)
        value.tokenizer.encode.assert_called_once_with("formatted prompt", add_special_tokens=True)
        self.assertEqual(value.last_generation["prompt_tokens"], 17)
        self.assertEqual(value.last_generation["generation_tokens"], 2)
        self.assertGreaterEqual(value.last_generation["ttft_ms"], 0)

    def test_incomplete_native_iterator_cannot_claim_completed_generation(self):
        self.native.side_effect = lambda *args, **kwargs: iter([SimpleNamespace(text="partial", finish_reason=None)])
        value = state()
        with self.assertRaisesRegex(RuntimeError, "completed response"):
            value.generate([{"role": "user", "content": "hello"}], 2, 0.1)
        self.assertEqual(value.total_requests, 0)
        self.assertEqual(value.total_tokens, 0)

    def test_eos_at_budget_keeps_native_metadata_and_stop(self):
        self.native.side_effect = lambda *args, **kwargs: responses("stop")
        obj = handler(state(), {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 2})
        obj._handle_chat()
        result = obj.results[0][1]
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        self.assertIs(result["mlx_flash_compress"].get("native_generation_metadata"), True)
        self.assertEqual(result["mlx_flash_compress"].get("usage_source"), "exact_mlx_lm_generation")

    def test_http_usage_and_finish_reason_are_native_not_placeholder(self):
        obj = handler(
            state(),
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 10,
            },
        )
        obj._handle_chat()
        status, result = obj.results[0]
        self.assertEqual(status, 200)
        self.assertEqual(result["usage"], {"prompt_tokens": 17, "completion_tokens": 2, "total_tokens": 19})
        self.assertEqual(result["choices"][0]["finish_reason"], "length")
        self.sampler.assert_called_once_with(temp=0.1, top_p=0.9, top_k=10)

    def test_unsupported_controls_and_invalid_numbers_fail_before_generation(self):
        for extra in (
            {"stop": ["end"]},
            {"seed": 1},
            {"temperature": -1},
            {"temperature": float("nan")},
            {"top_p": 2},
            {"top_k": -1},
            {"max_tokens": False},
            {"n": 2},
        ):
            with self.subTest(extra=extra):
                obj = handler(state(), {"messages": [{"role": "user", "content": "hello"}], **extra})
                obj._handle_chat()
                self.assertEqual(obj.results[0][0], 400)
        self.native.assert_not_called()

    def test_buffered_sse_uses_same_counts_and_actual_finish(self):
        value = state()
        obj = handler(
            value,
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "max_tokens": 2,
                "stream_options": {"include_usage": True},
            },
        )
        obj._handle_chat()
        events = [
            json.loads(line[6:]) for line in obj.wfile.getvalue().decode().splitlines() if line.startswith("data: {")
        ]
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "length")
        self.assertEqual(events[-1].get("usage"), {"prompt_tokens": 17, "completion_tokens": 2, "total_tokens": 19})
        self.assertEqual(value.total_tokens, 2)
        self.assertIn(
            ("X-MLX-Generation", "buffered; chunks are not live model tokens"),
            [call.args for call in obj.send_header.call_args_list],
        )

    def test_batch_decode_records_eos_and_budget_separately(self):
        for token, max_tokens, expected in ((7, 1, "stop"), (4, 1, "length")):
            engine = object.__new__(batching.ContinuousBatchingEngine)
            engine.tokenizer = SimpleNamespace(eos_token_id=7)
            engine._padded_forward = MagicMock()
            engine._sample = lambda *args: SimpleNamespace(item=lambda: token)
            engine._finish_request = MagicMock()
            engine._total_tokens_generated = 0
            request = batching.InferenceRequest("one", [1, 2], max_tokens=max_tokens)
            engine._decode_step(request)
            self.assertEqual(getattr(request, "finish_reason", None), expected)
            self.assertEqual(request.generated_tokens, [token])

    def test_batch_http_uses_actual_finish_reason_and_rejects_filters(self):
        value = state()
        value.batching = True
        value.engine = MagicMock()
        request = batching.InferenceRequest("one", [1, 2], max_tokens=2)
        request.status = batching.RequestStatus.COMPLETED
        request.generated_tokens = [3, 4]
        request.finish_reason = "length"
        value.engine.submit.return_value = request
        value.engine.wait_for_completion.return_value = request
        value.tokenizer.decode.return_value = "result"
        obj = handler(value, {"messages": [{"role": "user", "content": "hello"}]})
        with patch.dict(sys.modules, {"mlx_flash_compress.continuous_batching": batching}):
            obj._handle_chat()
        self.assertEqual(obj.results[0][1]["choices"][0]["finish_reason"], "length")
        obj = handler(value, {"messages": [{"role": "user", "content": "hello"}], "top_p": 0.9})
        obj._handle_chat()
        self.assertEqual(obj.results[0][0], 400)

    def test_batch_stream_counts_once_and_reports_native_stop(self):
        value = state()
        value.batching = True
        value.engine = MagicMock()
        request = batching.InferenceRequest("one", [1, 2], max_tokens=2)
        request.status = batching.RequestStatus.COMPLETED
        request.generated_tokens = [3, 4]
        request.finish_reason = "length"
        value.engine.submit.return_value = request
        value.engine.stream_tokens.return_value = iter([3, 4])
        value.tokenizer.decode.return_value = "x"
        obj = handler(
            value,
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        with patch.dict(sys.modules, {"mlx_flash_compress.continuous_batching": batching}):
            obj._handle_chat()
        self.assertEqual(value.total_requests, 1)
        self.assertEqual(value.total_tokens, 2)
        events = [
            json.loads(line[6:]) for line in obj.wfile.getvalue().decode().splitlines() if line.startswith("data: {")
        ]
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "length")
        self.assertEqual(events[-1]["usage"]["total_tokens"], 4)

    def test_speculative_engine_rejects_unimplemented_sampling(self):
        value = state()
        value.spec_engine = MagicMock()
        obj = handler(value, {"messages": [{"role": "user", "content": "hello"}], "temperature": 0.2})
        obj._handle_chat()
        self.assertEqual(obj.results[0][0], 400)
        value.spec_engine.generate.assert_not_called()

    def test_buffered_sse_preserves_cross_origin_browser_access(self):
        obj = handler(state(), {"messages": [{"role": "user", "content": "fixture"}], "stream": True})
        obj._handle_chat()
        self.assertIn(("Access-Control-Allow-Origin", "*"), [call.args for call in obj.send_header.call_args_list])

    def test_speculative_eos_inside_accepted_block_reports_stop(self):
        value = state()
        value.spec_engine = MagicMock()
        value.tokenizer.eos_token_ids = [7]
        value.tokenizer.eos_token_id = 7
        serve.mx.array.return_value = [123]
        value.spec_engine.generate.return_value = SimpleNamespace(tolist=lambda: [123, 8, 7, 9])
        for budget in (3, 5):
            with self.subTest(budget=budget):
                result = value.generate([{"role": "user", "content": "fixture"}], budget, 0)
                self.assertEqual(result["finish_reason"], "stop")
                self.assertEqual(result["tokens"], 3)
                self.assertIs(result["native_generation_metadata"], False)


if __name__ == "__main__":
    unittest.main()
