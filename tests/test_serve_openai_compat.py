"""HTTP tests for OpenAI-compatible capability and structured-output behavior."""

import io
import json
import sys
import threading
import types
import unittest
import urllib.error
import urllib.request


def _import_serve_without_mlx():
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError:
        mlx = types.ModuleType("mlx")
        mlx.__path__ = []
        core = types.ModuleType("mlx.core")
        mlx.core = core
        sys.modules.setdefault("mlx", mlx)
        sys.modules.setdefault("mlx.core", core)
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.generate = lambda *args, **kwargs: None
        mlx_lm.load = lambda *args, **kwargs: None
        sys.modules.setdefault("mlx_lm", mlx_lm)
    from mlx_flash_compress.serve import ChatHandler, ThreadedHTTPServer

    return ChatHandler, ThreadedHTTPServer


ChatHandler, ThreadedHTTPServer = _import_serve_without_mlx()


class FakeState:
    def __init__(self, outputs):
        self.chat_lock = threading.RLock()
        self.batching = False
        self.engine = None
        self.speculative = "none"
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, messages, max_tokens, temperature):
        self.calls.append((messages, max_tokens, temperature))
        return next(self.outputs)


def _handler(state, payload, *, raw_body=None, content_length=None):
    body = raw_body if raw_body is not None else json.dumps(payload).encode()
    handler = object.__new__(ChatHandler)
    handler.headers = {"Content-Length": str(len(body) if content_length is None else content_length)}
    handler.rfile = io.BytesIO(body)
    handler.server_state = state
    handler.responses = []
    handler._send_json = lambda data, status=200: handler.responses.append((status, data))
    return handler


class OpenAIChatContractTests(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        self.response_format = {
            "type": "json_schema",
            "json_schema": {"name": "probe", "strict": True, "schema": self.schema},
        }
        self.payload = {"messages": [{"role": "user", "content": "Confirm"}], "response_format": self.response_format}

    def test_capability_endpoint_returns_machine_readable_actual_support(self):
        handler = object.__new__(ChatHandler)
        handler.path = "/v1/capabilities"
        handler.responses = []
        handler._send_json = lambda data, status=200: handler.responses.append((status, data))

        handler.do_GET()

        status, capabilities = handler.responses[0]
        self.assertEqual(status, 200)
        self.assertTrue(capabilities["structured_outputs"]["supported"])
        self.assertFalse(capabilities["vision"]["supported"])

    def test_legacy_completions_endpoint_returns_openai_shape(self):
        state = FakeState([{"output": "Hello", "prompt_tokens": 2, "tokens": 1, "tok_per_s": 2, "memory_pressure": "normal"}])
        handler = _handler(state, {"prompt": "Say hello", "max_tokens": 32})

        handler._handle_legacy_completion()

        status, response = handler.responses[0]
        self.assertEqual(status, 200)
        self.assertEqual(response["object"], "text_completion")
        self.assertEqual(response["choices"][0]["text"], "Hello")
        self.assertEqual(response["usage"]["total_tokens"], 3)

    def test_responses_endpoint_accepts_text_input_and_instructions(self):
        state = FakeState([{"output": "Witaj", "prompt_tokens": 4, "tokens": 2, "tok_per_s": 2, "memory_pressure": "normal"}])
        handler = _handler(state, {"input": "Odpowiedz po polsku", "instructions": "Bądź zwięzły", "max_output_tokens": 24})

        handler._handle_responses()

        status, response = handler.responses[0]
        self.assertEqual(status, 200)
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["output"][0]["content"][0]["text"], "Witaj")
        self.assertEqual(state.calls[0][0][0], {"role": "system", "content": "Bądź zwięzły"})
        self.assertEqual(response["usage"]["total_tokens"], 6)

    def test_responses_endpoint_rejects_stream_and_image_input_without_inference(self):
        state = FakeState([])
        for payload in (
            {"input": "hello", "stream": True},
            {"input": [{"role": "user", "content": [{"type": "input_image"}]}]},
        ):
            handler = _handler(state, payload)
            handler._handle_responses()
            self.assertEqual(handler.responses[0][0], 400)
        self.assertEqual(state.calls, [])


class OpenAIHTTPIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.original_state = ChatHandler.server_state
        ChatHandler.server_state = FakeState(
            [
                {"output": '{"ok":true}', "prompt_tokens": 3, "tokens": 4, "tok_per_s": 2, "memory_pressure": "normal"},
                {"output": "Cześć", "prompt_tokens": 2, "tokens": 2, "tok_per_s": 2, "memory_pressure": "normal"},
            ]
        )
        try:
            self.server = ThreadedHTTPServer(("127.0.0.1", 0), ChatHandler)
        except OSError as error:
            ChatHandler.server_state = self.original_state
            self.skipTest(f"sandbox does not allow local HTTP sockets: {error}")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        ChatHandler.server_state = self.original_state

    def test_real_http_capabilities_and_structured_and_responses_requests(self):
        with urllib.request.urlopen(self.base + "/v1/capabilities", timeout=2) as response:
            capabilities = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertFalse(capabilities["vision"]["supported"])

        structured = {
            "model": "local",
            "messages": [{"role": "user", "content": "Say okay"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "confirmation",
                    "strict": True,
                    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False},
                },
            },
        }
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(structured).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as response:
            output = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(output["choices"][0]["message"]["content"]), {"ok": True})

        req = urllib.request.Request(self.base + "/v1/responses", data=b'{"input":"Say hi"}', headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as response:
            output = json.loads(response.read())
            self.assertEqual(output["object"], "response")
            self.assertEqual(output["output"][0]["content"][0]["text"], "Cześć")

class StructuredChatHandlerTests(unittest.TestCase):
    def setUp(self):
        self.schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
        self.response_format = {"type": "json_schema", "json_schema": {"name": "probe", "strict": True, "schema": self.schema}}
        self.payload = {"messages": [{"role": "user", "content": "Confirm"}], "response_format": self.response_format}

    def test_existing_chat_request_remains_compatible_and_accepts_max_completion_tokens(self):
        state = FakeState([{"output": "Confirmed", "tokens": 2, "tok_per_s": 1, "memory_pressure": "normal"}])
        payload = {"model": "local", "messages": [{"role": "user", "content": "Confirm"}], "max_completion_tokens": 64, "reasoning_effort": "none"}
        handler = _handler(state, payload)

        handler._handle_chat()

        self.assertEqual(handler.responses[0][0], 200)
        self.assertEqual(handler.responses[0][1]["choices"][0]["message"]["content"], "Confirmed")
        self.assertEqual(state.calls[0][1], 64)

    def test_invalid_schema_is_rejected_before_inference(self):
        state = FakeState([])
        payload = json.loads(json.dumps(self.payload))
        payload["response_format"]["json_schema"]["schema"]["type"] = "invalid-test-schema"
        handler = _handler(state, payload)

        handler._handle_chat()

        self.assertEqual(handler.responses[0][0], 400)
        self.assertEqual(state.calls, [])

    def test_strict_json_schema_response_is_validated_before_success(self):
        state = FakeState([{"output": '{"ok":true}', "tokens": 4, "tok_per_s": 2, "memory_pressure": "normal"}])
        handler = _handler(state, self.payload)

        handler._handle_chat()

        status, response = handler.responses[0]
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["message"]["content"], '{"ok":true}')
        self.assertIn("JSON Schema", state.calls[0][0][0]["content"])

    def test_schema_mismatch_gets_one_bounded_repair_then_success(self):
        state = FakeState(
            [
                {"output": '{"ok":"yes"}', "tokens": 4, "tok_per_s": 2, "memory_pressure": "normal"},
                {"output": '{"ok":false}', "tokens": 4, "tok_per_s": 2, "memory_pressure": "normal"},
            ]
        )
        handler = _handler(state, self.payload)

        handler._handle_chat()

        self.assertEqual(handler.responses[0][0], 200)
        self.assertEqual(len(state.calls), 2)
        self.assertEqual(state.calls[1][2], 0)
        usage = handler.responses[0][1]["usage"]
        self.assertEqual(usage["completion_tokens"], 8)

    def test_schema_mismatch_after_repair_never_returns_success(self):
        state = FakeState(
            [
                {"output": "not JSON", "tokens": 4, "tok_per_s": 2, "memory_pressure": "normal"},
                {"output": '{"ok":"still wrong"}', "tokens": 4, "tok_per_s": 2, "memory_pressure": "normal"},
            ]
        )
        handler = _handler(state, self.payload)

        handler._handle_chat()

        self.assertEqual(handler.responses[0][0], 502)
        self.assertEqual(handler.responses[0][1]["error"]["code"], "invalid_model_output")

    def test_oversized_request_is_rejected_before_read_or_generation(self):
        state = FakeState([])
        handler = _handler(state, {}, raw_body=b"", content_length=ChatHandler.max_request_bytes + 1)

        handler._handle_chat()

        self.assertEqual(handler.responses[0][0], 413)
        self.assertEqual(state.calls, [])

    def test_json_object_mode_rejects_non_object_json(self):
        state = FakeState(
            [
                {"output": '[1,2]', "tokens": 4, "tok_per_s": 2, "memory_pressure": "normal"},
                {"output": '[1,2]', "tokens": 4, "tok_per_s": 2, "memory_pressure": "normal"},
            ]
        )
        payload = {"messages": [{"role": "user", "content": "Return JSON"}], "response_format": {"type": "json_object"}}
        handler = _handler(state, payload)

        handler._handle_chat()

        self.assertEqual(handler.responses[0][0], 502)
        self.assertEqual(len(state.calls), 2)


class CORSPolicyTests(unittest.TestCase):
    def test_cors_preflight_allows_only_same_localhost_port(self):
        handler = object.__new__(ChatHandler)
        handler.server = types.SimpleNamespace(server_port=8080)
        calls = []
        handler.send_response = lambda status: calls.append(("status", status))
        handler.send_header = lambda name, value: calls.append((name, value))
        handler.end_headers = lambda: calls.append(("end", None))

        handler.headers = {"Origin": "http://localhost:8080"}
        handler.do_OPTIONS()
        self.assertIn(("status", 204), calls)
        self.assertIn(("Access-Control-Allow-Origin", "http://localhost:8080"), calls)

        calls.clear()
        handler.headers = {"Origin": "https://attacker.example"}
        handler.do_OPTIONS()
        self.assertIn(("status", 403), calls)
        self.assertFalse(any(name == "Access-Control-Allow-Origin" for name, _ in calls))

    def test_json_responses_never_emit_wildcard_cors(self):
        handler = object.__new__(ChatHandler)
        handler.server = types.SimpleNamespace(server_port=8080)
        handler.headers = {"Origin": "https://attacker.example"}
        handler.wfile = io.BytesIO()
        calls = []
        handler.send_response = lambda status: calls.append(("status", status))
        handler.send_header = lambda name, value: calls.append((name, value))
        handler.end_headers = lambda: calls.append(("end", None))

        handler._send_json({"ok": True})

        self.assertFalse(any(name == "Access-Control-Allow-Origin" and value == "*" for name, value in calls))
        self.assertFalse(any(name == "Access-Control-Allow-Origin" for name, _ in calls))


if __name__ == "__main__":
    unittest.main()
