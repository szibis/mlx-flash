"""HTTP vision contracts run without native model inference."""

import http.client
import importlib.util
import json
import socket
import threading
import unittest
import urllib.error
import urllib.request

from tests.test_serve_openai_compat import ChatHandler, FakeState, ThreadedHTTPServer, _handler
from tests.test_vision_runtime import messages

APIRequestError = ChatHandler._handle_chat.__globals__["APIRequestError"]


class VisionState(FakeState):
    vision = True
    vision_runtime = type("Runtime", (), {"ready": True})()
    vision_admission = threading.BoundedSemaphore(1)

    def capabilities(self):
        return {"thinking_control": True}


@unittest.skipUnless(importlib.util.find_spec("PIL"), "optional vision dependency Pillow unavailable")
class VisionHandlerTests(unittest.TestCase):
    def test_image_request_reaches_vision_runtime_and_returns_openai_shape(self):
        state = VisionState([{"output": "red", "tokens": 1, "prompt_tokens": 80, "finish_reason": "stop"}])
        handler = _handler(
            state,
            {"messages": messages(), "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"},
        )
        handler._handle_chat()
        status, response = handler.responses[0]
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["message"]["content"], "red")
        self.assertEqual(response["usage"]["total_tokens"], 81)

    def test_vision_stream_and_cache_requests_are_rejected_before_inference(self):
        state = VisionState([])
        for options in [
            {"stream": True},
            {"cache_scope": "tenant"},
            {"chat_template_kwargs": {"enable_thinking": True}},
        ]:
            handler = _handler(state, {"messages": messages(), **options})
            handler._handle_chat()
            self.assertEqual(handler.responses[0][0], 400)
        self.assertEqual(state.calls, [])


class VisionBodyTimeoutTests(unittest.TestCase):
    def test_read_timeout_returns_safe_408_without_model_work(self):
        state = VisionState([])
        handler = _handler(state, {"messages": [{"role": "user", "content": "hello"}]})

        class TimedOutBody:
            def read(self, length):
                raise TimeoutError("private socket details")

        handler.rfile = TimedOutBody()
        handler._handle_chat()
        self.assertEqual(handler.responses[0][0], 408)
        self.assertEqual(handler.responses[0][1]["error"]["code"], "request_timeout")
        self.assertNotIn("private socket details", json.dumps(handler.responses))
        self.assertEqual(state.calls, [])

    def test_stalled_body_times_out_and_releases_the_only_admission_slot(self):
        state = VisionState([{"output": "red", "tokens": 1, "prompt_tokens": 80, "finish_reason": "stop"}])
        state.vision_admission = threading.BoundedSemaphore(1)
        state.request_timeout = 0.25
        admitted = threading.Event()

        class Handler(ChatHandler):
            server_state = state

            def _dispatch_post(self):
                admitted.set()
                return super()._dispatch_post()

        try:
            server = ThreadedHTTPServer(("127.0.0.1", 0), Handler)
        except OSError as error:
            self.skipTest(f"sandbox does not allow local HTTP sockets: {error}")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        stalled = socket.create_connection(server.server_address, timeout=2)
        base = f"http://127.0.0.1:{server.server_port}"
        payload = json.dumps({"messages": [{"role": "user", "content": "Identify red"}]}).encode()
        try:
            stalled.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1000\r\n\r\n{")
            self.assertTrue(admitted.wait(1), "stalled request did not acquire admission")
            req = urllib.request.Request(
                base + "/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}
            )
            with self.assertRaises(urllib.error.HTTPError) as busy:
                urllib.request.urlopen(req, timeout=2)
            self.assertEqual(busy.exception.code, 503)
            response = http.client.HTTPResponse(stalled)
            response.begin()
            self.assertEqual(response.status, 408)
            self.assertEqual(json.loads(response.read())["error"]["code"], "request_timeout")
            self.assertEqual(state.calls, [])
            with urllib.request.urlopen(req, timeout=2) as following:
                self.assertEqual(following.status, 200)
                self.assertEqual(json.loads(following.read())["choices"][0]["message"]["content"], "red")
            self.assertEqual(len(state.calls), 1)
        finally:
            stalled.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "optional vision dependency Pillow unavailable")
    def test_bad_vision_output_has_explicit_502(self):
        state = VisionState([])

        def fail(*args, **kwargs):
            raise APIRequestError("unresolved control tokens", status=502, code="invalid_model_output")

        state.generate = fail
        handler = _handler(state, {"messages": messages()})
        handler._handle_chat()
        self.assertEqual(handler.responses[0][0], 502)
        self.assertEqual(handler.responses[0][1]["error"]["code"], "invalid_model_output")

    def test_capabilities_remain_available_for_verified_nonresident_model(self):
        handler = _handler(VisionState([]), {})
        handler.path = "/v1/capabilities"
        handler.do_GET()
        self.assertTrue(handler.responses[0][1]["vision"]["supported"])

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "optional vision dependency Pillow unavailable")
    def test_busy_runtime_rejects_work_without_reading_or_generating(self):
        state = VisionState([])
        handler = _handler(state, {"messages": messages()})
        handler.path = "/v1/chat/completions"
        state.vision_admission.acquire()
        try:
            handler.do_POST()
        finally:
            state.vision_admission.release()
        self.assertEqual(handler.responses[0][0], 503)
        self.assertEqual(handler.responses[0][1]["error"]["code"], "runtime_busy")
        self.assertEqual(handler.rfile.tell(), 0)
        self.assertEqual(state.calls, [])
