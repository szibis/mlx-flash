"""Model-free lifecycle tests; these never claim real inference coverage."""

import json
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import qwen_metal_smoke as smoke


def contend(path, queue):
    try:
        with smoke.machine_lock(path, timeout=0.15):
            queue.put("acquired")
    except RuntimeError:
        queue.put("timeout")


class LifecycleTests(unittest.TestCase):
    def test_shared_lock_blocks_another_process_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "metal.lock")
            context = multiprocessing.get_context("spawn")
            queue = context.Queue()
            with smoke.machine_lock(path):
                child = context.Process(target=contend, args=(path, queue))
                child.start()
                child.join(timeout=5)
                self.assertFalse(child.is_alive())
                self.assertEqual(queue.get(timeout=1), "timeout")
            with smoke.machine_lock(path, timeout=0.1):
                self.assertTrue(Path(path).exists())
            queue.close()

    def test_readiness_reports_server_death(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait(timeout=5)
        with self.assertRaisesRegex(RuntimeError, "exited"):
            smoke.wait_ready(process, "http://127.0.0.1:1/health", timeout=0.2)

    def test_readiness_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            with smoke.owned_server(
                [sys.executable, "-c", "import time; time.sleep(60)"], Path(directory) / "server.log", os.environ.copy()
            ) as process:
                with patch.object(smoke, "request", side_effect=OSError("not listening")):
                    started = time.monotonic()
                    with self.assertRaisesRegex(RuntimeError, "Timed out"):
                        smoke.wait_ready(process, "unused", timeout=0.1)
                    self.assertLess(time.monotonic() - started, 1)
            self.assertIsNotNone(process.poll())

    def test_http_request_timeout_is_bounded(self):
        class StalledHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                time.sleep(1)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), StalledHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                smoke.request(f"http://127.0.0.1:{server.server_port}/health", timeout=0.1)
            self.assertLess(time.monotonic() - started, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_cleanup_reaps_owned_server_even_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                with smoke.owned_server(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    Path(directory) / "server.log",
                    os.environ.copy(),
                ) as process:
                    raise RuntimeError("generation failed")
            self.assertIsNotNone(process.poll())
            with self.assertRaises(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)

    def test_cleanup_stops_descendant_after_leader_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "child.pid"
            ready = Path(directory) / "ready"
            child_code = (
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"open({str(ready)!r},'w').close(); time.sleep(60)"
            )
            code = (
                f"import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                "open(sys.argv[1],'w').write(str(p.pid)); "
                f"\nwhile not __import__('os').path.exists({str(ready)!r}): time.sleep(0.01)"
            )
            with smoke.owned_server(
                [sys.executable, "-c", code, str(pid_path)], Path(directory) / "server.log", os.environ.copy()
            ) as process:
                process.wait(timeout=5)
                child_pid = int(pid_path.read_text())
            # macOS reaps orphaned children asynchronously; wait for signal delivery.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                # Linux CI can retain orphan zombies; they have stopped executing.
                status = Path(f"/proc/{child_pid}/stat")
                if status.exists() and status.read_text().split()[2] == "Z":
                    break
                time.sleep(0.05)
            else:
                self.fail("Owned descendant survived process-group cleanup")

    def test_port_preflight_rejects_live_listener(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen()
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                smoke.preflight_ports([server.getsockname()[1]])

    def test_port_preflight_accepts_closed_connection_time_wait(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
            server.listen()
            with socket.create_connection(("127.0.0.1", port)) as client:
                accepted, _ = server.accept()
                with accepted:
                    accepted.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
                    client.shutdown(socket.SHUT_WR)
                    self.assertEqual(accepted.recv(1), b"")
        smoke.preflight_ports([port])

    def test_exact_marker_requires_completed_thinking(self):
        smoke.final_text("reasoning</think>" + smoke.MARKER, True)
        with self.assertRaisesRegex(RuntimeError, "reasoning"):
            smoke.final_text(smoke.MARKER, True)
        with self.assertRaisesRegex(RuntimeError, "exact"):
            smoke.final_text(smoke.MARKER + " extra", False)

    def test_model_preflight_rejects_missing_index_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            for name in ("config.json", "tokenizer.json"):
                (model / name).write_text("{}")
            (model / "tokenizer_config.json").write_text('{"chat_template":"enable_thinking"}')
            (model / "weights.safetensors").write_bytes(b"weights")
            (model / "model.safetensors.index.json").write_text('{"weight_map":{"a":"missing.safetensors"}}')
            with self.assertRaisesRegex(RuntimeError, "shards"):
                smoke.validate_model(str(model))


class NativeFamilyTests(unittest.TestCase):
    def test_completed_gemma_and_think_sections_require_exact_final(self):
        for text in ("<|channel>thought\nprivate<channel|>" + smoke.MARKER, "private<channel|>" + smoke.MARKER):
            smoke.final_text(text, True, "gemma")
        for text in (
            "<|channel>analysis\nprivate<channel|>" + smoke.MARKER,
            "<|channel>thought\nunfinished",
            "<think>unfinished",
            "<think>private</think><think>" + smoke.MARKER,
            "<think broken</think>" + smoke.MARKER,
            "<think>private<channel|>" + smoke.MARKER,
            "<|channel>thought\n<|channel>analysis<channel|>" + smoke.MARKER,
            "<|channel>thought\nprivate<channel|>" + smoke.MARKER + "<|channel",
        ):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                smoke.final_text(text, False)

    def test_model_type_and_template_must_agree(self):
        for family, template, valid in (
            ("qwen3_5", "enable_thinking <think> </think>", True),
            ("gemma4", "enable_thinking <|channel>thought <channel|>", True),
            ("lfm2_moe", "<think> </think>", True),
            ("lfm2_moe", "enable_thinking <think> </think>", False),
            ("gemma4", "enable_thinking <think> </think>", False),
            ("unknown", "enable_thinking", False),
        ):
            with self.subTest(family=family, valid=valid), tempfile.TemporaryDirectory() as directory:
                model = Path(directory)
                (model / "config.json").write_text(json.dumps({"model_type": family}))
                (model / "tokenizer.json").write_text("{}")
                (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
                (model / "weights.safetensors").write_bytes(b"weights")
                if valid:
                    self.assertEqual(smoke.validate_model(str(model)), model)
                else:
                    with self.assertRaises(RuntimeError):
                        smoke.validate_model(str(model))

    def test_native_lfm_default_omits_toggle_and_sampler_overrides(self):
        caps = {
            "model_family": "lfm2_moe",
            "thinking_control": False,
            "reasoning_format": "think",
            "chat_template_kwargs": [],
        }
        result = {
            "choices": [{"finish_reason": "stop", "message": {"content": "<think>done</think>" + smoke.MARKER}}],
            "usage": {"completion_tokens": 15},
        }
        with patch.object(smoke, "request", return_value=result) as request:
            smoke.generate_chat(None, caps)
            payload = request.call_args.args[1]
            self.assertNotIn("chat_template_kwargs", payload)
            self.assertNotIn("repetition_penalty", payload)
            self.assertNotIn("top_p", payload)
        with self.assertRaises(RuntimeError):
            smoke.generate_chat(True, caps)

    def test_native_controlled_profiles_forward_boolean_toggle(self):
        for family, fmt in (("qwen3_5", "think"), ("gemma4", "gemma")):
            caps = {
                "model_family": family,
                "thinking_control": True,
                "reasoning_format": fmt,
                "chat_template_kwargs": ["enable_thinking"],
            }
            for thinking in (False, True):
                content = (
                    ("private</think>" if fmt == "think" else "<|channel>thought\nprivate<channel|>") + smoke.MARKER
                    if thinking
                    else smoke.MARKER
                )
                result = {
                    "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                    "usage": {"completion_tokens": 10},
                }
                with patch.object(smoke, "request", return_value=result) as request:
                    self.assertEqual(smoke.generate_chat(thinking, caps), {"completion_tokens": 10})
                    self.assertEqual(request.call_args.args[1]["chat_template_kwargs"], {"enable_thinking": thinking})

    def test_sonnet_tool_bridge_rejects_wrong_marker_or_multiple_calls(self):
        call = {"type": "tool_use", "name": "record_marker", "input": {"marker": smoke.MARKER}}
        with patch.object(
            smoke, "request", return_value={"stop_reason": "tool_use", "content": [call], "usage": {"input_tokens": 10}}
        ):
            self.assertEqual(smoke.generate_role("sonnet", tool=True), {"input_tokens": 10})
        for content in ([dict(call, input={"marker": "WRONG"})], [call, call]):
            with (
                patch.object(smoke, "request", return_value={"stop_reason": "tool_use", "content": content}),
                self.assertRaises(RuntimeError),
            ):
                smoke.generate_role("sonnet", tool=True)

    def test_health_profiles_accept_legacy_qwen_and_coherent_native_families(self):
        self.assertEqual(smoke.thinking_profiles({"chat_template_kwargs": ["enable_thinking"]}), (False, True))
        for family, fmt, control, profiles in (
            ("qwen3_5", "think", True, (False, True)),
            ("gemma4", "gemma", True, (False, True)),
            ("lfm2_moe", "think", False, (None,)),
        ):
            caps = {
                "model_family": family,
                "reasoning_format": fmt,
                "thinking_control": control,
                "chat_template_kwargs": ["enable_thinking"] if control else [],
            }
            self.assertEqual(smoke.thinking_profiles(caps), profiles)
        for caps in (
            {
                "model_family": "lfm2_moe",
                "reasoning_format": "think",
                "thinking_control": True,
                "chat_template_kwargs": ["enable_thinking"],
            },
            {
                "model_family": "gemma4",
                "reasoning_format": "think",
                "thinking_control": True,
                "chat_template_kwargs": ["enable_thinking"],
            },
            {},
        ):
            with self.assertRaises(RuntimeError):
                smoke.thinking_profiles(caps)


if __name__ == "__main__":
    unittest.main()
