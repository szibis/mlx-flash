"""Tests for the inference server (no model loading)."""

import io
import json
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

# Pre-import the module so mock.patch can resolve dotted paths.
# The serve module imports mlx which may not be available.
try:
    import mlx_flash_compress.serve

    HAS_SERVE = True
except (ImportError, ModuleNotFoundError):
    HAS_SERVE = False

pytestmark = pytest.mark.skipif(not HAS_SERVE, reason="serve module requires mlx")


# ---------------------------------------------------------------------------
# Helpers: build fake objects that stand in for hardware / memory state
# ---------------------------------------------------------------------------


def _make_fake_hw():
    hw = MagicMock()
    hw.chip = "Apple M2 Pro"
    hw.total_ram_gb = 32.0
    return hw


def _make_fake_mem(pressure="normal"):
    mem = MagicMock()
    mem.total_gb = 32.0
    mem.free_gb = 8.0
    mem.inactive_gb = 4.0
    mem.available_gb = 10.0
    mem.pressure_level = pressure
    mem.swap_used_gb = 0.0
    return mem


def _make_fake_mem_mgr():
    mgr = MagicMock()
    mgr.get_cache_budget_gb.return_value = 4.0
    mgr.get_optimization_hints.return_value = []
    mgr.auto_release_if_needed.return_value = {"released": False}
    return mgr


# ---------------------------------------------------------------------------
# Patch targets — all resolved at import time of mlx_flash_compress.serve
# ---------------------------------------------------------------------------

HW_PATCH = "mlx_flash_compress.serve.detect_hardware"
MGR_PATCH = "mlx_flash_compress.serve.MemoryManager"
MEM_PATCH = "mlx_flash_compress.serve.get_memory_state"
MX_PATCH = "mlx_flash_compress.serve.mx"
LOAD_PATCH = "mlx_flash_compress.serve.load"
GEN_PATCH = "mlx_flash_compress.serve.generate"


def _make_state(model_name="test-model", pressure="normal"):
    """Create an InferenceState with all external dependencies mocked."""
    fake_mgr = _make_fake_mem_mgr()
    fake_mem = _make_fake_mem(pressure)

    with (
        patch(HW_PATCH, return_value=_make_fake_hw()),
        patch(MGR_PATCH, return_value=fake_mgr),
        patch(MEM_PATCH, return_value=fake_mem),
    ):
        from mlx_flash_compress.serve import InferenceState

        state = InferenceState(model_name)

    # Keep references so tests can inspect them
    state._fake_mgr = fake_mgr
    state._fake_mem = fake_mem
    return state


class TestInferenceState:
    def test_status_returns_valid_structure(self):
        state = _make_state()
        with patch(MEM_PATCH, return_value=_make_fake_mem()):
            status = state.get_status()
        assert "model" in status
        assert "hardware" in status
        assert "memory" in status
        assert "stats" in status
        assert "optimization_hints" in status

    def test_status_memory_has_required_fields(self):
        state = _make_state()
        with patch(MEM_PATCH, return_value=_make_fake_mem()):
            mem = state.get_status()["memory"]
        for field in ["total_gb", "free_gb", "available_gb", "pressure", "cache_budget_gb"]:
            assert field in mem, f"Missing field: {field}"

    def test_status_pressure_is_valid(self):
        state = _make_state()
        with patch(MEM_PATCH, return_value=_make_fake_mem("warning")):
            pressure = state.get_status()["memory"]["pressure"]
        assert pressure in ("normal", "warning", "critical")

    def test_stats_start_at_zero(self):
        state = _make_state()
        with patch(MEM_PATCH, return_value=_make_fake_mem()):
            stats = state.get_status()["stats"]
        assert stats["requests"] == 0
        assert stats["tokens_generated"] == 0

    def test_generate_without_model_loads_model(self):
        state = _make_state()
        # load() raises because "nonexistent-model" can't be found;
        # we let it propagate to confirm load_model() is called.
        with (
            patch(MEM_PATCH, return_value=_make_fake_mem()),
            patch(LOAD_PATCH, side_effect=Exception("model not found")),
            patch(MX_PATCH),
        ):
            with pytest.raises(Exception):
                state.generate([{"role": "user", "content": "hi"}], max_tokens=5)

    def test_format_messages_fallback(self):
        state = _make_state("test")
        state.tokenizer = MagicMock()
        state.tokenizer.apply_chat_template = MagicMock(side_effect=Exception("nope"))
        result = state._format_messages([{"role": "user", "content": "hello"}])
        assert "hello" in result

    def test_optimization_hints_in_status(self):
        state = _make_state("test")
        with patch(MEM_PATCH, return_value=_make_fake_mem()):
            hints = state.get_status()["optimization_hints"]
        assert isinstance(hints, list)

    def test_model_name_in_status(self):
        state = _make_state("my-custom-model")
        with patch(MEM_PATCH, return_value=_make_fake_mem()):
            assert state.get_status()["model"] == "my-custom-model"


class TestChatRequestProfiles:
    @staticmethod
    def handler(state, payload):
        from mlx_flash_compress.serve import ChatHandler

        body = json.dumps(payload).encode()
        handler = object.__new__(ChatHandler)
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.server_state = state
        handler.responses = []
        handler._send_json = lambda data, status=200: handler.responses.append((status, data))
        return handler

    def test_http_profile_changes_actual_prompt(self):
        from mlx_flash_compress.chat_profiles import current_chat_template_kwargs

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"] + (" THINK" if kwargs.get("enable_thinking") else " DIRECT")

        state = _make_state()
        state.tokenizer = Tokenizer()
        state.generate = lambda messages, *args: {
            "output": state._format_messages(messages),
            "tokens": 2,
            "tok_per_s": 1,
            "memory_pressure": "normal",
        }
        for thinking, suffix in ((True, "THINK"), (False, "DIRECT")):
            handler = self.handler(
                state,
                {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "chat_template_kwargs": {"enable_thinking": thinking},
                },
            )
            handler._handle_chat()
            status, response = handler.responses[0]
            assert status == 200
            assert response["choices"][0]["message"]["content"] == "Hello " + suffix
            assert current_chat_template_kwargs() == {}

    def test_http_bad_profile_and_batching_never_generate(self):
        state = _make_state()

        def forbidden(*args):
            pytest.fail("invalid/unsupported profile reached generation")

        state.generate = forbidden
        for options in ({"enable_thinking": "false"}, {"unknown": True}, None):
            handler = self.handler(
                state, {"messages": [{"role": "user", "content": "Hello"}], "chat_template_kwargs": options}
            )
            handler._handle_chat()
            assert handler.responses[0][0] == 400
        state.batching = True
        handler = self.handler(
            state,
            {"messages": [{"role": "user", "content": "Hello"}], "chat_template_kwargs": {"enable_thinking": False}},
        )
        handler._handle_chat()
        assert handler.responses[0][0] == 400

    def test_http_generation_failure_restores_profile(self):
        from mlx_flash_compress.chat_profiles import current_chat_template_kwargs

        state = _make_state()

        def fail(*args):
            assert current_chat_template_kwargs() == {"enable_thinking": True}
            raise RuntimeError("native generation failed")

        state.generate = fail
        handler = self.handler(
            state,
            {"messages": [{"role": "user", "content": "Hello"}], "chat_template_kwargs": {"enable_thinking": True}},
        )
        with pytest.raises(RuntimeError, match="generation failed"):
            handler._handle_chat()
        assert current_chat_template_kwargs() == {}

    def test_model_switch_waits_for_shared_native_state(self):
        state = _make_state("original")
        state.model = object()
        state.tokenizer = object()
        original_tokenizer = state.tokenizer
        entered = threading.Event()
        completed = threading.Event()

        def load():
            entered.set()

        state.load_model = load
        handler = self.handler(state, {"model": "replacement"})

        def switch():
            handler._handle_switch()
            completed.set()

        with patch(MX_PATCH):
            thread = threading.Thread(target=switch)
            try:
                with state.chat_lock:
                    thread.start()
                    assert not entered.wait(0.05)
                    assert state.model_name == "original"
                    assert state.tokenizer is original_tokenizer
            finally:
                thread.join(timeout=2)
        assert completed.is_set()
        assert state.model_name == "replacement"
