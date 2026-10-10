"""Native generation boundaries, accounting and cache invalidation."""

from types import SimpleNamespace
from unittest.mock import patch

from tests.test_generation_measurements import handler as make_handler
from tests.test_generation_measurements import serve
from tests.test_generation_measurements import state as fixture_state

InferenceState = serve.InferenceState
from tests.test_prompt_reuse import State


class Tokenizer:
    bos_token = None

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


def native_generator(model, tokenizer, *, prompt, prompt_cache=None, prompt_progress_callback=None, **kwargs):
    # Native MLX-LM reports all-but-last prefill before the decoding pipeline.
    if prompt_cache is not None:
        prompt_cache[0].values += list(prompt[:-1])
        prompt_progress_callback(len(prompt) - 1, len(prompt))
        prompt_cache[0].values.append(prompt[-1])
        prompt_cache[0].values.append(99)  # lookahead is NOT safe to snapshot
        prompt_progress_callback(len(prompt), len(prompt))
    yield SimpleNamespace(
        text="ok",
        token=99,
        generation_tokens=1,
        prompt_tokens=len(prompt),
        finish_reason="stop",
        generation_tps=10,
        prompt_tps=20,
        peak_memory=0.1,
    )


def test_native_prefill_snapshots_exclude_decode_lookahead_and_report_full_input():
    memory = SimpleNamespace(pressure_level="normal", available_gb=20)
    with (
        patch.object(serve, "detect_hardware"),
        patch.object(serve, "MemoryManager"),
        patch.object(serve, "HardwareTelemetry"),
        patch.object(serve, "get_memory_state", return_value=memory),
        patch.object(serve.mx, "synchronize"),
        patch.object(serve, "stream_generate", side_effect=native_generator),
        patch.object(serve, "make_prompt_cache", return_value=[State([])]),
    ):
        state = InferenceState("local-model")
        state.model = object()
        state.tokenizer = Tokenizer()
        cold = state.generate([{"role": "user", "content": "abcd"}], temperature=0)
        warm = state.generate([{"role": "user", "content": "abcd"}], temperature=0)
        assert cold.get("cached_prompt_tokens") == 0
        assert warm.get("cached_prompt_tokens") == 3
        assert warm["prompt_tokens"] == 4
        assert warm["processed_prompt_tokens"] == 1
        assert state.prompt_reuse.lookup(("local-model", (), "default"), [0, 1, 2, 3])[0][0].values == [0, 1, 2]


def test_configured_family_does_not_advertise_an_ignored_thinking_toggle(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"lfm2_moe"}')
    with (
        patch.object(serve, "detect_hardware"),
        patch.object(serve, "MemoryManager"),
        patch.object(serve, "HardwareTelemetry"),
    ):
        state = InferenceState(str(tmp_path))
        assert state.capabilities()["model_family"] == "lfm2_moe"
        assert state.capabilities()["thinking_control"] is False
        assert state.capabilities()["chat_template_kwargs"] == []


def test_http_cache_scope_reaches_generation_and_rejects_nonstring_scope():
    import json

    ChatHandler = serve.ChatHandler
    state = fixture_state()
    state.generate = lambda *args, **kwargs: {"error": "scope:" + str(kwargs.get("cache_scope"))}
    body = {"messages": [{"role": "user", "content": "hello"}], "cache_scope": "client-session"}
    handler = make_handler(state, body)
    ChatHandler._handle_chat_data(handler, body)
    assert handler.results[-1][1] == {"error": {"message": "scope:client-session", "type": "server_error"}}
    body["cache_scope"] = {"bad": "scope"}
    ChatHandler._handle_chat_data(handler, body)
    assert handler.results[-1][0] == 400


def test_failed_generation_does_not_report_unprocessed_tokens_as_savings():
    import pytest

    value = fixture_state()
    memory = SimpleNamespace(pressure_level="normal", available_gb=20)
    with (
        patch.object(serve, "get_memory_state", return_value=memory),
        patch.object(serve, "mx"),
        patch.object(serve, "stream_generate", return_value=iter([])),
    ):
        with pytest.raises(RuntimeError):
            value.generate([{"role": "user", "content": "hello"}])
    assert value.prompt_reuse.stats()["processed_tokens"] == 0
    assert value.prompt_reuse.stats()["reused_tokens"] == 0


def test_oversized_native_prefill_is_rejected_before_snapshot_copy():
    class Oversized(State):
        def __deepcopy__(self, memo):
            raise AssertionError("oversized native state was copied")

    value = fixture_state()
    value.tokenizer = Tokenizer()
    memory = SimpleNamespace(pressure_level="normal", available_gb=20)
    with (
        patch.object(serve, "get_memory_state", return_value=memory),
        patch.object(serve, "mx"),
        patch.object(serve, "make_prompt_cache", return_value=[Oversized([], nbytes=1024**3)]),
        patch.object(serve, "stream_generate", side_effect=native_generator),
    ):
        result = value.generate([{"role": "user", "content": "abcd"}])
    assert result["tokens"] == 1
    assert value.prompt_reuse.stats()["entries"] == 0


def test_cached_sampling_receives_the_full_prompt_history():
    seen = []

    def processor(tokens, logits):
        seen.append(list(tokens))
        return logits

    def generator(model, tokenizer, **kwargs):
        for fn in kwargs.get("logits_processors") or []:
            fn(list(kwargs["prompt"]), None)
        yield from native_generator(model, tokenizer, **kwargs)

    value = fixture_state()
    value.tokenizer = Tokenizer()
    value._format_messages = lambda messages: "abcd"
    memory = SimpleNamespace(pressure_level="normal", available_gb=20)
    with (
        patch.object(serve, "get_memory_state", return_value=memory),
        patch.object(serve, "mx"),
        patch.object(serve, "make_prompt_cache", side_effect=lambda model: [State([])]),
        patch.object(serve, "make_logits_processors", create=True, return_value=[processor]),
        patch.object(serve, "stream_generate", side_effect=generator),
    ):
        serve.mx.array.side_effect = list
        serve.mx.concatenate.side_effect = lambda parts: sum(parts, [])
        for _ in range(2):
            value.generate([{"role": "user", "content": "abcd"}], repetition_penalty=1.05)
    assert seen == [[0, 1, 2, 3], [0, 1, 2, 3]]


def test_http_repetition_penalty_is_validated_and_forwarded():
    state = fixture_state()
    state.generate = lambda *args, **kwargs: {"error": "penalty:" + str(kwargs.get("repetition_penalty"))}
    for value, status in [(1.05, 503), (-1, 400), (False, 400), (float("nan"), 400)]:
        body = {"messages": [{"role": "user", "content": "hello"}], "repetition_penalty": value}
        handler = make_handler(state, body)
        serve.ChatHandler._handle_chat_data(handler, body)
        assert handler.results[-1][0] == status
        if status == 503:
            assert handler.results[-1][1] == {"error": {"message": "penalty:1.05", "type": "server_error"}}
