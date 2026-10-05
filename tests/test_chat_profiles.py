"""Request profile tests require neither Metal nor model weights."""

import concurrent.futures
import threading
import time

import pytest

from mlx_flash_compress.chat_profiles import (
    chat_template_profile,
    current_chat_template_kwargs,
    format_chat_messages,
    validate_chat_template_kwargs,
)


class TemplateTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        prefix = "|".join(message["content"] for message in messages)
        return prefix + ("<think>" if kwargs.get("enable_thinking", True) else "<think></think>")


def test_profiles_change_the_formatted_prompt_and_restore_defaults():
    tokenizer = TemplateTokenizer()
    messages = [{"role": "user", "content": "Hello"}]
    assert format_chat_messages(tokenizer, messages) == "Hello<think>"
    with chat_template_profile({"enable_thinking": False}):
        assert format_chat_messages(tokenizer, messages) == "Hello<think></think>"
    with chat_template_profile({"enable_thinking": True}):
        assert format_chat_messages(tokenizer, messages) == "Hello<think>"
    assert current_chat_template_kwargs() == {}


def test_profile_is_restored_after_generation_failure():
    with pytest.raises(RuntimeError):
        with chat_template_profile({"enable_thinking": True}):
            raise RuntimeError("generation failed")
    assert current_chat_template_kwargs() == {}


def test_concurrent_requests_do_not_share_thinking_profile():
    barrier = threading.Barrier(2)

    def request(thinking):
        with chat_template_profile({"enable_thinking": thinking}):
            barrier.wait(timeout=2)
            time.sleep(0.01)
            return format_chat_messages(TemplateTokenizer(), [{"role": "user", "content": "Hello"}])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(request, True), pool.submit(request, False)
        assert a.result() == "Hello<think>"
        assert b.result() == "Hello<think></think>"


@pytest.mark.parametrize(
    "value", [None, [], "false", {"enable_thinking": "false"}, {"enable_thinking": 1}, {"unknown": True}]
)
def test_invalid_or_unknown_options_are_rejected(value):
    with pytest.raises(ValueError):
        validate_chat_template_kwargs(value)


def test_requested_profile_never_silently_falls_back():
    class BrokenTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("template failed")

    messages = [{"role": "user", "content": "Hello"}]
    assert format_chat_messages(BrokenTokenizer(), messages) == "user: Hello"
    with chat_template_profile({"enable_thinking": False}):
        with pytest.raises(ValueError, match="template"):
            format_chat_messages(BrokenTokenizer(), messages)
        with pytest.raises(ValueError, match="template"):
            format_chat_messages(object(), messages)


def test_profile_options_are_copied():
    options = {"enable_thinking": False}
    with chat_template_profile(options):
        options["enable_thinking"] = True
        returned = current_chat_template_kwargs()
        returned["enable_thinking"] = True
        assert current_chat_template_kwargs() == {"enable_thinking": False}
