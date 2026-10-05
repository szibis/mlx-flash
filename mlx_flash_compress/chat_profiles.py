"""Validated, request-scoped tokenizer profiles without native dependencies."""

from contextlib import contextmanager
from contextvars import ContextVar

_CHAT_TEMPLATE_KWARGS: ContextVar[dict] = ContextVar("mlx_flash_chat_template_kwargs", default={})


class ChatTemplateProfileError(ValueError):
    """The selected tokenizer cannot honor an explicitly requested profile."""


def validate_chat_template_kwargs(value: object) -> dict:
    if not isinstance(value, dict) or set(value) - {"enable_thinking"}:
        raise ValueError("chat_template_kwargs accepts only enable_thinking")
    if "enable_thinking" in value and type(value["enable_thinking"]) is not bool:
        raise ValueError("chat_template_kwargs.enable_thinking must be a boolean")
    return dict(value)


@contextmanager
def chat_template_profile(options: dict):
    token = _CHAT_TEMPLATE_KWARGS.set(validate_chat_template_kwargs(options))
    try:
        yield
    finally:
        _CHAT_TEMPLATE_KWARGS.reset(token)


def current_chat_template_kwargs() -> dict:
    return dict(_CHAT_TEMPLATE_KWARGS.get())


def format_chat_messages(tokenizer, messages: list[dict]) -> str:
    options = current_chat_template_kwargs()
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **options)
        except Exception as error:
            if options:
                raise ChatTemplateProfileError(
                    "Tokenizer failed to apply the requested thinking chat template"
                ) from error
    elif options:
        raise ChatTemplateProfileError("Tokenizer has no chat template for the requested thinking profile")
    # Preserve legacy behavior only when no explicit profile was requested.
    return "\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages)
