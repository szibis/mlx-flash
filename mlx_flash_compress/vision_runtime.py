"""Optional, bounded mlx-vlm image-understanding adapter (no image editing)."""
from __future__ import annotations

import base64
import binascii
import gc
import io
import re
import warnings

from mlx_flash_compress.openai_compat import APIRequestError

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_194_304
MAX_IMAGES = 1
_DATA_URL = re.compile(r"^data:image/(png|jpeg);base64,([A-Za-z0-9+/=]+)$")
_CONTROL = re.compile(r"<\|[^>]*>|<[^>]*\|>|</?think>|<(?:start_of_turn|end_of_turn|eos|bos)>")


def decode_image_url(value):
    if not isinstance(value, dict) or set(value) - {"url", "detail"} or value.get("detail", "auto") not in ("auto", "low", "high"):
        raise APIRequestError("image_url accepts url and detail=auto/low/high only.")
    url = value.get("url")
    if not isinstance(url, str) or len(url) > (MAX_IMAGE_BYTES + 2) // 3 * 4 + 64:
        raise APIRequestError("Image data URL exceeds the 4 MiB decoded limit.")
    match = _DATA_URL.fullmatch(url)
    if not match:
        raise APIRequestError("Only base64 PNG/JPEG data URLs are supported; remote and local URLs are disabled.")
    from PIL import Image
    try:
        raw = base64.b64decode(match[2], validate=True)
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("image bytes limit")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                expected = "PNG" if match[1] == "png" else "JPEG"
                if source.format != expected or source.width * source.height > MAX_IMAGE_PIXELS or getattr(source, "n_frames", 1) != 1:
                    raise ValueError("image format, dimensions or frame count")
                source.load()
                return source.convert("RGB")
    except (ValueError, OSError, binascii.Error, Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise APIRequestError("Image must be a valid single-frame PNG/JPEG of at most 4 MiB and 4 megapixels.") from None


def prepare_vision_messages(messages):
    """Preserve ordered markers on their original user turn; never fetch URLs."""
    normalized, images = [], []
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "developer":
            role = "system"
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue
        if not isinstance(content, list) or not content or len(content) > 64:
            raise APIRequestError("Vision content must contain 1–64 text/image_url parts.")
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise APIRequestError("Vision content parts must be objects.")
            if part.get("type") == "text" and set(part) == {"type", "text"} and isinstance(part["text"], str):
                parts.append(dict(part))
            elif part.get("type") == "image_url" and set(part) == {"type", "image_url"} and role == "user":
                if len(images) >= MAX_IMAGES:
                    raise APIRequestError("Vision requests accept at most one image.")
                images.append(decode_image_url(part["image_url"]))
                parts.append({"type": "image"})
            else:
                raise APIRequestError("Only text and user image_url content parts are supported.")
        normalized.append({"role": role, "content": parts})
    return normalized, images


def validate_vision_output(output):
    if not isinstance(output, str) or not output.strip() or _CONTROL.search(output):
        raise APIRequestError("Vision model returned empty content or unresolved control tokens.", status=502, code="invalid_model_output")


class VisionRuntime:
    """One resident model at a time; owned resources released after every call."""
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.processor = None
        self.ready = False
        self.family = "unknown"

    def load(self):
        if self.model is not None:
            return
        try:
            from mlx_vlm import load
        except ImportError:
            raise RuntimeError("Vision requires the optional dependency: install mlx-flash[vision].") from None
        try:
            self.model, self.processor = load(self.model_name)
            config = self.model.config
            config = config if isinstance(config, dict) else vars(config)
            self.family = config.get("model_type", "unknown")
            if not config.get("vision_config") or not getattr(self.processor, "image_processor", None):
                raise RuntimeError("The selected model has no verified vision configuration/processor.")
            self.ready = True
        except Exception:
            self.release()
            raise

    def release(self):
        self.model = self.processor = None
        gc.collect()
        import mlx.core as mx
        mx.synchronize()
        mx.clear_cache()

    def generate(self, messages, max_tokens, temperature, **sampling):
        normalized, images = prepare_vision_messages(messages)
        try:
            self.load()
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template
            prompt = apply_chat_template(self.processor, self.model.config, normalized, num_images=len(images), enable_thinking=False)
            result = generate(self.model, self.processor, prompt=prompt, image=images or None, max_tokens=max_tokens,
                              temperature=temperature, verbose=False, enable_thinking=False, **sampling)
            validate_vision_output(result.text)
            if result.finish_reason not in ("stop", "length"):
                raise APIRequestError("Vision generator did not complete.", status=502, code="invalid_model_output")
            return {"output": result.text, "tokens": result.generation_tokens, "prompt_tokens": result.prompt_tokens,
                    "finish_reason": result.finish_reason, "tok_per_s": result.generation_tps,
                    "generation_tps": result.generation_tps, "prompt_tps": result.prompt_tps,
                    "peak_memory_gb": result.peak_memory, "native_generation_metadata": True,
                    "usage_source": "exact_mlx_vlm_generation", "memory_pressure": "normal"}
        finally:
            self.release()
