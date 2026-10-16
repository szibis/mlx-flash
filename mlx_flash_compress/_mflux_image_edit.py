"""Optional native MFLUX adapter; imported only for an actual image request."""

from __future__ import annotations

import gc
import importlib.metadata
import os
import resource
import sys
from types import SimpleNamespace

MEMORY_LIMIT_BYTES = 8 * 1024**3
CACHE_LIMIT_BYTES = 512 * 1024**2


def _dependencies():
    from .image_edit import ImageEditError

    try:
        version = importlib.metadata.version("mflux")
        if version != "0.23.0":
            raise ImageEditError("backend_unavailable", "This prototype requires mflux 0.23.0.")
        import mlx.core as mx
        from mflux.callbacks.instances.memory_saver import MemorySaver
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
        from mflux.utils.exceptions import StopImageGenerationException

        return SimpleNamespace(
            mx=mx,
            Model=Flux2KleinEdit,
            ModelConfig=ModelConfig,
            MemorySaver=MemorySaver,
            TilingConfig=TilingConfig,
            StopImageGenerationException=StopImageGenerationException,
            mflux_version=version,
            mlx_version=importlib.metadata.version("mlx"),
        )
    except (ImportError, importlib.metadata.PackageNotFoundError):
        raise ImageEditError(
            "backend_unavailable", "Install the image-edit optional dependencies on Apple Silicon."
        ) from None


class _Progress:
    def __init__(self, mx, emit, steps):
        self.mx, self.emit, self.steps, self.completed = mx, emit, steps, 0

    def call_before_loop(self, **kwargs):
        self.emit("denoising", step=0, total_steps=self.steps)

    def call_in_loop(self, t, latents, **kwargs):
        # MFLUX invokes the callback before its own lazy mx.eval. Materialize
        # here so the counter reports completed compute, not scheduled work.
        self.mx.eval(latents)
        self.completed += 1
        self.emit("denoising", step=self.completed, total_steps=self.steps)

    def call_after_loop(self, **kwargs):
        self.emit("decode")


def run(*, model_path, image_paths, prompt, seed, width, height, steps, emit):
    from .image_edit import ImageEditError

    # Complete local paths plus local-only tokenizer resolution are the first
    # boundary. These flags additionally disable Hub/Transformers fallback for
    # the isolated CLI process. No downloader is provided by this adapter.
    offline = {
        key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
    }
    for key in offline:
        os.environ[key] = "1"
    mx, previous_memory, previous_cache = None, None, None
    model, saver, deps = None, None, None
    try:
        emit("loading")
        deps = _dependencies()
        mx = deps.mx
        if not mx.metal.is_available():
            raise ImageEditError("backend_unavailable", "Image editing requires an available Apple Metal runtime.")
        previous_memory = mx.set_memory_limit(MEMORY_LIMIT_BYTES)
        previous_cache = mx.set_cache_limit(CACHE_LIMIT_BYTES)
        mx.clear_cache()
        mx.reset_peak_memory()  # Before weights; loading is part of the peak.
        model = deps.Model(model_path=str(model_path), quantize=4, model_config=deps.ModelConfig.flux2_klein_4b())
        if model.bits != 4:
            raise ImageEditError("invalid_model", "The loaded checkpoint is not 4-bit quantized.")
        if getattr(model.vae, "supports_implicit_tiling", True):
            model.tiling_config = deps.TilingConfig()
        # Passing a cache limit into upstream MemorySaver resets peak memory
        # AFTER weight loading. Configure the limit above and avoid that reset.
        saver = deps.MemorySaver(model=model, keep_transformer=False, cache_limit_bytes=None, num_seeds=1)
        model.callbacks.register(saver)
        model.callbacks.register(_Progress(mx, emit, steps))
        emit("conditioning")
        generated = model.generate_image(
            image_paths=[str(path) for path in image_paths],
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance=1.0,
            use_kv_cache=False,
        )
        mx.synchronize()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        statistics = {
            "mflux_version": deps.mflux_version,
            "mlx_version": deps.mlx_version,
            "peak_mlx_bytes": int(mx.get_peak_memory()),
            "peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "cache_limit_bytes": CACHE_LIMIT_BYTES,
            "peak_rss_scope": "process_lifetime",
            "peak_mlx_scope": "request_including_model_load",
        }
        return generated.image, statistics
    except Exception as error:
        if deps is not None and isinstance(error, getattr(deps, "StopImageGenerationException", ())):
            raise ImageEditError("cancelled", "Image editing was cancelled.") from None
        raise
    finally:
        model, saver = None, None
        gc.collect()
        if mx is not None and previous_memory is not None:
            mx.clear_cache()
            if previous_cache is not None:
                mx.set_cache_limit(previous_cache)
            mx.set_memory_limit(previous_memory)
        for key, value in offline.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
