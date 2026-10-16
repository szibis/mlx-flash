"""Exercise the runtime adapter with a fake model, not downloaded weights."""

import importlib
import os
from types import SimpleNamespace

import pytest
from PIL import Image


def test_native_adapter_reports_completed_steps_and_preserves_loading_peak(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous-value")
    try:
        module = importlib.import_module("mlx_flash_compress._mflux_image_edit")
    except ModuleNotFoundError:
        pytest.fail("Native MFLUX adapter is missing")
    history, progress = [], []

    class Core:
        metal = SimpleNamespace(is_available=lambda: True)

        def set_memory_limit(self, value):
            history.append(("limit", value))
            return 123

        def set_cache_limit(self, value):
            history.append(("cache", value))
            return 456

        def clear_cache(self):
            history.append("clear")

        def reset_peak_memory(self):
            history.append("reset")

        def eval(self, value):
            history.append("eval")

        def synchronize(self):
            history.append("sync")

        def get_peak_memory(self):
            return 999

    class Model:
        bits = 4
        vae = SimpleNamespace(supports_implicit_tiling=True)
        tiling_config = None

        def __init__(self, **kwargs):
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert kwargs == {"model_path": str(tmp_path), "quantize": 4, "model_config": "klein4"}
            history.append("load")
            self.callbacks = SimpleNamespace(register=lambda callback: self.subscribers.append(callback))
            self.subscribers = []

        def generate_image(self, **kwargs):
            assert kwargs["use_kv_cache"] is False
            assert kwargs["guidance"] == 1.0 and kwargs["num_inference_steps"] == 4
            for callback in self.subscribers:
                callback.call_before_loop(latents="latent")
            for t in range(4):
                for callback in self.subscribers:
                    callback.call_in_loop(t=t, latents="latent")
            for callback in self.subscribers:
                callback.call_after_loop(latents="latent")
            return SimpleNamespace(image=Image.new("RGB", (512, 640)))

    class Saver:
        def __init__(self, **kwargs):
            assert kwargs["cache_limit_bytes"] is None
            assert kwargs["keep_transformer"] is False
            assert kwargs["num_seeds"] == 1

        def call_before_loop(self, **kwargs):
            pass

        def call_in_loop(self, **kwargs):
            pass

        def call_after_loop(self, **kwargs):
            pass

    dependencies = SimpleNamespace(
        mx=Core(),
        Model=Model,
        ModelConfig=SimpleNamespace(flux2_klein_4b=lambda: "klein4"),
        MemorySaver=Saver,
        TilingConfig=lambda: "tiles",
        mflux_version="0.23.0",
        mlx_version="0.32.4",
    )
    monkeypatch.setattr(module, "_dependencies", lambda: dependencies)
    image, stats = module.run(
        model_path=tmp_path,
        image_paths=[tmp_path / "ref.png"],
        prompt="SECRET",
        seed=42,
        width=512,
        height=640,
        steps=4,
        emit=lambda stage, **kwargs: progress.append({"stage": stage, **kwargs}),
    )
    assert image.size == (512, 640)
    assert history.count("reset") == 1 and history.index("reset") < history.index("load")
    assert ("limit", 8 * 1024**3) in history and ("cache", 512 * 1024**2) in history
    assert progress == [
        {"stage": "loading"},
        {"stage": "conditioning"},
        {"stage": "denoising", "step": 0, "total_steps": 4},
        {"stage": "denoising", "step": 1, "total_steps": 4},
        {"stage": "denoising", "step": 2, "total_steps": 4},
        {"stage": "denoising", "step": 3, "total_steps": 4},
        {"stage": "denoising", "step": 4, "total_steps": 4},
        {"stage": "decode"},
    ]
    assert stats["peak_mlx_bytes"] == 999 and stats["memory_limit_bytes"] == 8 * 1024**3
    assert history[-2:] == [("cache", 456), ("limit", 123)]
    assert os.environ["HF_HUB_OFFLINE"] == "previous-value"

    class CancelledError(Exception):
        pass

    dependencies.StopImageGenerationException = CancelledError

    def cancelled(self, **kwargs):
        raise CancelledError("PRIVATE upstream detail")

    monkeypatch.setattr(Model, "generate_image", cancelled)
    from mlx_flash_compress.image_edit import ImageEditError

    with pytest.raises(ImageEditError) as error:
        module.run(
            model_path=tmp_path,
            image_paths=[tmp_path / "ref.png"],
            prompt="SECRET",
            seed=42,
            width=512,
            height=640,
            steps=4,
            emit=lambda *a, **k: None,
        )
    assert error.value.code == "cancelled"
    monkeypatch.setattr(Model, "bits", 8)
    with pytest.raises(ImageEditError) as error:
        module.run(
            model_path=tmp_path,
            image_paths=[],
            prompt="edit",
            seed=42,
            width=512,
            height=640,
            steps=4,
            emit=lambda *a, **k: None,
        )
    assert error.value.code == "invalid_model"
    assert history[-2:] == [("cache", 456), ("limit", 123)]


def test_native_adapter_refuses_non_metal_runtime_without_loading_model(monkeypatch, tmp_path):
    try:
        module = importlib.import_module("mlx_flash_compress._mflux_image_edit")
    except ModuleNotFoundError:
        pytest.fail("Native MFLUX adapter is missing")
    dependencies = SimpleNamespace(mx=SimpleNamespace(metal=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(module, "_dependencies", lambda: dependencies)
    with pytest.raises(Exception, match="Metal"):
        module.run(
            model_path=tmp_path,
            image_paths=[],
            prompt="edit",
            seed=42,
            width=512,
            height=640,
            steps=4,
            emit=lambda *a, **k: None,
        )


@pytest.mark.skipif(
    not os.environ.get("MLX_FLASH_IMAGE_EDIT_REAL_MODEL"),
    reason="Opt-in local checkpoint/Metal check; no model download",
)
def test_real_local_model_encodes_image_and_reports_actual_steps(tmp_path):
    """Technical smoke check only; does not approve the generated appearance."""
    from mlx_flash_compress.image_edit import edit_image

    source = tmp_path / "reference.png"
    Image.new("RGB", (256, 256), (80, 100, 120)).save(source)
    output = tmp_path / "generated.png"
    events = []
    receipt = edit_image(
        source,
        output,
        os.environ["MLX_FLASH_IMAGE_EDIT_REAL_MODEL"],
        "Make the background a softly lit neutral gray.",
        width=256,
        height=256,
        steps=4,
        progress=events.append,
    )
    with Image.open(output) as image:
        assert image.size == (256, 256) and image.format == "PNG"
    assert [event["step"] for event in events if event["stage"] == "denoising"] == [0, 1, 2, 3, 4]
    assert 0 < receipt["peak_mlx_bytes"] <= 8 * 1024**3
    assert receipt["provenance"] == "generated" and receipt["visual_quality_approved"] is False
