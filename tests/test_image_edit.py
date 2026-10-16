"""Boundary tests: no large models, network calls, or Metal allocations."""

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def api():
    try:
        return importlib.import_module("mlx_flash_compress.image_edit")
    except ModuleNotFoundError:
        pytest.fail("The universal image-edit API is missing")


@pytest.fixture
def files(tmp_path):
    source = tmp_path / "reference.png"
    Image.new("RGB", (900, 600), (30, 60, 90)).save(source)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "flux2-klein-4b", "quantization": {"bits": 4}}))
    for component in ("vae", "transformer", "text_encoder"):
        (model / component).mkdir()
        (model / component / "0.safetensors").write_bytes(b"test-only-not-real-weights")
    (model / "tokenizer").mkdir()
    (model / "tokenizer" / "tokenizer.json").write_text("{}")
    return source, tmp_path / "output.png", model


@pytest.mark.parametrize(
    "options",
    [
        {"width": 255},
        {"height": 784},
        {"width": 513},
        {"width": True},
        {"steps": 0},
        {"steps": 9},
        {"steps": 2.5},
        {"seed": -1},
        {"seed": True},
        {"prompt": ""},
        {"prompt": "x" * 4001},
    ],
)
def test_rejects_unbounded_requests_before_loading_backend(api, files, monkeypatch, options):
    def forbidden(**kwargs):
        pytest.fail("Invalid input reached the backend")

    monkeypatch.setattr(api, "_run_backend", forbidden)
    source, output, model = files
    arguments = {"prompt": "private prompt", **options}
    with pytest.raises(api.ImageEditError) as error:
        api.edit_image(source, output, model, **arguments)
    assert error.value.code == "invalid_request"
    assert not output.exists()


def test_generated_output_has_source_bound_receipt_and_private_progress(api, files, monkeypatch):
    source, output, model = files
    original = source.read_bytes()
    events = []

    def backend(**kwargs):
        assert kwargs["model_path"] == model.resolve()
        assert kwargs["width"] == 512 and kwargs["height"] == 640
        with Image.open(kwargs["image_paths"][0]) as image:
            assert image.size == (768, 512)
        kwargs["emit"]("loading")
        kwargs["emit"]("conditioning")
        kwargs["emit"]("denoising", step=1, total_steps=4)
        kwargs["emit"]("decode")
        return Image.new("RGB", (512, 640), "red"), {
            "peak_mlx_bytes": 1234,
            "peak_rss_bytes": 5678,
            "mflux_version": "0.23.0",
            "mlx_version": "0.32.4",
        }

    monkeypatch.setattr(api, "_run_backend", backend)
    receipt = api.edit_image(source, output, model, "SECRET PROMPT", progress=events.append)
    assert receipt["input_sha256"] == [hashlib.sha256(original).hexdigest()]
    assert receipt["prompt_sha256"] == hashlib.sha256(b"SECRET PROMPT").hexdigest()
    assert receipt["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["provenance"] == "generated"
    assert receipt["visual_quality_approved"] is False
    assert receipt["peak_mlx_bytes"] == 1234
    assert receipt["width"] == 512 and receipt["height"] == 640
    assert receipt["seed"] == 42 and receipt["steps"] == 4
    assert len(receipt["model_sha256"]) == 64
    assert events[-1]["stage"] == "complete"
    assert "SECRET" not in json.dumps(events)
    assert source.read_bytes() == original
    assert not list(output.parent.glob("*.tmp"))


def test_accepts_decoded_image_bytes_and_multiple_references(api, files, monkeypatch):
    source, output, model = files

    def backend(**kwargs):
        assert len(kwargs["image_paths"]) == 2
        return Image.new("RGB", (256, 256)), {"peak_mlx_bytes": 0, "peak_rss_bytes": 0}

    monkeypatch.setattr(api, "_run_backend", backend)
    result = api.edit_image([source.read_bytes(), source], output, model, "edit", width=256, height=256)
    assert len(result["input_sha256"]) == 2


@pytest.mark.parametrize("fault", ["overwrite", "corrupt", "too_many", "remote_model", "not_four_bit"])
def test_rejects_unsafe_assets_before_backend(api, files, monkeypatch, fault):
    source, output, model = files
    if fault == "overwrite":
        output.write_bytes(b"keep")
    if fault == "corrupt":
        source.write_bytes(b"not an image")
    if fault == "too_many":
        source = [source] * 5
    if fault == "remote_model":
        model = "org/remote-model"
    if fault == "not_four_bit":
        (model / "config.json").write_text('{"quantization":{"bits":8}}')
    monkeypatch.setattr(api, "_run_backend", lambda **kwargs: pytest.fail("Bad assets reached backend"))
    with pytest.raises(api.ImageEditError):
        api.edit_image(source, output, model, "edit")
    if fault == "overwrite":
        assert output.read_bytes() == b"keep"


def test_backend_failure_does_not_leak_prompt_or_leave_output(api, files, monkeypatch):
    source, output, model = files

    def broken(**kwargs):
        raise RuntimeError("SECRET PRIVATE PROMPT and /private/customer/photo")

    monkeypatch.setattr(api, "_run_backend", broken)
    with pytest.raises(api.ImageEditError) as error:
        api.edit_image(source, output, model, "SECRET PRIVATE PROMPT")
    assert error.value.code == "backend_failed"
    assert "SECRET" not in str(error.value)
    assert not output.exists()


def test_wrong_output_dimensions_are_rejected(api, files, monkeypatch):
    source, output, model = files
    monkeypatch.setattr(api, "_run_backend", lambda **kwargs: (Image.new("RGB", (16, 16)), {}))
    with pytest.raises(api.ImageEditError):
        api.edit_image(source, output, model, "edit")
    assert not output.exists()


@pytest.mark.parametrize("fault", ["file", "large_config", "missing_weights", "missing_tokenizer", "large_weights"])
def test_incomplete_or_oversized_checkpoint_never_reaches_model_loader(api, files, monkeypatch, fault):
    source, output, model = files
    if fault == "file":
        model = source
    elif fault == "large_config":
        (model / "config.json").write_bytes(b" " * (1024**2 + 1))
    elif fault == "missing_weights":
        (model / "text_encoder" / "0.safetensors").unlink()
    elif fault == "missing_tokenizer":
        (model / "tokenizer" / "tokenizer.json").unlink()
    elif fault == "large_weights":
        # Sparse fixture exercises the byte gate without allocating 8 GiB.
        with (model / "transformer" / "0.safetensors").open("wb") as stream:
            stream.truncate(8 * 1024**3 + 1)
    monkeypatch.setattr(api, "_run_backend", lambda **kwargs: pytest.fail("Invalid checkpoint reached backend"))
    with pytest.raises(api.ImageEditError) as error:
        api.edit_image(source, output, model, "edit")
    assert error.value.code == "invalid_request"
    assert not output.exists()


@pytest.mark.parametrize("fault", ["object", "tiny", "animated"])
def test_invalid_reference_shapes_and_animation_are_rejected(api, files, monkeypatch, fault):
    source, output, model = files
    if fault == "object":
        source = [object()]
    elif fault == "tiny":
        Image.new("RGB", (15, 32)).save(source)
    elif fault == "animated":
        Image.new("RGB", (32, 32), "red").save(
            source, format="WEBP", save_all=True, append_images=[Image.new("RGB", (32, 32), "blue")], duration=100
        )
    monkeypatch.setattr(api, "_run_backend", lambda **kwargs: pytest.fail("Invalid reference reached backend"))
    with pytest.raises(api.ImageEditError) as error:
        api.edit_image(source, output, model, "edit")
    assert error.value.code == "invalid_request"


def test_process_lock_refuses_overlap_without_consuming_or_writing_assets(api, files):
    source, output, model = files
    assert api._RUN_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(api.ImageEditError) as error:
            api.edit_image(source, output, model, "edit")
        assert error.value.code == "busy"
        assert not output.exists()
    finally:
        api._RUN_LOCK.release()


def test_bad_output_type_has_safe_validation_error(api, files):
    source, _, model = files
    with pytest.raises(api.ImageEditError) as error:
        api.edit_image(source, None, model, "edit")
    assert error.value.code == "invalid_request"


def test_output_collision_during_generation_preserves_existing_file_and_cleans_staging(api, files, monkeypatch):
    source, output, model = files

    def backend(**kwargs):
        output.write_bytes(b"another writer owns this")
        return Image.new("RGB", (512, 640)), {}

    monkeypatch.setattr(api, "_run_backend", backend)
    with pytest.raises(api.ImageEditError):
        api.edit_image(source, output, model, "edit")
    assert output.read_bytes() == b"another writer owns this"
    assert not list(output.parent.glob(".image-edit-*.tmp"))


@pytest.mark.parametrize("version", ["0.22.0", None])
def test_optional_backend_failure_is_safe_without_loading_metal(api, files, monkeypatch, version):
    from mlx_flash_compress import _mflux_image_edit as backend

    source, output, model = files

    def version_lookup(name):
        if version is None:
            raise backend.importlib.metadata.PackageNotFoundError("PRIVATE environment detail")
        return version

    monkeypatch.setattr(backend.importlib.metadata, "version", version_lookup)
    with pytest.raises(api.ImageEditError) as error:
        api.edit_image(source, output, model, "PRIVATE prompt")
    assert error.value.code == "backend_unavailable"
    assert "PRIVATE" not in str(error.value)
    assert not output.exists()


def test_cli_success_emits_progress_then_receipt(api, files, monkeypatch, capsys):
    source, output, model = files

    def backend(**kwargs):
        kwargs["emit"]("conditioning")
        return Image.new("RGB", (512, 640)), {}

    monkeypatch.setattr(api, "_run_backend", backend)
    assert (
        api.main(["--input", str(source), "--output", str(output), "--model-path", str(model), "--prompt", "edit"]) == 0
    )
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["type"] for event in events] == ["progress", "progress", "result"]
    assert events[-1]["receipt"]["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_cli_keyboard_interrupt_has_safe_cancellation_json(api, files, monkeypatch, capsys):
    source, output, model = files

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(api, "edit_image", interrupted)
    assert (
        api.main(["--input", str(source), "--output", str(output), "--model-path", str(model), "--prompt", "edit"])
        == 130
    )
    assert json.loads(capsys.readouterr().out)["code"] == "cancelled"


def test_cli_bad_option_uses_safe_argument_error_in_process(api, capsys):
    assert api.main(["--unknown=PRIVATE"]) == 2
    text = capsys.readouterr().out
    assert json.loads(text)["code"] == "invalid_request" and "PRIVATE" not in text


def test_oversized_encoded_reference_is_rejected_before_backend(api, files, monkeypatch):
    _, output, model = files
    monkeypatch.setattr(api, "_run_backend", lambda **kwargs: pytest.fail("Oversized input reached backend"))
    with pytest.raises(api.ImageEditError) as error:
        api.edit_image(b"x" * (12 * 1024**2 + 1), output, model, "edit")
    assert error.value.code == "invalid_request"


def test_changed_model_weights_change_receipt_identity(api, files):
    _, _, model = files
    _, before = api._model_identity(model)
    (model / "transformer" / "0.safetensors").write_bytes(b"different model")
    _, after = api._model_identity(model)
    assert before != after


def test_cli_bad_argument_never_echoes_private_value():
    result = subprocess.run(
        [sys.executable, "-m", "mlx_flash_compress.image_edit", "--private-option=SECRET"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["code"] == "invalid_request"
    assert "SECRET" not in result.stdout + result.stderr


def test_cli_normalized_backend_cancellation_uses_cancellation_exit(api, files, monkeypatch, capsys):
    source, output, model = files

    def cancelled(*args, **kwargs):
        raise api.ImageEditError("cancelled", "Image editing was cancelled.")

    monkeypatch.setattr(api, "edit_image", cancelled)
    result = api.main(["--input", str(source), "--output", str(output), "--model-path", str(model), "--prompt", "edit"])
    assert result == 130
    assert json.loads(capsys.readouterr().out)["code"] == "cancelled"


def test_cli_invalid_request_returns_safe_json_without_loading_models(files):
    source, output, model = files
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_flash_compress.image_edit",
            "--input",
            str(source),
            "--output",
            str(output),
            "--model-path",
            str(model),
            "--prompt",
            "SECRET",
            "--width",
            "900",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    event = json.loads(result.stdout)
    assert event["type"] == "error" and event["code"] == "invalid_request"
    assert "SECRET" not in result.stdout + result.stderr
