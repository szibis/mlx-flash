"""Bounded, local image editing. Run one request per CLI process on Apple Silicon."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

MAX_INPUT_BYTES = 12 * 1024**2
MAX_INPUT_PIXELS = 40_000_000
_RUN_LOCK = threading.Lock()


class ImageEditError(ValueError):
    """Public, redacted failure. Backend exceptions are never public messages."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _fail(message="Invalid image-edit request."):
    raise ImageEditError("invalid_request", message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _validate_parameters(prompt, seed, width, height, steps, max_dimension=768):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000 or "\0" in prompt:
        _fail("Provide a prompt of 1–4000 characters.")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        _fail("Seed must be an unsigned 32-bit integer.")
    if type(max_dimension) is not int or not 256 <= max_dimension <= 1600 or max_dimension % 16:
        _fail("Resolution budget must be a multiple of 16 between 256 and 1600.")
    if any(type(v) is not int or not 256 <= v <= max_dimension or v % 16 for v in (width, height)):
        _fail("Dimensions must be multiples of 16 within the explicit resolution budget.")
    if type(steps) is not int or not 1 <= steps <= 8:
        _fail("Steps must be an integer between 1 and 8.")


def _model_identity(model_path):
    """Hash the actual local checkpoint; an HF-looking directory is not proof."""
    try:
        root = Path(model_path).expanduser().resolve(strict=True)
        if not root.is_dir():
            _fail("Model must be a prepared local directory.")
        config_file = root / "config.json"
        if config_file.stat().st_size > 1024**2:
            _fail("Model configuration exceeds its size limit.")
        config = json.loads(config_file.read_text())
        if config.get("model_type") != "flux2-klein-4b" or config.get("quantization", {}).get("bits") != 4:
            _fail("Only a local 4-bit FLUX.2 Klein 4B checkpoint is supported.")
        for component in ("vae", "transformer", "text_encoder"):
            if not any((root / component).glob("*.safetensors")):
                _fail("Local checkpoint is incomplete.")
        if not (root / "tokenizer" / "tokenizer.json").is_file():
            _fail("Local checkpoint tokenizer is missing.")
        files = sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".txt", ".jinja"}
        )
        if len(files) > 512 or sum(p.stat().st_size for p in files) > 8 * 1024**3:
            _fail("Local checkpoint exceeds prototype limits.")
        inventory = []
        for path in files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    digest.update(chunk)
            inventory.append(
                {"name": path.relative_to(root).as_posix(), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
            )
        identity = _sha(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode())
        return root, identity
    except ImageEditError:
        raise
    except (OSError, ValueError, TypeError):
        _fail("Model must be a complete prepared local directory.")


def _prepare_references(inputs, directory, max_dimension=768):
    from PIL import Image, ImageOps

    if isinstance(inputs, (str, os.PathLike, bytes)):
        inputs = [inputs]
    if not isinstance(inputs, (tuple, list)) or not 1 <= len(inputs) <= 4:
        _fail("Provide between one and four reference images.")
    hashes, paths, conditions = [], [], []
    for index, source in enumerate(inputs):
        try:
            if isinstance(source, bytes):
                raw = source
            elif isinstance(source, (str, os.PathLike)):
                with Path(source).open("rb") as stream:
                    raw = stream.read(MAX_INPUT_BYTES + 1)
            else:
                _fail("Reference must be a local image path or encoded image bytes.")
            if not 1 <= len(raw) <= MAX_INPUT_BYTES:
                _fail("Reference exceeds its encoded size limit.")
            with Image.open(io.BytesIO(raw)) as original:
                if original.format not in {"JPEG", "PNG", "WEBP"} or getattr(original, "n_frames", 1) != 1:
                    _fail("Reference must be a still JPEG, PNG or WebP.")
                if min(original.size) < 16 or original.width * original.height > MAX_INPUT_PIXELS:
                    _fail("Reference dimensions exceed prototype limits.")
                original.load()
                image = ImageOps.exif_transpose(original).convert("RGBA")
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            size = tuple(((v + 15) // 16) * 16 for v in image.size)
            # Preserve the whole reference: pad the sub-16 remainder instead of
            # letting MFLUX center-crop thin details at the source boundaries.
            normalized = Image.new("RGB", size, "white")
            normalized.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2), image)
            path = Path(directory) / f"reference-{index}.png"
            normalized.save(path, format="PNG")
            os.chmod(path, 0o600)
            hashes.append(_sha(raw))
            paths.append(path)
            conditions.append({"sha256": _sha(path.read_bytes()), "width": size[0], "height": size[1]})
        except ImageEditError:
            raise
        except Exception:
            _fail("Reference image could not be decoded safely.")
    if max_dimension > 768 and sum(c["width"] * c["height"] for c in conditions) > max_dimension**2:
        _fail("Combined reference images exceed the native-resolution area budget.")
    return paths, hashes, conditions


def _run_backend(**kwargs):
    from ._mflux_image_edit import run

    return run(**kwargs)


def _write_output(image, destination):
    fd, name = tempfile.mkstemp(prefix=".image-edit-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            image.convert("RGB").save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        # Link is atomic and refuses an existing destination, including symlinks.
        os.link(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def edit_image(
    input_path,
    output_path,
    model_path,
    prompt,
    *,
    seed=42,
    width=512,
    height=640,
    steps=4,
    progress=None,
    max_dimension=768,
):
    """Generate a PNG and receipt from local references, without downloading.

    ``input_path`` accepts a path, encoded bytes, or a list of up to four of
    these. The generated image can change reference content. This API does not
    assert product fidelity or visual acceptance. Use an isolated CLI process;
    its exit releases the Metal context as well as all model objects.
    """
    _validate_parameters(prompt, seed, width, height, steps, max_dimension)
    started = time.monotonic()
    try:
        destination = Path(output_path).expanduser().absolute()
        if destination.suffix.lower() != ".png" or not destination.parent.is_dir() or os.path.lexists(destination):
            _fail("Output must be a new PNG file in an existing directory.")
    except (TypeError, OSError):
        _fail("Output path is invalid.")
    if not _RUN_LOCK.acquire(blocking=False):
        raise ImageEditError("busy", "An image-edit request is already active.")

    def emit(stage, **fields):
        event = {"type": "progress", "stage": stage, "elapsed_seconds": round(time.monotonic() - started, 3), **fields}
        if progress:
            progress(event)

    try:
        with tempfile.TemporaryDirectory(prefix="mlx-image-edit-") as temporary:
            paths, hashes, conditions = _prepare_references(input_path, temporary, max_dimension)
            root, model_sha = _model_identity(model_path)
            # Upstream diagnostics can contain private prompt/path data. Only
            # structured, allowlisted progress is forwarded to the caller.
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                image, statistics = _run_backend(
                    model_path=root,
                    image_paths=paths,
                    prompt=prompt,
                    seed=seed,
                    width=width,
                    height=height,
                    steps=steps,
                    emit=emit,
                    **({"reference_max_area": max_dimension**2} if max_dimension > 768 else {}),
                )
            if image.size != (width, height):
                raise ImageEditError("invalid_output", "Backend returned unexpected output dimensions.")
            _write_output(image, destination)
            receipt = {
                "version": "mlx-image-edit-v1",
                "backend": "mflux",
                "model": "flux2-klein-4b",
                "quantization_bits": 4,
                "model_sha256": model_sha,
                "input_sha256": hashes,
                "prompt_sha256": _sha(prompt.encode("utf-8")),
                "conditioning": conditions,
                "conditioning_alpha_background": "#ffffff",
                "output_sha256": _sha(destination.read_bytes()),
                "format": "png",
                "width": width,
                "height": height,
                "max_dimension": max_dimension,
                "seed": seed,
                "steps": steps,
                "guidance": 1.0,
                "provenance": "generated",
                "visual_quality_approved": False,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                **statistics,
            }
            emit("complete")
            return receipt
    except ImageEditError:
        raise
    except Exception as error:
        code = (
            "memory_limit"
            if isinstance(error, MemoryError)
            or any(word in str(error).lower() for word in ("out of memory", "allocation failed", "memory limit"))
            else "backend_failed"
        )
        raise ImageEditError(code, "Local image editing failed; original inputs were preserved.") from None
    finally:
        _RUN_LOCK.release()


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ImageEditError("invalid_request", "Invalid command arguments; use --help for supported options.")


def main(argv=None):
    output_stream = sys.stdout

    def send(value):
        print(json.dumps(value, ensure_ascii=False), file=output_stream, flush=True)

    parser = _Parser(description=__doc__)
    parser.add_argument(
        "--input", action="append", required=True, help="Local reference image (repeat up to four times)"
    )
    parser.add_argument("--output", required=True, help="New PNG output path")
    parser.add_argument("--model-path", required=True, help="Prepared local 4-bit FLUX.2 Klein 4B checkpoint directory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=768,
        help="Explicit input/output edge budget, 256–1600 in multiples of 16; higher values need more memory/time",
    )
    try:
        args = parser.parse_args(argv)
        result = edit_image(
            args.input,
            args.output,
            args.model_path,
            args.prompt,
            seed=args.seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
            progress=send,
            max_dimension=args.max_dimension,
        )
        send({"type": "result", "receipt": result})
        return 0
    except ImageEditError as error:
        send({"type": "error", "code": error.code, "message": str(error)})
        return 130 if error.code == "cancelled" else 2
    except KeyboardInterrupt:
        send({"type": "error", "code": "cancelled", "message": "Image editing was cancelled."})
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
