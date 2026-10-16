# Local image-edit prototype

`mlx-flash-image-edit` runs one bounded image-edit request natively on Apple
Silicon through MFLUX 0.23.0 and MLX. It is independent of the text/vision server:
there is no image-edit HTTP endpoint and it does not require an LLM to plan the
request. The output is generated content; reference-image fidelity requires
inspection by the caller.

Install the optional dependencies from the current source checkout in an
isolated macOS environment. These instructions do not assume that the latest
PyPI release already contains this prototype:

```sh
python -m pip install -e '.[image-edit]'
```

Prepare a complete, trusted local **4-bit FLUX.2 Klein 4B MFLUX checkpoint**
separately. Its `config.json` must specify `model_type: flux2-klein-4b` and
`quantization.bits: 4`; provide the `vae`, `transformer`, `text_encoder`
safetensors components and `tokenizer/tokenizer.json` with associated tokenizer
assets. Acquisition, revision pinning and license review belong to the caller.
Inference accepts an existing local directory, enables offline Hub/Transformers
mode, and provides no implicit downloader. The receipt hashes the actual local
model inventory instead of treating the directory name as proof of its weights.

```sh
mlx-flash-image-edit \
  --model-path /private/models/klein-4b-4bit \
  --input /private/images/reference.png \
  --output /private/results/edit.png \
  --prompt 'Change only the backdrop to a softly lit neutral studio.' \
  --seed 42 --width 512 --height 640 --steps 4
```

Equivalent entry point: `python -m mlx_flash_compress.image_edit`. Repeat
`--input` for up to four references. Inputs are still JPEG, PNG or WebP, up to
12 MiB encoded and 40 MP per image. References are oriented, converted to RGB,
downscaled to at most 768 pixels per side, and padded to multiples of 16 rather
than cropped. Input transparency is composited onto white **for conditioning
only**; source files remain untouched. The output is a new RGB PNG.

Dimensions must be multiples of 16 between 256 and 768, steps integers from 1
through 8, seeds unsigned 32-bit integers, and prompts at most 4000 characters.
Guidance is fixed at 1.0 and reference KV caching is disabled.

## Progress and receipt

Standard output is newline-delimited JSON. Successful requests emit
`type: progress` records with stages `loading`, `conditioning`, `denoising`,
`decode` and `complete`, followed by `type: result` with a `receipt`.
Denoising records contain `step` and `total_steps`: zero marks the start and
subsequent counts follow actual MLX evaluation of that step's latents. No
percentage or remaining duration is invented. Prompts, input paths and upstream
diagnostic output are excluded from these records.

```json
{"type":"progress","stage":"denoising","step":2,"total_steps":4,"elapsed_seconds":12.3}
```

The receipt contains source/conditioning/output and UTF-8 prompt SHA-256, an aggregate hash of the
local model inventory, backend/library versions, seed, dimensions, steps,
guidance, elapsed duration and peak MLX/RSS memory. It always records
`provenance: generated` and `visual_quality_approved: false`. Encoding success
does not establish visual quality or content preservation.

Failures emit `type: error`, a code and a redacted message, with exit status 2.
Cancellation exits 130. Invalid CLI arguments also return safe JSON; help is
ordinary text. Model diagnostics and exception details are not forwarded.

## Python API

```python
from mlx_flash_compress.image_edit import edit_image, ImageEditError

receipt = edit_image(
    "/private/images/reference.png",
    "/private/results/edit.png",
    "/private/models/klein-4b-4bit",
    "Change only the backdrop to a softly lit neutral studio.",
    seed=42, width=512, height=640, steps=4,
    progress=consume_progress,
)
```

The first argument accepts one path, encoded image bytes, or a list of up to four
paths/encoded byte strings. Dependencies are lazy: importing the API and
validating inputs do not import MFLUX or allocate Metal resources. The adapter
serializes image-edit requests in its process but does not coordinate with an
already resident text/vision model. The caller must unload other heavy models
and use a dedicated request subprocess.

## Integration with an orchestrator

Any orchestrator, including LLM Sentinel, can use the CLI as a separate image
tool. Configure the native executable and local checkpoint directory in that
orchestrator. An LLM may propose the prompt and bounded request parameters; the
orchestrator invokes the subprocess and consumes its structured result. Image
editing does not require a text-model connection, an OpenAI-compatible endpoint,
or a project-specific integration in MLX-Flash.

The caller owns the lifecycle:

1. Acquire its shared hardware reservation and release other resident models.
   Wait for that release before starting the image-edit process.
2. Start one CLI subprocess, with argument values passed as an argument list.
   Read stdout incrementally as NDJSON; use `type: progress` for status.
3. Require exit status zero and a single `type: result` receipt before accepting
   success. Verify the output hash against the actual PNG. A partial progress
   stream or a written file by itself is not success.
4. On cancellation or timeout, terminate the owned process group and wait for
   it to exit. Restore any previously running model and release the hardware
   reservation after cleanup.

The per-process image-edit lock does **not** serialize independent CLI processes
or coordinate the text/vision server. This prototype provides neither an HTTP
image endpoint nor a cross-process scheduler. Callers must supply those resource
and cancellation policies. Sentinel's existing Metal CI reservation and lab
handoff are documented in [Real Qwen Metal CI](qwen-metal-ci.md); this image CLI
does not acquire that reservation or pause/resume a Sentinel lab automatically.

A successful receipt establishes generation and file integrity. An orchestrator
must separately evaluate content preservation, appearance and any requested
mask/compositing constraints before applying or publishing a generated image.

## Memory and validation scope

Before loading weights, the prototype sets an 8 GiB **MLX allocation limit** and
a 512 MiB MLX cache limit. Loading is included in the peak measurement. MFLUX's
memory-saver releases the text encoder before denoising and the transformer
before decode; supported VAE tiling is enabled. The adapter releases model
references/cache afterward; CLI process exit releases the Metal context. Python
API calls restore previous allocation/cache limits.

The allocation limit does not cap total RAM, CPU arrays, RSS, swap or macOS
memory pressure, and cannot guarantee an OOM-free request. `peak_rss_bytes`
covers the process lifetime; `peak_mlx_bytes` covers the request including model
loading. Keep heavy model execution sequential on memory-constrained machines.

Model-free adapter tests cover bounds, immutable sources, safe failures,
receipts, progress timing and memory-limit ordering. They do not establish
perceptual quality, material preservation, real shadow recovery, or fit on a
16 GB M4. Report real-model checks separately with hardware, checkpoint,
latency, peak memory/pressure and visual inspection.

```sh
python -m pytest tests/test_image_edit.py tests/test_mflux_image_edit.py -q
```

An optional real Metal smoke check is skipped unless the caller explicitly sets
`MLX_FLASH_IMAGE_EDIT_REAL_MODEL` to its prepared checkpoint directory. It runs a
generated 256×256 reference through four actual steps and checks encoding,
completed-step events and memory accounting; it does not score appearance.

Current model-free coverage, full-suite results and the separately enabled
native Metal smoke are recorded in the [prototype verification proof](testing/image-edit-prototype-2026-10-16.md).

For an experimental way to retain a supplied foreground and estimated generated
scene shadow over live backdrop colors, see [studio-scene layers](studio-layers.md).
This separate CPU API requires an aligned matte and refuses uncertain scenes.
The same module also supports masked RGB backdrop recoloring, retaining the
generated scene's existing gradients and shadow without requiring decomposition.
