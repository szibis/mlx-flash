# Native image-edit prototype verification — 2026-10-16

The new image-edit CLI/API is an optional universal runtime capability. It runs
one local request through MFLUX 0.23.0 and MLX 0.32.4 using a prepared 4-bit
FLUX.2 Klein 4B checkpoint. It adds no image-edit HTTP endpoint, cross-process
scheduler, or automatic handoff to an existing text/vision server or Sentinel
lab. Orchestrator lifecycle requirements are in [the integration guide](../image-edit.md).

## Environment and real-model evidence

The native host was a MacBook Pro with an M5 Pro and 64 GiB unified memory.
Metal was available. This is **not** verification on an M4 with 16 GiB.

An earlier focused invocation with the opt-in real Metal smoke enabled passed
**28 tests**. Its real-model check used the synthetic 256×256 reference created
by `test_real_local_model_encodes_image_and_reports_actual_steps`: four actual
denoising steps, encoded PNG dimensions, completed-step events and bounded MLX
memory accounting. The test leaves `visual_quality_approved: false`; it does
not evaluate perceptual quality or exact preservation of reference content.
Checkpoint acquisition occurred separately from inference.

No customer images, application-specific cases, private paths or photo-comparison
metrics are included in this universal runtime proof. Application quality and
compositing evidence belong to the consuming application's repository.

A final opt-in real Metal rerun after the edge-case additions passed **44 tests in 11.11 seconds**, including the real smoke. This separately confirms the final source still executes native image generation.

## Fresh model-free verification

After formatting and edge-case coverage work, the focused suite passed
**43 tests, with one real Metal test explicitly skipped** because the opt-in
checkpoint variable was absent. This fresh run did not repeat image inference.

```sh
python -m pytest tests/test_image_edit.py tests/test_mflux_image_edit.py \
  --cov=mlx_flash_compress.image_edit \
  --cov=mlx_flash_compress._mflux_image_edit \
  --cov-config=.coveragerc --cov-report=term-missing --cov-fail-under=90 -q
```

| Module | Coverage |
| --- | ---: |
| `image_edit.py` | 100% |
| `_mflux_image_edit.py` | 91% |
| Both new modules | 97.27% |

The native adapter's uncovered lines import the actual optional MFLUX/MLX
classes; their real execution is represented by the separately reported Metal
smoke, rather than being counted as model-free coverage. The existing CI
coverage threshold remains 90%; no exclusions were added or weakened.

Behavioral checks cover parameter limits, image size/animation/format rejection,
checkpoint completeness and quantization, changed-weight fingerprints, prompt
hashing without prompt disclosure, staging cleanup, an output-creation race,
process-local overlap refusal, safe dependency/version errors, and structured
CLI success/cancellation. Runtime-adapter checks verify limits before loading,
retention of the loading memory peak, completed-step evaluation before events,
offline-environment restoration, and cleanup after an invalid loaded model or
cancelled request.

Ruff check and format verification pass for the two runtime modules and two
test files. `git diff --check` passes. Formatting changes are confined to these
new Python files.

## Full CI-style coverage gate

The fresh full repository run passed **2164 tests and 306 subtests**, with
**19 tests skipped**, in 33.58 seconds. It used native execution and the existing
coverage configuration:

```sh
python -m pytest tests/ -q --tb=short \
  --cov=mlx_flash_compress --cov-config=.coveragerc \
  --cov-report=term-missing --cov-fail-under=90
```

Repository coverage was **93.68%**, exceeding the unchanged 90% gate. The opt-in
image-edit real-model test was skipped in this run; the earlier enabled Metal
smoke remains separate evidence. Existing hardware/optional tests retained
their normal skips; this run is not a claim that every available model path
executed.

## Limits of the evidence

Neither file/hash validity, test coverage nor a completed smoke establishes
natural relighting, source-shadow recovery, reference-object fidelity or parity
with another image model. The 8 GiB allocation limit is for MLX, not total RAM,
RSS, swap or system memory pressure. Multiple resident models remain the
orchestrator's responsibility. CLI subprocess exit releases the Metal context;
the process-local lock does not coordinate independent processes.
