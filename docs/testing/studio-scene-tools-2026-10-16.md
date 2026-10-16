# Studio-scene tools — verification, 2026-10-16

This proof covers two independent CPU APIs in `studio_layers.py`:

- `recolor_studio_scene` applies alpha-weighted channel gains to the existing
  scene backdrop, preserving opaque scene pixels and existing spatial shading.
- `decompose_studio_scene` estimates transparent attenuation only for a supported
  light neutral smooth scene. Uncertain scenes fail with a stable stage/reason;
  they do not receive a fabricated substitute shadow.

Neither API performs inference, registers images, reconstructs foreground RGB or
establishes visual quality. Recoloring does not assert shadow extraction.
Decomposition records `generated_scene_attenuation_estimate`, never original
photographic-shadow provenance. See [the API guide](../studio-layers.md).

## Synthetic verification

**28 tests passed**, with **98.28%** statement coverage of the new module:

```sh
python -m pytest tests/test_studio_layers.py \
  --cov=mlx_flash_compress.studio_layers --cov-report=term-missing \
  --cov-fail-under=90 -q
```

The controlled fixture includes a smooth backdrop illumination field, a known
attenuation field, a dark object, and partial-alpha pixels. Tests verify:

- immutable scene/matte data and unchanged returned foreground bytes;
- known shadow density and contact, no-shadow behavior and black layer RGB;
- refusal of heterogeneous/dark/colored backdrops, detached marks, excessive
  density, missing support and invalid types/dimensions;
- stable safe `needs_review` stage/reason fields;
- arbitrary target colors including black, identity at the reported reference,
  exact opaque scene RGB, and source/matte immutability;
- retained shadow contrast for a darker target and exact documented gain/alpha
  arithmetic at partial alpha;
- no use of matte RGB to reconstruct scene pixels and rejection of unavailable
  neutral/light reference samples.

A contact-density test first failed because ordinary blur treated the excluded
object interior as zero attenuation. Normalized feathering corrected that
specific failure. No decomposition tolerances were loosened to force difficult
application scenes through the estimator.

Ruff check and format verification pass for the new module and tests.
`git diff --check` passes. Dependencies remain lazy NumPy/Pillow imports; no MLX
or image-model session is needed for these APIs.

## Repository regression

The native full repository run passed **2192 tests and 306 subtests**, with
**19 tests skipped**, in 45.46 seconds. Coverage was **93.76%**, above the
unchanged 90% gate:

```sh
python -m pytest tests/ -q --tb=short \
  --cov=mlx_flash_compress --cov-config=.coveragerc \
  --cov-report=term-missing --cov-fail-under=90
```

This run did not enable the optional image-model smoke. Existing hardware and
optional checks retained their normal skips. Application-specific images,
comparison metrics and visual-review evidence remain in the consuming
application, outside this generic proof.

## Acceptance limits

Same dimensions do not prove correct alignment. Matte errors can affect edges,
glass and omitted components. A low hanging component can invalidate inferred
floor contact. Recolor targets are approximate relative to a sampled illuminated
reference; clipping can flatten highlights, and black hides shadow contrast.
No perfect matting, exact shadow recovery, physical relighting, commercial-model
parity or M4/16 GB performance is claimed.
