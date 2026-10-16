# Experimental studio-scene layers

## Live background color without shadow extraction

For a generated RGB scene whose existing shading should remain visible,
`recolor_studio_scene(scene_rgb, foreground_rgba, target_rgb)` changes backdrop
color through a supplied aligned matte. It returns a new RGB image and receipt.
This path does not require successful shadow decomposition: it keeps the scene's
spatial shading and already rendered shadow in the RGB image.

```python
from mlx_flash_compress.studio_layers import recolor_studio_scene

recolored, receipt = recolor_studio_scene(scene, matte, [100, 160, 220])
```

The target is three integer channels from 0 to 255. The reference is the rounded
median color of illuminated neutral backdrop samples outside a dilated matte.
Insufficient neutral/light samples are refused. Reference channels are at least
128, bounding gains below 2. Opaque **scene RGB** pixels remain byte-identical;
the matte's RGB is not used to reconstruct or replace them. Partial alpha blends
the adjustment, so existing matte errors and glass contamination remain relevant.

The receipt exposes integer `reference_rgb`, `target_rgb`, `channel_gains` and
method `display-rgb-alpha-weighted-channel-gains-v1`. For each channel:

```text
gain = target / reference
output = clamp(round_to_even(scene * (alpha + (1 - alpha) * gain)), 0, 255)
```

Alpha is the supplied matte byte divided by 255. A live browser preview can
apply the same arithmetic directly into a `Uint8ClampedArray`, whose conversion
rounds ties to even. Use the original scene pixels for every update to avoid
compounding corrections:

```javascript
const gains = target.map((value, c) => value / receipt.reference_rgb[c]);
const output = new Uint8ClampedArray(scenePixels);
for (let i = 0; i < output.length; i += 4) {
  const alpha = mattePixels[i + 3] / 255;
  for (let c = 0; c < 3; c++) {
    output[i + c] = scenePixels[i + c] * (alpha + (1 - alpha) * gains[c]);
  }
}
```

This is an approximate color target relative to an illuminated whitepoint, not
a promise that every backdrop pixel equals the target hex value. Existing
gradients remain, subject to rounding and channel clipping; black hides visible
shadow contrast. It does not create new illumination or restore a photographed
product if the generated scene already changed it.

The receipt records decoded scene/matte/output pixel hashes,
`provenance: masked_scene_background_recolor`, `alignment_required: true`,
`alignment_verified: false`, `opaque_scene_pixels_preserved: true`,
`foreground_rgb_reconstructed: false`, `shadow_extraction_performed: false`
and `visual_quality_approved: false`. These fields distinguish recoloring an
existing scene from extracting a transparent shadow layer.

## Conservative transparent layer decomposition

`decompose_studio_scene` separates a supported light, neutral RGB studio scene
into a caller-supplied foreground and an estimated black RGBA shadow. The layers
can be composed over arbitrary live backdrop colors without repainting the
foreground. It does not run a model or modify the supplied foreground pixels.

This is a conservative experimental API. It works on controlled smooth scenes;
many generated scenes and imperfect mattes are refused. It does not establish
exact shadow segmentation, physical relighting, glass fidelity, or production
visual acceptance.

```python
from PIL import Image
from mlx_flash_compress.studio_layers import decompose_studio_scene, StudioLayersError

scene = Image.open("generated-scene.png")       # RGB
matte = Image.open("aligned-foreground.png")    # RGBA, same size and geometry
try:
    foreground, shadow, receipt = decompose_studio_scene(scene, matte)
except StudioLayersError as error:
    # Safe UI/orchestrator fields: no image contents or filesystem paths.
    show_unavailable(error.code, error.stage, error.reason)
else:
    editable = Image.alpha_composite(shadow, foreground)
    editable.save("editable.png")
```

The resulting transparent PNG can sit above a CSS background controlled by a
color picker. Alternatively, use `Image.alpha_composite` to place the same
shadow and foreground above another RGBA background. Foreground RGB and alpha
are copied exactly, including partially transparent pixels and hidden RGB.
No product recoloring or global RGB repaint is performed. A black alpha shadow
darkens the selected backdrop; its estimated shape remains fixed as colors
change. Arbitrary backdrop colors do not mean arbitrary new light sources.

## Input and estimation contract

Inputs must be PIL images in RGB and RGBA modes, respectively, with identical
dimensions, 32–1600 pixels per side. NumPy and Pillow are imported lazily; no
MLX, MFLUX or model weights are needed. The API performs no file writes.

The caller must supply a correctly aligned foreground, excluding the backdrop
and shadow. Equal image dimensions do not prove alignment or correct
segmentation. The function cannot repair omitted accessories, a misplaced
matte, old backdrop color inside glass, or missing transparent material detail.

The estimator dilates the foreground before selecting distributed backdrop
samples. It fits a smooth quadratic RGB surface, iteratively excluding dark
outliers that could be shadows. Sample coverage, neutral color, remaining lit
sample count, residual error and predicted surface bounds must pass. Only
observed attenuation in a lower region, connected to the actual lower matte,
can become a shadow. Detached dark marks, ambiguous colors/density and excess
darkness outside that region are refused. Feathering normalizes around missing
foreground evidence so the excluded object interior does not dilute contact.

No ellipse, projected silhouette, synthetic drop shadow or fallback shadow is
drawn. These are display-RGB attenuation estimates, not calibrated physical
light transport. A low hanging component can make inferred support ambiguous;
such geometry needs review rather than automatic approval.

## Receipt and refusal

Successful returns include a JSON-safe receipt with pixel hashes, dimensions,
sample counts and fit errors. It records:

- `provenance: generated_scene_attenuation_estimate`;
- `original_photographic_shadow: false`;
- `foreground_preserved: true`;
- `alignment_required: true` and `alignment_verified: false`;
- `visual_quality_approved: false`;
- `status: estimated` or `none`, with `shadow_pixels`.

Hashes cover decoded raster bytes; the shared dimensions and image modes form
part of this API's contract. They are not hashes of original encoded files.
`status: none` means the supported scene contained no accepted shadow evidence;
the API does not synthesize grounding to replace it.

Invalid modes, types or dimensions raise `StudioLayersError` with code
`invalid_request`. Uncertain decomposition raises code `needs_review` with no
output layers. Stable `stage` and `reason` fields identify the gate:

| Stage | Reasons |
| --- | --- |
| `foreground` | `missing_foreground_support` |
| `background_samples` | `insufficient_background_samples`, `nonneutral_background`, `background_reference_unavailable` |
| `surface_fit` | `insufficient_lit_samples`, `surface_fit_error`, `surface_extrapolation` |
| `shadow_support` | `darkness_outside_support`, `disconnected_shadow` |
| `shadow_density` | `ambiguous_shadow_density` |

On refusal, keep the full RGB scene or present an honest unavailable state.
Removing the shadow, bypassing a gate, or drawing a substitute does not establish
successful decomposition. Shadow extraction from an original photograph is a
different operation and must not reuse this generated-scene provenance claim.

## Verification

Synthetic tests cover a smooth illumination gradient with known attenuation,
retained contact density, no-shadow behavior around a dark object, immutable
foreground/source pixels, independent backdrop colors, heterogeneity, detached
marks, excessive density, invalid input bounds and safe refusal reasons.
They do not substitute for visual assessment of real generated scenes.
Recolor tests independently cover reference-color identity, exact opaque source
pixels, source/matte immutability, retained shadow contrast for darker targets,
the reported alpha/gain formula, black, invalid colors and unavailable reference
samples.

```sh
python -m pytest tests/test_studio_layers.py \
  --cov=mlx_flash_compress.studio_layers --cov-report=term-missing --cov-fail-under=90 -q
```
