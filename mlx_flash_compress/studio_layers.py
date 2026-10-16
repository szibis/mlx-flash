"""Experimental, conservative attenuation layers for aligned neutral studio scenes."""

import hashlib
from collections import deque


class StudioLayersError(ValueError):
    def __init__(self, code, message, *, stage="validation", reason="invalid_request"):
        self.code = code
        self.stage, self.reason = stage, reason
        super().__init__(message)


def _review(reason, stage):
    raise StudioLayersError(
        "needs_review",
        "Studio backdrop or shadow evidence is uncertain; review the full scene.",
        stage=stage,
        reason=reason,
    )


def _connected(candidate, seeds):
    """Keep observed components touching the lower foreground, never its silhouette."""
    import numpy as np

    height, width = candidate.shape
    selected = np.zeros_like(candidate)
    queue = deque(zip(*np.nonzero(candidate & seeds)))
    for y, x in queue:
        selected[y, x] = True
    while queue:
        y, x = queue.popleft()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < height and 0 <= xx < width and candidate[yy, xx] and not selected[yy, xx]:
                selected[yy, xx] = True
                queue.append((yy, xx))
    return selected


def decompose_studio_scene(scene_rgb, foreground_rgba):
    """Return (unchanged foreground copy, black RGBA shadow, JSON-safe receipt).

    Both PIL images must already share exact geometry. Equal dimensions do not
    establish semantic alignment. Only light, neutral, smoothly illuminated
    studio backgrounds are supported; uncertain scenes raise ``needs_review``.
    No model, file I/O, registration, object edit or shadow synthesis occurs.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    if (
        not isinstance(scene_rgb, Image.Image)
        or not isinstance(foreground_rgba, Image.Image)
        or scene_rgb.mode != "RGB"
        or foreground_rgba.mode != "RGBA"
        or scene_rgb.size != foreground_rgba.size
        or min(scene_rgb.size) < 32
        or max(scene_rgb.size) > 1600
    ):
        raise StudioLayersError("invalid_request", "Provide aligned RGB/RGBA PIL images, 32–1600 pixels per side.")
    width, height = scene_rgb.size
    rgb = np.asarray(scene_rgb, dtype=np.float32)
    alpha = np.asarray(foreground_rgba.getchannel("A"))
    rows, columns = np.nonzero(alpha >= 192)
    if not len(rows):
        _review("missing_foreground_support", "foreground")
    left, right, top, base = int(columns.min()), int(columns.max()), int(rows.min()), int(rows.max())
    radius = max(1, min(16, round(min(width, height) * 0.01)))
    expanded = np.asarray(foreground_rgba.getchannel("A").filter(ImageFilter.MaxFilter(radius * 2 + 1))) > 2
    yy, xx = np.mgrid[:height, :width]
    stride = max(1, int(np.ceil(np.sqrt(width * height / 4096))))
    sample = (~expanded) & (yy % stride == 0) & (xx % stride == 0)
    sy, sx = np.nonzero(sample)
    values = rgb[sample].astype(np.float64)
    if (
        len(values) < 128
        or np.ptp(sx) < width * 0.6
        or np.ptp(sy) < height * 0.6
        or np.count_nonzero(sy > base + radius) < 20
    ):
        _review("insufficient_background_samples", "background_samples")
    if np.max(np.ptp(values, axis=1)) > 24:
        _review("nonneutral_background", "background_samples")
    x, y = sx / width, sy / height
    design = np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))
    keep = np.ones(len(values), dtype=bool)
    for _ in range(6):
        coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        residual = np.median(values - design @ coefficients, axis=1)
        center = np.median(residual[keep])
        sigma = 1.4826 * np.median(np.abs(residual[keep] - center))
        keep &= residual >= center - max(2.0, 3 * sigma)
    if keep.sum() < max(128, 0.7 * len(keep)):
        _review("insufficient_lit_samples", "surface_fit")
    coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
    error = np.max(np.abs(values[keep] - design[keep] @ coefficients), axis=1)
    if np.percentile(error, 95) > 3 or error.max() > 8 or values[keep].min() < 128:
        _review("surface_fit_error", "surface_fit")
    x, y = xx[..., None] / width, yy[..., None] / height
    lit = (
        coefficients[0]
        + x * coefficients[1]
        + y * coefficients[2]
        + x * x * coefficients[3]
        + x * y * coefficients[4]
        + y * y * coefficients[5]
    )
    if lit.min() < 120 or lit.max() > 260 or np.max(np.ptp(lit, axis=2)) > 24:
        _review("surface_extrapolation", "surface_fit")
    ratios = rgb / np.maximum(lit, 1)
    density = np.clip(1 - ratios.mean(axis=2), 0, 1)
    known = alpha <= 2
    candidate = known & (density > 0.012)
    margin = max((right - left + 1) * 0.9, width * 0.18)
    region = (
        (xx >= left - margin)
        & (xx <= right + margin)
        & (yy >= base - max(4, (base - top + 1) * 0.2))
        & (yy <= base + max(height * 0.12, (base - top + 1) * 0.55))
    )
    tolerance = max(16, width * height * 0.001)
    if np.count_nonzero(candidate & ~region) > tolerance:
        _review("darkness_outside_support", "shadow_support")
    if np.any(candidate & region & ((np.ptp(ratios, axis=2) > 0.04) | (density > 0.65))):
        _review("ambiguous_shadow_density", "shadow_density")
    seeds = expanded & (yy >= base - max(radius, 4)) & (yy <= base + radius)
    selected = _connected(candidate & region, seeds)
    if np.count_nonzero(candidate & region & ~selected) > tolerance:
        _review("disconnected_shadow", "shadow_support")
    # Feather measured attenuation only; no geometry-derived shadow is drawn.
    observed = Image.fromarray(np.rint(np.where(selected, density, 0) * 255).astype(np.uint8))
    feather = ImageFilter.GaussianBlur(max(0.5, min(2, min(width, height) * 0.004)))
    softened = np.asarray(observed.filter(feather), dtype=np.float32)
    weights = np.asarray(Image.fromarray(known.astype(np.uint8) * 255).filter(feather), dtype=np.float32) / 255
    # The object's excluded interior is unknown, not a zero-density floor.
    # Normalization keeps that missing evidence from diluting measured contact.
    softened = softened / np.maximum(weights, 1 / 255)
    shadow_alpha = np.clip(np.rint(np.where(known & region, softened, 0)), 0, 255).astype(np.uint8)
    shadow = Image.new("RGBA", scene_rgb.size)
    shadow.putalpha(Image.fromarray(shadow_alpha))
    pixels = int(np.count_nonzero(shadow_alpha))
    receipt = {
        "version": "studio-layers-v1",
        "width": width,
        "height": height,
        "provenance": "generated_scene_attenuation_estimate",
        "status": "estimated" if pixels else "none",
        "alignment_required": True,
        "alignment_verified": False,
        "foreground_preserved": True,
        "visual_quality_approved": False,
        "original_photographic_shadow": False,
        "scene_pixels_sha256": hashlib.sha256(scene_rgb.tobytes()).hexdigest(),
        "foreground_pixels_sha256": hashlib.sha256(foreground_rgba.tobytes()).hexdigest(),
        "shadow_pixels_sha256": hashlib.sha256(shadow.tobytes()).hexdigest(),
        "shadow_pixels": pixels,
        "fit": {
            "model": "quadratic-display-rgb",
            "samples": int(len(values)),
            "lit_samples": int(keep.sum()),
            "residual_p95": float(np.percentile(error, 95)),
            "residual_max": float(error.max()),
            "excluded_fraction": float(1 - keep.mean()),
        },
    }
    return foreground_rgba.copy(), shadow, receipt


def prepare_studio_edge_correction(scene_rgb, foreground_rgba):
    """Return a target-independent local background evidence map and receipt.

    Map RGB stores a bounded B estimate and A=255 marks supported partial edge
    pixels only. Alpha is confidence, not a replacement foreground matte.
    """
    import numpy as np
    from PIL import Image

    from ._studio_edges import prepare

    if (
        not isinstance(scene_rgb, Image.Image)
        or not isinstance(foreground_rgba, Image.Image)
        or scene_rgb.mode != "RGB"
        or foreground_rgba.mode != "RGBA"
        or scene_rgb.size != foreground_rgba.size
        or min(scene_rgb.size) < 32
        or max(scene_rgb.size) > 1600
    ):
        raise StudioLayersError("invalid_request", "Provide aligned RGB/RGBA PIL images, 32–1600 pixels per side.")
    pixels, stats = prepare(np.asarray(scene_rgb, dtype=np.float64), np.asarray(foreground_rgba.getchannel("A")))
    basis = Image.fromarray(pixels)
    receipt = {
        "version": "studio-edge-basis-v1",
        "method": "local-inverse-compositing-evidence-v1",
        "provenance": "local_scene_background_estimate",
        "width": scene_rgb.width,
        "height": scene_rgb.height,
        "alignment_required": True,
        "alignment_verified": False,
        "visual_quality_approved": False,
        **stats,
        "scene_pixels_sha256": hashlib.sha256(scene_rgb.tobytes()).hexdigest(),
        "matte_pixels_sha256": hashlib.sha256(foreground_rgba.tobytes()).hexdigest(),
        "basis_pixels_sha256": hashlib.sha256(basis.tobytes()).hexdigest(),
    }
    return basis, receipt


def recolor_studio_scene(scene_rgb, foreground_rgba, target_rgb, *, edge_mode="none"):
    """Recolor the masked backdrop while retaining scene geometry and darkness.

    This does not extract a shadow or replace foreground RGB. The supplied
    matte must already be aligned. ``local_unmix`` optionally replaces a
    supported partial edge's estimated original backdrop contribution only.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    if not isinstance(edge_mode, str) or edge_mode not in {"none", "local_unmix"}:
        raise StudioLayersError("invalid_request", "Choose an available edge correction mode.")

    if (
        not isinstance(scene_rgb, Image.Image)
        or not isinstance(foreground_rgba, Image.Image)
        or scene_rgb.mode != "RGB"
        or foreground_rgba.mode != "RGBA"
        or scene_rgb.size != foreground_rgba.size
        or min(scene_rgb.size) < 32
        or max(scene_rgb.size) > 1600
    ):
        raise StudioLayersError("invalid_request", "Provide aligned RGB/RGBA PIL images, 32–1600 pixels per side.")
    if (
        not isinstance(target_rgb, (tuple, list))
        or len(target_rgb) != 3
        or any(type(value) is not int or not 0 <= value <= 255 for value in target_rgb)
    ):
        raise StudioLayersError("invalid_request", "Target color must contain three integers between 0 and 255.")
    width, height = scene_rgb.size
    rgb = np.asarray(scene_rgb, dtype=np.float64)
    alpha = np.asarray(foreground_rgba.getchannel("A"), dtype=np.float64) / 255
    radius = max(1, min(16, round(min(width, height) * 0.01)))
    outside = np.asarray(foreground_rgba.getchannel("A").filter(ImageFilter.MaxFilter(radius * 2 + 1))) <= 2
    stride = max(1, int(np.ceil(np.sqrt(width * height / 8192))))
    samples = rgb[::stride, ::stride][outside[::stride, ::stride]]
    neutral = samples[(np.ptp(samples, axis=1) <= 24) & (samples.min(axis=1) >= 128)]
    if len(neutral) < max(128, 0.5 * len(samples)):
        _review("background_reference_unavailable", "background_samples")
    lightness = neutral @ np.array([0.2126, 0.7152, 0.0722])
    illuminated = neutral[lightness >= np.percentile(lightness, 75)]
    reference = np.rint(np.median(illuminated, axis=0)).astype(int)
    gains = np.asarray(target_rgb, dtype=np.float64) / reference
    # Reference channels are >=128, so every legal target has gain <=2.
    # Apply exactly the same arithmetic/order as the documented browser formula.
    base = rgb * (alpha[..., None] + (1 - alpha[..., None]) * gains)
    edge_receipt = None
    if edge_mode == "local_unmix":
        basis, edge_receipt = prepare_studio_edge_correction(scene_rgb, foreground_rgba)
        evidence = np.asarray(basis)
        # C=aF+(1-a)B -> C'=C+(1-a)(gain-1)B. No F reconstruction
        # or alpha change is needed to replace just the original B contribution.
        unmixed = rgb + (1 - alpha[..., None]) * (gains - 1) * evidence[..., :3]
        allowed = (evidence[..., 3] == 255) & (np.max(np.abs(unmixed - base), axis=2) <= 64)
        edge_receipt["eligible_pixels"] = int(allowed.sum())
        changed = np.any(np.clip(np.rint(unmixed), 0, 255) != np.clip(np.rint(base), 0, 255), axis=2)
        edge_receipt["corrected_pixels"] = int((allowed & changed).sum())
        edge_receipt["correction_limit_refused_pixels"] = int(((evidence[..., 3] == 255) & ~allowed).sum())
        if edge_receipt["correction_limit_refused_pixels"]:
            edge_receipt["quality_flags"].append("correction_bound_exceeded")
        base = np.where(allowed[..., None], unmixed, base)
    corrected = np.clip(np.rint(base), 0, 255).astype(np.uint8)
    output = Image.fromarray(corrected)
    receipt = {
        "version": "studio-background-recolor-v1",
        "width": width,
        "height": height,
        "method": "display-rgb-alpha-weighted-channel-gains-v1",
        "provenance": "masked_scene_background_recolor",
        "reference_rgb": reference.tolist(),
        "reference_method": "neutral-light-upper-quartile-median-v1",
        "target_rgb": list(target_rgb),
        "channel_gains": gains.tolist(),
        "channel_gain_limit": 2.0,
        "neutral_light_samples": int(len(neutral)),
        "illuminated_samples": int(len(illuminated)),
        "alignment_required": True,
        "alignment_verified": False,
        "matte_required": True,
        "opaque_scene_pixels_preserved": True,
        "foreground_rgb_reconstructed": False,
        "shadow_extraction_performed": False,
        "visual_quality_approved": False,
        "scene_pixels_sha256": hashlib.sha256(scene_rgb.tobytes()).hexdigest(),
        "matte_pixels_sha256": hashlib.sha256(foreground_rgba.tobytes()).hexdigest(),
        "output_pixels_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
    }
    if edge_receipt is not None:
        receipt.update(
            version="studio-background-recolor-v2",
            method="display-rgb-local-edge-unmix-v1",
            edge_mode=edge_mode,
            edge_correction=edge_receipt,
        )
    return output, receipt
