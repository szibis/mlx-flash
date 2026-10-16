"""Local inverse-compositing evidence; never alters a matte or a source image."""


def _box_sum(array, radius):
    import numpy as np

    padding = [(radius + 1, radius), (radius + 1, radius)] + [(0, 0)] * (array.ndim - 2)
    integral = np.pad(array, padding).cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]


def prepare(rgb, alpha):
    """Return uint8 RGBA B evidence (A=255 accepted) plus refusal counters."""
    import numpy as np

    height, width = alpha.shape
    radius = max(3, min(8, round(min(width, height) * 0.01)))
    known = (alpha == 0).astype(np.float64)
    count = _box_sum(known, radius)
    denominator = np.maximum(count, 1)[..., None]
    background = _box_sum(rgb * known[..., None], radius) / denominator
    second = _box_sum(rgb**2 * known[..., None], radius) / denominator
    deviation = np.sqrt(np.maximum(0, second - background**2)).max(axis=2)
    background = np.clip(np.rint(background), 0, 255)
    partial = (alpha > 0) & (alpha < 255)
    candidates = partial & (alpha >= 32)
    supported = (
        candidates
        & (count >= 4)
        & (deviation <= 6)
        & (background.min(axis=2) >= 96)
        & (np.ptp(background, axis=2) <= 48)
    )
    yy, xx = np.nonzero(supported)
    a = alpha[yy, xx, None] / 255
    recovered = (rgb[yy, xx] - (1 - a) * background[yy, xx]) / a
    physical = ((recovered >= -3) & (recovered <= 258)).all(axis=1)
    invalid_gamut = int((~physical).sum())
    yy, xx, recovered = yy[physical], xx[physical], recovered[physical]
    found = np.zeros(len(yy), dtype=bool)
    reference = np.zeros((len(yy), 3), dtype=np.float64)
    offsets = sorted(
        (dy * dy + dx * dx, dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy * dy + dx * dx <= radius * radius
    )
    for _, dy, dx in offsets:
        if found.all():
            break
        indices = np.nonzero(~found & (yy + dy >= 0) & (yy + dy < height) & (xx + dx >= 0) & (xx + dx < width))[0]
        indices = indices[alpha[yy[indices] + dy, xx[indices] + dx] == 255]
        reference[indices] = rgb[yy[indices] + dy, xx[indices] + dx]
        found[indices] = True
    agreement = np.max(np.abs(np.clip(recovered, 0, 255) - reference), axis=1) <= 32
    accepted = found & agreement
    basis = np.zeros((height, width, 4), dtype=np.uint8)
    basis[yy[accepted], xx[accepted], :3] = background[yy[accepted], xx[accepted]]
    basis[yy[accepted], xx[accepted], 3] = 255
    accepted_count = int(accepted.sum())
    stats = {
        "status": "estimated" if accepted_count else "skipped",
        "partial_pixels": int(partial.sum()),
        "supported_pixels": accepted_count,
        "alpha_changed_pixels": 0,
        "refused": {
            "low_alpha": int((partial & ~candidates).sum()),
            "local_background": int((candidates & ~supported).sum()),
            "foreground_gamut": invalid_gamut,
            "opaque_anchor_unavailable": int((~found).sum()),
            "opaque_anchor_mismatch": int((found & ~agreement).sum()),
        },
        "parameters": {
            "radius": radius,
            "background_alpha": 0,
            "foreground_alpha": 255,
            "minimum_partial_alpha": 32,
            "minimum_background_samples": 4,
            "background_std_max": 6,
            "background_channel_min": 96,
            "background_channel_spread_max": 48,
            "foreground_channel_tolerance": 3,
            "foreground_anchor_error_max": 32,
            "max_correction_per_channel": 64,
        },
        "quality_flags": ["glass_and_unresolved_edges_require_review"],
    }
    if accepted_count < int(partial.sum()):
        stats["quality_flags"].append("partial_edges_skipped_for_uncertainty")
    return basis, stats
