"""Known mixtures expose bright-backdrop contamination without changing mattes."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from mlx_flash_compress.studio_layers import StudioLayersError, recolor_studio_scene


def mixture_fixture():
    height, width = 128, 160
    background = np.zeros((height, width, 3), dtype=float) + [240, 242, 238]
    foreground = np.zeros_like(background)
    alpha = np.zeros((height, width), dtype=np.uint8)
    # Dark narrow line, bright petal/rim, saturated material. Each has known
    # opaque support and fractional coverage, including one-pixel details.
    for x0, x1, color in ((30, 35, [30, 65, 95]), (70, 90, [248, 244, 238]), (115, 135, [150, 40, 180])):
        foreground[20:108, x0 - 1 : x1 + 2] = color
        alpha[20:108, x0 : x1 + 1] = 255
        alpha[20:108, x0 - 1] = 128
        alpha[20:108, x1 + 1] = 64
    a = alpha[..., None] / 255
    scene = np.rint(a * foreground + (1 - a) * background).astype(np.uint8)
    matte = np.concatenate((foreground.astype(np.uint8), alpha[..., None]), axis=2)
    return Image.fromarray(scene), Image.fromarray(matte), foreground, alpha


@pytest.mark.parametrize("target", [(24, 36, 30), (250, 240, 220), (150, 35, 220)])
def test_local_unmix_recovers_known_fractional_edges_on_three_backgrounds(target):
    scene, matte, foreground, alpha = mixture_fixture()
    before, mask_before = scene.tobytes(), matte.tobytes()
    result, receipt = recolor_studio_scene(scene, matte, target, edge_mode="local_unmix")
    a = alpha[..., None] / 255
    expected = np.rint(a * foreground + (1 - a) * np.asarray(target)).astype(np.uint8)
    # Avoid the deliberately abrupt ends of the synthetic strips.
    edge = (alpha > 0) & (alpha < 255)
    edge[:25] = False
    edge[103:] = False
    error = np.abs(np.asarray(result).astype(int) - expected.astype(int))[edge]
    assert error.max() <= 1
    assert receipt["edge_correction"]["corrected_pixels"] > 0
    assert receipt["edge_mode"] == "local_unmix"
    assert np.array_equal(np.asarray(result)[alpha == 255], np.asarray(scene)[alpha == 255])
    assert scene.tobytes() == before and matte.tobytes() == mask_before


def test_local_unmix_reduces_dark_target_error_instead_of_blurring_or_eroding():
    scene, matte, foreground, alpha = mixture_fixture()
    target = (24, 36, 30)
    old, _ = recolor_studio_scene(scene, matte, target)
    new, receipt = recolor_studio_scene(scene, matte, target, edge_mode="local_unmix")
    a = alpha[..., None] / 255
    expected = np.rint(a * foreground + (1 - a) * target).astype(np.uint8)
    y, x = 64, 29  # One-pixel edge beside the dark line.
    assert np.abs(np.asarray(old)[y, x].astype(int) - expected[y, x]).max() > 25
    assert np.abs(np.asarray(new)[y, x].astype(int) - expected[y, x]).max() <= 1
    assert receipt["edge_correction"]["alpha_changed_pixels"] == 0
    assert receipt["visual_quality_approved"] is False


def test_zero_alpha_scene_shading_is_identical_to_legacy_recolor():
    scene, matte, _, alpha = mixture_fixture()
    rgb = np.array(scene)
    rgb[112:120, 50:110] = [175, 176, 173]  # Existing rendered shadow, outside matte.
    scene = Image.fromarray(rgb)
    legacy, _ = recolor_studio_scene(scene, matte, (42, 57, 47))
    corrected, _ = recolor_studio_scene(scene, matte, (42, 57, 47), edge_mode="local_unmix")
    assert np.array_equal(np.asarray(legacy)[alpha == 0], np.asarray(corrected)[alpha == 0])


@pytest.mark.parametrize("target", [(24, 36, 30), (250, 240, 220), (150, 35, 220)])
def test_enclosed_glass_and_nonphysical_glints_are_skipped_not_claimed_fixed(target):
    scene, matte, _, _ = mixture_fixture()
    rgb, rgba = np.array(scene), np.array(matte)
    rgba[30:95, 45:65, 3] = 128
    rgb[30:95, 45:65] = [225, 227, 224]
    rgb[30, 45] = [255, 255, 255]  # Inverse mixture is out of gamut.
    scene, matte = Image.fromarray(rgb), Image.fromarray(rgba)
    old, _ = recolor_studio_scene(scene, matte, target)
    new, receipt = recolor_studio_scene(scene, matte, target, edge_mode="local_unmix")
    for point in ((55, 60), (45, 30)):
        assert new.getpixel(point) == old.getpixel(point)
    assert "glass_and_unresolved_edges_require_review" in receipt["edge_correction"]["quality_flags"]


def test_no_opaque_anchor_refuses_unmix_for_an_isolated_translucent_line():
    scene, matte, _, _ = mixture_fixture()
    rgb, rgba = np.array(scene), np.array(matte)
    rgba[..., 3] = 0
    rgba[20:108, 50, 3] = 128
    rgb[:] = [240, 242, 238]
    rgb[20:108, 50] = [140, 142, 138]
    scene, matte = Image.fromarray(rgb), Image.fromarray(rgba)
    old, _ = recolor_studio_scene(scene, matte, (42, 57, 47))
    new, receipt = recolor_studio_scene(scene, matte, (42, 57, 47), edge_mode="local_unmix")
    assert new.tobytes() == old.tobytes()
    assert receipt["edge_correction"]["status"] == "skipped"
    assert receipt["edge_correction"]["corrected_pixels"] == 0


def test_heterogeneous_local_background_is_refused_without_global_fit_rejection():
    scene, matte, _, _ = mixture_fixture()
    rgb = np.array(scene)
    alpha = np.asarray(matte)[..., 3]
    yy, xx = np.mgrid[: scene.height, : scene.width]
    rgb[(alpha == 0) & ((xx + yy) % 2 == 0)] = [190, 192, 188]
    scene = Image.fromarray(rgb)
    old, _ = recolor_studio_scene(scene, matte, (42, 57, 47))
    new, receipt = recolor_studio_scene(scene, matte, (42, 57, 47), edge_mode="local_unmix")
    assert new.tobytes() == old.tobytes()
    assert receipt["edge_correction"]["corrected_pixels"] == 0


def test_prepared_edge_evidence_is_target_independent_and_hash_bound():
    from mlx_flash_compress.studio_layers import prepare_studio_edge_correction

    scene, matte, _, alpha = mixture_fixture()
    basis, receipt = prepare_studio_edge_correction(scene, matte)
    assert basis.mode == "RGBA" and basis.size == scene.size
    assert basis.getpixel((29, 64)) == (240, 242, 238, 255)
    assert np.count_nonzero(np.asarray(basis)[alpha == 0, 3]) == 0
    assert np.count_nonzero(np.asarray(basis)[alpha == 255, 3]) == 0
    assert receipt["scene_pixels_sha256"] == hashlib.sha256(scene.tobytes()).hexdigest()
    assert receipt["matte_pixels_sha256"] == hashlib.sha256(matte.tobytes()).hexdigest()


def test_reference_target_is_exact_identity_including_partial_edges():
    scene, matte, _, _ = mixture_fixture()
    _, original_receipt = recolor_studio_scene(scene, matte, [24, 24, 24])
    result, receipt = recolor_studio_scene(scene, matte, original_receipt["reference_rgb"], edge_mode="local_unmix")
    assert result.tobytes() == scene.tobytes()
    assert receipt["edge_correction"]["corrected_pixels"] == 0
    assert receipt["edge_correction"]["eligible_pixels"] > 0


def test_matte_rgb_never_substitutes_scene_rgb_for_edge_estimation():
    scene, matte, _, _ = mixture_fixture()
    different = np.array(matte)
    different[..., :3] = [255, 0, 123]
    first, _ = recolor_studio_scene(scene, matte, [24, 36, 30], edge_mode="local_unmix")
    second, _ = recolor_studio_scene(scene, Image.fromarray(different), [24, 36, 30], edge_mode="local_unmix")
    assert first.tobytes() == second.tobytes()


@pytest.mark.parametrize("failure", ["mode", "alignment", "small", "large", "type"])
def test_prepared_edge_evidence_rejects_invalid_inputs(failure):
    from mlx_flash_compress.studio_layers import prepare_studio_edge_correction

    scene, matte, _, _ = mixture_fixture()
    if failure == "mode":
        scene = scene.convert("RGBA")
    elif failure == "alignment":
        matte = matte.resize((64, 64))
    elif failure == "small":
        scene, matte = scene.resize((31, 40)), matte.resize((31, 40))
    elif failure == "large":
        scene, matte = scene.resize((1601, 32)), matte.resize((1601, 32))
    else:
        scene = "untrusted-path-is-not-a-PIL-image"
    with pytest.raises(StudioLayersError) as error:
        prepare_studio_edge_correction(scene, matte)
    assert error.value.code == "invalid_request"


def test_no_partial_pixels_has_no_edge_corrections():
    scene, matte, _, _ = mixture_fixture()
    matte = np.array(matte)
    matte[..., 3] = np.where(matte[..., 3] >= 128, 255, 0)
    matte = Image.fromarray(matte)
    old, _ = recolor_studio_scene(scene, matte, [24, 36, 30])
    result, receipt = recolor_studio_scene(scene, matte, [24, 36, 30], edge_mode="local_unmix")
    assert result.tobytes() == old.tobytes()
    assert receipt["edge_correction"]["partial_pixels"] == 0
    assert receipt["edge_correction"]["status"] == "skipped"


def test_correction_bound_keeps_default_for_a_gamut_tolerance_edge():
    # Slightly negative recovered F is within the rounding tolerance but must
    # not bypass the independent maximum-correction gate on a black target.
    rgb = np.full((128, 160, 3), 255, dtype=np.uint8)
    alpha = np.zeros((128, 160), dtype=np.uint8)
    rgb[20:108, 30:36] = 0
    alpha[20:108, 30:36] = 255
    rgb[20:108, 29] = 126
    alpha[20:108, 29] = 128
    scene = Image.fromarray(rgb)
    matte = Image.fromarray(np.concatenate((rgb, alpha[..., None]), axis=2))
    default, _ = recolor_studio_scene(scene, matte, [0, 0, 0])
    result, receipt = recolor_studio_scene(scene, matte, [0, 0, 0], edge_mode="local_unmix")
    assert result.tobytes() == default.tobytes()
    assert receipt["edge_correction"]["correction_limit_refused_pixels"] > 0
    assert "correction_bound_exceeded" in receipt["edge_correction"]["quality_flags"]


@pytest.mark.parametrize("mode", [True, "blur", {}, None])
def test_invalid_edge_modes_are_not_silently_applied(mode):
    scene, matte, _, _ = mixture_fixture()
    with pytest.raises(StudioLayersError) as error:
        recolor_studio_scene(scene, matte, (42, 57, 47), edge_mode=mode)
    assert error.value.code == "invalid_request"
