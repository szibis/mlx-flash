"""Synthetic evidence for conservative scene attenuation, not visual acceptance."""

import importlib

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def api():
    try:
        return importlib.import_module("mlx_flash_compress.studio_layers")
    except ModuleNotFoundError:
        pytest.fail("Studio scene decomposition is missing")


def scene_fixture(shadow=True):
    width, height = 240, 240
    y, x = np.mgrid[:height, :width]
    lit = 230 + 5 * x / width - 7 * y / height + 3 * (x / width) ** 2
    density = 0.34 * np.exp(-(((x - 120) / 46) ** 2) - ((y - 165) / 13) ** 2) if shadow else np.zeros_like(lit)
    rgb = np.repeat(np.rint(lit * (1 - density))[..., None], 3, axis=2).astype(np.uint8)
    foreground = np.zeros((height, width, 4), dtype=np.uint8)
    foreground[55:161, 91:149] = [28, 36, 44, 255]
    foreground[70:120, 90] = [40, 70, 90, 128]
    occupied = foreground[..., 3] > 0
    rgb[occupied] = foreground[occupied, :3]
    return Image.fromarray(rgb), Image.fromarray(foreground), density


def test_recovers_observed_shadow_and_preserves_foreground_exactly(api):
    scene, foreground, density = scene_fixture()
    before_scene, before_foreground = scene.tobytes(), foreground.tobytes()
    kept, shadow, receipt = api.decompose_studio_scene(scene, foreground)
    alpha = np.asarray(shadow)[..., 3]
    assert kept is not foreground and kept.tobytes() == before_foreground
    assert scene.tobytes() == before_scene and foreground.tobytes() == before_foreground
    assert shadow.mode == "RGBA" and shadow.size == scene.size
    assert np.count_nonzero(np.asarray(shadow)[..., :3]) == 0
    assert alpha[170, 120] == pytest.approx(density[170, 120] * 255, abs=7)
    assert alpha[161, 120] == pytest.approx(density[161, 120] * 255, abs=7)
    assert alpha[60, 40] == 0 and alpha[100, 120] == 0
    assert receipt["provenance"] == "generated_scene_attenuation_estimate"
    assert receipt["alignment_required"] is True and receipt["alignment_verified"] is False
    assert receipt["visual_quality_approved"] is False
    assert receipt["status"] == "estimated" and receipt["shadow_pixels"] > 0
    assert receipt["fit"]["lit_samples"] >= 128
    assert receipt["fit"]["residual_p95"] < 3


def test_no_shadow_does_not_create_grounding_under_dark_object(api):
    scene, foreground, _ = scene_fixture(shadow=False)
    kept, shadow, receipt = api.decompose_studio_scene(scene, foreground)
    assert shadow.getchannel("A").getbbox() is None
    assert receipt["status"] == "none" and receipt["shadow_pixels"] == 0
    assert kept.tobytes() == foreground.tobytes()


def test_review_failure_has_stable_stage_and_reason_without_input_data(api):
    scene, foreground, _ = scene_fixture()
    rgb = np.array(scene)
    rgb[:40] = [235, 180, 140]
    with pytest.raises(api.StudioLayersError) as error:
        api.decompose_studio_scene(Image.fromarray(rgb), foreground)
    assert error.value.stage == "background_samples"
    assert error.value.reason == "nonneutral_background"


@pytest.mark.parametrize(
    "fault",
    [
        "heterogeneous",
        "colored",
        "dark",
        "unattached_patch",
        "detached_floor_mark",
        "dense_contact",
        "no_floor",
        "no_foreground",
    ],
)
def test_uncertain_backdrop_or_support_refuses_instead_of_inventing_shadow(api, fault):
    scene, foreground, _ = scene_fixture()
    rgb = np.array(scene)
    if fault == "heterogeneous":
        rgb[:, :80] = [155, 155, 155]
    elif fault == "colored":
        rgb[:40, :] = [235, 180, 140]
    elif fault == "dark":
        rgb = np.minimum(rgb, 95)
    elif fault == "unattached_patch":
        rgb[10:35, 30:70] = 155
    elif fault == "detached_floor_mark":
        rgb[205:216, 110:131] = 155
    elif fault == "dense_contact":
        rgb[161:166, 100:140] = 40
    elif fault == "no_floor":
        foreground = Image.new("RGBA", scene.size, (10, 10, 10, 255))
    elif fault == "no_foreground":
        foreground = Image.new("RGBA", scene.size)
    with pytest.raises(api.StudioLayersError) as error:
        api.decompose_studio_scene(Image.fromarray(rgb), foreground)
    assert error.value.code == "needs_review"


@pytest.mark.parametrize("fault", ["mode", "matte_mode", "size", "too_large", "too_small", "not_image"])
def test_rejects_invalid_or_unaligned_inputs(api, fault):
    scene, foreground, _ = scene_fixture()
    if fault == "mode":
        scene = scene.convert("RGBA")
    elif fault == "matte_mode":
        foreground = foreground.convert("RGB")
    elif fault == "size":
        foreground = foreground.resize((120, 120))
    elif fault == "too_large":
        scene = Image.new("RGB", (1601, 32))
        foreground = Image.new("RGBA", scene.size)
    elif fault == "too_small":
        scene = Image.new("RGB", (16, 16))
        foreground = Image.new("RGBA", scene.size)
    elif fault == "not_image":
        scene = b"not pixels"
    with pytest.raises(api.StudioLayersError) as error:
        api.decompose_studio_scene(scene, foreground)
    assert error.value.code == "invalid_request"


def test_backdrop_recolor_is_independent_of_opaque_foreground(api):
    scene, foreground, _ = scene_fixture()
    kept, shadow, _ = api.decompose_studio_scene(scene, foreground)
    outputs = []
    for color in ("white", "#2a392f", "#eac4d5"):
        backdrop = Image.new("RGBA", scene.size, color)
        outputs.append(Image.alpha_composite(Image.alpha_composite(backdrop, shadow), kept))
    assert all(image.getpixel((120, 100)) == (28, 36, 44, 255) for image in outputs)
    assert len({image.getpixel((20, 20)) for image in outputs}) == 3
    assert all(
        image.getpixel((120, 170))[:3] != Image.new("RGB", (1, 1), color).getpixel((0, 0))
        for image, color in zip(outputs, ("white", "#2a392f", "#eac4d5"))
    )


def test_scene_recolor_identity_opaque_source_and_immutable_inputs(api):
    scene, foreground, _ = scene_fixture()
    before, matte_before = scene.tobytes(), foreground.tobytes()
    output, receipt = api.recolor_studio_scene(scene, foreground, (100, 160, 220))
    opaque = np.asarray(foreground)[..., 3] == 255
    assert np.array_equal(np.asarray(output)[opaque], np.asarray(scene)[opaque])
    assert scene.tobytes() == before and foreground.tobytes() == matte_before
    identity, identity_receipt = api.recolor_studio_scene(scene, foreground, receipt["reference_rgb"])
    assert identity.tobytes() == before and identity_receipt["channel_gains"] == [1.0, 1.0, 1.0]
    assert receipt["provenance"] == "masked_scene_background_recolor"
    assert receipt["method"] == "display-rgb-alpha-weighted-channel-gains-v1"
    assert receipt["visual_quality_approved"] is False
    assert receipt["alignment_required"] is True and receipt["alignment_verified"] is False
    assert receipt["shadow_extraction_performed"] is False
    # Only matte alpha is a control input; its RGB must not replace scene RGB.
    different_rgb = np.array(foreground)
    different_rgb[..., :3] = [255, 0, 255]
    same_output, _ = api.recolor_studio_scene(scene, Image.fromarray(different_rgb), (100, 160, 220))
    assert same_output.tobytes() == output.tobytes()


def test_scene_recolor_preserves_existing_darkness_and_matches_reported_formula(api):
    scene, foreground, _ = scene_fixture()
    result, receipt = api.recolor_studio_scene(scene, foreground, (100, 130, 160))
    # Two completely uncovered backdrop pixels, one inside observed shadow.
    for channel in range(3):
        original_ratio = scene.getpixel((120, 170))[channel] / scene.getpixel((120, 220))[channel]
        actual_ratio = result.getpixel((120, 170))[channel] / result.getpixel((120, 220))[channel]
        assert actual_ratio == pytest.approx(original_ratio, abs=0.015)
    a = 128 / 255
    expected = np.rint(np.array([40, 70, 90]) * (a + (1 - a) * np.array(receipt["channel_gains"]))).astype(np.uint8)
    assert result.getpixel((90, 80)) == tuple(expected)
    black, _ = api.recolor_studio_scene(scene, foreground, (0, 0, 0))
    assert black.getpixel((120, 170)) == (0, 0, 0)
    assert black.getpixel((120, 100)) == scene.getpixel((120, 100))


@pytest.mark.parametrize("target", [(0, 0), (256, 20, 30), (-1, 20, 30), (True, 20, 30), (20.5, 20, 30), "#ffffff"])
def test_scene_recolor_rejects_invalid_colors(api, target):
    scene, foreground, _ = scene_fixture()
    with pytest.raises(api.StudioLayersError) as error:
        api.recolor_studio_scene(scene, foreground, target)
    assert error.value.code == "invalid_request"


def test_scene_recolor_refuses_without_neutral_light_reference(api):
    scene, foreground, _ = scene_fixture()
    with pytest.raises(api.StudioLayersError) as error:
        api.recolor_studio_scene(Image.new("RGB", scene.size, (90, 55, 20)), foreground, (100, 100, 100))
    assert error.value.code == "needs_review" and error.value.reason == "background_reference_unavailable"


def test_scene_recolor_rejects_unaligned_image_modes_and_dimensions(api):
    scene, foreground, _ = scene_fixture()
    with pytest.raises(api.StudioLayersError) as error:
        api.recolor_studio_scene(scene, foreground.resize((120, 120)), (100, 100, 100))
    assert error.value.code == "invalid_request"
