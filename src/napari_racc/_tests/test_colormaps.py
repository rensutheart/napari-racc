from __future__ import annotations

import numpy as np

from napari_racc._colormaps import (
    overlay_channel_colormap,
    racc_colormap,
    racc_display_contrast_limits,
    racc_volume_colormap,
)


def test_racc_colormap_makes_zero_transparent():
    colormap = racc_colormap("turbo")
    mapped = colormap.map(np.array([0.0, 0.5], dtype=np.float32))

    assert colormap.name == "racc-turbo-transparent"
    assert mapped[0, 3] == 0.0
    assert mapped[1, 3] == 1.0


def test_racc_colormap_floor_hides_background_without_remapping_colors():
    baseline = racc_colormap("turbo")
    floored = racc_colormap("turbo", floor=0.4)

    baseline_signal = baseline.map(np.array([0.8], dtype=np.float32))[0]
    mapped = floored.map(np.array([0.2, 0.4, 0.8], dtype=np.float32))

    assert mapped[0, 3] == 0.0
    assert mapped[1, 3] == 0.0
    assert mapped[2, 3] == 1.0
    np.testing.assert_allclose(mapped[2, :3], baseline_signal[:3], atol=1e-6)


def test_racc_volume_colormap_uses_opacity_and_quadratic_suppression():
    colormap = racc_volume_colormap(
        "turbo",
        opacity=0.4,
        suppression=2.0,
        floor=0.0,
    )
    mapped = colormap.map(np.array([0.0, 0.5, 1.0], dtype=np.float32))

    assert colormap.name == "racc-turbo-volume-0.400-s2.00-floor-0.000"
    assert mapped[0, 3] == 0.0
    np.testing.assert_allclose(mapped[:, 3], [0.0, 0.1, 0.4], atol=1e-6)


def test_racc_volume_colormap_applies_floor_before_opacity_transfer():
    colormap = racc_volume_colormap(
        "turbo",
        opacity=0.4,
        suppression=2.0,
        floor=0.25,
    )
    mapped = colormap.map(np.array([0.25, 0.625, 1.0], dtype=np.float32))

    np.testing.assert_allclose(mapped[:, 3], [0.0, 0.1, 0.4], atol=2e-4)

    stronger_suppression = racc_volume_colormap(
        "turbo",
        opacity=0.4,
        suppression=3.0,
        floor=0.25,
    ).map(np.array([0.625, 1.0], dtype=np.float32))
    assert stronger_suppression[0, 3] < mapped[1, 3]
    assert np.isclose(stronger_suppression[1, 3], mapped[2, 3])


def test_racc_volume_colormap_maps_full_opacity_directly_to_one():
    colormap = racc_volume_colormap(
        "turbo",
        opacity=1.0,
        suppression=2.0,
        floor=0.0,
    )

    assert np.isclose(colormap.map([1.0])[0, 3], 1.0)


def test_racc_display_limits_exclude_exact_zero():
    low, high = racc_display_contrast_limits()

    assert 0.0 < low < high
    assert high == 1.0


def test_overlay_channel_colormap_uses_alpha_transfer():
    colormap = overlay_channel_colormap("green", alpha=0.37)
    mapped = colormap.map(np.array([0.0, 0.5], dtype=np.float32))

    assert colormap.name == "racc-overlay-green-0.37"
    assert mapped[0, 3] == 0.0
    assert 0.0 < mapped[1, 3] < 0.37
    assert np.isclose(colormap.map([1.0])[0, 3], 0.37)


def test_overlay_channel_colormap_can_use_gain_transfer():
    colormap = overlay_channel_colormap("red", alpha=1.8, transfer="gain")
    mapped = colormap.map(np.array([0.0, 0.5, 0.8], dtype=np.float32))

    assert colormap.name == "racc-overlay-red-1.80-gain"
    assert mapped[0, 3] == 0.0
    assert 0.85 < mapped[1, 3] < 0.95
    assert np.isclose(mapped[2, 3], 1.0)


def test_overlay_channel_colormap_can_use_step_transfer():
    colormap = overlay_channel_colormap("red", alpha=0.5, transfer="step")
    mapped = colormap.map(np.array([0.0, 0.01, 1.0], dtype=np.float32))

    assert colormap.name == "racc-overlay-red-0.50-step"
    assert mapped[0, 3] == 0.0
    assert np.isclose(mapped[1, 3], 0.5)
    assert np.isclose(mapped[2, 3], 0.5)


def test_overlay_channel_colormap_can_use_volume_transfer():
    colormap = overlay_channel_colormap(
        "red",
        alpha=0.5,
        transfer="volume",
        suppression=2.0,
    )
    mapped = colormap.map(np.array([0.0, 0.4, 1.0], dtype=np.float32))

    assert colormap.name == "racc-overlay-red-0.500-volume-s2.00"
    assert mapped[0, 3] == 0.0
    assert np.isclose(mapped[1, 3], 0.5 * 0.4**2)
    assert np.isclose(mapped[2, 3], 0.5)
