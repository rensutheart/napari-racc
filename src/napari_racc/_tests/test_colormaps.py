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


def test_racc_volume_colormap_uses_low_alpha_ramp():
    colormap = racc_volume_colormap("turbo", alpha=0.12)
    mapped = colormap.map(np.array([0.0, 0.5, 1.0], dtype=np.float32))

    assert colormap.name == "racc-turbo-volume-0.12"
    assert mapped[0, 3] == 0.0
    assert 0.0 < mapped[1, 3] < mapped[2, 3]
    assert np.isclose(mapped[2, 3], 0.12)


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
    colormap = overlay_channel_colormap("red", alpha=0.5, transfer="volume")
    mapped = colormap.map(np.array([0.0, 0.25, 1.0], dtype=np.float32))

    assert colormap.name == "racc-overlay-red-0.50-volume"
    assert mapped[0, 3] == 0.0
    assert 0.125 < mapped[1, 3] < 0.5
    assert np.isclose(mapped[2, 3], 0.5)
