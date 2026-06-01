from __future__ import annotations

import numpy as np
from napari.layers import Image

from napari_racc._views import (
    _overlay_channel_volumes,
    _overlay_rgba_volume,
    show_mips,
    show_overlay,
    show_racc_only,
    show_side_by_side,
)


class _LayerList(list):
    def __contains__(self, item):
        if isinstance(item, str):
            return any(layer.name == item for layer in self)
        return super().__contains__(item)

    def __getitem__(self, item):
        if isinstance(item, str):
            for layer in self:
                if layer.name == item:
                    return layer
            raise KeyError(item)
        return super().__getitem__(item)


class _Dims:
    ndisplay = 3


class _Grid:
    enabled = False
    shape = (-1, -1)
    stride = 1
    spacing = 0.0


class _Viewer:
    def __init__(self, layers):
        self.layers = _LayerList(layers)
        self.dims = _Dims()
        self.grid = _Grid()
        self.reset_view_margin = None

    def add_image(self, data, **kwargs):
        layer = Image(data, **kwargs)
        self.layers.append(layer)
        return layer

    def reset_view(self, *, margin=0.05):
        self.reset_view_margin = margin


def test_show_mips_creates_flat_overlay_and_racc_layers():
    channel_1 = Image(
        np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5),
        name="red",
        scale=(2.0, 0.5, 0.5),
    )
    channel_2 = Image(
        np.flip(channel_1.data, axis=0),
        name="green",
        scale=(2.0, 0.5, 0.5),
    )
    result = Image(
        np.ones((3, 4, 5), dtype=np.float32),
        name="RACC: red x green",
        scale=(2.0, 0.5, 0.5),
    )
    viewer = _Viewer([channel_1, channel_2, result])

    show_mips(viewer, channel_1, channel_2, result)

    overlay = viewer.layers["RACC overlay MIP: red x green"]
    result_mip = viewer.layers["RACC: red x green MIP"]
    visible_layers = [layer for layer in viewer.layers if layer.visible]
    overlay_column = viewer.layers.index(overlay) % 2
    result_column = viewer.layers.index(result_mip) % 2

    assert viewer.dims.ndisplay == 2
    assert viewer.grid.enabled is True
    assert viewer.grid.shape == (1, 2)
    assert viewer.grid.stride == 1
    assert viewer.grid.spacing == 0.02
    assert viewer.reset_view_margin == 0.02
    assert set(visible_layers) == {overlay, result_mip}
    assert overlay_column == 0
    assert result_column == 1
    assert overlay.rgb is True
    assert overlay.data.shape == (4, 5, 3)
    assert tuple(overlay.scale) == (0.5, 0.5)
    assert result_mip.data.shape == (4, 5)
    assert result_mip.contrast_limits[0] > 0.0
    assert result_mip.colormap.map([0.0])[0, 3] == 0.0


def test_single_panel_views_disable_grid_shape_and_reset_camera():
    channel_1 = Image(np.ones((3, 4, 5), dtype=np.float32), name="red")
    channel_2 = Image(np.ones((3, 4, 5), dtype=np.float32), name="green")
    result = Image(np.ones((3, 4, 5), dtype=np.float32), name="RACC")
    viewer = _Viewer([channel_1, channel_2, result])
    viewer.grid.enabled = True
    viewer.grid.shape = (1, 2)
    viewer.grid.stride = 2
    viewer.grid.spacing = 0.05

    show_overlay(viewer, channel_1, channel_2, result)

    assert viewer.dims.ndisplay == 3
    assert viewer.grid.enabled is False
    assert viewer.grid.shape == (-1, -1)
    assert viewer.grid.stride == 1
    assert viewer.grid.spacing == 0.0
    assert viewer.reset_view_margin == 0.02
    overlay = viewer.layers["RACC overlay volume: red x green"]
    assert channel_1.visible is False
    assert channel_2.visible is False
    assert result.visible is False
    assert overlay.visible is True
    assert overlay.rgb is True
    assert overlay.data.shape == (3, 4, 5, 4)
    assert overlay.opacity == 1.0
    assert overlay.blending == "translucent"
    assert overlay.rendering == "translucent"
    assert overlay.depiction == "volume"
    assert overlay.data[..., 3].min() == 1.0

    viewer.grid.enabled = True
    viewer.grid.shape = (1, 2)
    viewer.reset_view_margin = None

    show_racc_only(viewer, result)

    assert viewer.grid.enabled is False
    assert viewer.grid.shape == (-1, -1)
    assert viewer.reset_view_margin == 0.02
    assert channel_1.visible is False
    assert channel_2.visible is False
    assert result.visible is True
    assert result.rendering == "translucent"
    assert result.depiction == "volume"
    assert result.contrast_limits[0] > 0.0


def test_show_racc_only_handles_contrast_thumbnail_failure(monkeypatch):
    result = Image(np.ones((3, 4, 5), dtype=np.float32), name="RACC")
    viewer = _Viewer([result])
    thumbnail_calls = 0

    def fail_thumbnail_update():
        nonlocal thumbnail_calls
        thumbnail_calls += 1
        if thumbnail_calls > 1:
            raise RuntimeError("sequence argument must have length equal to input rank")

    monkeypatch.setattr(result, "_update_thumbnail", fail_thumbnail_update)

    show_racc_only(viewer, result)

    assert result.visible is True
    assert result.contrast_limits[0] > 0.0
    assert result.rendering == "translucent"


def test_translucent_volume_status_reader_ignores_empty_ray():
    result = Image(np.ones((3, 4, 5), dtype=np.float32), name="RACC")
    viewer = _Viewer([result])

    show_racc_only(viewer, result)

    assert result._calculate_value_from_ray(np.array([], dtype=np.float32)) is None
    assert result._calculate_value_from_ray(np.array([0.5], dtype=np.float32)) == 0.5


def test_side_by_side_arranges_overlay_volume_left_and_racc_right():
    channel_1 = Image(np.ones((3, 4, 5), dtype=np.float32), name="red")
    channel_2 = Image(np.ones((3, 4, 5), dtype=np.float32), name="green")
    result = Image(np.ones((3, 4, 5), dtype=np.float32), name="RACC")
    mip = Image(np.ones((4, 5), dtype=np.float32), name="RACC MIP")
    viewer = _Viewer([mip, result, channel_2, channel_1])

    show_side_by_side(viewer, channel_1, channel_2, result, overlay_alpha=0.42)

    overlay = viewer.layers["RACC overlay volume: red x green"]

    assert viewer.dims.ndisplay == 3
    assert viewer.grid.enabled is True
    assert viewer.grid.shape == (1, 2)
    assert viewer.grid.stride == 1
    assert viewer.layers.index(overlay) == 0
    assert viewer.layers.index(result) == 1
    assert overlay.visible is True
    assert result.visible is True
    assert channel_1.visible is False
    assert channel_2.visible is False
    assert mip.visible is False
    assert overlay.rgb is True
    assert overlay.data.shape == (3, 4, 5, 4)
    assert overlay.opacity == 1.0
    assert np.isclose(overlay.data[..., 3].max(), 0.42)
    assert overlay.rendering == "translucent"


def test_side_by_side_after_mips_clears_stale_mip_view_state():
    channel_1 = Image(np.ones((3, 4, 5), dtype=np.float32), name="red")
    channel_2 = Image(np.ones((3, 4, 5), dtype=np.float32), name="green")
    result = Image(np.ones((3, 4, 5), dtype=np.float32), name="RACC")
    viewer = _Viewer([channel_1, channel_2, result])

    channel_1.bounding_box.visible = True
    channel_2.bounding_box.visible = True
    result.bounding_box.visible = True

    show_mips(viewer, channel_1, channel_2, result)
    show_side_by_side(viewer, channel_1, channel_2, result, overlay_alpha=0.42)

    overlay_mip = viewer.layers["RACC overlay MIP: red x green"]
    result_mip = viewer.layers["RACC MIP"]
    overlay = viewer.layers["RACC overlay volume: red x green"]

    assert viewer.dims.ndisplay == 3
    assert viewer.grid.enabled is True
    assert viewer.grid.shape == (1, 2)
    assert viewer.grid.stride == 1
    assert overlay_mip.visible is False
    assert result_mip.visible is False
    assert overlay_mip.bounding_box.visible is False
    assert result_mip.bounding_box.visible is False
    assert viewer.layers.index(overlay) == 0
    assert viewer.layers.index(result) == 1


def test_overlay_volume_thresholds_channels_and_hides_background():
    channel_1 = Image(
        np.array([[[0, 10], [255, 3]]], dtype=np.float32),
        name="red",
    )
    channel_2 = Image(
        np.array([[[0, 2], [0, 128]]], dtype=np.float32),
        name="green",
    )

    red, green = _overlay_channel_volumes(
        channel_1,
        channel_2,
        threshold_1=5,
        threshold_2=5,
    )

    assert red.shape == (1, 2, 2)
    assert green.shape == (1, 2, 2)
    assert red[0, 0, 0] == 0.0
    assert green[0, 0, 0] == 0.0
    assert red[0, 0, 1] > 0.0
    assert green[0, 0, 1] == 0.0
    assert red[0, 1, 1] == 0.0
    assert green[0, 1, 1] > 0.0


def test_overlay_rgba_volume_uses_selected_probe_colors_and_transparency():
    channel_1 = Image(
        np.array([[[0, 255], [0, 128]]], dtype=np.float32),
        name="red",
    )
    channel_2 = Image(
        np.array([[[0, 0], [255, 128]]], dtype=np.float32),
        name="green",
    )

    rgba = _overlay_rgba_volume(
        channel_1,
        channel_2,
        threshold_1=5,
        threshold_2=5,
        overlay_color_1="magenta",
        overlay_color_2="cyan",
        overlay_alpha=0.5,
    )

    assert rgba.shape == (1, 2, 2, 4)
    np.testing.assert_allclose(rgba[0, 0, 0], [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(rgba[0, 0, 1], [1.0, 0.0, 1.0, 0.5])
    np.testing.assert_allclose(rgba[0, 1, 0], [0.0, 1.0, 1.0, 0.5])
    np.testing.assert_allclose(rgba[0, 1, 1, :3], [0.5, 0.5, 1.0], atol=0.01)
    assert 0.0 < rgba[0, 1, 1, 3] < 0.5
