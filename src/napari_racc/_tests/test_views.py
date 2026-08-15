from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers import Image

from napari_racc._views import (
    RGB_VOLUME_RENDERING_METHODS,
    SCALAR_ADDITIVE_RENDERING_METHOD,
    _apply_rgb_volume_shader,
    _apply_scalar_volume_shader,
    _overlay_channel_volumes,
    compute_overlay_rgb,
    configure_overlay_display,
    configure_racc_display,
    racc_colormap_for_layer,
    show_mips,
    show_overlay,
    show_racc_only,
    show_side_by_side,
    update_mips,
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

    show_mips(
        viewer,
        channel_1,
        channel_2,
        result,
        intensity_opacity=0.64,
        racc_opacity=0.37,
    )

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
    assert overlay.opacity == 1.0
    assert overlay.blending == "opaque"
    assert tuple(overlay.scale) == (0.5, 0.5)
    assert result_mip.data.shape == (4, 5)
    assert result_mip.opacity == 1.0
    assert result_mip.contrast_limits[0] > 0.0
    assert result_mip.colormap.map([0.0])[0, 3] == 0.0


def test_mips_apply_channel_cutoffs_and_racc_floor_before_projection():
    channel_1 = Image(
        np.array(
            [
                [[55, 100, 255]],
                [[20, 120, 200]],
            ],
            dtype=np.float32,
        ),
        name="red",
    )
    channel_2 = Image(np.zeros((2, 1, 3), dtype=np.float32), name="green")
    result_data = np.array(
        [
            [[0.2, 0.6, 0.9]],
            [[0.5, 0.8, 0.4]],
        ],
        dtype=np.float32,
    )
    result = Image(result_data.copy(), name="RACC")
    viewer = _Viewer([channel_1, channel_2, result])

    update_mips(
        viewer,
        channel_1,
        channel_2,
        result,
        display_cutoff_1=55,
        display_cutoff_2=5,
        racc_display_floor=0.5,
    )

    overlay = viewer.layers["RACC overlay MIP: red x green"]
    result_mip = viewer.layers["RACC MIP"]
    expected_red = np.array([0.0, (120.0 - 55.0) / 200.0, 1.0])
    np.testing.assert_allclose(overlay.data[0, :, 0], expected_red, atol=1e-6)
    np.testing.assert_array_equal(overlay.data[0, :, 1:], 0.0)
    np.testing.assert_allclose(result_mip.data, [[0.5, 0.8, 0.9]])
    np.testing.assert_array_equal(result.data, result_data)
    assert result_mip.colormap.map([0.5])[0, 3] == 0.0
    assert result_mip.contrast_limits == [1e-6, 1.0]


def test_mip_updates_reuse_source_projections_and_keep_raw_result_data():
    channel_1 = Image(
        np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
        name="red",
    )
    channel_2 = Image(np.flip(channel_1.data, axis=0), name="green")
    result = Image(
        np.linspace(0.0, 1.0, 2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
        name="RACC",
    )
    viewer = _Viewer([channel_1, channel_2, result])

    overlay, result_mip = update_mips(
        viewer,
        channel_1,
        channel_2,
        result,
        racc_display_floor=0.2,
    )
    overlay_cache = overlay._racc_channel_mip_cache
    result_cache = result_mip._racc_result_mip_cache
    raw_result_mip = result_mip.data
    result_set_data_events = 0

    def count_result_set_data(*args):
        nonlocal result_set_data_events
        result_set_data_events += 1

    result_mip.events.set_data.connect(count_result_set_data)

    update_mips(
        viewer,
        channel_1,
        channel_2,
        result,
        overlay_color_1="magenta",
        display_cutoff_1=25,
        racc_display_floor=0.6,
    )

    assert overlay._racc_channel_mip_cache is overlay_cache
    assert result_mip._racc_result_mip_cache is result_cache
    assert result_mip.data is raw_result_mip
    assert result_set_data_events == 0
    assert result_mip.colormap.map([0.6])[0, 3] == 0.0


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
    assert overlay.data.dtype == np.uint8
    assert overlay.opacity == 1.0
    assert overlay.blending == "translucent"
    assert overlay.rendering == "translucent"
    assert overlay.depiction == "volume"
    np.testing.assert_array_equal(overlay.data[0, 0, 0], [255, 255, 0, 255])

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

    show_side_by_side(
        viewer,
        channel_1,
        channel_2,
        result,
        intensity_opacity=0.42,
        racc_opacity=0.36,
    )

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
    assert overlay.data.dtype == np.uint8
    assert overlay.opacity == 1.0
    assert np.isclose(overlay._racc_volume_opacity, 0.42)
    assert overlay.rendering == "translucent"
    assert result.opacity == 1.0
    assert np.isclose(result.colormap.map([1.0])[0, 3], 0.36)


def test_side_by_side_2d_ignores_volume_opacity_and_suppression():
    channel_1 = Image(
        np.array([[0.0, 255.0], [128.0, 0.0]], dtype=np.float32),
        name="red",
    )
    channel_2 = Image(
        np.array([[0.0, 0.0], [128.0, 255.0]], dtype=np.float32),
        name="green",
    )
    result_data = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    result = Image(result_data.copy(), name="RACC")
    viewer = _Viewer([channel_1, channel_2, result])

    show_side_by_side(
        viewer,
        channel_1,
        channel_2,
        result,
        intensity_opacity=0.72,
        racc_opacity=0.41,
        background_suppression=4.0,
        rendering_mode="additive",
    )

    overlay = viewer.layers["RACC overlay volume: red x green"]
    assert viewer.dims.ndisplay == 2
    assert overlay.opacity == 1.0
    assert result.opacity == 1.0
    assert overlay.blending == "opaque"
    np.testing.assert_array_equal(
        overlay.data,
        np.array(
            [
                [[0, 0, 0, 0], [255, 0, 0, 255]],
                [[128, 128, 0, 128], [0, 255, 0, 255]],
            ],
            dtype=np.uint8,
        ),
    )
    assert np.isclose(result.colormap.map([1.0])[0, 3], 1.0)
    np.testing.assert_array_equal(result.data, result_data)


def test_side_by_side_after_mips_clears_stale_mip_view_state():
    channel_1 = Image(np.ones((3, 4, 5), dtype=np.float32), name="red")
    channel_2 = Image(np.ones((3, 4, 5), dtype=np.float32), name="green")
    result = Image(np.ones((3, 4, 5), dtype=np.float32), name="RACC")
    viewer = _Viewer([channel_1, channel_2, result])

    channel_1.bounding_box.visible = True
    channel_2.bounding_box.visible = True
    result.bounding_box.visible = True

    show_mips(viewer, channel_1, channel_2, result)
    show_side_by_side(
        viewer,
        channel_1,
        channel_2,
        result,
        intensity_opacity=0.42,
        racc_opacity=0.36,
    )

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


@pytest.mark.parametrize(
    (
        "rendering_mode",
        "expected_racc_colormap_kind",
    ),
    [
        ("translucent", "volume"),
        ("mip", "floor"),
        ("additive", "volume"),
    ],
)
def test_side_by_side_applies_one_render_mode_to_both_volumes(
    rendering_mode,
    expected_racc_colormap_kind,
):
    channel_1 = Image(np.ones((2, 3, 4), dtype=np.float32), name="red")
    channel_2 = Image(np.ones((2, 3, 4), dtype=np.float32), name="green")
    result = Image(np.ones((2, 3, 4), dtype=np.float32), name="RACC")
    viewer = _Viewer([channel_1, channel_2, result])

    show_side_by_side(
        viewer,
        channel_1,
        channel_2,
        result,
        intensity_opacity=0.4,
        racc_opacity=0.65,
        colormap_name="viridis",
        racc_display_floor=0.2,
        background_suppression=2.0,
        rendering_mode=rendering_mode,
    )

    overlay = viewer.layers["RACC overlay volume: red x green"]
    assert overlay.rendering == rendering_mode
    assert result.rendering == rendering_mode
    assert overlay.data.dtype == np.uint8
    assert overlay.data.shape == (2, 3, 4, 4)
    uses_transfer_alpha = rendering_mode in {"translucent", "additive"}
    assert np.isclose(overlay._racc_volume_opacity, 0.4)
    assert np.isclose(overlay.opacity, 1.0 if uses_transfer_alpha else 0.4)
    assert np.isclose(result.opacity, 1.0 if uses_transfer_alpha else 0.65)
    assert expected_racc_colormap_kind in result.colormap.name
    assert result.colormap.map([0.2])[0, 3] == 0.0
    assert np.isclose(
        result.colormap.map([1.0])[0, 3],
        0.65 if uses_transfer_alpha else 1.0,
    )
    assert result.contrast_limits == [1e-6, 1.0]


def test_racc_colormap_for_layer_preserves_selected_colormap_in_every_mode():
    result = Image(np.ones((2, 3, 4), dtype=np.float32), name="RACC")

    for rendering_mode in ("translucent", "mip", "additive"):
        colormap = racc_colormap_for_layer(
            result,
            "plasma",
            racc_display_floor=0.3,
            racc_opacity=0.2,
            background_suppression=2.5,
            rendering_mode=rendering_mode,
        )
        assert colormap.name.startswith("racc-plasma-")
        assert colormap.map([0.3])[0, 3] == 0.0
        expected_alpha = 1.0 if rendering_mode == "mip" else 0.2
        assert np.isclose(colormap.map([1.0])[0, 3], expected_alpha)

    slice_colormap = racc_colormap_for_layer(
        result,
        "plasma",
        racc_display_floor=0.3,
        racc_opacity=0.2,
        background_suppression=2.5,
        rendering_mode="translucent",
        ndisplay=2,
    )
    assert "volume" not in slice_colormap.name


def test_rgb_volume_shader_tracks_the_selected_global_render_mode():
    layer = Image(np.ones((2, 3, 4, 4), dtype=np.uint8), rgb=True, name="overlay")
    layer._racc_background_suppression = 3.2
    layer._racc_volume_opacity = 0.37
    node = SimpleNamespace(_rendering_methods={"translucent": {}}, method=None)
    visual = SimpleNamespace(node=node)

    class _VisualMap:
        def get(self, requested_layer):
            assert requested_layer is layer
            return visual

    viewer = SimpleNamespace(
        window=SimpleNamespace(
            _qt_viewer=SimpleNamespace(layer_to_visual=_VisualMap()),
        )
    )

    for rendering_mode, shader_method in RGB_VOLUME_RENDERING_METHODS.items():
        layer.rendering = rendering_mode
        assert _apply_rgb_volume_shader(viewer, layer) is True
        assert node.method == shader_method
        assert shader_method in node._rendering_methods
        assert np.isclose(node.gamma, 3.2)
        assert np.isclose(node.attenuation, 0.37)
        if rendering_mode != "mip":
            assert "float signal = color.a" in node._rendering_methods[
                shader_method
            ]["in_loop"]
            assert "u_attenuation" in node._rendering_methods[shader_method][
                "in_loop"
            ]


def test_overlay_display_style_switches_dimensions_without_replacing_data():
    data = np.full((2, 3, 4, 4), 128, dtype=np.uint8)
    layer = Image(data, rgb=True, name="overlay")
    viewer = _Viewer([layer])
    original_data = layer.data

    configure_overlay_display(
        viewer,
        layer,
        intensity_opacity=0.35,
        background_suppression=2.7,
        rendering_mode="translucent",
    )

    assert layer.data is original_data
    assert layer.opacity == 1.0
    assert np.isclose(layer._racc_volume_opacity, 0.35)
    assert layer.blending == "translucent"
    assert np.isclose(layer._racc_background_suppression, 2.7)

    viewer.dims.ndisplay = 2
    configure_overlay_display(
        viewer,
        layer,
        intensity_opacity=0.35,
        background_suppression=2.7,
        rendering_mode="translucent",
    )

    assert layer.data is original_data
    assert layer.opacity == 1.0
    assert layer.blending == "opaque"

    viewer.dims.ndisplay = 3
    configure_overlay_display(
        viewer,
        layer,
        intensity_opacity=0.35,
        background_suppression=2.7,
        rendering_mode="mip",
    )

    assert layer.data is original_data
    assert np.isclose(layer.opacity, 0.35)
    assert np.isclose(layer._racc_volume_opacity, 0.35)
    assert layer.rendering == "mip"


def test_racc_display_style_is_thumbnail_free_and_neutral_in_2d(monkeypatch):
    data = np.ones((2, 3, 4), dtype=np.float32)
    layer = Image(
        data,
        name="RACC",
        contrast_limits=(1e-6, 1.0),
    )
    viewer = _Viewer([layer])
    original_data = layer.data

    def fail_thumbnail_update():
        raise AssertionError("display style must not rebuild the thumbnail")

    monkeypatch.setattr(layer, "_update_thumbnail", fail_thumbnail_update)

    configure_racc_display(
        viewer,
        layer,
        colormap_name="viridis",
        racc_display_floor=0.2,
        racc_opacity=0.4,
        background_suppression=2.5,
        rendering_mode="translucent",
    )

    assert layer.data is original_data
    assert layer.opacity == 1.0
    assert "volume" in layer.colormap.name
    assert np.isclose(layer.colormap.map([1.0])[0, 3], 0.4)

    viewer.dims.ndisplay = 2
    configure_racc_display(
        viewer,
        layer,
        colormap_name="viridis",
        racc_display_floor=0.2,
        racc_opacity=0.4,
        background_suppression=2.5,
        rendering_mode="translucent",
    )

    assert layer.data is original_data
    assert layer.opacity == 1.0
    assert "volume" not in layer.colormap.name


def test_scalar_additive_shader_restores_standard_modes_when_switched():
    layer = Image(np.ones((2, 3, 4), dtype=np.float32), name="RACC")
    node = SimpleNamespace(
        _rendering_methods={"translucent": {}, "mip": {}, "additive": {}},
        method=None,
    )
    visual = SimpleNamespace(node=node)

    class _VisualMap:
        def get(self, requested_layer):
            assert requested_layer is layer
            return visual

    viewer = SimpleNamespace(
        window=SimpleNamespace(
            _qt_viewer=SimpleNamespace(layer_to_visual=_VisualMap()),
        )
    )

    layer.rendering = "additive"
    assert _apply_scalar_volume_shader(viewer, layer) is True
    assert node.method == SCALAR_ADDITIVE_RENDERING_METHOD

    layer.rendering = "mip"
    assert _apply_scalar_volume_shader(viewer, layer) is True
    assert node.method == "mip"

    node.method = SCALAR_ADDITIVE_RENDERING_METHOD
    layer.rendering = "translucent"
    assert _apply_scalar_volume_shader(viewer, layer) is True
    assert node.method == "translucent"


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
        display_cutoff_1=5,
        display_cutoff_2=5,
    )

    assert red.shape == (1, 2, 2)
    assert green.shape == (1, 2, 2)
    assert red[0, 0, 0] == 0.0
    assert green[0, 0, 0] == 0.0
    assert np.isclose(red[0, 0, 1], (10.0 - 5.0) / (255.0 - 5.0))
    assert green[0, 0, 1] == 0.0
    assert red[0, 1, 1] == 0.0
    assert np.isclose(green[0, 1, 1], (128.0 - 5.0) / (255.0 - 5.0))


def test_overlay_volume_maps_the_cutoff_itself_to_zero():
    channel_1 = Image(
        np.array([[[55, 155, 255]]], dtype=np.float32),
        name="red",
    )
    channel_2 = Image(np.zeros((1, 1, 3), dtype=np.float32), name="green")

    red, green = _overlay_channel_volumes(
        channel_1,
        channel_2,
        display_cutoff_1=55,
        display_cutoff_2=0,
    )

    np.testing.assert_allclose(red, [[[0.0, 0.5, 1.0]]], atol=1e-6)
    np.testing.assert_array_equal(green, 0.0)


def test_compute_overlay_rgb_keeps_color_independent_signal_component():
    channel_1 = np.array([[[0, 255], [0, 128]]], dtype=np.float32)
    channel_2 = np.array([[[0, 0], [255, 128]]], dtype=np.float32)

    rgba = compute_overlay_rgb(
        channel_1,
        channel_2,
        display_cutoff_1=5,
        display_cutoff_2=5,
        overlay_color_1="magenta",
        overlay_color_2="cyan",
    )

    midpoint = (128.0 - 5.0) / (255.0 - 5.0)
    assert rgba.shape == (1, 2, 2, 4)
    assert rgba.dtype == np.uint8
    np.testing.assert_array_equal(rgba[0, 0, 0], [0, 0, 0, 0])
    np.testing.assert_array_equal(rgba[0, 0, 1], [255, 0, 255, 255])
    np.testing.assert_array_equal(rgba[0, 1, 0], [0, 255, 255, 255])
    np.testing.assert_allclose(
        rgba[0, 1, 1],
        np.rint(
            np.asarray(
                [midpoint, midpoint, min(2.0 * midpoint, 1.0), midpoint],
            )
            * 255.0
        ),
        atol=1.0,
    )
