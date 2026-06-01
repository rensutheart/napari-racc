"""Helpers for managing napari layer views used by the RACC widget."""

from __future__ import annotations

import numpy as np
from napari.utils.colormaps import ensure_colormap
from qtpy.QtCore import QTimer

from napari_racc._colormaps import (
    overlay_channel_colormap,
    overlay_color_rgb,
    racc_colormap,
    racc_display_contrast_limits,
)
from napari_racc._napari_compat import guard_empty_translucent_ray
from napari_racc._racc import normalize_channels, reduce_to_intensity

RGB_VOLUME_RENDERING_METHOD = "racc_rgb_translucent"
RGB_VOLUME_RENDERING_SNIPPETS = {
    "before_loop": """
        vec4 integrated_color = vec4(0., 0., 0., 0.);
        """,
    "in_loop": """
        color = $get_data(loc);
        color = clamp(color, 0.0, 1.0);
        float a1 = integrated_color.a;
        float a2 = color.a * (1.0 - a1);
        float alpha = a1 + a2;

        if (alpha > 0.0) {
            integrated_color.rgb = (
                integrated_color.rgb * a1 + color.rgb * a2
            ) / alpha;
            integrated_color.a = alpha;
        }

        if (alpha > 0.99) {
            iter = nsteps;
        }
        """,
    "after_loop": """
        if (integrated_color.a <= 0.0)
            discard;
        gl_FragColor = integrated_color;
        """,
}


def show_overlay(
    viewer,
    channel_1_layer,
    channel_2_layer,
    result_layer=None,
    overlay_alpha: float = 1.0,
    threshold_1: float = 0.0,
    threshold_2: float = 0.0,
    overlay_color_1: str = "red",
    overlay_color_2: str = "green",
) -> None:
    _prepare_view_change(viewer)
    _set_ndisplay_from_layers(viewer, channel_1_layer, channel_2_layer)
    overlay_layer = update_raw_overlay_volume(
        viewer,
        channel_1_layer,
        channel_2_layer,
        overlay_alpha,
        threshold_1,
        threshold_2,
        overlay_color_1,
        overlay_color_2,
    )
    overlay_layer.visible = True
    if result_layer is not None:
        result_layer.visible = False
    _select_layer(viewer, overlay_layer)
    _reset_view(viewer)


def show_racc_only(viewer, result_layer) -> None:
    _prepare_view_change(viewer)
    _set_ndisplay_from_layers(viewer, result_layer)
    if result_layer is not None:
        _show_result(result_layer)
        _select_layer(viewer, result_layer)
    _reset_view(viewer)


def show_side_by_side(
    viewer,
    channel_1_layer,
    channel_2_layer,
    result_layer,
    overlay_alpha: float = 1.0,
    threshold_1: float = 0.0,
    threshold_2: float = 0.0,
    overlay_color_1: str = "red",
    overlay_color_2: str = "green",
) -> None:
    _prepare_view_change(viewer)
    _set_ndisplay_from_layers(viewer, channel_1_layer, channel_2_layer, result_layer)
    overlay_layer = update_raw_overlay_volume(
        viewer,
        channel_1_layer,
        channel_2_layer,
        overlay_alpha,
        threshold_1,
        threshold_2,
        overlay_color_1,
        overlay_color_2,
    )
    overlay_layer.visible = True
    if result_layer is not None:
        _show_result(result_layer)
        _arrange_side_by_side(
            viewer,
            overlay_layer,
            result_layer,
        )
    viewer.grid.enabled = True
    viewer.grid.shape = (1, 2)
    viewer.grid.stride = 1
    viewer.grid.spacing = 0.05
    _select_layer(viewer, result_layer)
    _reset_view(viewer)


def show_mips(
    viewer,
    channel_1_layer,
    channel_2_layer,
    result_layer,
    colormap_name: str | None = None,
    overlay_color_1: str = "red",
    overlay_color_2: str = "green",
) -> None:
    _prepare_view_change(viewer)
    overlay_layer, result_mip = update_mips(
        viewer,
        channel_1_layer,
        channel_2_layer,
        result_layer,
        colormap_name,
        overlay_color_1,
        overlay_color_2,
    )

    viewer.dims.ndisplay = 2
    overlay_layer.visible = True
    result_mip.visible = True
    viewer.grid.enabled = True
    viewer.grid.shape = (1, 2)
    viewer.grid.stride = 1
    viewer.grid.spacing = 0.02
    _select_layer(viewer, result_mip)
    _reset_view(viewer)


def update_mips(
    viewer,
    channel_1_layer,
    channel_2_layer,
    result_layer,
    colormap_name: str | None = None,
    overlay_color_1: str = "red",
    overlay_color_2: str = "green",
):
    channel_1_mip = _mip(channel_1_layer)
    channel_2_mip = _mip(channel_2_layer)
    raw_overlay_mip = _overlay_mip(
        channel_1_mip,
        channel_2_mip,
        overlay_color_1,
        overlay_color_2,
    )

    overlay_layer = _add_or_update_image(
        viewer,
        f"RACC overlay MIP: {channel_1_layer.name} x {channel_2_layer.name}",
        raw_overlay_mip,
        {
            "rgb": True,
            "blending": "opaque",
            "metadata": {
                "napari_racc_kind": "mip",
                "racc_mip_kind": "raw_overlay",
            },
            "scale": _mip_scale(channel_1_layer),
        },
    )
    result_mip = _add_or_update_image(
        viewer,
        f"{result_layer.name} MIP",
        _mip(result_layer),
        {
            "colormap": racc_colormap(colormap_name),
            "contrast_limits": racc_display_contrast_limits(),
            "blending": "opaque",
            "metadata": {
                "napari_racc_kind": "mip",
                "racc_mip_kind": "racc",
            },
            "scale": _mip_scale(result_layer),
        },
    )
    _arrange_grid_pair(viewer, overlay_layer, result_mip)
    return overlay_layer, result_mip


def update_raw_overlay_volume(
    viewer,
    channel_1_layer,
    channel_2_layer,
    overlay_alpha: float = 1.0,
    threshold_1: float = 0.0,
    threshold_2: float = 0.0,
    overlay_color_1: str = "red",
    overlay_color_2: str = "green",
):
    data = _overlay_rgba_volume(
        channel_1_layer,
        channel_2_layer,
        threshold_1,
        threshold_2,
        overlay_color_1,
        overlay_color_2,
        overlay_alpha,
    )
    layer = _add_or_update_image(
        viewer,
        f"RACC overlay volume: {channel_1_layer.name} x {channel_2_layer.name}",
        data,
        {
            "rgb": True,
            "blending": "translucent",
            "opacity": 1.0,
            "colormap": "gray",
            "contrast_limits": (0.0, 1.0),
            "metadata": {
                "napari_racc_kind": "overlay",
                "racc_overlay_kind": "raw_volume",
                "racc_input_1": channel_1_layer.name,
                "racc_input_2": channel_2_layer.name,
            },
            "scale": tuple(float(value) for value in channel_1_layer.scale),
        },
    )
    _set_translucent_volume_rendering(layer)
    _ensure_rgb_volume_rendering(viewer, layer)
    _hide_stale_raw_overlay_layers(viewer, channel_1_layer, channel_2_layer)
    return layer


def update_overlay_volumes(
    viewer,
    channel_1_layer,
    channel_2_layer,
    overlay_alpha: float = 1.0,
    threshold_1: float = 0.0,
    threshold_2: float = 0.0,
    overlay_color_1: str = "red",
    overlay_color_2: str = "green",
):
    red_data, green_data = _overlay_channel_volumes(
        channel_1_layer,
        channel_2_layer,
        threshold_1,
        threshold_2,
    )
    common_kwargs = {
        "blending": "additive",
        "opacity": 1.0,
        "contrast_limits": racc_display_contrast_limits(),
        "metadata": {
            "napari_racc_kind": "overlay",
            "racc_overlay_kind": "side_by_side",
            "racc_input_1": channel_1_layer.name,
            "racc_input_2": channel_2_layer.name,
        },
        "scale": tuple(float(value) for value in channel_1_layer.scale),
    }
    red_layer = _add_or_update_image(
        viewer,
        f"RACC overlay red: {channel_1_layer.name} x {channel_2_layer.name}",
        red_data,
        {
            **common_kwargs,
            "colormap": overlay_channel_colormap(overlay_color_1, overlay_alpha),
            "metadata": {
                **common_kwargs["metadata"],
                "racc_overlay_channel": "channel_1",
            },
        },
    )
    green_layer = _add_or_update_image(
        viewer,
        f"RACC overlay green: {channel_1_layer.name} x {channel_2_layer.name}",
        green_data,
        {
            **common_kwargs,
            "colormap": overlay_channel_colormap(overlay_color_2, overlay_alpha),
            "metadata": {
                **common_kwargs["metadata"],
                "racc_overlay_channel": "channel_2",
            },
        },
    )
    _set_translucent_volume_rendering(red_layer)
    _set_translucent_volume_rendering(green_layer)
    _hide_stale_encoded_overlay(viewer, channel_1_layer, channel_2_layer)
    return red_layer, green_layer


def _hide_all(viewer) -> None:
    for layer in viewer.layers:
        if hasattr(layer, "bounding_box"):
            layer.bounding_box.visible = False
        layer.visible = False


def _show_channel(layer, color: str, alpha_gain: float = 1.0) -> None:
    layer.visible = True
    layer.blending = "additive"
    layer.opacity = 1.0
    if not getattr(layer, "rgb", False):
        layer.colormap = overlay_channel_colormap(
            color,
            alpha=alpha_gain,
            transfer="gain",
        )
    _set_translucent_volume_rendering(layer)


def _show_result(layer) -> None:
    layer.visible = True
    layer.blending = "translucent"
    _set_layer_contrast_limits(layer, racc_display_contrast_limits())
    _set_translucent_volume_rendering(layer)


def _set_translucent_volume_rendering(layer) -> None:
    if int(layer.ndim) < 3:
        return
    guard_empty_translucent_ray(layer)
    layer.depiction = "volume"
    layer.rendering = "translucent"


def _mip(layer):
    data = np.asarray(layer.data)
    if bool(getattr(layer, "rgb", False)):
        if data.ndim == 4:
            return data.max(axis=0)
        return data
    if data.ndim >= 3:
        return data.max(axis=0)
    return data


def _overlay_channel_volumes(
    channel_1_layer,
    channel_2_layer,
    threshold_1: float = 0.0,
    threshold_2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    channel_1, channel_2, _warnings = normalize_channels(
        reduce_to_intensity(
            channel_1_layer.data,
            rgb=bool(getattr(channel_1_layer, "rgb", False)),
            channel_mode="max",
        ),
        reduce_to_intensity(
            channel_2_layer.data,
            rgb=bool(getattr(channel_2_layer, "rgb", False)),
            channel_mode="max",
        ),
    )
    channel_1_mask = channel_1 >= float(threshold_1)
    channel_2_mask = channel_2 >= float(threshold_2)
    red = np.where(channel_1_mask, channel_1 / 255.0, 0.0)
    green = np.where(channel_2_mask, channel_2 / 255.0, 0.0)
    return red.astype(np.float32, copy=False), green.astype(np.float32, copy=False)


def _overlay_rgba_volume(
    channel_1_layer,
    channel_2_layer,
    threshold_1: float = 0.0,
    threshold_2: float = 0.0,
    overlay_color_1: str = "red",
    overlay_color_2: str = "green",
    overlay_alpha: float = 1.0,
) -> np.ndarray:
    channel_1, channel_2 = _overlay_channel_volumes(
        channel_1_layer,
        channel_2_layer,
        threshold_1,
        threshold_2,
    )
    color_1 = overlay_color_rgb(overlay_color_1)
    color_2 = overlay_color_rgb(overlay_color_2)
    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))

    rgb = np.clip(
        channel_1[..., np.newaxis] * color_1
        + channel_2[..., np.newaxis] * color_2,
        0.0,
        1.0,
    )
    volume_alpha = np.sqrt(np.maximum(channel_1, channel_2)) * alpha
    return np.concatenate(
        [rgb, volume_alpha[..., np.newaxis]],
        axis=-1,
    ).astype(np.float32, copy=False)


def _overlay_mip(
    channel_1_mip,
    channel_2_mip,
    overlay_color_1: str = "red",
    overlay_color_2: str = "green",
) -> np.ndarray:
    channel_1 = _normalize_channel(channel_1_mip)
    channel_2 = _normalize_channel(channel_2_mip)
    color_1 = overlay_color_rgb(overlay_color_1)
    color_2 = overlay_color_rgb(overlay_color_2)
    return np.clip(
        (channel_1[..., np.newaxis] * color_1)
        + (channel_2[..., np.newaxis] * color_2),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)


def _normalize_channel(data) -> np.ndarray:
    channel = np.asarray(data, dtype=np.float32)
    if channel.ndim >= 3 and channel.shape[-1] in {3, 4}:
        channel = channel.max(axis=-1)
    channel = np.nan_to_num(channel, copy=False)
    max_value = float(channel.max()) if channel.size else 0.0
    if max_value > 0:
        channel = channel / max_value
    return np.clip(channel, 0.0, 1.0)


def _mip_scale(layer) -> tuple[float, ...]:
    scale = tuple(float(value) for value in layer.scale)
    if len(scale) >= 2:
        return scale[-2:]
    if len(scale) == 1:
        return (scale[0],)
    return (1.0, 1.0)


def _set_ndisplay_from_layers(viewer, *layers) -> None:
    if not hasattr(viewer, "dims"):
        return
    viewer.dims.ndisplay = 3 if any(int(layer.ndim) >= 3 for layer in layers) else 2


def _reset_view(viewer) -> None:
    if hasattr(viewer, "reset_view"):
        viewer.reset_view(margin=0.02)


def _disable_grid(viewer) -> None:
    viewer.grid.enabled = False
    viewer.grid.shape = (-1, -1)
    viewer.grid.stride = 1
    viewer.grid.spacing = 0.0


def _prepare_view_change(viewer) -> None:
    _disable_grid(viewer)
    _hide_all(viewer)


def _add_or_update_image(viewer, name: str, data, kwargs: dict):
    if name in viewer.layers:
        layer = viewer.layers[name]
        if "rgb" in kwargs:
            layer.rgb = kwargs["rgb"]
        layer.data = data
        for key, value in kwargs.items():
            if key == "rgb":
                continue
            try:
                if key == "colormap":
                    _set_layer_colormap(layer, value)
                elif key == "contrast_limits":
                    _set_layer_contrast_limits(layer, value)
                else:
                    setattr(layer, key, value)
            except Exception:
                pass
        return layer
    return viewer.add_image(data, name=name, **kwargs)


def _set_layer_colormap(layer, colormap) -> None:
    try:
        layer.colormap = colormap
    except RuntimeError as error:
        if "sequence argument must have length equal to input rank" not in str(error):
            raise
        layer._colormap = ensure_colormap(colormap)
        layer.events.colormap()


def _set_layer_contrast_limits(layer, contrast_limits) -> None:
    try:
        layer.contrast_limits = contrast_limits
    except RuntimeError as error:
        if "sequence argument must have length equal to input rank" not in str(error):
            raise
        layer._contrast_limits = contrast_limits
        current_range = list(layer.contrast_limits_range)
        layer._contrast_limits_range = [
            min(current_range[0], contrast_limits[0]),
            max(current_range[1], contrast_limits[1]),
        ]
        layer.events.contrast_limits()


def _ensure_rgb_volume_rendering(viewer, layer) -> None:
    if not getattr(layer, "rgb", False) or int(layer.ndim) < 3:
        return
    _bind_rgb_volume_shader_refresh(viewer, layer)
    if not _apply_rgb_volume_shader(viewer, layer):
        QTimer.singleShot(0, lambda: _apply_rgb_volume_shader(viewer, layer))


def _bind_rgb_volume_shader_refresh(viewer, layer) -> None:
    if getattr(layer, "_racc_rgb_volume_shader_bound", False):
        return

    def refresh_rgb_shader(*args):
        QTimer.singleShot(0, lambda: _apply_rgb_volume_shader(viewer, layer))

    layer.events.set_data.connect(refresh_rgb_shader)
    layer.events.rendering.connect(refresh_rgb_shader)
    dims_events = getattr(getattr(viewer, "dims", None), "events", None)
    if dims_events is not None and hasattr(dims_events, "ndisplay"):
        dims_events.ndisplay.connect(refresh_rgb_shader)
    layer._racc_rgb_volume_shader_bound = True


def _apply_rgb_volume_shader(viewer, layer) -> bool:
    visual = _vispy_layer_for(viewer, layer)
    node = getattr(visual, "node", None)
    if node is None or not hasattr(node, "_rendering_methods"):
        return False
    if int(getattr(layer, "ndim", 0)) < 3:
        return False
    if RGB_VOLUME_RENDERING_METHOD not in node._rendering_methods:
        node._rendering_methods = {
            **node._rendering_methods,
            RGB_VOLUME_RENDERING_METHOD: RGB_VOLUME_RENDERING_SNIPPETS,
        }
    if getattr(node, "method", None) != RGB_VOLUME_RENDERING_METHOD:
        node.method = RGB_VOLUME_RENDERING_METHOD
    return True


def _vispy_layer_for(viewer, layer):
    window = getattr(viewer, "window", None)
    qt_viewer = getattr(window, "_qt_viewer", None)
    if qt_viewer is None:
        qt_viewer = getattr(window, "qt_viewer", None)
    layer_to_visual = getattr(qt_viewer, "layer_to_visual", None)
    if layer_to_visual is None:
        return None
    return layer_to_visual.get(layer)


def _arrange_grid_pair(viewer, left_layer, right_layer) -> None:
    layers = viewer.layers
    if left_layer not in layers or right_layer not in layers:
        return

    base_index = min(layers.index(left_layer), layers.index(right_layer))
    desired_order = (
        (left_layer, right_layer)
        if base_index % 2 == 0
        else (right_layer, left_layer)
    )
    for offset, layer in enumerate(desired_order):
        _move_layer(layers, layer, base_index + offset)


def _arrange_side_by_side(
    viewer,
    overlay_layer,
    result_layer,
) -> None:
    layers = viewer.layers
    for destination, layer in enumerate((overlay_layer, result_layer)):
        if layer in layers:
            _move_layer(layers, layer, destination)


def _hide_stale_encoded_overlay(viewer, channel_1_layer, channel_2_layer) -> None:
    old_name = f"RACC overlay volume: {channel_1_layer.name} x {channel_2_layer.name}"
    if old_name in viewer.layers:
        viewer.layers[old_name].visible = False


def _hide_stale_raw_overlay_layers(viewer, channel_1_layer, channel_2_layer) -> None:
    for color in ("red", "green"):
        raw_name = (
            f"RACC overlay raw {color}: "
            f"{channel_1_layer.name} x {channel_2_layer.name}"
        )
        legacy_name = (
            f"RACC overlay {color}: "
            f"{channel_1_layer.name} x {channel_2_layer.name}"
        )
        for old_name in (raw_name, legacy_name):
            if old_name in viewer.layers:
                viewer.layers[old_name].visible = False


def _move_layer(layers, layer, destination: int) -> None:
    source = layers.index(layer)
    if source == destination:
        return
    if hasattr(layers, "move"):
        layers.move(source, destination)
        return
    item = layers.pop(source)
    layers.insert(destination, item)


def _select_layer(viewer, layer) -> None:
    selection = getattr(getattr(viewer, "layers", None), "selection", None)
    if selection is None or layer is None:
        return
    try:
        selection.clear()
        selection.add(layer)
    except Exception:
        try:
            selection.active = layer
        except Exception:
            return
