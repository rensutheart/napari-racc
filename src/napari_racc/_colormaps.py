"""Colormap helpers."""

from __future__ import annotations

import numpy as np
from napari.utils.colormaps import Colormap, ensure_colormap

DEFAULT_RACC_COLORMAP = "magma"
RACC_DISPLAY_MIN = 1e-6
DEFAULT_RACC_VOLUME_ALPHA = 0.12

RACC_COLORMAPS = (
    "magma",
    "inferno",
    "plasma",
    "viridis",
    "turbo",
    "fire",
    "ice",
    "gray",
)

_OVERLAY_COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.35, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "orange": (1.0, 0.45, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "gray": (0.8, 0.8, 0.8),
}


def racc_colormap_choices() -> tuple[str, ...]:
    """Return colormaps exposed by the RACC widget."""

    return RACC_COLORMAPS


def overlay_color_choices() -> tuple[str, ...]:
    """Return named probe colors exposed by the overlay controls."""

    return tuple(_OVERLAY_COLORS)


def overlay_color_name(name: str | None = None, fallback: str = "red") -> str:
    """Return a validated overlay probe color name."""

    if name in _OVERLAY_COLORS:
        return str(name)
    return fallback


def overlay_color_rgb(name: str | None = None) -> np.ndarray:
    """Return an RGB triplet for a named overlay probe color."""

    color_name = overlay_color_name(name)
    return np.asarray(_OVERLAY_COLORS[color_name], dtype=np.float32)


def racc_colormap_name(name: str | None = None) -> str:
    """Return a validated napari colormap name for RACC layers."""

    if name in RACC_COLORMAPS:
        return str(name)
    return DEFAULT_RACC_COLORMAP


def racc_display_contrast_limits() -> tuple[float, float]:
    """Return display limits that discard exact-zero RACC voxels."""

    return (RACC_DISPLAY_MIN, 1.0)


def racc_colormap(name: str | None = None) -> Colormap:
    """Return a RACC display colormap with zero mapped to transparent."""

    colormap_name = racc_colormap_name(name)
    return _racc_colormap_with_alpha(
        colormap_name,
        alpha_values=None,
        internal_name=f"racc-{colormap_name}-transparent",
    )


def racc_volume_colormap(
    name: str | None = None,
    alpha: float = DEFAULT_RACC_VOLUME_ALPHA,
) -> Colormap:
    """Return a low-opacity RACC colormap for accumulated volume rendering."""

    colormap_name = racc_colormap_name(name)
    base = ensure_colormap(colormap_name)
    controls = np.asarray(base.controls, dtype=np.float32)
    colors = np.asarray(base.colors, dtype=np.float32).copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))

    if len(controls) == len(colors):
        alpha_positions = controls
    else:
        alpha_positions = np.linspace(0.0, 1.0, len(colors), dtype=np.float32)
    colors[:, 3] = np.sqrt(np.clip(alpha_positions, 0.0, 1.0)) * alpha
    colors[0, 3] = 0.0

    return _racc_colormap_with_alpha(
        colormap_name,
        alpha_values=colors[:, 3],
        internal_name=f"racc-{colormap_name}-volume-{alpha:.2f}",
    )


def overlay_channel_colormap(
    color: str,
    alpha: float = 1.0,
    *,
    transfer: str = "linear",
) -> Colormap:
    """Return a zero-transparent overlay colormap for one probe color."""

    rgb = overlay_color_rgb(color)
    alpha = float(max(alpha, 0.0))
    if transfer == "gain":
        gain = float(np.clip(alpha, 0.0, 4.0))
        if gain <= 1.0:
            controls = np.array([0.0, 1.0], dtype=np.float32)
            colors = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [rgb[0], rgb[1], rgb[2], gain],
                ],
                dtype=np.float32,
            )
        else:
            controls = np.array([0.0, 1.0 / gain, 1.0], dtype=np.float32)
            colors = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [rgb[0], rgb[1], rgb[2], 1.0],
                    [rgb[0], rgb[1], rgb[2], 1.0],
                ],
                dtype=np.float32,
            )
        name_suffix = f"{gain:.2f}-gain"
    elif transfer == "volume":
        alpha = float(np.clip(alpha, 0.0, 1.0))
        controls = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        alpha_values = np.sqrt(controls) * alpha
        alpha_values[0] = 0.0
        colors = np.column_stack(
            [
                np.full_like(controls, rgb[0]),
                np.full_like(controls, rgb[1]),
                np.full_like(controls, rgb[2]),
                alpha_values,
            ]
        ).astype(np.float32, copy=False)
        name_suffix = f"{alpha:.2f}-volume"
    elif transfer == "step":
        alpha = float(np.clip(alpha, 0.0, 1.0))
        controls = np.array([0.0, 1e-6, 1.0], dtype=np.float32)
        colors = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [rgb[0], rgb[1], rgb[2], alpha],
                [rgb[0], rgb[1], rgb[2], alpha],
            ],
            dtype=np.float32,
        )
        name_suffix = f"{alpha:.2f}-step"
    else:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        controls = np.array([0.0, 1.0], dtype=np.float32)
        colors = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [rgb[0], rgb[1], rgb[2], alpha],
            ],
            dtype=np.float32,
        )
        name_suffix = f"{alpha:.2f}"

    return Colormap(
        colors,
        name=f"racc-overlay-{color}-{name_suffix}",
        display_name=f"RACC overlay {color}",
        controls=controls,
        interpolation="linear",
        nan_color=[0.0, 0.0, 0.0, 0.0],
        low_color=[0.0, 0.0, 0.0, 0.0],
        high_color=colors[-1],
    )


def _racc_colormap_with_alpha(
    name: str | None,
    alpha_values: np.ndarray | None,
    internal_name: str,
) -> Colormap:
    colormap_name = racc_colormap_name(name)
    base = ensure_colormap(colormap_name)
    colors = np.asarray(base.colors, dtype=np.float32).copy()
    if alpha_values is not None:
        colors[:, 3] = np.asarray(alpha_values, dtype=np.float32)

    return Colormap(
        colors,
        name=internal_name,
        display_name=getattr(base, "_display_name", colormap_name),
        controls=np.asarray(base.controls, dtype=np.float32),
        interpolation=base.interpolation,
        nan_color=[0.0, 0.0, 0.0, 0.0],
        low_color=[0.0, 0.0, 0.0, 0.0],
        high_color=base.high_color,
    )
