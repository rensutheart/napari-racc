"""Colormap helpers."""

from __future__ import annotations

import numpy as np
from napari.utils.colormaps import Colormap, ensure_colormap

DEFAULT_RACC_COLORMAP = "magma"
RACC_DISPLAY_MIN = 1e-6
DEFAULT_VOLUME_OPACITY = 1.0
DEFAULT_BACKGROUND_SUPPRESSION = 2.0

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
    """Return fixed RACC display limits that preserve the 0..1 color meaning."""

    return (RACC_DISPLAY_MIN, 1.0)


def racc_colormap(name: str | None = None, floor: float = 0.0) -> Colormap:
    """Return a fixed-scale RACC colormap with values through ``floor`` hidden."""

    colormap_name = racc_colormap_name(name)
    floor = _display_floor(floor)
    controls, colors = _sample_racc_colormap(colormap_name, floor)
    colors[:, 3] = np.where(controls <= floor, 0.0, colors[:, 3])
    return _make_racc_colormap(
        colormap_name,
        controls,
        colors,
        internal_name=(
            f"racc-{colormap_name}-transparent"
            if floor == 0.0
            else f"racc-{colormap_name}-floor-{floor:.3f}"
        ),
    )


def racc_volume_colormap(
    name: str | None = None,
    opacity: float = DEFAULT_VOLUME_OPACITY,
    suppression: float = DEFAULT_BACKGROUND_SUPPRESSION,
    floor: float = 0.0,
) -> Colormap:
    """Return an opacity-scaled RACC transfer function for volume rendering."""

    colormap_name = racc_colormap_name(name)
    opacity = float(np.clip(opacity, 0.0, 1.0))
    suppression = max(float(suppression), 1.0)
    floor = _display_floor(floor)
    controls, colors = _sample_racc_colormap(colormap_name, floor)
    visible = np.clip((controls - floor) / max(1.0 - floor, RACC_DISPLAY_MIN), 0, 1)
    colors[:, 3] = visible**suppression * opacity
    colors[0, 3] = 0.0

    return _make_racc_colormap(
        colormap_name,
        controls,
        colors,
        internal_name=(
            f"racc-{colormap_name}-volume-{opacity:.3f}"
            f"-s{suppression:.2f}-floor-{floor:.3f}"
        ),
    )


def overlay_channel_colormap(
    color: str,
    alpha: float = 1.0,
    *,
    transfer: str = "linear",
    suppression: float = DEFAULT_BACKGROUND_SUPPRESSION,
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
        suppression = max(float(suppression), 1.0)
        controls = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        alpha_values = controls**suppression * alpha
        alpha_values[0] = 0.0
        colors = np.column_stack(
            [
                np.full_like(controls, rgb[0]),
                np.full_like(controls, rgb[1]),
                np.full_like(controls, rgb[2]),
                alpha_values,
            ]
        ).astype(np.float32, copy=False)
        name_suffix = f"{alpha:.3f}-volume-s{suppression:.2f}"
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


def _display_floor(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0 - RACC_DISPLAY_MIN))


def _sample_racc_colormap(
    name: str,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    base = ensure_colormap(name)
    controls = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 257, dtype=np.float32),
                np.asarray(base.controls, dtype=np.float32),
                np.asarray([floor], dtype=np.float32),
            ]
        )
    )
    colors = np.asarray(base.map(controls), dtype=np.float32).copy()
    return controls, colors


def _make_racc_colormap(
    name: str | None,
    controls: np.ndarray,
    colors: np.ndarray,
    internal_name: str,
) -> Colormap:
    colormap_name = racc_colormap_name(name)
    base = ensure_colormap(colormap_name)

    return Colormap(
        colors,
        name=internal_name,
        display_name=getattr(base, "_display_name", colormap_name),
        controls=controls,
        interpolation="linear",
        nan_color=[0.0, 0.0, 0.0, 0.0],
        low_color=[0.0, 0.0, 0.0, 0.0],
        high_color=colors[-1],
    )
