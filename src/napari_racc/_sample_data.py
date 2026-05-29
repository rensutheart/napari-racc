"""Sample data contributions for local RACC development."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile

from napari_racc._racc import reduce_to_intensity


def _legacy_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "RACC_v0.8_2"


def _has_legacy_data(*filenames: str) -> bool:
    data_dir = _legacy_data_dir()
    return all((data_dir / filename).is_file() for filename in filenames)


def make_sample_data_2d():
    """Return a 2D red/green RACC example as napari layers."""

    if _has_legacy_data("Red2D.png", "Green2D.png"):
        data_dir = _legacy_data_dir()
        red = reduce_to_intensity(iio.imread(data_dir / "Red2D.png"), rgb=True)
        green = reduce_to_intensity(iio.imread(data_dir / "Green2D.png"), rgb=True)
    else:
        red, green = _synthetic_2d_example()
    return [
        (
            red,
            {
                "name": "RACC Red 2D",
                "colormap": "red",
                "blending": "additive",
                "contrast_limits": (0, 255),
            },
            "image",
        ),
        (
            green,
            {
                "name": "RACC Green 2D",
                "colormap": "green",
                "blending": "additive",
                "contrast_limits": (0, 255),
            },
            "image",
        ),
    ]


def make_sample_data_3d():
    """Return a synthetic 3D sphere RACC example as napari layers."""

    if _has_legacy_data("RedSphere.tif", "GreenSphere.tif"):
        data_dir = _legacy_data_dir()
        red = reduce_to_intensity(tifffile.imread(data_dir / "RedSphere.tif"), rgb=True)
        green = reduce_to_intensity(
            tifffile.imread(data_dir / "GreenSphere.tif"),
            rgb=True,
        )
    else:
        red, green = _synthetic_3d_example()
    return [
        (
            red,
            {
                "name": "RACC Red Sphere",
                "colormap": "red",
                "blending": "additive",
                "contrast_limits": (0, 255),
                "depiction": "volume",
                "rendering": "translucent",
            },
            "image",
        ),
        (
            green,
            {
                "name": "RACC Green Sphere",
                "colormap": "green",
                "blending": "additive",
                "contrast_limits": (0, 255),
                "depiction": "volume",
                "rendering": "translucent",
            },
            "image",
        ),
    ]


def _synthetic_2d_example() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices((256, 256), dtype=np.float32)
    red = _soft_disk(y, x, center=(128, 112), radius=72)
    green = _soft_disk(y, x, center=(128, 144), radius=72)
    return red, green


def _synthetic_3d_example() -> tuple[np.ndarray, np.ndarray]:
    z, y, x = np.indices((31, 128, 128), dtype=np.float32)
    red = _soft_sphere(z, y, x, center=(15, 64, 54), radius=38)
    green = _soft_sphere(z, y, x, center=(15, 64, 74), radius=38)
    return red, green


def _soft_disk(
    y: np.ndarray,
    x: np.ndarray,
    *,
    center: tuple[float, float],
    radius: float,
) -> np.ndarray:
    distance = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    intensity = np.clip(1.0 - distance / radius, 0.0, 1.0)
    return (np.sqrt(intensity) * 255.0).astype(np.float32, copy=False)


def _soft_sphere(
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    *,
    center: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    distance = np.sqrt(
        (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    )
    intensity = np.clip(1.0 - distance / radius, 0.0, 1.0)
    return (np.sqrt(intensity) * 255.0).astype(np.float32, copy=False)
