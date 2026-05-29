"""Sample data contributions for local RACC development."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import tifffile

from napari_racc._racc import reduce_to_intensity


def _legacy_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "RACC_v0.8_2"


def make_sample_data_2d():
    """Return the supplied 2D red/green examples as napari layers."""

    data_dir = _legacy_data_dir()
    red = reduce_to_intensity(iio.imread(data_dir / "Red2D.png"), rgb=True)
    green = reduce_to_intensity(iio.imread(data_dir / "Green2D.png"), rgb=True)
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
    """Return the supplied synthetic 3D sphere examples as napari layers."""

    data_dir = _legacy_data_dir()
    red = reduce_to_intensity(tifffile.imread(data_dir / "RedSphere.tif"), rgb=True)
    green = reduce_to_intensity(tifffile.imread(data_dir / "GreenSphere.tif"), rgb=True)
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
