from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import tifffile

from napari_racc._racc import compute_racc, reduce_to_intensity


def main() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "RACC_v0.8_2"

    red_2d = reduce_to_intensity(iio.imread(data_dir / "Red2D.png"), rgb=True)
    green_2d = reduce_to_intensity(iio.imread(data_dir / "Green2D.png"), rgb=True)
    result_2d = compute_racc(red_2d, green_2d, threshold_1=5, threshold_2=5)
    print(
        "2D:",
        result_2d.index.shape,
        result_2d.index.dtype,
        result_2d.parameters.overlap_voxels,
        float(result_2d.index.max()),
    )

    red_3d = tifffile.imread(data_dir / "RedSphere.tif")
    green_3d = tifffile.imread(data_dir / "GreenSphere.tif")
    red_3d = reduce_to_intensity(red_3d, rgb=True)
    green_3d = reduce_to_intensity(green_3d, rgb=True)
    result_3d = compute_racc(red_3d, green_3d, threshold_1=5, threshold_2=5)
    print(
        "3D:",
        result_3d.index.shape,
        result_3d.index.dtype,
        result_3d.parameters.overlap_voxels,
        float(result_3d.index.max()),
    )


if __name__ == "__main__":
    main()

