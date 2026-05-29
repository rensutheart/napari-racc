from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import tifffile

from napari_racc._racc import (
    RaccError,
    compute_costes_thresholds,
    compute_racc,
    reduce_to_intensity,
)


def _example_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "RACC_v0.8_2"


def _has_legacy_examples() -> bool:
    data_dir = _example_dir()
    return all(
        (data_dir / filename).is_file()
        for filename in ("Red2D.png", "Green2D.png", "RedSphere.tif", "GreenSphere.tif")
    )


def test_perfect_correlation_returns_expected_regression():
    values = np.linspace(0, 255, 256, dtype=np.float32).reshape(16, 16)
    result = compute_racc(values, values, threshold_1=10, threshold_2=10)

    assert result.index.shape == values.shape
    assert result.index.dtype == np.float32
    assert result.parameters.slope == pytest.approx(1.0)
    assert result.parameters.intercept == pytest.approx(0.0, abs=1e-5)
    assert result.index.max() > 0.9


def test_regression_uses_overlap_population_not_single_channel_context():
    ch1 = np.zeros((12, 12), dtype=np.float32)
    ch2 = np.zeros((12, 12), dtype=np.float32)

    overlap_x = np.array([20, 40, 60, 80, 100], dtype=np.float32)
    overlap_y = overlap_x * 2
    ch1[0, :5] = overlap_x
    ch2[0, :5] = overlap_y

    ch1[1:6, 6:] = 255
    ch2[6:, 1:6] = 255

    result = compute_racc(ch1, ch2, threshold_1=5, threshold_2=5)

    assert result.parameters.overlap_voxels == 5
    assert result.parameters.slope == pytest.approx(2.0, rel=1e-6)
    assert np.count_nonzero(result.index[ch1 < 5]) == 0
    assert np.count_nonzero(result.index[ch2 < 5]) == 0


def test_empty_overlap_mask_raises():
    ch1 = np.ones((8, 8), dtype=np.float32) * 10
    ch2 = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(RaccError, match="Fewer than two voxels"):
        compute_racc(ch1, ch2, threshold_1=5, threshold_2=5)


def test_costes_thresholds_follow_regression_to_zero_below_threshold():
    rng = np.random.default_rng(0)
    background_x = np.linspace(0, 90, 200, dtype=np.float32)
    background_y = 90 - background_x
    signal_x = np.linspace(100, 240, 200, dtype=np.float32)
    signal_y = signal_x + rng.normal(0, 3, 200).astype(np.float32)
    ch1 = np.concatenate([background_x, signal_x]).reshape(20, 20)
    ch2 = np.concatenate([background_y, signal_y]).reshape(20, 20)

    thresholds = compute_costes_thresholds(ch1, ch2)

    assert 0 <= thresholds.threshold_1 <= 255
    assert 0 <= thresholds.threshold_2 <= 255
    assert thresholds.threshold_1 == pytest.approx(thresholds.threshold_2, abs=2)
    assert abs(thresholds.pearson_below) < 0.05
    assert thresholds.iterations > 0


def test_costes_thresholds_for_perfect_correlation_reach_low_threshold():
    values = np.linspace(0, 255, 256, dtype=np.float32).reshape(16, 16)

    thresholds = compute_costes_thresholds(values, values)

    assert thresholds.threshold_1 <= 1
    assert thresholds.threshold_2 <= 1
    assert thresholds.slope == pytest.approx(1.0)


@pytest.mark.skipif(
    not _has_legacy_examples(),
    reason="legacy RACC example files are not present",
)
def test_supplied_2d_examples_compute():
    data_dir = _example_dir()
    red = reduce_to_intensity(iio.imread(data_dir / "Red2D.png"), rgb=True)
    green = reduce_to_intensity(iio.imread(data_dir / "Green2D.png"), rgb=True)

    result = compute_racc(red, green, threshold_1=5, threshold_2=5)

    assert result.index.shape == red.shape
    assert result.parameters.overlap_voxels > 0
    assert result.index.max() > 0


@pytest.mark.skipif(
    not _has_legacy_examples(),
    reason="legacy RACC example files are not present",
)
def test_supplied_3d_stack_crop_computes():
    data_dir = _example_dir()
    red = tifffile.imread(data_dir / "RedSphere.tif")
    green = tifffile.imread(data_dir / "GreenSphere.tif")

    red = reduce_to_intensity(red[10:20, 384:640, 384:640], rgb=True)
    green = reduce_to_intensity(green[10:20, 384:640, 384:640], rgb=True)
    result = compute_racc(red, green, threshold_1=5, threshold_2=5)

    assert result.index.shape == red.shape
    assert result.parameters.overlap_voxels > 0
    assert result.index.max() > 0
