"""Core RACC calculation.

This module intentionally has no napari or Qt imports. The default algorithm
uses one overlap mask consistently for regression, threshold statistics,
distance filtering, and output eligibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

ChannelMode = Literal["max", "red", "green", "blue", "luminance"]


class RaccError(ValueError):
    """Raised when RACC cannot be computed for the supplied data."""


@dataclass(frozen=True)
class RaccParameters:
    threshold_1: float
    threshold_2: float
    theta_degrees: float
    include_percentile: float
    intensity_max: float
    slope: float
    intercept: float
    p0: tuple[float, float]
    p1: tuple[float, float]
    pmax: tuple[float, float]
    distance_threshold: float
    overlap_voxels: int
    total_voxels: int
    overlap_fraction: float
    algorithm: str = "overlap_mask_3d"

    def to_dict(self) -> dict[str, float | int | str | tuple[float, float]]:
        return asdict(self)


@dataclass(frozen=True)
class RaccResult:
    index: np.ndarray
    parameters: RaccParameters
    scatter_histogram: np.ndarray
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CostesThresholds:
    threshold_1: float
    threshold_2: float
    slope: float
    intercept: float
    pearson_below: float
    iterations: int
    algorithm: str = "costes_bisection"


def reduce_to_intensity(
    data: np.ndarray,
    *,
    rgb: bool | None = None,
    channel_mode: ChannelMode = "max",
) -> np.ndarray:
    """Return a scalar intensity image from scalar or RGB/RGBA-like data."""

    array = np.asarray(data)
    if array.ndim < 2:
        raise RaccError("RACC requires at least 2D image data.")

    looks_rgb = array.ndim >= 3 and array.shape[-1] in (3, 4)
    use_rgb = looks_rgb if rgb is None else rgb
    if not use_rgb:
        return array

    rgb_array = array[..., :3]
    if channel_mode == "max":
        return np.max(rgb_array, axis=-1)
    if channel_mode == "red":
        return rgb_array[..., 0]
    if channel_mode == "green":
        return rgb_array[..., 1]
    if channel_mode == "blue":
        return rgb_array[..., 2]
    if channel_mode == "luminance":
        return (
            0.2126 * rgb_array[..., 0]
            + 0.7152 * rgb_array[..., 1]
            + 0.0722 * rgb_array[..., 2]
        )
    raise RaccError(f"Unsupported channel mode: {channel_mode!r}")


def normalize_channels(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    *,
    output_max: float = 255.0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Normalize two channels jointly into the 0..output_max range."""

    ch1 = np.asarray(channel_1, dtype=np.float32)
    ch2 = np.asarray(channel_2, dtype=np.float32)

    if ch1.shape != ch2.shape:
        raise RaccError(f"Input shapes differ: {ch1.shape} vs {ch2.shape}.")

    if not np.isfinite(ch1).any() or not np.isfinite(ch2).any():
        raise RaccError("Both channels must contain finite values.")

    warnings: list[str] = []
    max_intensity = float(np.nanmax([np.nanmax(ch1), np.nanmax(ch2)]))
    min_intensity = float(np.nanmin([np.nanmin(ch1), np.nanmin(ch2)]))

    if min_intensity < 0:
        warnings.append("Negative intensities were clipped to zero.")
        ch1 = np.clip(ch1, 0, None)
        ch2 = np.clip(ch2, 0, None)
        max_intensity = float(np.nanmax([np.nanmax(ch1), np.nanmax(ch2)]))

    if max_intensity <= 0:
        raise RaccError("Both channels are empty or zero-valued.")

    if max_intensity <= 1.0:
        scale = output_max
        warnings.append("Input appeared to be normalized float data; scaled to 0..255.")
    elif max_intensity > output_max:
        scale = output_max / max_intensity
        warnings.append(
            f"Input maximum {max_intensity:g} exceeded 255; jointly scaled to 0..255."
        )
    else:
        scale = 1.0

    ch1 = np.nan_to_num(ch1 * scale, nan=0.0, posinf=output_max, neginf=0.0)
    ch2 = np.nan_to_num(ch2 * scale, nan=0.0, posinf=output_max, neginf=0.0)
    ch1 = np.clip(ch1, 0.0, output_max).astype(np.float32, copy=False)
    ch2 = np.clip(ch2, 0.0, output_max).astype(np.float32, copy=False)

    return ch1, ch2, tuple(warnings)


def compute_scatter_histogram(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    *,
    bins: int = 256,
    intensity_max: float = 255.0,
) -> np.ndarray:
    """Compute a 2D channel-intensity histogram for diagnostics."""

    hist, _, _ = np.histogram2d(
        np.ravel(channel_1),
        np.ravel(channel_2),
        bins=bins,
        range=((0.0, intensity_max), (0.0, intensity_max)),
    )
    return hist.astype(np.float32, copy=False)


def compute_costes_thresholds(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    *,
    intensity_max: float = 255.0,
    max_iterations: int = 100,
    tolerance: float = 1.0,
) -> CostesThresholds:
    """Compute Costes automatic thresholds using bisection.

    The threshold point is moved along the orthogonal regression line until
    Pearson's correlation for pixels below either channel threshold approaches
    zero. Inputs are normalized with the same 0..``intensity_max`` convention as
    ``compute_racc``.
    """

    ch1, ch2, _warnings = normalize_channels(
        channel_1,
        channel_2,
        output_max=intensity_max,
    )
    if ch1.size < 2:
        raise RaccError("Costes thresholding requires at least two voxels.")

    mean_x, mean_y, var_xx, var_yy, var_xy = _channel_moments(ch1, ch2)
    slope = _costes_regression_slope(var_xx, var_yy, var_xy)
    intercept = mean_y - slope * mean_x

    max_x = float(np.nanmax(ch1))
    max_y = float(np.nanmax(ch2))
    min_x = float(np.nanmin(ch1))
    min_y = float(np.nanmin(ch2))

    if -1.0 < slope < 1.0:
        threshold = abs(max_x + min_x) * 0.5
        previous_threshold = max_x

        def map_threshold(value: float) -> tuple[float, float]:
            return value, value * slope + intercept

    else:
        threshold = abs(max_y + min_y) * 0.5
        previous_threshold = max_y

        def map_threshold(value: float) -> tuple[float, float]:
            return (value - intercept) / slope, value

    diff = abs(threshold - previous_threshold)
    pearson_below = float("nan")
    iterations = 0
    threshold_1 = threshold_2 = 0.0

    while iterations <= max_iterations and diff >= tolerance:
        threshold_1, threshold_2 = map_threshold(threshold)
        threshold_1 = _clamp(threshold_1, 0.0, intensity_max)
        threshold_2 = _clamp(threshold_2, 0.0, intensity_max)
        pearson_below = _pearson_below_threshold(ch1, ch2, threshold_1, threshold_2)

        previous_threshold, old_diff = threshold, diff
        if np.isfinite(pearson_below) and pearson_below > 0:
            threshold -= old_diff * 0.5
        else:
            threshold += old_diff * 0.5
        diff = abs(threshold - previous_threshold)
        iterations += 1

    threshold_1, threshold_2 = map_threshold(threshold)
    threshold_1 = float(_round_and_clamp(threshold_1, 0.0, intensity_max))
    threshold_2 = float(_round_and_clamp(threshold_2, 0.0, intensity_max))
    pearson_below = _pearson_below_threshold(ch1, ch2, threshold_1, threshold_2)

    return CostesThresholds(
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        slope=float(slope),
        intercept=float(intercept),
        pearson_below=float(pearson_below),
        iterations=int(iterations),
    )


def compute_racc(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    *,
    threshold_1: float = 5.0,
    threshold_2: float = 5.0,
    theta_degrees: float = 45.0,
    include_percentile: float = 99.0,
    intensity_max: float = 255.0,
    output_dtype: Literal["float32", "uint8"] = "float32",
) -> RaccResult:
    """Compute a scalar RACC index image.

    The returned index is in 0..1 for ``float32`` output and 0..255 for
    ``uint8`` output.
    """

    if not 0 <= threshold_1 <= intensity_max:
        raise RaccError("threshold_1 must be in the normalized intensity range.")
    if not 0 <= threshold_2 <= intensity_max:
        raise RaccError("threshold_2 must be in the normalized intensity range.")
    if not 0 <= theta_degrees < 90:
        raise RaccError("theta_degrees must satisfy 0 <= theta < 90.")
    if not 0 < include_percentile <= 100:
        raise RaccError("include_percentile must satisfy 0 < p <= 100.")

    ch1, ch2, warnings = normalize_channels(
        channel_1, channel_2, output_max=intensity_max
    )
    total_voxels = int(ch1.size)
    mask = (ch1 >= threshold_1) & (ch2 >= threshold_2)
    overlap_voxels = int(np.count_nonzero(mask))
    if overlap_voxels < 2:
        raise RaccError(
            "Fewer than two voxels pass both channel thresholds; RACC is undefined."
        )

    x = ch1[mask].astype(np.float64, copy=False)
    y = ch2[mask].astype(np.float64, copy=False)

    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    covariance = np.cov(x, y)
    var_xx = float(covariance[0, 0])
    var_yy = float(covariance[1, 1])
    var_xy = float(covariance[0, 1])

    slope = _positive_deming_slope(var_xx, var_yy, var_xy)
    intercept = mean_y - slope * mean_x

    p0 = _line_threshold_intersection(threshold_1, threshold_2, slope, intercept)
    p1 = _line_intensity_intersection(intensity_max, slope, intercept)

    q = np.column_stack((x, y))
    t_full = _projection_fraction(q, p0, p1)
    distances = _normalized_line_distance(q, p0, p1, intensity_max)

    percentile_fraction = include_percentile / 100.0
    t_max = float(np.quantile(t_full, percentile_fraction))
    if not np.isfinite(t_max):
        raise RaccError("Could not calculate pmax from the overlap population.")
    t_max = max(t_max, np.finfo(float).eps)
    pmax = (p0[0] + t_max * (p1[0] - p0[0]), p0[1] + t_max * (p1[1] - p0[1]))

    distance_threshold = float(np.quantile(distances, percentile_fraction))

    output = _calculate_index(
        ch1,
        ch2,
        mask,
        p0,
        pmax,
        distance_threshold,
        theta_degrees,
        intensity_max,
    )

    if output_dtype == "uint8":
        index = np.round(output * 255.0).astype(np.uint8)
    elif output_dtype == "float32":
        index = output.astype(np.float32, copy=False)
    else:
        raise RaccError(f"Unsupported output dtype: {output_dtype!r}")

    parameters = RaccParameters(
        threshold_1=float(threshold_1),
        threshold_2=float(threshold_2),
        theta_degrees=float(theta_degrees),
        include_percentile=float(include_percentile),
        intensity_max=float(intensity_max),
        slope=float(slope),
        intercept=float(intercept),
        p0=(float(p0[0]), float(p0[1])),
        p1=(float(p1[0]), float(p1[1])),
        pmax=(float(pmax[0]), float(pmax[1])),
        distance_threshold=float(distance_threshold),
        overlap_voxels=overlap_voxels,
        total_voxels=total_voxels,
        overlap_fraction=overlap_voxels / total_voxels,
    )

    return RaccResult(
        index=index,
        parameters=parameters,
        scatter_histogram=compute_scatter_histogram(
            ch1,
            ch2,
            intensity_max=intensity_max,
        ),
        warnings=warnings,
    )


def _channel_moments(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
) -> tuple[float, float, float, float, float]:
    n = float(channel_1.size)
    sum_x = float(np.sum(channel_1, dtype=np.float64))
    sum_y = float(np.sum(channel_2, dtype=np.float64))
    mean_x = sum_x / n
    mean_y = sum_y / n
    sum_xx = float(np.sum(channel_1 * channel_1, dtype=np.float64))
    sum_yy = float(np.sum(channel_2 * channel_2, dtype=np.float64))
    sum_xy = float(np.sum(channel_1 * channel_2, dtype=np.float64))
    var_xx = max(sum_xx / n - mean_x * mean_x, 0.0)
    var_yy = max(sum_yy / n - mean_y * mean_y, 0.0)
    var_xy = sum_xy / n - mean_x * mean_y
    return mean_x, mean_y, var_xx, var_yy, var_xy


def _pearson_below_threshold(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    threshold_1: float,
    threshold_2: float,
) -> float:
    mask = (channel_1 < threshold_1) | (channel_2 < threshold_2)
    count = int(np.count_nonzero(mask))
    if count < 2:
        return float("nan")

    inv_count = 1.0 / count
    sum_x = float(np.sum(channel_1, where=mask, dtype=np.float64))
    sum_y = float(np.sum(channel_2, where=mask, dtype=np.float64))
    sum_xx = float(np.sum(channel_1 * channel_1, where=mask, dtype=np.float64))
    sum_yy = float(np.sum(channel_2 * channel_2, where=mask, dtype=np.float64))
    sum_xy = float(np.sum(channel_1 * channel_2, where=mask, dtype=np.float64))

    numerator = sum_xy - sum_x * sum_y * inv_count
    denom_x = sum_xx - sum_x * sum_x * inv_count
    denom_y = sum_yy - sum_y * sum_y * inv_count
    denominator = denom_x * denom_y
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(numerator / np.sqrt(denominator))


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _round_and_clamp(value: float, lower: float, upper: float) -> int:
    return int(_clamp(np.floor(value + 0.5), lower, upper))


def _positive_deming_slope(var_xx: float, var_yy: float, var_xy: float) -> float:
    eps = np.finfo(float).eps
    if abs(var_xy) <= eps:
        if var_xx > eps and var_yy > eps:
            return float(np.sqrt(var_yy / var_xx))
        return 1.0

    delta = var_yy - var_xx
    root = float(np.sqrt(delta * delta + 4.0 * var_xy * var_xy))
    if var_xy >= 0:
        slope = (delta + root) / (2.0 * var_xy)
    else:
        slope = (delta - root) / (2.0 * var_xy)

    if not np.isfinite(slope) or slope <= 0:
        raise RaccError("Could not fit a positive Deming regression slope.")
    return float(slope)


def _costes_regression_slope(var_xx: float, var_yy: float, var_xy: float) -> float:
    eps = np.finfo(float).eps
    if abs(var_xy) <= eps:
        if var_xx > eps and var_yy > eps:
            return float(np.sqrt(var_yy / var_xx))
        return 1.0

    delta = var_yy - var_xx
    root = float(np.sqrt(delta * delta + 4.0 * var_xy * var_xy))
    slope = (delta + root) / (2.0 * var_xy)
    if not np.isfinite(slope) or abs(slope) <= eps:
        raise RaccError("Could not fit a Costes regression slope.")
    return float(slope)


def _line_threshold_intersection(
    threshold_1: float,
    threshold_2: float,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
    y_at_threshold_1 = threshold_1 * slope + intercept
    if threshold_2 <= y_at_threshold_1:
        return (float(threshold_1), float(y_at_threshold_1))
    return (float((threshold_2 - intercept) / slope), float(threshold_2))


def _line_intensity_intersection(
    intensity_max: float,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
    if intercept >= intensity_max * (1.0 - slope):
        return (float((intensity_max - intercept) / slope), float(intensity_max))
    return (float(intensity_max), float(intensity_max * slope + intercept))


def _projection_fraction(
    points: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> np.ndarray:
    p0_arr = np.asarray(p0, dtype=np.float64)
    vector = np.asarray(p1, dtype=np.float64) - p0_arr
    denom = float(np.dot(vector, vector))
    if denom <= np.finfo(float).eps:
        raise RaccError("Regression line has degenerate endpoints.")
    return ((points - p0_arr) @ vector) / denom


def _normalized_line_distance(
    points: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    intensity_max: float,
) -> np.ndarray:
    p0_arr = np.asarray(p0, dtype=np.float64)
    vector = np.asarray(p1, dtype=np.float64) - p0_arr
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).eps:
        raise RaccError("Regression line has degenerate endpoints.")
    relative = points - p0_arr
    distance = np.abs(vector[0] * relative[:, 1] - vector[1] * relative[:, 0]) / norm
    return distance / intensity_max


def _calculate_index(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    mask: np.ndarray,
    p0: tuple[float, float],
    pmax: tuple[float, float],
    distance_threshold: float,
    theta_degrees: float,
    intensity_max: float,
) -> np.ndarray:
    output = np.zeros(channel_1.shape, dtype=np.float32)
    if not np.any(mask):
        return output

    x = channel_1[mask].astype(np.float64, copy=False)
    y = channel_2[mask].astype(np.float64, copy=False)
    points = np.column_stack((x, y))
    t = _projection_fraction(points, p0, pmax)
    distances = _normalized_line_distance(points, p0, pmax, intensity_max)

    theta = np.deg2rad(theta_degrees)
    values = np.minimum(t, 1.0) - distances * np.tan(theta)
    values[(t <= 0.0) | (distances > distance_threshold)] = 0.0
    values = np.clip(values, 0.0, 1.0)
    output[mask] = values.astype(np.float32, copy=False)
    return output
