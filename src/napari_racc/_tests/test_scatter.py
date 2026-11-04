from __future__ import annotations

from math import log1p
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtGui import QImage, QPainter

from napari_racc._scatter import (
    ScatterHistogramWidget,
    _format_axis_value,
    _line_square_intersections,
    _max_limiter_intersections,
    _strip_square_polygon,
)


class _Result:
    scatter_histogram = np.eye(256, dtype=np.float32)
    parameters = SimpleNamespace(
        intensity_max=255.0,
        threshold_1=5.0,
        threshold_2=5.0,
        slope=1.0,
        intercept=0.0,
        p0=(5.0, 5.0),
        pmax=(200.0, 200.0),
        distance_threshold=0.05,
    )


def test_scatter_histogram_uses_selected_colormap(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)

    widget.set_colormap("viridis")
    widget.set_result(_Result())

    assert widget._colormap_name == "viridis"
    assert widget._image is not None
    assert widget._image.width() == 256
    assert widget._image.height() == 256


def test_scatter_plot_area_is_square(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)
    widget.resize(320, 260)

    plot_rect = widget._plot_rect()

    assert plot_rect.width() == plot_rect.height()


def test_scatter_contrast_is_independent_of_empty_background_size(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)
    hist = np.zeros((256, 256), dtype=np.float32)
    occupied = [(80, 120, 8), (90, 130, 64), (100, 140, 512)]
    for x, y, count in occupied:
        hist[x, y] = count
    baseline = widget._make_image(hist)
    hist[0, 0] = 1_000_000
    original = hist.copy()
    with_background = widget._make_image(hist)

    for x, y, _count in occupied:
        assert baseline.pixelColor(x, 255 - y) == with_background.pixelColor(x, 255 - y)
    assert with_background.pixelColor(80, 255 - 120) != with_background.pixelColor(0, 0)
    np.testing.assert_array_equal(hist, original)


def test_scatter_background_only_and_empty_histograms_render_safely(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)
    empty = np.zeros((256, 256), dtype=np.float32)
    background = empty.copy()
    background[0, 0] = 1_000_000

    empty_image = widget._make_image(empty)
    background_image = widget._make_image(background)
    assert not empty_image.isNull()
    assert not background_image.isNull()
    assert background_image.pixelColor(0, 0) == empty_image.pixelColor(0, 0)
    assert background_image.pixelColor(0, 255) != empty_image.pixelColor(0, 255)


def test_scatter_plot_area_leaves_room_for_axis_values(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)
    widget.resize(320, 260)

    plot_rect = widget._plot_rect()

    assert plot_rect.left() >= 70
    assert widget.height() - plot_rect.bottom() >= 60


def test_scatter_axis_labels_store_source_names(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)

    widget.set_axis_labels("RACC Red Sphere", "RACC Green Sphere")

    assert widget._x_label == "RACC Red Sphere intensity"
    assert widget._y_label == "RACC Green Sphere intensity"


def test_scatter_paints_axes_and_endpoint_values(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)
    widget.resize(360, 320)
    widget.set_result(_Result())
    image = QImage(widget.size(), QImage.Format_ARGB32)
    painter = QPainter(image)

    widget.render(painter)
    painter.end()

    assert image.width() == 360
    assert image.height() == 320


def test_scatter_renders_steep_regression_at_its_actual_slope(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)
    widget.resize(420, 340)
    params = vars(_Result.parameters) | {"slope": 13.0, "intercept": -3165.0}
    widget.set_result(
        SimpleNamespace(
            scatter_histogram=np.zeros((256, 256)),
            parameters=SimpleNamespace(**params),
        )
    )

    image = widget.grab().toImage()
    plot = widget._plot_rect()

    def at_intensity(x, y):
        col = round(plot.left() + (plot.width() - 1) * x / 255)
        row = round(plot.bottom() - (plot.height() - 1) * y / 255)
        return image.pixelColor(col, row)

    assert at_intensity(250, 85).red() > 200
    assert at_intensity(127.5, 75).red() < 100


@pytest.mark.parametrize(
    ("offset", "far_corner_is_filled"), [(200.0, False), (400.0, True)]
)
def test_scatter_fills_band_when_a_boundary_misses_plot(
    qtbot, offset, far_corner_is_filled
):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)
    widget.resize(420, 340)
    params = vars(_Result.parameters) | {
        "intercept": 100.0,
        "distance_threshold": offset / (255.0 * np.sqrt(2.0)),
    }
    widget.set_result(
        SimpleNamespace(
            scatter_histogram=np.zeros((256, 256)),
            parameters=SimpleNamespace(**params),
        )
    )
    unfilled = widget.grab().toImage()
    widget.set_show_percentile_fill(True)
    filled = widget.grab().toImage()
    plot = widget._plot_rect()

    def is_shaded(x, y):
        col = round(plot.left() + (plot.width() - 1) * x / 255)
        row = round(plot.bottom() - (plot.height() - 1) * y / 255)
        return filled.pixelColor(col, row) != unfilled.pixelColor(col, row)

    assert is_shaded(45, 175)
    assert is_shaded(240, 50) is far_corner_is_filled


def test_scatter_log_axes_keep_sparse_high_intensity_bins_and_curve_the_fit(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)
    widget.resize(420, 340)
    plot = widget._plot_rect()
    histogram = np.zeros((256, 256), dtype=np.float32)
    histogram[254, 10] = 15
    params = vars(_Result.parameters) | {"slope": 0.5, "intercept": 0.0}
    widget.set_result(
        SimpleNamespace(
            scatter_histogram=histogram, parameters=SimpleNamespace(**params)
        )
    )
    linear_image = widget._image
    widget.set_log_intensity(True)
    image = widget.grab().toImage()

    def pixel_at(x, y):
        return image.pixelColor(
            round(plot.left() + (plot.width() - 1) * log1p(x) / log1p(255)),
            round(plot.bottom() - (plot.height() - 1) * log1p(y) / log1p(255)),
        )

    assert pixel_at(254, 10).red() > 200
    # At x=10, the fitted line has y=5. Mapping just the two endpoints
    # would draw a chord much higher up, through roughly y=7.
    assert pixel_at(10, 5).red() > 150
    assert pixel_at(10, 7).red() < 100
    widget.set_log_intensity(False)
    assert widget._image == linear_image


def test_scatter_focus_removes_ineligible_bins_and_reveals_sparse_bins(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)
    widget.resize(420, 340)
    plot = widget._plot_rect()
    histogram = np.zeros((256, 256), dtype=np.float32)
    histogram[20, 35] = 1_000_000
    histogram[100, 125] = 1
    histogram[255, 120] = 566
    overlap = histogram.copy()
    overlap[20, 35] = 0
    params = vars(_Result.parameters) | {"threshold_1": 27.0, "threshold_2": 58.0}
    result = SimpleNamespace(
        scatter_histogram=histogram,
        overlap_scatter_histogram=overlap,
        parameters=SimpleNamespace(**params),
    )
    widget.set_result(result)
    initial = widget.grab().toImage()
    widget.set_focus_thresholds(True)
    focused = widget.grab().toImage()

    def pixel_at(image, x, y):
        return image.pixelColor(
            round(plot.left() + (plot.width() - 1) * x / 255),
            round(plot.bottom() - (plot.height() - 1) * y / 255),
        ).red()

    assert pixel_at(focused, 100, 125) > pixel_at(initial, 100, 125) + 60
    assert pixel_at(focused, 20, 35) == pixel_at(focused, 18, 35)
    assert pixel_at(focused, 255, 120) > 200
    assert histogram[20, 35] == 1_000_000
    assert overlap[100, 125] == 1


def test_scatter_percentile_fill_defaults_off_and_can_toggle(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)

    assert widget._show_percentile_fill is False

    widget.set_show_percentile_fill(True)

    assert widget._show_percentile_fill is True


def test_line_square_intersections_for_percentile_band():
    intersections = _line_square_intersections(1.0, 0.0, 255.0)

    assert intersections == ((0.0, 0.0), (255.0, 255.0))


def test_max_limiter_intersections_are_perpendicular_to_regression():
    intersections = _max_limiter_intersections(1.0, (127.5, 127.5), 255.0)

    assert intersections == ((0.0, 255.0), (255.0, 0.0))


def test_strip_square_polygon_returns_percentile_band_area():
    polygon = _strip_square_polygon(1.0, 0.0, 30.0, 255.0)

    assert len(polygon) >= 3


def test_format_axis_value_prefers_clean_endpoints():
    assert _format_axis_value(0.0) == "0"
    assert _format_axis_value(255.0) == "255"
    assert _format_axis_value(1.23456) == "1.23"
