from __future__ import annotations

import numpy as np

from napari_racc._scatter import (
    ScatterHistogramWidget,
    _line_square_intersections,
    _max_limiter_intersections,
    _strip_square_polygon,
)


class _Result:
    scatter_histogram = np.eye(256, dtype=np.float32)


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


def test_scatter_axis_labels_store_source_names(qtbot):
    widget = ScatterHistogramWidget()
    qtbot.addWidget(widget)

    widget.set_axis_labels("RACC Red Sphere", "RACC Green Sphere")

    assert widget._x_label == "RACC Red Sphere intensity"
    assert widget._y_label == "RACC Green Sphere intensity"


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
