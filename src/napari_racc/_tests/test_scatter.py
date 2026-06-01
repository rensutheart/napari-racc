from __future__ import annotations

from types import SimpleNamespace

import numpy as np
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
