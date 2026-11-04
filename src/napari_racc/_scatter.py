"""Lightweight scatter histogram widget for RACC diagnostics."""

from __future__ import annotations

from math import log1p

import numpy as np
from napari.utils.colormaps import ensure_colormap
from qtpy.QtCore import QMargins, QPointF, QRect, Qt
from qtpy.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import QSizePolicy, QWidget

from napari_racc._colormaps import racc_colormap_name
from napari_racc._racc import RaccResult


class ScatterHistogramWidget(QWidget):
    """Display a 2D channel-intensity histogram and RACC overlays."""

    _plot_margins = QMargins(72, 14, 14, 70)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 280)
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self._result: RaccResult | None = None
        self._image: QImage | None = None
        self._colormap_name = racc_colormap_name()
        self._x_label = "Channel 1 intensity"
        self._y_label = "Channel 2 intensity"
        self._show_percentile_fill = False
        self._log_intensity = False
        self._focus_thresholds = False
        self._show_fit_guides = True
        self.setToolTip(
            "Colour shows log-scaled voxel counts. Focus above thresholds plots "
            "only voxels that meet both thresholds and brightens sparse bins. "
            "Log intensity spaces the axes logarithmically, including zero.\n\n"
            "Red: fitted regression. Yellow dashes and optional shading: "
            "perpendicular distance cutoff. Orange dots: projected-intensity "
            "cap, drawn perpendicular to the fitted line."
        )

    def set_result(self, result: RaccResult | None) -> None:
        self._result = result
        self._refresh_image()
        self.update()

    def set_colormap(self, colormap_name: str) -> None:
        selected = racc_colormap_name(colormap_name)
        if selected == self._colormap_name:
            return
        self._colormap_name = selected
        self._refresh_image()
        self.update()

    def set_axis_labels(self, x_label: str, y_label: str) -> None:
        self._x_label = f"{x_label} intensity"
        self._y_label = f"{y_label} intensity"
        self.update()

    def set_show_percentile_fill(self, show: bool) -> None:
        self._show_percentile_fill = bool(show)
        self.update()

    def set_log_intensity(self, enabled: bool) -> None:
        if self._log_intensity != enabled:
            self._log_intensity = enabled
            self._refresh_image()
            self.update()

    def set_focus_thresholds(self, enabled: bool) -> None:
        if self._focus_thresholds != enabled:
            self._focus_thresholds = enabled
            self._refresh_image()
            self.update()

    def set_show_fit_guides(self, enabled: bool) -> None:
        self._show_fit_guides = enabled
        self.update()

    def _refresh_image(self) -> None:
        if self._result is None:
            self._image = None
            return

        histogram = self._result.scatter_histogram
        if self._focus_thresholds:
            overlap = getattr(self._result, "overlap_scatter_histogram", None)
            if overlap is not None:
                self._image = self._make_image(overlap, focused=True)
                return
        self._image = self._make_image(histogram)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if (self._log_intensity or self._focus_thresholds) and self._image is not None:
            if self._image.width() != self._plot_rect().width():
                self._refresh_image()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(8, 8, 10))

        if self._result is None or self._image is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignCenter, "No RACC result yet")
            return

        target = self._plot_rect()
        painter.drawImage(target, self._image)
        self._draw_overlays(painter, target)
        self._draw_axes(painter, target)
        self._draw_axis_labels(painter, target)

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        margins = self._plot_margins
        plot_width = max(1, width - margins.left() - margins.right())
        return max(self.minimumHeight(), plot_width + margins.top() + margins.bottom())

    def _plot_rect(self) -> QRect:
        margins = self.rect().marginsRemoved(self._plot_margins)
        side = max(1, min(margins.width(), margins.height()))
        left = margins.left() + max(0, (margins.width() - side) // 2)
        top = margins.top() + max(0, (margins.height() - side) // 2)
        return QRect(left, top, side, side)

    def _make_image(self, histogram: np.ndarray, *, focused: bool = False) -> QImage:
        hist = np.asarray(histogram, dtype=np.float32)
        values = np.log1p(hist.T) if hist.size else np.zeros((256, 256), np.float32)
        peak = float(values.max()) or 1.0
        if focused:
            # Keep singleton bins visible even beside a dense saturated column.
            values = np.where(values > 0, 0.32 + 0.68 * (values / peak), 0.0)
        else:
            # Empty background must not set the contrast for other bins.
            contrast = values.copy()
            contrast[0, 0] = 0.0
            values = np.clip(values / (float(contrast.max()) or peak), 0.0, 1.0)

        if self._log_intensity or focused:
            max_intensity = (
                self._result.parameters.intensity_max if self._result else 255.0
            )
            values = _resample_bins(
                values, self._plot_rect().width(), max_intensity, self._log_intensity
            )
        values = np.flipud(values)
        image_rgb = self._map_colormap(values)
        height, width, _channels = image_rgb.shape
        return QImage(
            image_rgb.data,
            width,
            height,
            width * 3,
            QImage.Format_RGB888,
        ).copy()

    def _map_colormap(self, values: np.ndarray) -> np.ndarray:
        rgba = ensure_colormap(self._colormap_name).map(values.ravel())
        rgb = rgba[:, :3].reshape((*values.shape, 3))
        return np.ascontiguousarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))

    def _plot_point(
        self, target: QRect, point: tuple[float, float], max_intensity: float
    ) -> QPointF:
        x, y = point
        if self._log_intensity:
            maximum = log1p(max_intensity)
            x, y = log1p(x) / maximum, log1p(y) / maximum
        else:
            x, y = x / max_intensity, y / max_intensity
        return QPointF(
            target.left() + (target.width() - 1) * x,
            target.bottom() - (target.height() - 1) * y,
        )

    def _plot_segment(
        self,
        target: QRect,
        start: tuple[float, float],
        end: tuple[float, float],
        max_intensity: float,
    ) -> list[QPointF]:
        first = self._plot_point(target, start, max_intensity)
        last = self._plot_point(target, end, max_intensity)
        if not self._log_intensity or start[0] == end[0] or start[1] == end[1]:
            return [first, last]

        # A straight line in data space generally bends on logarithmic axes.
        # Subdivide until each displayed chord is accurate to a fraction of a pixel.
        points = [first]

        def append_chords(a, b, p, q, depth):
            midpoint = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            mapped = self._plot_point(target, midpoint, max_intensity)
            error = max(
                abs(mapped.x() - (p.x() + q.x()) / 2.0),
                abs(mapped.y() - (p.y() + q.y()) / 2.0),
            )
            if depth < 12 and error > 0.25:
                append_chords(a, midpoint, p, mapped, depth + 1)
                append_chords(midpoint, b, mapped, q, depth + 1)
            else:
                points.append(q)

        append_chords(start, end, first, last, 0)
        return points

    def _draw_segment(self, painter, target, endpoints, max_intensity) -> None:
        painter.drawPolyline(
            QPolygonF(self._plot_segment(target, *endpoints, max_intensity))
        )

    def _draw_overlays(self, painter: QPainter, target: QRect) -> None:
        params = self._result.parameters
        max_intensity = params.intensity_max

        painter.save()
        painter.setClipRect(target, Qt.IntersectClip)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._show_fit_guides:
            self._draw_percentile_limits(painter, target)
            self._draw_max_limiter(painter, target)

        threshold_pen = QPen(QColor(255, 255, 255, 180), 1)
        painter.setPen(threshold_pen)
        painter.drawLine(
            self._plot_point(
                target, (params.threshold_1, max_intensity), max_intensity
            ),
            self._plot_point(target, (params.threshold_1, 0.0), max_intensity),
        )
        painter.drawLine(
            self._plot_point(target, (0.0, params.threshold_2), max_intensity),
            self._plot_point(
                target, (max_intensity, params.threshold_2), max_intensity
            ),
        )

        if self._show_fit_guides:
            self._draw_regression(painter, target)
        painter.restore()

    def _draw_regression(self, painter: QPainter, target: QRect) -> None:
        params = self._result.parameters
        max_intensity = params.intensity_max
        regression = _line_square_intersections(
            params.slope, params.intercept, max_intensity
        )
        if regression is not None:
            painter.setPen(QPen(QColor(255, 70, 70), 2))
            self._draw_segment(painter, target, regression, max_intensity)

        point_pen = QPen(QColor(255, 190, 50), 5)
        painter.setPen(point_pen)
        for point in (params.p0, params.pmax):
            if all(0.0 <= value <= max_intensity for value in point):
                painter.drawPoint(self._plot_point(target, point, max_intensity))

    def _draw_percentile_limits(self, painter: QPainter, target: QRect) -> None:
        params = self._result.parameters
        max_intensity = params.intensity_max
        distance = params.distance_threshold * max_intensity
        if distance <= 0 or not np.isfinite(distance):
            return

        offset = distance * float(np.hypot(params.slope, 1.0))
        if self._show_percentile_fill:
            polygon = _strip_square_polygon(
                params.slope, params.intercept, offset, max_intensity
            )
            if len(polygon) >= 3:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(255, 220, 40, 34)))
                points = []
                for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
                    segment = self._plot_segment(target, start, end, max_intensity)
                    points.extend(segment[:-1])
                painter.drawPolygon(QPolygonF(points))

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 220, 40, 120), 1, Qt.DashLine))
        for intercept in (params.intercept - offset, params.intercept + offset):
            boundary = _line_square_intersections(
                params.slope, intercept, max_intensity
            )
            if boundary is not None:
                self._draw_segment(painter, target, boundary, max_intensity)

    def _draw_max_limiter(self, painter: QPainter, target: QRect) -> None:
        params = self._result.parameters
        max_intensity = params.intensity_max
        intersections = _max_limiter_intersections(
            params.slope, params.pmax, max_intensity
        )
        if intersections is None:
            return

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 150, 30, 165), 1, Qt.DotLine))
        self._draw_segment(painter, target, intersections, max_intensity)

    def _draw_axes(self, painter: QPainter, target: QRect) -> None:
        max_intensity = float(self._result.parameters.intensity_max)
        ticks = [0.0, max_intensity]
        if self._log_intensity:
            ticks[1:1] = [
                value
                for value in (1, 3, 10, 30, 100, 300, 1000)
                if log1p(max_intensity) - log1p(value) >= 0.45
            ]

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(95, 95, 105, 160), 1))
        # Keep the top and right frames off the last intensity bins.
        right, top = target.right() + 1, target.top() - 1
        painter.drawLine(target.left(), top, right, top)
        painter.drawLine(right, top, right, target.bottom())

        axis_pen = QPen(QColor(235, 235, 235, 230), 1)
        painter.setPen(axis_pen)
        painter.drawLine(target.bottomLeft(), target.bottomRight())
        painter.drawLine(target.bottomLeft(), target.topLeft())

        label_font = painter.font()
        point_size = label_font.pointSizeF()
        if point_size > 0:
            label_font.setPointSizeF(max(7.0, point_size * 0.78))
        painter.setFont(label_font)
        metrics = painter.fontMetrics()
        label_color = QColor(230, 230, 230, 235)

        for value in ticks:
            point = self._plot_point(target, (value, value), max_intensity)
            x, y = round(point.x()), round(point.y())
            label = _format_axis_value(value)
            width = max(metrics.horizontalAdvance(label) + 4, 24)
            painter.setPen(axis_pen)
            painter.drawLine(x, target.bottom(), x, target.bottom() + 5)
            painter.drawLine(target.left() - 5, y, target.left(), y)
            painter.setPen(label_color)
            painter.drawText(
                QRect(x - width // 2, target.bottom() + 8, width, 18),
                Qt.AlignHCenter | Qt.AlignTop,
                label,
            )
            painter.drawText(
                QRect(target.left() - 12 - width, y - 9, width, 18),
                Qt.AlignRight | Qt.AlignVCenter,
                label,
            )

        painter.restore()

    def _draw_axis_labels(self, painter: QPainter, target: QRect) -> None:
        painter.setPen(QColor(220, 220, 220))
        metrics = painter.fontMetrics()

        suffix = " (log)" if self._log_intensity else ""
        x_text = metrics.elidedText(
            self._x_label + suffix, Qt.ElideMiddle, target.width()
        )
        x_rect = QRect(
            target.left(),
            target.bottom() + 34,
            target.width(),
            24,
        )
        painter.drawText(x_rect, Qt.AlignCenter, x_text)

        y_text = metrics.elidedText(
            self._y_label + suffix, Qt.ElideMiddle, target.height()
        )
        painter.save()
        painter.translate(target.left() - 58, target.center().y())
        painter.rotate(-90)
        y_rect = QRect(-target.height() // 2, -12, target.height(), 24)
        painter.drawText(y_rect, Qt.AlignCenter, y_text)
        painter.restore()


def _resample_bins(
    values: np.ndarray, size: int, max_intensity: float, logarithmic: bool
) -> np.ndarray:
    # Map display pixel edges back into the original intensity histogram.
    # Keep the brightest overlapping bin so a sparse high-intensity bin
    # remains visible even when several original bins share a display pixel.
    fractions = np.linspace(0.0, 1.0, size + 1)
    if logarithmic:
        fractions = np.expm1(fractions * log1p(max_intensity)) / max_intensity
    for axis in (0, 1):
        bins = values.shape[axis]
        edges = fractions * bins
        starts = np.clip(np.floor(edges[:-1]).astype(int), 0, bins - 1)
        stops = np.clip(np.ceil(edges[1:]).astype(int), starts + 1, bins)
        rows = np.moveaxis(values, axis, 0)
        projected = np.stack(
            [rows[a:b].max(axis=0) for a, b in zip(starts, stops, strict=True)]
        )
        values = np.moveaxis(projected, 0, axis)
    return values


def _line_square_intersections(
    slope: float,
    intercept: float,
    max_intensity: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    points: list[tuple[float, float]] = []

    for x_value in (0.0, max_intensity):
        y_value = slope * x_value + intercept
        if 0.0 <= y_value <= max_intensity:
            points.append((float(x_value), float(y_value)))

    if abs(slope) > np.finfo(float).eps:
        for y_value in (0.0, max_intensity):
            x_value = (y_value - intercept) / slope
            if 0.0 <= x_value <= max_intensity:
                points.append((float(x_value), float(y_value)))

    deduped: list[tuple[float, float]] = []
    for point in points:
        if not any(np.allclose(point, existing) for existing in deduped):
            deduped.append(point)

    if len(deduped) < 2:
        return None
    return deduped[0], deduped[1]


def _max_limiter_intersections(
    slope: float,
    pmax: tuple[float, float],
    max_intensity: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not np.all(np.isfinite([slope, pmax[0], pmax[1], max_intensity])):
        return None
    if abs(slope) <= np.finfo(float).eps:
        return _vertical_square_intersections(pmax[0], max_intensity)

    limiter_slope = -1.0 / slope
    limiter_intercept = pmax[1] - limiter_slope * pmax[0]
    return _line_square_intersections(
        limiter_slope,
        limiter_intercept,
        max_intensity,
    )


def _vertical_square_intersections(
    x_value: float,
    max_intensity: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not 0.0 <= x_value <= max_intensity:
        return None
    return (float(x_value), 0.0), (float(x_value), float(max_intensity))


def _strip_square_polygon(
    slope: float,
    intercept: float,
    offset: float,
    max_intensity: float,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    corners = [
        (0.0, 0.0),
        (float(max_intensity), 0.0),
        (float(max_intensity), float(max_intensity)),
        (0.0, float(max_intensity)),
    ]
    tolerance = np.finfo(float).eps * max(1.0, max_intensity)
    for point in corners:
        signed_offset = point[1] - (slope * point[0] + intercept)
        if abs(signed_offset) <= offset + tolerance:
            points.append(point)

    for boundary_intercept in (intercept - offset, intercept + offset):
        intersections = _line_square_intersections(
            slope,
            boundary_intercept,
            max_intensity,
        )
        if intersections is not None:
            points.extend(intersections)

    deduped = _unique_points(points)
    if len(deduped) < 3:
        return deduped

    center_x = float(np.mean([point[0] for point in deduped]))
    center_y = float(np.mean([point[1] for point in deduped]))
    return sorted(
        deduped,
        key=lambda point: np.arctan2(point[1] - center_y, point[0] - center_x),
    )


def _unique_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not any(np.allclose(point, existing) for existing in deduped):
            deduped.append((float(point[0]), float(point[1])))
    return deduped


def _format_axis_value(value: float) -> str:
    if not np.isfinite(value):
        return ""
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-6:
        return str(int(rounded))
    return f"{float(value):.3g}"
