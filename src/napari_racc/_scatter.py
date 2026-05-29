"""Lightweight scatter histogram widget for RACC diagnostics."""

from __future__ import annotations

import numpy as np
from napari.utils.colormaps import ensure_colormap
from qtpy.QtCore import QPointF, QRect, Qt
from qtpy.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import QWidget

from napari_racc._colormaps import racc_colormap_name
from napari_racc._racc import RaccResult


class ScatterHistogramWidget(QWidget):
    """Display a 2D channel-intensity histogram and RACC overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 280)
        self._result: RaccResult | None = None
        self._image: QImage | None = None
        self._colormap_name = racc_colormap_name()
        self._x_label = "Channel 1 intensity"
        self._y_label = "Channel 2 intensity"
        self._show_percentile_fill = False

    def set_result(self, result: RaccResult | None) -> None:
        self._result = result
        self._image = self._make_image(result.scatter_histogram) if result else None
        self.update()

    def set_colormap(self, colormap_name: str) -> None:
        self._colormap_name = racc_colormap_name(colormap_name)
        if self._result is not None:
            self._image = self._make_image(self._result.scatter_histogram)
        self.update()

    def set_axis_labels(self, x_label: str, y_label: str) -> None:
        self._x_label = f"{x_label} intensity"
        self._y_label = f"{y_label} intensity"
        self.update()

    def set_show_percentile_fill(self, show: bool) -> None:
        self._show_percentile_fill = bool(show)
        self.update()

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
        self._draw_axis_labels(painter, target)

    def _plot_rect(self) -> QRect:
        margins = self.rect().adjusted(42, 10, -10, -44)
        side = max(1, min(margins.width(), margins.height()))
        left = margins.left() + max(0, (margins.width() - side) // 2)
        top = margins.top() + max(0, (margins.height() - side) // 2)
        return QRect(left, top, side, side)

    def _make_image(self, histogram: np.ndarray) -> QImage:
        hist = np.asarray(histogram, dtype=np.float32)
        if hist.size == 0 or hist.max() <= 0:
            values = np.zeros((256, 256), dtype=np.float32)
        else:
            values = np.log1p(hist.T)
            values = values / values.max()
            values = np.flipud(values).astype(np.float32)
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

    def _draw_overlays(self, painter: QPainter, target) -> None:
        params = self._result.parameters
        max_intensity = params.intensity_max

        def px(x_value: float) -> float:
            return target.left() + target.width() * x_value / max_intensity

        def py(y_value: float) -> float:
            return target.bottom() - target.height() * y_value / max_intensity

        painter.setRenderHint(QPainter.Antialiasing)

        self._draw_percentile_limits(painter, target)
        self._draw_max_limiter(painter, target)

        threshold_pen = QPen(QColor(255, 255, 255, 180), 1)
        painter.setPen(threshold_pen)
        painter.drawLine(
            QPointF(px(params.threshold_1), target.top()),
            QPointF(px(params.threshold_1), target.bottom()),
        )
        painter.drawLine(
            QPointF(target.left(), py(params.threshold_2)),
            QPointF(target.right(), py(params.threshold_2)),
        )

        line_pen = QPen(QColor(255, 70, 70), 2)
        painter.setPen(line_pen)
        x0 = 0.0
        x1 = max_intensity
        y0 = np.clip(params.slope * x0 + params.intercept, 0, max_intensity)
        y1 = np.clip(params.slope * x1 + params.intercept, 0, max_intensity)
        painter.drawLine(QPointF(px(x0), py(y0)), QPointF(px(x1), py(y1)))

        point_pen = QPen(QColor(255, 190, 50), 5)
        painter.setPen(point_pen)
        painter.drawPoint(QPointF(px(params.p0[0]), py(params.p0[1])))
        painter.drawPoint(QPointF(px(params.pmax[0]), py(params.pmax[1])))

    def _draw_percentile_limits(self, painter: QPainter, target: QRect) -> None:
        params = self._result.parameters
        max_intensity = params.intensity_max
        distance = params.distance_threshold * max_intensity
        if distance <= 0 or not np.isfinite(distance):
            return

        offset = distance * float(np.sqrt(params.slope * params.slope + 1.0))
        lower = _line_square_intersections(
            params.slope,
            params.intercept - offset,
            max_intensity,
        )
        upper = _line_square_intersections(
            params.slope,
            params.intercept + offset,
            max_intensity,
        )
        if lower is None or upper is None:
            return

        def point(data_point: tuple[float, float]) -> QPointF:
            x_value, y_value = data_point
            return QPointF(
                target.left() + target.width() * x_value / max_intensity,
                target.bottom() - target.height() * y_value / max_intensity,
            )

        lower_points = [point(lower[0]), point(lower[1])]
        upper_points = [point(upper[1]), point(upper[0])]
        if self._show_percentile_fill:
            polygon_points = _strip_square_polygon(
                params.slope,
                params.intercept,
                offset,
                max_intensity,
            )
            if len(polygon_points) >= 3:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(255, 220, 40, 34)))
                painter.drawPolygon(QPolygonF([point(p) for p in polygon_points]))

        band_pen = QPen(QColor(255, 220, 40, 120), 1, Qt.DashLine)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(band_pen)
        painter.drawLine(lower_points[0], lower_points[1])
        painter.drawLine(upper_points[1], upper_points[0])

    def _draw_max_limiter(self, painter: QPainter, target: QRect) -> None:
        params = self._result.parameters
        max_intensity = params.intensity_max
        intersections = _max_limiter_intersections(
            params.slope,
            params.pmax,
            max_intensity,
        )
        if intersections is None:
            return

        def point(data_point: tuple[float, float]) -> QPointF:
            x_value, y_value = data_point
            return QPointF(
                target.left() + target.width() * x_value / max_intensity,
                target.bottom() - target.height() * y_value / max_intensity,
            )

        max_pen = QPen(QColor(255, 150, 30, 165), 1, Qt.DotLine)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(max_pen)
        painter.drawLine(point(intersections[0]), point(intersections[1]))

    def _draw_axis_labels(self, painter: QPainter, target: QRect) -> None:
        painter.setPen(QColor(220, 220, 220))
        metrics = painter.fontMetrics()

        x_text = metrics.elidedText(self._x_label, Qt.ElideMiddle, target.width())
        x_rect = QRect(
            target.left(),
            target.bottom() + 12,
            target.width(),
            24,
        )
        painter.drawText(x_rect, Qt.AlignCenter, x_text)

        y_text = metrics.elidedText(self._y_label, Qt.ElideMiddle, target.height())
        painter.save()
        painter.translate(target.left() - 30, target.center().y())
        painter.rotate(-90)
        y_rect = QRect(-target.height() // 2, -12, target.height(), 24)
        painter.drawText(y_rect, Qt.AlignCenter, y_text)
        painter.restore()


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
