"""napari dock widget for interactive RACC calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from napari.layers import Image
from napari.qt.threading import create_worker
from napari.utils.colormaps import ensure_colormap
from qtpy.QtCore import QSignalBlocker, Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_racc._colormaps import (
    overlay_channel_colormap,
    overlay_color_choices,
    overlay_color_name,
    racc_colormap,
    racc_colormap_choices,
    racc_colormap_name,
    racc_display_contrast_limits,
    racc_volume_colormap,
)
from napari_racc._racc import (
    CostesThresholds,
    RaccError,
    compute_costes_thresholds,
    compute_racc,
    reduce_to_intensity,
)
from napari_racc._scatter import ScatterHistogramWidget
from napari_racc._views import (
    show_mips,
    show_overlay,
    show_racc_only,
    show_side_by_side,
    update_mips,
    update_raw_overlay_volume,
)

if TYPE_CHECKING:
    import napari

OVERLAY_VOLUME_ALPHA_MAX = 0.08
OVERLAY_VOLUME_ALPHA_EXPONENT = 3.0


@dataclass(frozen=True)
class _ComputeRequest:
    threshold_1: int
    threshold_2: int
    theta_degrees: int
    include_percentile: float


class RaccWidget(QWidget):
    """Interactive RACC analysis widget."""

    def __init__(self, viewer: napari.viewer.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._result_layer_name: str | None = None
        self._result_pair: tuple[str, str] | None = None
        self._pending_jobs: dict[int, tuple[str, str]] = {}
        self._latest_job_id = 0
        self._worker = None
        self._threshold_worker = None
        self._costes_pair: tuple[str, str] | None = None

        self.channel_1_combo = QComboBox()
        self.channel_2_combo = QComboBox()

        self.threshold_1_slider, self.threshold_1_spin = self._make_int_control(5)
        self.threshold_2_slider, self.threshold_2_spin = self._make_int_control(5)
        self.theta_slider, self.theta_spin = self._make_int_control(
            45,
            minimum=0,
            maximum=89,
        )

        self.percentile_spin = QDoubleSpinBox()
        self.percentile_spin.setRange(0.1, 100.0)
        self.percentile_spin.setSingleStep(0.5)
        self.percentile_spin.setDecimals(1)
        self.percentile_spin.setValue(99.0)

        self.xy_scale_slider, self.xy_scale_spin = self._make_scale_control(1.0)
        self.z_scale_slider, self.z_scale_spin = self._make_scale_control(1.0)
        self.bounding_box_checkbox = QCheckBox("Bounding box")
        self.channel_1_color_combo = QComboBox()
        self.channel_1_color_combo.addItems(overlay_color_choices())
        self.channel_1_color_combo.setCurrentText("red")
        self.channel_2_color_combo = QComboBox()
        self.channel_2_color_combo.addItems(overlay_color_choices())
        self.channel_2_color_combo.setCurrentText("green")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(racc_colormap_choices())
        self.colormap_combo.setCurrentText(racc_colormap_name())
        self.overlay_alpha_slider, self.overlay_alpha_spin = self._make_int_control(
            2,
            minimum=0,
            maximum=100,
        )
        self.volume_alpha_slider, self.volume_alpha_spin = self._make_int_control(
            2,
            minimum=1,
            maximum=100,
        )

        self.live_checkbox = QCheckBox("Live")
        self.live_checkbox.setChecked(True)
        self.replace_checkbox = QCheckBox("Update existing result")
        self.replace_checkbox.setChecked(True)

        self.costes_button = QPushButton("Costes thresholds")
        self.run_button = QPushButton("Run RACC")
        self.overlay_button = QPushButton("Overlay")
        self.racc_button = QPushButton("RACC")
        self.side_by_side_button = QPushButton("Side by side")
        self.mip_button = QPushButton("MIPs")

        self.status_label = QLabel("Select two image layers.")
        self.status_label.setWordWrap(True)
        self.scatter_widget = ScatterHistogramWidget()
        self.percentile_fill_checkbox = QCheckBox("Fill percentile band")
        self.percentile_fill_checkbox.setChecked(False)

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setInterval(250)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.run_racc)

        self._build_layout()
        self._connect_signals()
        self._refresh_layer_choices()

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)

        input_group = QGroupBox("Inputs")
        input_layout = QFormLayout(input_group)
        input_layout.addRow("Channel 1", self.channel_1_combo)
        input_layout.addRow("Channel 2", self.channel_2_combo)
        root.addWidget(input_group)

        parameter_group = QGroupBox("RACC parameters")
        parameter_layout = QGridLayout(parameter_group)
        self._add_labeled_control(
            parameter_layout,
            0,
            "Channel 1 threshold",
            self.threshold_1_slider,
            self.threshold_1_spin,
        )
        self._add_labeled_control(
            parameter_layout,
            1,
            "Channel 2 threshold",
            self.threshold_2_slider,
            self.threshold_2_spin,
        )
        self._add_labeled_control(
            parameter_layout,
            2,
            "Penalization theta",
            self.theta_slider,
            self.theta_spin,
        )
        parameter_layout.addWidget(QLabel("Include percentile"), 3, 0)
        parameter_layout.addWidget(self.percentile_spin, 3, 2)
        parameter_layout.addWidget(self.costes_button, 4, 0, 1, 3)
        root.addWidget(parameter_group)

        scale_group = QGroupBox("Display scale")
        scale_layout = QGridLayout(scale_group)
        self._add_labeled_control(
            scale_layout,
            0,
            "XY scale",
            self.xy_scale_slider,
            self.xy_scale_spin,
        )
        self._add_labeled_control(
            scale_layout,
            1,
            "Z scale",
            self.z_scale_slider,
            self.z_scale_spin,
        )
        scale_layout.addWidget(QLabel("Channel 1 color"), 2, 0)
        scale_layout.addWidget(self.channel_1_color_combo, 2, 1, 1, 2)
        scale_layout.addWidget(QLabel("Channel 2 color"), 3, 0)
        scale_layout.addWidget(self.channel_2_color_combo, 3, 1, 1, 2)
        self._add_labeled_control(
            scale_layout,
            4,
            "Overlay alpha",
            self.overlay_alpha_slider,
            self.overlay_alpha_spin,
        )
        scale_layout.addWidget(QLabel("RACC colormap"), 5, 0)
        scale_layout.addWidget(self.colormap_combo, 5, 1, 1, 2)
        self._add_labeled_control(
            scale_layout,
            6,
            "RACC volume alpha",
            self.volume_alpha_slider,
            self.volume_alpha_spin,
        )
        scale_layout.addWidget(self.bounding_box_checkbox, 7, 0, 1, 3)
        root.addWidget(scale_group)

        option_row = QHBoxLayout()
        option_row.addWidget(self.live_checkbox)
        option_row.addWidget(self.replace_checkbox)
        option_row.addStretch(1)
        root.addLayout(option_row)

        scatter_option_row = QHBoxLayout()
        scatter_option_row.addWidget(self.percentile_fill_checkbox)
        scatter_option_row.addStretch(1)
        root.addLayout(scatter_option_row)
        root.addWidget(self.scatter_widget)

        action_row = QHBoxLayout()
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.overlay_button)
        action_row.addWidget(self.racc_button)
        root.addLayout(action_row)

        view_row = QHBoxLayout()
        view_row.addWidget(self.side_by_side_button)
        view_row.addWidget(self.mip_button)
        root.addLayout(view_row)

        root.addWidget(self.status_label)
        root.addStretch(1)

    def _connect_signals(self) -> None:
        for combo in (self.channel_1_combo, self.channel_2_combo):
            combo.currentIndexChanged.connect(self._schedule_if_live)
            combo.currentIndexChanged.connect(self._sync_scale_from_selected)

        controls = (
            self.threshold_1_slider,
            self.threshold_1_spin,
            self.threshold_2_slider,
            self.threshold_2_spin,
            self.theta_slider,
            self.theta_spin,
            self.percentile_spin,
        )
        for control in controls:
            control.valueChanged.connect(self._schedule_if_live)

        self.xy_scale_slider.valueChanged.connect(self._apply_display_scale)
        self.xy_scale_spin.valueChanged.connect(self._apply_display_scale)
        self.z_scale_slider.valueChanged.connect(self._apply_display_scale)
        self.z_scale_spin.valueChanged.connect(self._apply_display_scale)
        self.bounding_box_checkbox.toggled.connect(
            self._apply_bounding_box_visibility
        )
        self.overlay_alpha_slider.valueChanged.connect(self._apply_overlay_alpha)
        self.overlay_alpha_spin.valueChanged.connect(self._apply_overlay_alpha)
        self.channel_1_color_combo.currentTextChanged.connect(self._apply_overlay_alpha)
        self.channel_2_color_combo.currentTextChanged.connect(self._apply_overlay_alpha)
        self.colormap_combo.currentTextChanged.connect(self._apply_colormap)
        self.volume_alpha_slider.valueChanged.connect(self._apply_colormap)
        self.volume_alpha_spin.valueChanged.connect(self._apply_colormap)
        self.percentile_fill_checkbox.toggled.connect(
            self.scatter_widget.set_show_percentile_fill
        )

        self.costes_button.clicked.connect(self.apply_costes_thresholds)
        self.run_button.clicked.connect(self.run_racc)
        self.overlay_button.clicked.connect(self._show_overlay)
        self.racc_button.clicked.connect(self._show_racc)
        self.side_by_side_button.clicked.connect(self._show_side_by_side)
        self.mip_button.clicked.connect(self._show_mips)

        events = self.viewer.layers.events
        events.inserted.connect(self._refresh_layer_choices)
        events.removed.connect(self._refresh_layer_choices)
        events.reordered.connect(self._refresh_layer_choices)

        dims_events = getattr(getattr(self.viewer, "dims", None), "events", None)
        if dims_events is not None and hasattr(dims_events, "ndisplay"):
            dims_events.ndisplay.connect(self._apply_bounding_box_visibility)

    def _refresh_layer_choices(self, event=None) -> None:
        current_1 = self.channel_1_combo.currentText()
        current_2 = self.channel_2_combo.currentText()
        image_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image) and not _is_racc_generated_layer(layer)
        ]

        for combo, previous in (
            (self.channel_1_combo, current_1),
            (self.channel_2_combo, current_2),
        ):
            with QSignalBlocker(combo):
                combo.clear()
                combo.addItems(image_names)
                if previous in image_names:
                    combo.setCurrentText(previous)

        if len(image_names) >= 2 and not current_2:
            with QSignalBlocker(self.channel_2_combo):
                self.channel_2_combo.setCurrentIndex(1)
        self._sync_scale_from_selected()

    def _make_int_control(
        self,
        value: int,
        *,
        minimum: int = 0,
        maximum: int = 255,
    ) -> tuple[QSlider, QSpinBox]:
        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        return slider, spin

    def _make_scale_control(self, value: float) -> tuple[QSlider, QDoubleSpinBox]:
        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.setRange(1, 2000)
        slider.setValue(_scale_to_slider(value))

        spin = QDoubleSpinBox()
        spin.setRange(0.01, 20.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setValue(value)

        def update_spin(raw_value: int) -> None:
            scale_value = _slider_to_scale(raw_value)
            if abs(spin.value() - scale_value) > 0.005:
                with QSignalBlocker(spin):
                    spin.setValue(scale_value)

        def update_slider(scale_value: float) -> None:
            raw_value = _scale_to_slider(scale_value)
            if slider.value() != raw_value:
                with QSignalBlocker(slider):
                    slider.setValue(raw_value)

        slider.valueChanged.connect(update_spin)
        spin.valueChanged.connect(update_slider)
        return slider, spin

    def _add_labeled_control(self, layout, row: int, label: str, slider, spin) -> None:
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(spin, row, 2)

    def _schedule_if_live(self, *args) -> None:
        if self.live_checkbox.isChecked():
            self._debounce_timer.start()

    def _sync_scale_from_selected(self, *args) -> None:
        if not hasattr(self, "xy_scale_spin") or self.channel_1_combo.count() == 0:
            return
        try:
            channel_1_layer, _channel_2_layer = self._selected_layers()
        except RaccError:
            return

        scale = tuple(float(value) for value in channel_1_layer.scale)
        xy_scale = scale[-1] if len(scale) >= 1 else 1.0
        z_scale = scale[-3] if len(scale) >= 3 else 1.0

        with QSignalBlocker(self.xy_scale_spin), QSignalBlocker(self.xy_scale_slider):
            self.xy_scale_spin.setValue(xy_scale)
            self.xy_scale_slider.setValue(_scale_to_slider(xy_scale))
        with QSignalBlocker(self.z_scale_spin), QSignalBlocker(self.z_scale_slider):
            self.z_scale_spin.setValue(z_scale)
            self.z_scale_slider.setValue(_scale_to_slider(z_scale))

    def _apply_display_scale(self, *args) -> None:
        try:
            channel_1_layer, channel_2_layer = self._selected_layers()
        except RaccError:
            return

        xy_scale = float(self.xy_scale_spin.value())
        z_scale = float(self.z_scale_spin.value())
        for layer in (channel_1_layer, channel_2_layer, self._result_layer()):
            if layer is not None:
                layer.scale = _scale_tuple_for_layer(layer, xy_scale, z_scale)
        self._refresh_existing_overlay_volume()

    def _selected_colormap(self) -> str:
        return racc_colormap_name(self.colormap_combo.currentText())

    def _selected_volume_alpha(self) -> float:
        return max(float(self.volume_alpha_spin.value()) / 100.0, 0.01)

    def _selected_overlay_gain(self) -> float:
        return float(self.overlay_alpha_spin.value()) / 100.0

    def _selected_overlay_colors(self) -> tuple[str, str]:
        return (
            overlay_color_name(self.channel_1_color_combo.currentText(), "red"),
            overlay_color_name(self.channel_2_color_combo.currentText(), "green"),
        )

    def _selected_overlay_volume_alpha(self) -> float:
        gain = min(self._selected_overlay_gain(), 1.0)
        if gain <= 0:
            return 0.0
        return OVERLAY_VOLUME_ALPHA_MAX * gain**OVERLAY_VOLUME_ALPHA_EXPONENT

    def _selected_thresholds(self) -> tuple[float, float]:
        return (
            float(self.threshold_1_spin.value()),
            float(self.threshold_2_spin.value()),
        )

    def _apply_overlay_alpha(self, *args) -> None:
        try:
            channel_1_layer, channel_2_layer = self._selected_layers()
        except RaccError:
            return
        gain = self._selected_overlay_gain()
        volume_alpha = self._selected_overlay_volume_alpha()
        raw_overlay_name = (
            f"RACC overlay volume: {channel_1_layer.name} x {channel_2_layer.name}"
        )
        if (
            raw_overlay_name in self.viewer.layers
            and self.viewer.layers[raw_overlay_name].visible
        ):
            overlay_color_1, overlay_color_2 = self._selected_overlay_colors()
            update_raw_overlay_volume(
                self.viewer,
                channel_1_layer,
                channel_2_layer,
                gain,
                *self._selected_thresholds(),
                overlay_color_1,
                overlay_color_2,
            )
        overlay_color_1, overlay_color_2 = self._selected_overlay_colors()
        for layer, color in (
            (channel_1_layer, overlay_color_1),
            (channel_2_layer, overlay_color_2),
        ):
            if layer.visible:
                layer.opacity = 1.0
                if not getattr(layer, "rgb", False):
                    _set_layer_colormap(
                        layer,
                        overlay_channel_colormap(
                            color,
                            gain,
                            transfer="gain",
                        ),
                    )
        for layer in self.viewer.layers:
            if layer.metadata.get("racc_overlay_kind") == "side_by_side":
                if not layer.visible:
                    continue
                if layer.opacity != 1.0:
                    layer.opacity = 1.0
                channel = layer.metadata.get("racc_overlay_channel")
                if channel in {"channel_1", "red"}:
                    color = overlay_color_1
                elif channel in {"channel_2", "green"}:
                    color = overlay_color_2
                else:
                    color = None
                if color is not None:
                    _set_layer_colormap(
                        layer,
                        overlay_channel_colormap(color, volume_alpha),
                    )
        result_layer = self._result_layer()
        if result_layer is not None:
            self._refresh_existing_mips(
                (channel_1_layer.name, channel_2_layer.name),
                result_layer,
            )

    def _racc_colormap_for_layer(self, layer):
        if int(layer.ndim) >= 3:
            return racc_volume_colormap(
                self._selected_colormap(),
                self._selected_volume_alpha(),
            )
        return racc_colormap(self._selected_colormap())

    def _apply_colormap(self, *args) -> None:
        colormap_name = self._selected_colormap()
        self.scatter_widget.set_colormap(colormap_name)
        result_layer = self._result_layer()
        if result_layer is not None:
            _set_layer_colormap(
                result_layer,
                self._racc_colormap_for_layer(result_layer),
            )
        for layer in self.viewer.layers:
            if layer.metadata.get("racc_mip_kind") == "racc":
                _set_layer_colormap(layer, racc_colormap(colormap_name))

    def _apply_bounding_box_visibility(self, *args) -> None:
        self._set_bounding_box_visibility(force_refresh=False)

    def _refresh_bounding_box_visibility(self) -> None:
        self._set_bounding_box_visibility(force_refresh=True)

    def _set_bounding_box_visibility(self, force_refresh: bool) -> None:
        show_box = bool(self.bounding_box_checkbox.isChecked())
        ndisplay = getattr(getattr(self.viewer, "dims", None), "ndisplay", 3)
        display_is_3d = int(ndisplay) == 3
        for layer in self.viewer.layers:
            if hasattr(layer, "bounding_box"):
                visible = (
                    show_box
                    and display_is_3d
                    and layer.visible
                    and int(layer.ndim) >= 3
                )
                if force_refresh and visible:
                    layer.bounding_box.visible = False
                if visible:
                    layer.bounding_box.blending = "translucent_no_depth"
                layer.bounding_box.visible = visible

    def _refresh_bounding_box_after_view_change(self) -> None:
        self._refresh_bounding_box_visibility()
        QTimer.singleShot(0, self._refresh_bounding_box_visibility)

    def run_racc(self) -> None:
        try:
            channel_1_layer, channel_2_layer = self._selected_layers()
        except RaccError as error:
            self._set_status(str(error))
            return

        request = _ComputeRequest(
            threshold_1=self.threshold_1_spin.value(),
            threshold_2=self.threshold_2_spin.value(),
            theta_degrees=self.theta_spin.value(),
            include_percentile=float(self.percentile_spin.value()),
        )
        self._latest_job_id += 1
        job_id = self._latest_job_id
        self._pending_jobs[job_id] = (channel_1_layer.name, channel_2_layer.name)
        self._set_status("Computing full RACC volume...")

        self._worker = create_worker(
            _compute_racc_job,
            job_id,
            channel_1_layer.data,
            channel_2_layer.data,
            bool(getattr(channel_1_layer, "rgb", False)),
            bool(getattr(channel_2_layer, "rgb", False)),
            request,
            _connect={"returned": self._handle_result, "errored": self._handle_error},
            _start_thread=True,
        )

    def apply_costes_thresholds(self) -> None:
        try:
            channel_1_layer, channel_2_layer = self._selected_layers()
        except RaccError as error:
            self._set_status(str(error))
            return

        self._costes_pair = (channel_1_layer.name, channel_2_layer.name)
        self.costes_button.setEnabled(False)
        self._set_status("Computing Costes thresholds...")
        self._threshold_worker = create_worker(
            _compute_costes_thresholds_job,
            channel_1_layer.data,
            channel_2_layer.data,
            bool(getattr(channel_1_layer, "rgb", False)),
            bool(getattr(channel_2_layer, "rgb", False)),
            _connect={
                "returned": self._handle_costes_thresholds,
                "errored": self._handle_costes_error,
            },
            _start_thread=True,
        )

    def _handle_costes_thresholds(self, thresholds: CostesThresholds) -> None:
        try:
            channel_1_layer, channel_2_layer = self._selected_layers()
        except RaccError:
            self.costes_button.setEnabled(True)
            return
        if self._costes_pair != (channel_1_layer.name, channel_2_layer.name):
            self.costes_button.setEnabled(True)
            self._set_status("Ignored stale Costes thresholds for previous inputs.")
            return

        threshold_1 = _threshold_to_control_value(thresholds.threshold_1)
        threshold_2 = _threshold_to_control_value(thresholds.threshold_2)
        self.threshold_1_spin.setValue(threshold_1)
        self.threshold_2_spin.setValue(threshold_2)
        self.costes_button.setEnabled(True)
        self._set_status(
            "Costes thresholds applied: "
            f"channel 1 {threshold_1}, channel 2 {threshold_2}."
        )

    def _handle_costes_error(self, error: Exception) -> None:
        self.costes_button.setEnabled(True)
        self._set_status(f"Costes thresholding failed: {error}")

    def _handle_result(self, returned) -> None:
        job_id, result = returned
        pair = self._pending_jobs.pop(job_id, None)
        if job_id != self._latest_job_id:
            return
        if pair is None:
            return

        name = _result_name(pair)
        layer = None
        metadata = result.parameters.to_dict()
        metadata.update(
            {
                "napari_racc_kind": "result",
                "racc_input_1": pair[0],
                "racc_input_2": pair[1],
            }
        )
        if self.replace_checkbox.isChecked() and name in self.viewer.layers:
            layer = self.viewer.layers[name]
            layer.data = result.index
            layer.metadata = metadata
            layer.scale = self._scale_for_result(pair)
            _set_layer_contrast_limits(layer, racc_display_contrast_limits())
            _set_layer_colormap(layer, self._racc_colormap_for_layer(layer))
            _configure_racc_volume_layer(layer)
        else:
            result_colormap = (
                racc_volume_colormap(
                    self._selected_colormap(),
                    self._selected_volume_alpha(),
                )
                if result.index.ndim >= 3
                else racc_colormap(self._selected_colormap())
            )
            layer = self.viewer.add_image(
                result.index,
                name=name,
                colormap=result_colormap,
                contrast_limits=racc_display_contrast_limits(),
                blending="translucent",
                metadata=metadata,
                scale=self._scale_for_result(pair),
                depiction="volume",
                rendering="translucent",
        )
        self._result_layer_name = layer.name
        self._result_pair = pair
        self.scatter_widget.set_axis_labels(pair[0], pair[1])
        self.scatter_widget.set_result(result)
        self.scatter_widget.set_colormap(self._selected_colormap())
        self._refresh_existing_mips(pair, layer)
        self._refresh_existing_overlay_volume()
        self._apply_bounding_box_visibility()

        warnings = " ".join(result.warnings)
        self._set_status(
            f"RACC complete: {result.parameters.overlap_voxels} overlap voxels "
            f"({result.parameters.overlap_fraction:.2%}). {warnings}".strip()
        )

    def _handle_error(self, error: Exception) -> None:
        self._set_status(f"RACC failed: {error}")

    def _selected_layers(self):
        if self.channel_1_combo.count() == 0 or self.channel_2_combo.count() == 0:
            raise RaccError("Open or select two image layers first.")
        name_1 = self.channel_1_combo.currentText()
        name_2 = self.channel_2_combo.currentText()
        if not name_1 or not name_2:
            raise RaccError("Select two image layers first.")
        if name_1 == name_2:
            raise RaccError("Select two different image layers.")
        return self.viewer.layers[name_1], self.viewer.layers[name_2]

    def _result_layer(self):
        try:
            ch1, ch2 = self._selected_layers()
        except RaccError:
            return None

        pair = (ch1.name, ch2.name)
        expected_name = _result_name(pair)
        if expected_name in self.viewer.layers:
            self._result_layer_name = expected_name
            self._result_pair = pair
            return self.viewer.layers[expected_name]

        if (
            self._result_pair == pair
            and self._result_layer_name
            and self._result_layer_name in self.viewer.layers
        ):
            return self.viewer.layers[self._result_layer_name]
        return None

    def _scale_for_result(self, pair: tuple[str, str]) -> tuple[float, ...]:
        if pair[0] in self.viewer.layers:
            return tuple(float(value) for value in self.viewer.layers[pair[0]].scale)
        return (1.0, 1.0)

    def _refresh_existing_mips(self, pair: tuple[str, str], result_layer) -> None:
        overlay_name = f"RACC overlay MIP: {pair[0]} x {pair[1]}"
        result_mip_name = f"{result_layer.name} MIP"
        if (
            overlay_name not in self.viewer.layers
            and result_mip_name not in self.viewer.layers
        ):
            return
        if pair[0] not in self.viewer.layers or pair[1] not in self.viewer.layers:
            return

        update_mips(
            self.viewer,
            self.viewer.layers[pair[0]],
            self.viewer.layers[pair[1]],
            result_layer,
            self._selected_colormap(),
            *self._selected_overlay_colors(),
        )

    def _refresh_existing_overlay_volume(self) -> None:
        try:
            channel_1_layer, channel_2_layer = self._selected_layers()
        except RaccError:
            return

        raw_overlay_name = (
            f"RACC overlay volume: {channel_1_layer.name} x {channel_2_layer.name}"
        )
        if raw_overlay_name in self.viewer.layers:
            update_raw_overlay_volume(
                self.viewer,
                channel_1_layer,
                channel_2_layer,
                self._selected_overlay_gain(),
                *self._selected_thresholds(),
                *self._selected_overlay_colors(),
            )

    def _show_overlay(self) -> None:
        try:
            ch1, ch2 = self._selected_layers()
        except RaccError as error:
            self._set_status(str(error))
            return
        show_overlay(
            self.viewer,
            ch1,
            ch2,
            self._result_layer(),
            self._selected_overlay_gain(),
            *self._selected_thresholds(),
            *self._selected_overlay_colors(),
        )
        self._refresh_bounding_box_after_view_change()

    def _show_racc(self) -> None:
        result = self._result_layer()
        if result is None:
            self._set_status("Run RACC before switching to the RACC view.")
            return
        show_racc_only(self.viewer, result)
        self._refresh_bounding_box_after_view_change()

    def _show_side_by_side(self) -> None:
        try:
            ch1, ch2 = self._selected_layers()
        except RaccError as error:
            self._set_status(str(error))
            return
        result = self._result_layer()
        if result is None:
            self._set_status("Run RACC before using side-by-side view.")
            return
        show_side_by_side(
            self.viewer,
            ch1,
            ch2,
            result,
            self._selected_overlay_gain(),
            *self._selected_thresholds(),
            *self._selected_overlay_colors(),
        )
        self._refresh_bounding_box_after_view_change()

    def _show_mips(self) -> None:
        try:
            ch1, ch2 = self._selected_layers()
        except RaccError as error:
            self._set_status(str(error))
            return
        result = self._result_layer()
        if result is None:
            self._set_status("Run RACC before creating RACC MIPs.")
            return
        show_mips(
            self.viewer,
            ch1,
            ch2,
            result,
            self._selected_colormap(),
            *self._selected_overlay_colors(),
        )
        self._refresh_bounding_box_after_view_change()

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)


def _compute_racc_job(
    job_id: int,
    channel_1_data,
    channel_2_data,
    channel_1_rgb: bool,
    channel_2_rgb: bool,
    request: _ComputeRequest,
):
    ch1 = reduce_to_intensity(channel_1_data, rgb=channel_1_rgb, channel_mode="max")
    ch2 = reduce_to_intensity(channel_2_data, rgb=channel_2_rgb, channel_mode="max")
    result = compute_racc(
        ch1,
        ch2,
        threshold_1=request.threshold_1,
        threshold_2=request.threshold_2,
        theta_degrees=request.theta_degrees,
        include_percentile=request.include_percentile,
        output_dtype="float32",
    )
    return job_id, result


def _compute_costes_thresholds_job(
    channel_1_data,
    channel_2_data,
    channel_1_rgb: bool,
    channel_2_rgb: bool,
) -> CostesThresholds:
    ch1 = reduce_to_intensity(channel_1_data, rgb=channel_1_rgb, channel_mode="max")
    ch2 = reduce_to_intensity(channel_2_data, rgb=channel_2_rgb, channel_mode="max")
    return compute_costes_thresholds(ch1, ch2)


def _set_layer_colormap(layer, colormap) -> None:
    try:
        layer.colormap = colormap
    except RuntimeError as error:
        if "sequence argument must have length equal to input rank" not in str(error):
            raise
        layer._colormap = ensure_colormap(colormap)
        layer.events.colormap()


def _set_layer_contrast_limits(layer, contrast_limits) -> None:
    try:
        layer.contrast_limits = contrast_limits
    except RuntimeError as error:
        if "sequence argument must have length equal to input rank" not in str(error):
            raise
        layer._contrast_limits = contrast_limits
        current_range = list(layer.contrast_limits_range)
        layer._contrast_limits_range = [
            min(current_range[0], contrast_limits[0]),
            max(current_range[1], contrast_limits[1]),
        ]
        layer.events.contrast_limits()


def _result_name(pair: tuple[str, str]) -> str:
    return f"RACC: {pair[0]} x {pair[1]}"


def _threshold_to_control_value(value: float) -> int:
    return int(max(0, min(255, int(float(value) + 0.5))))


def _is_racc_generated_layer(layer) -> bool:
    if str(layer.name).startswith("RACC:"):
        return True
    return layer.metadata.get("napari_racc_kind") in {"result", "mip", "overlay"}


def _configure_racc_volume_layer(layer) -> None:
    _set_layer_contrast_limits(layer, racc_display_contrast_limits())
    if int(layer.ndim) < 3:
        return
    layer.depiction = "volume"
    layer.rendering = "translucent"
    layer.blending = "translucent"


def _slider_to_scale(raw_value: int) -> float:
    return max(raw_value / 100.0, 0.01)


def _scale_to_slider(scale_value: float) -> int:
    return max(1, min(2000, int(round(scale_value * 100))))


def _scale_tuple_for_layer(layer, xy_scale: float, z_scale: float) -> tuple[float, ...]:
    ndim = int(layer.ndim)
    if ndim <= 1:
        return (xy_scale,)
    if ndim == 2:
        return (xy_scale, xy_scale)
    return (*([1.0] * (ndim - 3)), z_scale, xy_scale, xy_scale)
