"""napari dock widget for interactive RACC calculation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from napari.layers import Image
from napari.qt.threading import create_worker
from napari.utils.colormaps import ensure_colormap
from napari.utils.theme import get_theme, increase
from qtpy.QtCore import QSignalBlocker, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napari_racc._colormaps import (
    overlay_color_choices,
    overlay_color_name,
    racc_colormap,
    racc_colormap_choices,
    racc_colormap_name,
    racc_display_contrast_limits,
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
    DEFAULT_VOLUME_RENDERING,
    compute_overlay_rgb,
    configure_overlay_display,
    configure_racc_display,
    normalize_volume_rendering,
    refresh_bounding_box_clipping,
    show_mips,
    show_overlay,
    show_racc_only,
    show_side_by_side,
    update_mips,
)

if TYPE_CHECKING:
    import napari

DEFAULT_INTENSITY_OPACITY_PERCENT = 25
DEFAULT_RACC_OPACITY_PERCENT = 25
DEFAULT_RACC_DISPLAY_FLOOR = 0.05
DEFAULT_BACKGROUND_SUPPRESSION = 2.0
DEFAULT_DISPLAY_SCALE = 1.0
MINIMUM_WIDGET_WIDTH = 430
RENDERING_LABELS = {
    "Translucent": "translucent",
    "Maximum intensity (MIP)": "mip",
    "Additive": "additive",
}


@dataclass(frozen=True)
class _ComputeRequest:
    threshold_1: int
    threshold_2: int
    theta_degrees: int
    include_percentile: float


@dataclass(frozen=True)
class _DisplayComputeRequest:
    job_id: int
    pair: tuple[str, str]
    channel_1_data: object
    channel_2_data: object
    channel_1_rgb: bool
    channel_2_rgb: bool
    display_cutoff_1: float
    display_cutoff_2: float
    overlay_color_1: str
    overlay_color_2: str


class _DisclosureSection(QWidget):
    """Show optional controls without disabling their current settings."""

    def __init__(self, title: str, body: QWidget):
        super().__init__()
        self.body = body
        self.toggle = QToolButton()
        self.toggle.setObjectName("raccDisclosure")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle.setMinimumHeight(34)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle)
        layout.addWidget(body)
        self.toggle.toggled.connect(self.set_expanded)
        self.set_expanded(False)

    def set_expanded(self, expanded: bool) -> None:
        with QSignalBlocker(self.toggle):
            self.toggle.setChecked(expanded)
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body.setVisible(expanded)


class _SettingsTabs(QTabWidget):
    """Share the available width between the two main settings tabs."""

    def __init__(self):
        super().__init__()
        self.setObjectName("raccSettings")
        self.setDocumentMode(True)
        self.tabBar().setObjectName("raccSettingsTabs")
        self.tabBar().setExpanding(True)
        self.tabBar().setDrawBase(False)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self.tabBar().setFixedWidth(self.width())


class RaccWidget(QWidget):
    """Interactive RACC analysis widget."""

    result_ready = Signal(object)

    def __init__(self, viewer: napari.viewer.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._result_layer_name: str | None = None
        self._result_pair: tuple[str, str] | None = None
        self._pending_jobs: dict[int, tuple[str, str]] = {}
        self._latest_job_id = 0
        self._worker = None
        self._threshold_worker = None
        self._display_worker = None
        self._display_worker_active = False
        self._pending_display_request: _DisplayComputeRequest | None = None
        self._latest_display_job_id = 0
        self._costes_pair: tuple[str, str] | None = None

        self.channel_1_combo = QComboBox()
        self.channel_2_combo = QComboBox()
        self._configure_layer_combo(self.channel_1_combo)
        self._configure_layer_combo(self.channel_2_combo)

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

        self.xy_scale_slider, self.xy_scale_spin = self._make_scale_control(
            DEFAULT_DISPLAY_SCALE
        )
        self.z_scale_slider, self.z_scale_spin = self._make_scale_control(
            DEFAULT_DISPLAY_SCALE
        )
        self.xy_scale_reset_button = QPushButton("Reset")
        self.z_scale_reset_button = QPushButton("Reset")
        for axis, button in (
            ("XY", self.xy_scale_reset_button), ("Z", self.z_scale_reset_button)
        ):
            button.setToolTip(f"Reset {axis} display scale to 1.00.")
            button.setAccessibleName(f"Reset {axis} scale")
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.bounding_box_checkbox = QCheckBox("Bounding box")
        self.rendering_combo = QComboBox()
        self.rendering_combo.addItems(RENDERING_LABELS)
        self.rendering_combo.setCurrentText("Translucent")
        self.rendering_combo.setToolTip(
            "Apply the same interactive 3D ray-casting method to the intensity "
            "overlay and RACC result."
        )
        self.link_display_cutoffs_checkbox = QCheckBox(
            "Link channel cutoffs to analysis thresholds"
        )
        self.link_display_cutoffs_checkbox.setChecked(True)
        self.link_display_cutoffs_checkbox.setToolTip(
            "Use the manual or Costes analysis thresholds as the intensity "
            "display black points."
        )
        self.display_cutoff_1_slider, self.display_cutoff_1_spin = (
            self._make_int_control(5)
        )
        self.display_cutoff_2_slider, self.display_cutoff_2_spin = (
            self._make_int_control(5)
        )
        self.racc_display_floor_slider, self.racc_display_floor_spin = (
            self._make_double_control(
                DEFAULT_RACC_DISPLAY_FLOOR,
                minimum=0.0,
                maximum=0.99,
                scale=100,
                decimals=2,
                step=0.01,
            )
        )
        self.intensity_opacity_slider, self.intensity_opacity_spin = (
            self._make_int_control(
                DEFAULT_INTENSITY_OPACITY_PERCENT,
                minimum=0,
                maximum=100,
            )
        )
        self.racc_opacity_slider, self.racc_opacity_spin = (
            self._make_int_control(
                DEFAULT_RACC_OPACITY_PERCENT,
                minimum=0,
                maximum=100,
            )
        )
        self.background_suppression_slider, self.background_suppression_spin = (
            self._make_double_control(
                DEFAULT_BACKGROUND_SUPPRESSION,
                minimum=1.0,
                maximum=4.0,
                scale=10,
                decimals=1,
                step=0.1,
            )
        )
        for control in (
            self.racc_display_floor_slider,
            self.racc_display_floor_spin,
        ):
            control.setToolTip(
                "Hide RACC values at or below this level without changing data "
                "or remapping the fixed 0..1 colors."
            )
        for control in (
            self.intensity_opacity_slider,
            self.intensity_opacity_spin,
        ):
            control.setToolTip(
                "Set the 3D intensity-volume opacity. At 100, the strongest "
                "retained voxels can be fully opaque. This is ignored in 2D."
            )
        for control in (
            self.racc_opacity_slider,
            self.racc_opacity_spin,
        ):
            control.setToolTip(
                "Set the 3D RACC-volume opacity. At 100, the strongest retained "
                "voxels can be fully opaque. This is ignored in 2D."
            )
        for control in (
            self.background_suppression_slider,
            self.background_suppression_spin,
        ):
            control.setToolTip(
                "In 3D translucent/additive rendering, higher values make "
                "weak retained signals more transparent. This is ignored in 2D."
            )
        self.channel_1_color_combo = QComboBox()
        self.channel_1_color_combo.addItems(overlay_color_choices())
        self.channel_1_color_combo.setCurrentText("red")
        self.channel_2_color_combo = QComboBox()
        self.channel_2_color_combo.addItems(overlay_color_choices())
        self.channel_2_color_combo.setCurrentText("green")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(racc_colormap_choices())
        self.colormap_combo.setCurrentText(racc_colormap_name())

        self.live_checkbox = QCheckBox("Update automatically")
        self.live_checkbox.setToolTip(
            "Recalculate RACC when analysis settings change. Appearance edits "
            "always update the view independently."
        )
        self.live_checkbox.setChecked(True)
        self.replace_checkbox = QCheckBox("Update existing result")
        self.replace_checkbox.setChecked(True)

        self.costes_button = QPushButton("Estimate thresholds (Costes)")
        self.costes_button.setToolTip(
            "Estimate channel thresholds on the shared 0–255 working scale. "
            "Review the estimate against your image background."
        )
        self.run_button = QPushButton("Run RACC")
        self.run_button.setMinimumHeight(40)
        self.run_button.setObjectName("raccRun")
        self.export_button = QPushButton("Export RACC TIFF")
        self.export_button.setEnabled(False)
        self.overlay_button = QPushButton("Channel overlay")
        self.racc_button = QPushButton("RACC map")
        self.side_by_side_button = QPushButton("Compare side by side")
        self.mip_button = QPushButton("2D Z-projections")
        self.overlay_button.setToolTip("Show the two intensity channels together.")
        self.racc_button.setToolTip("Show the calculated RACC colour map on its own.")

        self.status_label = QLabel("Select two image layers.")
        self.status_label.setWordWrap(True)
        self.scatter_widget = ScatterHistogramWidget()
        self.log_intensity_button = QPushButton("Log intensity")
        self.log_intensity_button.setCheckable(True)
        self.log_intensity_button.setToolTip(
            "Show both channel intensity axes on a log scale that includes zero."
        )
        self.focus_thresholds_checkbox = QCheckBox("Focus above thresholds")
        self.focus_thresholds_checkbox.setToolTip(
            "Plot only voxels that meet both thresholds and brighten sparse bins."
        )
        self.show_fit_guides_checkbox = QCheckBox("Show fit guides")
        self.show_fit_guides_checkbox.setChecked(True)
        self.show_fit_guides_checkbox.setToolTip(
            "Show the regression, percentile band and projected-intensity cap."
        )
        self.percentile_fill_checkbox = QCheckBox("Fill percentile band")
        self.percentile_fill_checkbox.setChecked(False)
        self.percentile_fill_checkbox.setToolTip(
            "Shade the area within the percentile distance cutoff of the fitted line."
        )

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setInterval(250)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.run_racc)

        # Rebuilding a combined color/signal volume is the one display operation
        # still has to touch every voxel.  Coalesce slider/colour edits so a
        # drag produces one update instead of a queue of large allocations.
        self._display_data_timer = QTimer(self)
        self._display_data_timer.setInterval(200)
        self._display_data_timer.setSingleShot(True)
        self._display_data_timer.timeout.connect(
            self._start_expensive_display_refresh
        )

        self._build_layout()
        self._apply_interface_style()
        self._connect_signals()
        self._refresh_layer_choices()
        self._apply_cutoff_link(refresh=False)
        self._update_display_control_availability()
        if self.run_button.isEnabled():
            self._set_status("Ready. Run RACC to calculate the selected channels.")

    def _build_layout(self) -> None:
        self.setObjectName("raccPanel")
        self.setMinimumWidth(MINIMUM_WIDGET_WIDTH)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_tabs = _SettingsTabs()
        outer_layout.addWidget(self.settings_tabs, 1)
        self._scroll_area, self._content_widget, analysis = self._settings_page()
        self._appearance_scroll_area, _, appearance = self._settings_page()
        self.settings_tabs.addTab(self._scroll_area, "Analysis")
        self.settings_tabs.addTab(self._appearance_scroll_area, "Appearance")
        analysis.addWidget(self._note(
            "Choose the inputs and settings used to calculate RACC."
        ))

        input_group = QGroupBox("1. Choose channels")
        input_layout = QFormLayout(input_group)
        input_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        input_layout.addRow("Channel 1", self.channel_1_combo)
        input_layout.addRow("Channel 2", self.channel_2_combo)
        analysis.addWidget(input_group)

        parameter_group = QGroupBox("2. Set background thresholds")
        parameter_layout = QGridLayout(parameter_group)
        self._add_labeled_control(
            parameter_layout,
            0,
            "Channel 1",
            self.threshold_1_slider,
            self.threshold_1_spin,
        )
        self._add_labeled_control(
            parameter_layout,
            1,
            "Channel 2",
            self.threshold_2_slider,
            self.threshold_2_spin,
        )
        parameter_layout.addWidget(self.costes_button, 2, 0, 1, 3)
        threshold_note = self._note(
            "Shared 0–255 working intensity scale. "
            "Costes provides an estimate to review against the background."
        )
        threshold_note.setToolTip(
            "Both channels share one working scale: normalized 0–1 data are "
            "scaled to 0–255; data above 255 are scaled by their shared maximum. "
            "Values already within 0–255 are retained."
        )
        parameter_layout.addWidget(threshold_note, 3, 0, 1, 3)
        analysis.addWidget(parameter_group)

        method_body = QWidget()
        method_layout = QGridLayout(method_body)
        self._add_labeled_control(
            method_layout, 0, "Penalization angle (θ)",
            self.theta_slider, self.theta_spin,
        )
        method_layout.addWidget(QLabel("Include percentile (%)"), 1, 0)
        method_layout.addWidget(self.percentile_spin, 1, 2)
        method_layout.addWidget(self.replace_checkbox, 2, 0, 1, 3)
        self.method_section = _DisclosureSection(
            "Advanced method settings", method_body
        )
        analysis.addWidget(self.method_section)

        diagnostic_body = QWidget()
        diagnostic_layout = QVBoxLayout(diagnostic_body)
        diagnostic_layout.addWidget(self._note(
            "Channel intensity distribution and regression used by RACC."
        ))
        scatter_options = QHBoxLayout()
        scatter_options.addWidget(self.log_intensity_button)
        scatter_options.addWidget(self.focus_thresholds_checkbox)
        scatter_options.addStretch()
        diagnostic_layout.addLayout(scatter_options)
        diagnostic_layout.addWidget(self.scatter_widget)
        guides_options = QHBoxLayout()
        guides_options.addWidget(self.show_fit_guides_checkbox)
        guides_options.addWidget(self.percentile_fill_checkbox)
        guides_options.addStretch()
        diagnostic_layout.addLayout(guides_options)
        self.diagnostics_section = _DisclosureSection(
            "Analysis diagnostics", diagnostic_body
        )
        analysis.addWidget(self.diagnostics_section)
        analysis.addStretch(1)

        appearance.addWidget(self._note(
            "Adjust the view. Calculated and exported values stay the same."
        ))
        volume_group = QWidget()
        volume_layout = QGridLayout(volume_group)
        volume_layout.addWidget(QLabel("Rendering method"), 0, 0)
        volume_layout.addWidget(self.rendering_combo, 0, 1, 1, 2)
        self._add_labeled_control(
            volume_layout, 1, "Intensity opacity (%)",
            self.intensity_opacity_slider, self.intensity_opacity_spin,
        )
        self._add_labeled_control(
            volume_layout, 2, "RACC opacity (%)",
            self.racc_opacity_slider, self.racc_opacity_spin,
        )
        self._add_labeled_control(
            volume_layout, 3, "Background suppression",
            self.background_suppression_slider, self.background_suppression_spin,
        )
        self.volume_note = self._note(
            "Volume controls apply in 3D. Slices and fixed Z-projections "
            "show retained pixels at full opacity."
        )
        volume_layout.addWidget(self.volume_note, 4, 0, 1, 3)
        self.volume_section = _DisclosureSection(
            "3D volume rendering", volume_group
        )
        self.volume_section.set_expanded(True)
        appearance.addWidget(self.volume_section)

        scale_group = QWidget()
        scale_layout = QGridLayout(scale_group)
        self._add_labeled_control(
            scale_layout,
            0,
            "XY scale",
            self.xy_scale_slider,
            self.xy_scale_spin,
            reset_button=self.xy_scale_reset_button,
        )
        self._add_labeled_control(
            scale_layout,
            1,
            "Z scale",
            self.z_scale_slider,
            self.z_scale_spin,
            reset_button=self.z_scale_reset_button,
        )
        scale_layout.addWidget(self.bounding_box_checkbox, 2, 0, 1, 3)

        mapping_group = QWidget()
        mapping_layout = QGridLayout(mapping_group)
        mapping_layout.addWidget(
            self.link_display_cutoffs_checkbox,
            0,
            0,
            1,
            3,
        )
        self._add_labeled_control(
            mapping_layout,
            1,
            "Channel 1 display cutoff",
            self.display_cutoff_1_slider,
            self.display_cutoff_1_spin,
        )
        self._add_labeled_control(
            mapping_layout,
            2,
            "Channel 2 display cutoff",
            self.display_cutoff_2_slider,
            self.display_cutoff_2_spin,
        )
        self._add_labeled_control(
            mapping_layout,
            3,
            "RACC minimum (0–1)",
            self.racc_display_floor_slider,
            self.racc_display_floor_spin,
        )
        mapping_layout.addWidget(self._note(
            "Visibility filters apply in both 2D and 3D. "
            "Channel cutoffs use the shared 0–255 working scale."
        ), 4, 0, 1, 3)
        self.visibility_section = _DisclosureSection(
            "Visibility filters", mapping_group
        )
        appearance.addWidget(self.visibility_section)

        color_body = QWidget()
        color_layout = QFormLayout(color_body)
        color_layout.addRow("Channel 1 color", self.channel_1_color_combo)
        color_layout.addRow("Channel 2 color", self.channel_2_color_combo)
        color_layout.addRow("RACC colormap", self.colormap_combo)
        self.colors_section = _DisclosureSection("Colors", color_body)
        appearance.addWidget(self.colors_section)
        self.geometry_section = _DisclosureSection(
            "Scale && bounding box", scale_group
        )
        appearance.addWidget(self.geometry_section)
        appearance.addStretch(1)

        # Primary actions stay visible even when a tab is scrolled.
        self.actions_panel = QWidget()
        self.actions_panel.setObjectName("raccActions")
        actions = QVBoxLayout(self.actions_panel)
        actions.setContentsMargins(12, 16, 12, 10)
        actions.setSpacing(8)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_button, 1)
        run_row.addWidget(self.live_checkbox)
        actions.addLayout(run_row)
        actions.addWidget(self.status_label)
        view_group = QGroupBox("Explore result")
        views = QGridLayout(view_group)
        views.addWidget(self.overlay_button, 0, 0)
        views.addWidget(self.racc_button, 0, 1)
        views.addWidget(self.side_by_side_button, 1, 0)
        views.addWidget(self.mip_button, 1, 1)
        self.projection_note = self._note(
            "Z-projection: maximum intensity along Z (MIP)."
        )
        views.addWidget(self.projection_note, 2, 0, 1, 2)
        actions.addWidget(view_group)
        actions.addWidget(self.export_button)
        outer_layout.addWidget(self.actions_panel)

    def _apply_interface_style(self, *args) -> None:
        """Use napari's current theme for a consistent, scoped control panel."""

        theme = get_theme(getattr(self.viewer, "theme", "dark"))
        background = theme.background.as_rgb()
        foreground = theme.foreground.as_rgb()
        primary = theme.primary.as_rgb()
        secondary = theme.secondary.as_rgb()
        text = theme.text.as_rgb()
        muted = theme.icon.as_rgb()
        accent = theme.current.as_rgb()
        section_font_size = increase(theme.font_size, "1")
        red, green, blue = theme.current.as_rgb_tuple()[:3]
        accent_luminance = red * 0.299 + green * 0.587 + blue * 0.114
        accent_text = "#111111" if accent_luminance > 160 else "#ffffff"
        self.setStyleSheet(f"""
            QTabWidget#raccSettings::pane {{ border: 0; }}
            QTabBar#raccSettingsTabs::tab {{
                min-height: 20px; padding: 11px 12px 3px 12px;
                margin: 0 2px 0 0;
                border: 0; border-bottom: 2px solid {foreground};
                border-top-left-radius: 3px; border-top-right-radius: 3px;
                background: {foreground}; color: {text};
                font-weight: normal;
            }}
            QTabBar#raccSettingsTabs::tab:selected {{
                background: {primary}; border-bottom: 2px solid {accent};
            }}
            QTabBar#raccSettingsTabs::tab:hover {{ background: {secondary}; }}
            QWidget#raccPanel QGroupBox {{
                margin-top: 10px; padding: 12px 10px 10px 10px;
                border: 1px solid {foreground}; border-radius: 5px;
            }}
            QWidget#raccPanel QGroupBox::title {{
                subcontrol-origin: margin; left: 10px; padding: 0 4px;
                color: {text}; font-weight: 600;
            }}
            QWidget#raccPanel QLabel#raccHint {{ color: {muted}; }}
            QToolButton#raccDisclosure {{
                text-align: left; padding: 7px 10px;
                background: {foreground}; color: {text};
                border: 1px solid {foreground}; border-radius: 4px;
                font-size: {section_font_size}; font-weight: normal;
            }}
            QToolButton#raccDisclosure:hover {{
                background: {primary}; border-color: {secondary};
            }}
            QToolButton#raccDisclosure:checked {{ border-left: 3px solid {accent}; }}
            QWidget#raccActions {{ border-top: 2px solid {secondary}; }}
            QWidget#raccActions QPushButton {{
                min-height: 20px; padding: 5px 8px; border-radius: 4px;
            }}
            QPushButton#raccRun {{
                background: {accent}; color: {accent_text};
                font-weight: 600; padding: 8px 12px;
            }}
            QPushButton#raccRun:hover {{ border: 1px solid {muted}; }}
            QPushButton#raccRun:disabled {{
                background: {foreground}; color: {muted};
            }}
            QWidget#raccPanel QPushButton:focus,
            QToolButton#raccDisclosure:focus {{ border: 1px solid {accent}; }}
            QWidget#raccPanel QScrollArea {{ background: {background}; border: 0; }}
        """)

    @staticmethod
    def _note(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("raccHint")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _settings_page() -> tuple[QScrollArea, QWidget, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(12)
        return scroll, content, layout

    def _configure_layer_combo(self, combo: QComboBox) -> None:
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        combo.setMinimumContentsLength(18)
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.currentTextChanged.connect(combo.setToolTip)

    def _connect_signals(self) -> None:
        theme_event = getattr(getattr(self.viewer, "events", None), "theme", None)
        if theme_event is not None:
            theme_event.connect(self._apply_interface_style)
        for combo in (self.channel_1_combo, self.channel_2_combo):
            combo.currentIndexChanged.connect(self._handle_input_change)
            combo.currentIndexChanged.connect(self._sync_scale_from_selected)
            combo.currentIndexChanged.connect(self._update_result_actions)
            combo.currentIndexChanged.connect(self._update_display_control_availability)

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
            control.valueChanged.connect(self._mark_stale_analysis_if_not_live)

        self.threshold_1_spin.valueChanged.connect(
            self._sync_linked_display_cutoffs
        )
        self.threshold_1_slider.valueChanged.connect(
            self._sync_linked_display_cutoffs
        )
        self.threshold_2_spin.valueChanged.connect(
            self._sync_linked_display_cutoffs
        )
        self.threshold_2_slider.valueChanged.connect(
            self._sync_linked_display_cutoffs
        )

        self.xy_scale_slider.valueChanged.connect(self._apply_display_scale)
        self.xy_scale_spin.valueChanged.connect(self._apply_display_scale)
        self.z_scale_slider.valueChanged.connect(self._apply_display_scale)
        self.z_scale_spin.valueChanged.connect(self._apply_display_scale)
        self.xy_scale_reset_button.clicked.connect(
            lambda: self.xy_scale_spin.setValue(DEFAULT_DISPLAY_SCALE)
        )
        self.z_scale_reset_button.clicked.connect(
            lambda: self.z_scale_spin.setValue(DEFAULT_DISPLAY_SCALE)
        )
        self.bounding_box_checkbox.toggled.connect(
            self._apply_bounding_box_visibility
        )
        self.link_display_cutoffs_checkbox.toggled.connect(self._apply_cutoff_link)
        expensive_display_controls = (
            self.display_cutoff_1_slider,
            self.display_cutoff_1_spin,
            self.display_cutoff_2_slider,
            self.display_cutoff_2_spin,
        )
        for control in expensive_display_controls:
            control.valueChanged.connect(self._schedule_expensive_display_refresh)
        # Prevent a drag over a large stack from requesting one full-volume
        # rebuild per pixel of slider movement.
        self.display_cutoff_1_slider.setTracking(False)
        self.display_cutoff_2_slider.setTracking(False)

        volume_style_controls = (
            self.intensity_opacity_slider,
            self.intensity_opacity_spin,
            self.racc_opacity_slider,
            self.racc_opacity_spin,
            self.background_suppression_slider,
            self.background_suppression_spin,
        )
        for control in volume_style_controls:
            control.valueChanged.connect(self._apply_volume_display_style)
        for control in (
            self.racc_display_floor_slider,
            self.racc_display_floor_spin,
        ):
            control.valueChanged.connect(self._apply_racc_display_style)
        self.rendering_combo.currentTextChanged.connect(
            self._apply_volume_display_style
        )
        self.rendering_combo.currentTextChanged.connect(
            self._update_display_control_availability
        )
        self.channel_1_color_combo.currentTextChanged.connect(
            self._schedule_expensive_display_refresh
        )
        self.channel_2_color_combo.currentTextChanged.connect(
            self._schedule_expensive_display_refresh
        )
        self.colormap_combo.currentTextChanged.connect(
            self._apply_racc_display_style
        )
        self.log_intensity_button.toggled.connect(self.scatter_widget.set_log_intensity)
        self.focus_thresholds_checkbox.toggled.connect(
            self.scatter_widget.set_focus_thresholds
        )
        self.show_fit_guides_checkbox.toggled.connect(
            self.scatter_widget.set_show_fit_guides
        )
        self.show_fit_guides_checkbox.toggled.connect(
            self.percentile_fill_checkbox.setEnabled
        )
        self.percentile_fill_checkbox.toggled.connect(
            self.scatter_widget.set_show_percentile_fill
        )

        self.costes_button.clicked.connect(self.apply_costes_thresholds)
        self.run_button.clicked.connect(self.run_racc)
        self.export_button.clicked.connect(self.export_racc_tiff)
        self.overlay_button.clicked.connect(self._show_overlay)
        self.racc_button.clicked.connect(self._show_racc)
        self.side_by_side_button.clicked.connect(self._show_side_by_side)
        self.mip_button.clicked.connect(self._show_mips)
        self.mip_button.setToolTip(
            "Create fixed 2D maximum-intensity projections along the Z axis."
        )

        events = self.viewer.layers.events
        events.inserted.connect(self._refresh_layer_choices)
        events.removed.connect(self._refresh_layer_choices)
        events.reordered.connect(self._refresh_layer_choices)

        dims_events = getattr(getattr(self.viewer, "dims", None), "events", None)
        if dims_events is not None and hasattr(dims_events, "ndisplay"):
            dims_events.ndisplay.connect(self._handle_ndisplay_change)

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
                for index, image_name in enumerate(image_names):
                    combo.setItemData(index, image_name, Qt.ToolTipRole)
                if previous in image_names:
                    combo.setCurrentText(previous)
                combo.setToolTip(combo.currentText())

        if len(image_names) >= 2 and not current_2:
            with QSignalBlocker(self.channel_2_combo):
                self.channel_2_combo.setCurrentIndex(1)
        self._sync_scale_from_selected()
        self._update_result_actions()

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
        def update_spin(raw_value: int) -> None:
            if spin.value() != raw_value:
                with QSignalBlocker(spin):
                    spin.setValue(raw_value)

        def update_slider(control_value: int) -> None:
            if slider.value() != control_value:
                with QSignalBlocker(slider):
                    slider.setValue(control_value)

        slider.valueChanged.connect(update_spin)
        spin.valueChanged.connect(update_slider)
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

    def _make_double_control(
        self,
        value: float,
        *,
        minimum: float,
        maximum: float,
        scale: int,
        decimals: int,
        step: float,
    ) -> tuple[QSlider, QDoubleSpinBox]:
        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.setRange(int(round(minimum * scale)), int(round(maximum * scale)))
        slider.setValue(int(round(value * scale)))

        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)

        def update_spin(raw_value: int) -> None:
            control_value = raw_value / scale
            if abs(spin.value() - control_value) > 0.5 / scale:
                with QSignalBlocker(spin):
                    spin.setValue(control_value)

        def update_slider(control_value: float) -> None:
            raw_value = int(round(control_value * scale))
            if slider.value() != raw_value:
                with QSignalBlocker(slider):
                    slider.setValue(raw_value)

        slider.valueChanged.connect(update_spin)
        spin.valueChanged.connect(update_slider)
        return slider, spin

    def _add_labeled_control(
        self, layout, row: int, label: str, slider, spin, *, reset_button=None
    ) -> None:
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(slider, row, 1)
        if reset_button is None:
            layout.addWidget(spin, row, 2)
        else:
            spin.setMaximumWidth(86)
            field = QHBoxLayout()
            field.setContentsMargins(0, 0, 0, 0)
            field.setSpacing(4)
            field.addWidget(spin)
            field.addWidget(reset_button)
            layout.addLayout(field, row, 2)

    def _schedule_if_live(self, *args) -> None:
        if self.live_checkbox.isChecked():
            self._debounce_timer.start()

    def _handle_input_change(self, *args) -> None:
        # An old pair's worker must not replace the current diagnostics while
        # the next pair's debounced calculation is waiting to start.
        self._latest_job_id += 1
        self.scatter_widget.set_result(None)
        self._set_status("Channels changed. Run RACC to calculate this pair.")
        self._schedule_if_live()

    def _mark_stale_analysis_if_not_live(self, *args) -> None:
        if self.live_checkbox.isChecked() or self._result_layer() is None:
            return
        self._set_status("Analysis parameters changed; press Run RACC to update.")

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
        result_layer = self._result_layer()
        pair = (channel_1_layer.name, channel_2_layer.name)
        volume_layers = [channel_1_layer, channel_2_layer, result_layer]
        raw_overlay_name = f"RACC overlay volume: {pair[0]} x {pair[1]}"
        if raw_overlay_name in self.viewer.layers:
            volume_layers.append(self.viewer.layers[raw_overlay_name])
        for layer in volume_layers:
            if layer is not None:
                _set_layer_scale_without_data_refresh(
                    layer,
                    _scale_tuple_for_layer(layer, xy_scale, z_scale),
                )

        mip_names = [f"RACC overlay MIP: {pair[0]} x {pair[1]}"]
        if result_layer is not None:
            mip_names.append(f"{result_layer.name} MIP")
        for name in mip_names:
            if name in self.viewer.layers:
                _set_layer_scale_without_data_refresh(
                    self.viewer.layers[name],
                    (xy_scale, xy_scale),
                )

    def _selected_colormap(self) -> str:
        return racc_colormap_name(self.colormap_combo.currentText())

    def _selected_overlay_colors(self) -> tuple[str, str]:
        return (
            overlay_color_name(self.channel_1_color_combo.currentText(), "red"),
            overlay_color_name(self.channel_2_color_combo.currentText(), "green"),
        )

    def _selected_analysis_thresholds(self) -> tuple[float, float]:
        return (
            float(self.threshold_1_spin.value()),
            float(self.threshold_2_spin.value()),
        )

    def _selected_display_cutoffs(self) -> tuple[float, float]:
        return (
            float(self.display_cutoff_1_spin.value()),
            float(self.display_cutoff_2_spin.value()),
        )

    def _selected_racc_display_floor(self) -> float:
        return float(self.racc_display_floor_spin.value())

    def _selected_intensity_opacity(self) -> float:
        return float(self.intensity_opacity_spin.value()) / 100.0

    def _selected_racc_opacity(self) -> float:
        return float(self.racc_opacity_spin.value()) / 100.0

    def _selected_background_suppression(self) -> float:
        return float(self.background_suppression_spin.value())

    def _selected_rendering_mode(self) -> str:
        return normalize_volume_rendering(
            RENDERING_LABELS.get(
                self.rendering_combo.currentText(),
                DEFAULT_VOLUME_RENDERING,
            )
        )

    def _apply_cutoff_link(self, *args, refresh: bool = True) -> None:
        linked = self.link_display_cutoffs_checkbox.isChecked()
        for control in (
            self.display_cutoff_1_slider,
            self.display_cutoff_1_spin,
            self.display_cutoff_2_slider,
            self.display_cutoff_2_spin,
        ):
            control.setEnabled(not linked)
        if linked:
            self._sync_linked_display_cutoffs(refresh=refresh)

    def _sync_linked_display_cutoffs(self, *args, refresh: bool = True) -> None:
        if not self.link_display_cutoffs_checkbox.isChecked():
            return
        threshold_1, threshold_2 = self._selected_analysis_thresholds()
        controls = (
            self.display_cutoff_1_slider,
            self.display_cutoff_1_spin,
            self.display_cutoff_2_slider,
            self.display_cutoff_2_spin,
        )
        with (
            QSignalBlocker(controls[0]),
            QSignalBlocker(controls[1]),
            QSignalBlocker(controls[2]),
            QSignalBlocker(controls[3]),
        ):
            self.display_cutoff_1_slider.setValue(int(round(threshold_1)))
            self.display_cutoff_1_spin.setValue(int(round(threshold_1)))
            self.display_cutoff_2_slider.setValue(int(round(threshold_2)))
            self.display_cutoff_2_spin.setValue(int(round(threshold_2)))
        if refresh:
            self._schedule_expensive_display_refresh()

    def _update_display_control_availability(self, *args) -> None:
        is_3d = False
        try:
            channel_1_layer, channel_2_layer = self._selected_layers()
        except RaccError:
            pass
        else:
            is_3d = any(
                int(layer.ndim) >= 3
                for layer in (channel_1_layer, channel_2_layer)
            )
        ndisplay = int(getattr(getattr(self.viewer, "dims", None), "ndisplay", 3))
        volume_is_active = is_3d and ndisplay == 3
        uses_transfer = (
            volume_is_active and self._selected_rendering_mode() != "mip"
        )
        self.rendering_combo.setEnabled(volume_is_active)
        if volume_is_active:
            self.volume_note.setText(
                "Volume controls apply in 3D. Slices and fixed Z-projections "
                "show retained pixels at full opacity."
            )
        elif is_3d:
            self.volume_note.setText(
                "Currently viewing 2D. Volume controls become available in "
                "a 3D view; visibility filters and colors still apply here."
            )
        else:
            self.volume_note.setText(
                "Select two volume channels to use 3D rendering. "
                "For 2D images, use visibility filters and colors below."
            )
        for control in (
            self.intensity_opacity_slider,
            self.intensity_opacity_spin,
            self.racc_opacity_slider,
            self.racc_opacity_spin,
        ):
            control.setEnabled(volume_is_active)
        for control in (
            self.background_suppression_slider,
            self.background_suppression_spin,
        ):
            control.setEnabled(uses_transfer)

    def _schedule_expensive_display_refresh(self, *args) -> None:
        # Invalidate any in-flight mapping immediately; only the most recent
        # slider/color state is allowed to reach the layer.
        self._latest_display_job_id += 1
        self._display_data_timer.start()

    def _start_expensive_display_refresh(self) -> None:
        """Queue the latest full-volume color mapping outside the GUI thread."""

        try:
            channel_1_layer, channel_2_layer = self._selected_layers()
        except RaccError:
            return
        result_layer = self._result_layer()
        pair = (channel_1_layer.name, channel_2_layer.name)
        overlay_name = f"RACC overlay volume: {pair[0]} x {pair[1]}"

        # Existing MIPs retain cached source projections, so their cutoff and
        # color mapping can update immediately even while a 3D overlay rebuild
        # continues in the background.
        if result_layer is not None:
            self._refresh_existing_mips(pair, result_layer)

        if overlay_name not in self.viewer.layers:
            self._apply_volume_display_style()
            return

        cutoff_1, cutoff_2 = self._selected_display_cutoffs()
        color_1, color_2 = self._selected_overlay_colors()
        self._pending_display_request = _DisplayComputeRequest(
            job_id=self._latest_display_job_id,
            pair=pair,
            channel_1_data=channel_1_layer.data,
            channel_2_data=channel_2_layer.data,
            channel_1_rgb=bool(getattr(channel_1_layer, "rgb", False)),
            channel_2_rgb=bool(getattr(channel_2_layer, "rgb", False)),
            display_cutoff_1=cutoff_1,
            display_cutoff_2=cutoff_2,
            overlay_color_1=color_1,
            overlay_color_2=color_2,
        )
        self._launch_pending_display_refresh()

    def _launch_pending_display_refresh(self) -> None:
        if self._display_worker_active or self._pending_display_request is None:
            return
        request = self._pending_display_request
        self._pending_display_request = None
        self._display_worker_active = True
        self._display_worker = create_worker(
            _compute_display_overlay_job,
            request,
            _connect={
                "returned": self._handle_display_overlay,
                "errored": self._handle_display_error,
            },
            _start_thread=True,
        )

    def _handle_display_overlay(self, returned) -> None:
        request, overlay_data = returned
        job_id = request.job_id
        pair = request.pair
        self._display_worker_active = False
        if job_id == self._latest_display_job_id:
            try:
                selected_pair = tuple(layer.name for layer in self._selected_layers())
            except RaccError:
                selected_pair = ()
            overlay_name = f"RACC overlay volume: {pair[0]} x {pair[1]}"
            if selected_pair == pair and overlay_name in self.viewer.layers:
                channel_1_layer, channel_2_layer = self._selected_layers()
                if (
                    channel_1_layer.data is not request.channel_1_data
                    or channel_2_layer.data is not request.channel_2_data
                ):
                    self._schedule_expensive_display_refresh()
                    return
                overlay_layer = self.viewer.layers[overlay_name]
                overlay_layer.data = overlay_data
                metadata = dict(overlay_layer.metadata)
                metadata.update(
                    {
                        "display_cutoff_1": request.display_cutoff_1,
                        "display_cutoff_2": request.display_cutoff_2,
                    }
                )
                overlay_layer.metadata = metadata
                self._apply_volume_display_style()
        self._launch_pending_display_refresh()

    def _handle_display_error(self, error: Exception) -> None:
        self._display_worker_active = False
        self._set_status(f"Display update failed: {error}")
        self._launch_pending_display_refresh()

    def _apply_overlay_display_style(self, *args) -> None:
        try:
            channel_1_layer, channel_2_layer = self._selected_layers()
        except RaccError:
            return
        raw_overlay_name = (
            f"RACC overlay volume: {channel_1_layer.name} x {channel_2_layer.name}"
        )
        if raw_overlay_name not in self.viewer.layers:
            return
        overlay_layer = self.viewer.layers[raw_overlay_name]
        configure_overlay_display(
            self.viewer,
            overlay_layer,
            intensity_opacity=self._selected_intensity_opacity(),
            background_suppression=self._selected_background_suppression(),
            rendering_mode=self._selected_rendering_mode(),
        )

    def _apply_racc_display_style(self, *args) -> None:
        colormap_name = self._selected_colormap()
        self.scatter_widget.set_colormap(colormap_name)
        result_layer = self._result_layer()
        if result_layer is not None:
            configure_racc_display(
                self.viewer,
                result_layer,
                colormap_name=colormap_name,
                racc_display_floor=self._selected_racc_display_floor(),
                racc_opacity=self._selected_racc_opacity(),
                background_suppression=self._selected_background_suppression(),
                rendering_mode=self._selected_rendering_mode(),
            )
            result_mip_name = f"{result_layer.name} MIP"
            if result_mip_name in self.viewer.layers:
                result_mip = self.viewer.layers[result_mip_name]
                _set_layer_colormap(
                    result_mip,
                    racc_colormap(
                        colormap_name,
                        self._selected_racc_display_floor(),
                    ),
                )
                if not np.isclose(float(result_mip.opacity), 1.0):
                    result_mip._opacity = 1.0
                    result_mip.events.opacity()
        self._update_display_control_availability()

    def _apply_volume_display_style(self, *args) -> None:
        self._apply_overlay_display_style()
        self._apply_racc_display_style()

    def _handle_ndisplay_change(self, *args) -> None:
        self._apply_bounding_box_visibility()
        self._apply_volume_display_style()
        self._update_display_control_availability()
        # napari swaps its ImageNode/VolumeNode after the model event. Reapply
        # the inexpensive style once the new visual exists.
        QTimer.singleShot(0, self._apply_volume_display_style)

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
                if visible:
                    refresh_bounding_box_clipping(self.viewer, layer)

    def _refresh_bounding_box_after_view_change(self) -> None:
        self._refresh_bounding_box_visibility()
        QTimer.singleShot(0, self._refresh_bounding_box_visibility)

    def _update_result_actions(self, *args) -> None:
        try:
            selected_layers = self._selected_layers()
        except RaccError:
            inputs_ready = False
            is_volume = False
        else:
            inputs_ready = True
            is_volume = any(int(layer.ndim) >= 3 for layer in selected_layers)
        dimension = "3D" if is_volume else "2D"
        self.overlay_button.setText(f"{dimension} channels")
        self.racc_button.setText(f"{dimension} RACC map")
        self.side_by_side_button.setText(f"{dimension} side by side")
        self.mip_button.setText("2D Z max projection")
        self.projection_note.setVisible(is_volume)
        self.run_button.setEnabled(inputs_ready)
        self.overlay_button.setEnabled(inputs_ready)
        has_result = self._result_layer() is not None
        for button in (
            self.export_button, self.racc_button,
            self.side_by_side_button,
        ):
            button.setEnabled(has_result)
        self.mip_button.setEnabled(has_result and is_volume)
        self.mip_button.setToolTip(
            "Compare fixed 2D maximum-intensity projections along Z, "
            "derived from the full 3D analysis."
            if is_volume else "Z projections require 3D input stacks."
        )
        self.side_by_side_button.setToolTip(
            f"Compare channel intensities with the RACC map in {dimension}."
            if has_result else "Run RACC for the selected channels first."
        )

    def run_racc(self) -> None:
        self._debounce_timer.stop()
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
        try:
            ch1, ch2 = self._selected_layers()
        except RaccError:
            return
        if pair != (ch1.name, ch2.name):
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
        else:
            layer = self.viewer.add_image(
                result.index,
                name=name,
                colormap=racc_colormap(
                    self._selected_colormap(),
                    self._selected_racc_display_floor(),
                ),
                contrast_limits=racc_display_contrast_limits(),
                blending="translucent",
                metadata=metadata,
                scale=self._scale_for_result(pair),
                depiction="volume",
                rendering=self._selected_rendering_mode(),
            )
        configure_racc_display(
            self.viewer,
            layer,
            colormap_name=self._selected_colormap(),
            racc_display_floor=self._selected_racc_display_floor(),
            racc_opacity=self._selected_racc_opacity(),
            background_suppression=self._selected_background_suppression(),
            rendering_mode=self._selected_rendering_mode(),
        )
        self._result_layer_name = layer.name
        self._result_pair = pair
        self._update_result_actions()
        self.scatter_widget.set_axis_labels(pair[0], pair[1])
        self.scatter_widget.set_result(result)
        self.scatter_widget.set_colormap(self._selected_colormap())
        self._refresh_existing_mips(pair, layer)
        self._apply_bounding_box_visibility()

        warnings = " ".join(result.warnings)
        self._set_status(
            f"RACC complete. Voxels meeting both thresholds: "
            f"{result.parameters.overlap_voxels} "
            f"({result.parameters.overlap_fraction:.2%} of all voxels). "
            f"{warnings}".strip()
        )
        self.result_ready.emit(layer)

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
            viewer=self.viewer,
            channel_1_layer=self.viewer.layers[pair[0]],
            channel_2_layer=self.viewer.layers[pair[1]],
            result_layer=result_layer,
            colormap_name=self._selected_colormap(),
            overlay_color_1=self._selected_overlay_colors()[0],
            overlay_color_2=self._selected_overlay_colors()[1],
            display_cutoff_1=self._selected_display_cutoffs()[0],
            display_cutoff_2=self._selected_display_cutoffs()[1],
            racc_display_floor=self._selected_racc_display_floor(),
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
            intensity_opacity=self._selected_intensity_opacity(),
            display_cutoff_1=self._selected_display_cutoffs()[0],
            display_cutoff_2=self._selected_display_cutoffs()[1],
            overlay_color_1=self._selected_overlay_colors()[0],
            overlay_color_2=self._selected_overlay_colors()[1],
            background_suppression=self._selected_background_suppression(),
            rendering_mode=self._selected_rendering_mode(),
        )
        self._refresh_bounding_box_after_view_change()

    def _show_racc(self) -> None:
        result = self._result_layer()
        if result is None:
            self._set_status("Run RACC before switching to the RACC view.")
            return
        show_racc_only(
            viewer=self.viewer,
            result_layer=result,
            colormap_name=self._selected_colormap(),
            racc_display_floor=self._selected_racc_display_floor(),
            racc_opacity=self._selected_racc_opacity(),
            background_suppression=self._selected_background_suppression(),
            rendering_mode=self._selected_rendering_mode(),
        )
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
            intensity_opacity=self._selected_intensity_opacity(),
            display_cutoff_1=self._selected_display_cutoffs()[0],
            display_cutoff_2=self._selected_display_cutoffs()[1],
            overlay_color_1=self._selected_overlay_colors()[0],
            overlay_color_2=self._selected_overlay_colors()[1],
            colormap_name=self._selected_colormap(),
            racc_display_floor=self._selected_racc_display_floor(),
            racc_opacity=self._selected_racc_opacity(),
            background_suppression=self._selected_background_suppression(),
            rendering_mode=self._selected_rendering_mode(),
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
            viewer=self.viewer,
            channel_1_layer=ch1,
            channel_2_layer=ch2,
            result_layer=result,
            colormap_name=self._selected_colormap(),
            overlay_color_1=self._selected_overlay_colors()[0],
            overlay_color_2=self._selected_overlay_colors()[1],
            display_cutoff_1=self._selected_display_cutoffs()[0],
            display_cutoff_2=self._selected_display_cutoffs()[1],
            racc_display_floor=self._selected_racc_display_floor(),
        )
        self._refresh_bounding_box_after_view_change()

    def export_racc_tiff(self) -> None:
        result = self._result_layer()
        if result is None:
            self._set_status("Run RACC before exporting a TIFF stack.")
            self._update_result_actions()
            return

        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export RACC TIFF",
            f"{_safe_export_stem(result.name)}.tif",
            "TIFF files (*.tif *.tiff);;All files (*)",
        )
        if not file_path:
            return

        try:
            written_path = self._export_racc_tiff_to_path(file_path, result)
        except Exception as error:  # noqa: BLE001
            self._set_status(f"RACC TIFF export failed: {error}")
            return
        self._set_status(f"Exported RACC TIFF: {written_path}")

    def _export_racc_tiff_to_path(self, file_path, result_layer=None) -> Path:
        layer = result_layer if result_layer is not None else self._result_layer()
        if layer is None:
            raise RaccError("Run RACC before exporting a TIFF stack.")

        path = Path(file_path).expanduser()
        if path.suffix.lower() not in {".tif", ".tiff"}:
            path = path.with_suffix(".tif")
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        data = np.asarray(layer.data)
        if data.ndim < 2:
            raise RaccError("RACC TIFF export requires a 2D image or 3D stack.")
        if data.dtype == np.float64 or np.issubdtype(data.dtype, np.floating):
            export_data = data.astype(np.float32, copy=False)
        else:
            export_data = data

        metadata = _tiff_metadata_for_layer(layer, export_data)
        tifffile.imwrite(
            path,
            export_data,
            photometric="minisblack",
            metadata={"axes": metadata["axes"]},
            description=json.dumps(metadata, sort_keys=True),
            software="napari-racc",
        )
        return path

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


def _compute_display_overlay_job(request: _DisplayComputeRequest):
    data = compute_overlay_rgb(
        request.channel_1_data,
        request.channel_2_data,
        channel_1_rgb=request.channel_1_rgb,
        channel_2_rgb=request.channel_2_rgb,
        display_cutoff_1=request.display_cutoff_1,
        display_cutoff_2=request.display_cutoff_2,
        overlay_color_1=request.overlay_color_1,
        overlay_color_2=request.overlay_color_2,
    )
    return request, data


def _set_layer_colormap(layer, colormap) -> None:
    selected = ensure_colormap(colormap)
    if getattr(getattr(layer, "colormap", None), "name", None) == selected.name:
        return
    layer._colormap = selected
    layer.events.colormap()


def _result_name(pair: tuple[str, str]) -> str:
    return f"RACC: {pair[0]} x {pair[1]}"


def _safe_export_stem(name: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {" ", ".", "_", "-"} else "_"
        for char in str(name)
    )
    safe = "_".join(safe.strip().split())
    return safe[:160] or "racc_result"


def _axes_for_data(data: np.ndarray) -> str:
    if data.ndim == 2:
        return "YX"
    if data.ndim == 3:
        return "ZYX"
    return "".join(f"Q{axis}" for axis in range(data.ndim - 2)) + "YX"


def _tiff_metadata_for_layer(layer, data: np.ndarray) -> dict:
    metadata = dict(layer.metadata)
    return {
        "axes": _axes_for_data(data),
        "napari_racc": {
            "layer_name": str(layer.name),
            "input_1": _json_safe(metadata.get("racc_input_1", "")),
            "input_2": _json_safe(metadata.get("racc_input_2", "")),
            "scale": [float(value) for value in getattr(layer, "scale", ())],
            "parameters": {
                str(key): _json_safe(value)
                for key, value in metadata.items()
                if key
                not in {
                    "napari_racc_kind",
                    "racc_input_1",
                    "racc_input_2",
                }
            },
        },
    }


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _threshold_to_control_value(value: float) -> int:
    return int(max(0, min(255, int(float(value) + 0.5))))


def _is_racc_generated_layer(layer) -> bool:
    if str(layer.name).startswith("RACC:"):
        return True
    return layer.metadata.get("napari_racc_kind") in {"result", "mip", "overlay"}


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


def _set_layer_scale_without_data_refresh(layer, scale: tuple[float, ...]) -> None:
    """Update the transform without reslicing or re-uploading image data."""

    if tuple(float(value) for value in layer.scale) == tuple(scale):
        return
    block_refresh = getattr(layer, "_block_refresh", None)
    if callable(block_refresh):
        with block_refresh():
            layer.scale = scale
        return
    layer.scale = scale
