from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import tifffile
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QScrollArea, QSizePolicy

from napari_racc._racc import CostesThresholds
from napari_racc._widget import (
    OVERLAY_VOLUME_ALPHA_EXPONENT,
    OVERLAY_VOLUME_ALPHA_MAX,
    RaccWidget,
)


class _Event:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback

    def emit(self):
        if self.callback is not None:
            self.callback()


class _Events:
    def __init__(self):
        self.inserted = _Event()
        self.removed = _Event()
        self.reordered = _Event()


class _DimsEvents:
    def __init__(self):
        self.ndisplay = _Event()


class _Dims:
    def __init__(self, ndisplay=3):
        self.ndisplay = ndisplay
        self.events = _DimsEvents()


class _LayerList(list):
    def __init__(self, layers):
        super().__init__(layers)
        self.events = _Events()

    def __contains__(self, item):
        if isinstance(item, str):
            return any(layer.name == item for layer in self)
        return super().__contains__(item)

    def __getitem__(self, item):
        if isinstance(item, str):
            for layer in self:
                if layer.name == item:
                    return layer
            raise KeyError(item)
        return super().__getitem__(item)


class _Grid:
    enabled = False
    stride = 1
    spacing = 0.0


def _image(name: str, shape=(8, 8), metadata=None):
    return Image(
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        name=name,
        metadata=metadata or {},
    )


class _Viewer:
    def __init__(self, layers=None, ndisplay=3):
        self.layers = _LayerList(layers or [_image("ch1"), _image("ch2")])
        self.dims = _Dims(ndisplay=ndisplay)
        self.grid = _Grid()

    def add_image(self, data, **kwargs):
        layer = Image(data, **kwargs)
        self.layers.append(layer)
        return layer


class _ResultParameters:
    overlap_voxels = 12
    overlap_fraction = 0.25

    def to_dict(self):
        return {
            "overlap_voxels": self.overlap_voxels,
            "overlap_fraction": self.overlap_fraction,
        }


def test_widget_instantiates_without_napari_canvas(qtbot):
    widget = RaccWidget(_Viewer())
    qtbot.addWidget(widget)

    assert widget.channel_1_combo.count() == 2
    assert widget.channel_2_combo.count() == 2
    assert widget.overlay_alpha_spin.value() == 2
    assert widget.volume_alpha_spin.value() == 2
    assert widget.export_button.isEnabled() is False
    assert isinstance(widget._scroll_area, QScrollArea)
    assert widget._scroll_area.widgetResizable() is True
    assert widget._scroll_area.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert np.isclose(widget._selected_overlay_gain(), 0.02)
    assert np.isclose(widget._selected_volume_alpha(), 0.02)


def test_input_combos_expand_and_expose_full_layer_names(qtbot):
    long_name = "RACC Red Sphere very long filename with acquisition metadata"
    viewer = _Viewer([_image(long_name), _image("green channel")])
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    assert (
        widget.channel_1_combo.sizePolicy().horizontalPolicy()
        == QSizePolicy.Policy.Expanding
    )
    assert widget.channel_1_combo.currentText() == long_name
    assert widget.channel_1_combo.toolTip() == long_name
    assert widget.channel_1_combo.itemData(0, Qt.ToolTipRole) == long_name


def test_widget_ignores_racc_generated_layers_in_input_choices(qtbot):
    viewer = _Viewer(
        [
            _image("ch1"),
            _image("ch2"),
            _image(
                "RACC: ch1 x ch2",
                metadata={"napari_racc_kind": "result"},
            ),
            _image(
                "RACC overlay red: ch1 x ch2",
                metadata={"napari_racc_kind": "overlay"},
            ),
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.channel_1_combo.count() == 2
    assert widget.channel_1_combo.findText("RACC: ch1 x ch2") == -1
    assert widget.channel_1_combo.findText("RACC overlay red: ch1 x ch2") == -1


def test_widget_does_not_return_stale_result_for_new_input_pair(qtbot):
    viewer = _Viewer(
        [
            _image("red 2d"),
            _image("green 2d"),
            _image("red sphere", shape=(2, 8, 8)),
            _image("green sphere", shape=(2, 8, 8)),
            _image(
                "RACC: red 2d x green 2d",
                metadata={
                    "napari_racc_kind": "result",
                    "racc_input_1": "red 2d",
                    "racc_input_2": "green 2d",
                },
            ),
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)
    widget._result_layer_name = "RACC: red 2d x green 2d"
    widget._result_pair = ("red 2d", "green 2d")

    widget.channel_1_combo.setCurrentText("red sphere")
    widget.channel_2_combo.setCurrentText("green sphere")

    assert widget._result_layer() is None


def test_widget_applies_costes_thresholds_to_controls(qtbot):
    viewer = _Viewer([_image("ch1"), _image("ch2")])
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)
    widget.live_checkbox.setChecked(False)
    widget.costes_button.setEnabled(False)
    widget._costes_pair = ("ch1", "ch2")

    widget._handle_costes_thresholds(
        CostesThresholds(
            threshold_1=12.4,
            threshold_2=56.6,
            slope=1.0,
            intercept=0.0,
            pearson_below=0.0,
            iterations=4,
        )
    )

    assert widget.threshold_1_spin.value() == 12
    assert widget.threshold_1_slider.value() == 12
    assert widget.threshold_2_spin.value() == 57
    assert widget.threshold_2_slider.value() == 57
    assert widget.costes_button.isEnabled()
    assert "Costes thresholds applied" in widget.status_label.text()


def test_widget_bounding_box_checkbox_updates_3d_layers(qtbot):
    viewer = _Viewer(
        [
            _image("ch1", shape=(2, 8, 8)),
            _image("ch2", shape=(2, 8, 8)),
            _image("flat", shape=(8, 8)),
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    widget.bounding_box_checkbox.setChecked(True)

    assert viewer.layers["ch1"].bounding_box.visible is True
    assert viewer.layers["ch2"].bounding_box.visible is True
    assert viewer.layers["ch1"].bounding_box.blending == "translucent_no_depth"
    assert viewer.layers["ch2"].bounding_box.blending == "translucent_no_depth"
    assert viewer.layers["flat"].bounding_box.visible is False

    widget.bounding_box_checkbox.setChecked(False)

    assert viewer.layers["ch1"].bounding_box.visible is False
    assert viewer.layers["ch2"].bounding_box.visible is False


def test_widget_bounding_box_only_applies_to_visible_3d_layers(qtbot):
    hidden_layer = _image("hidden", shape=(2, 8, 8))
    hidden_layer.visible = False
    viewer = _Viewer(
        [
            _image("ch1", shape=(2, 8, 8)),
            _image("ch2", shape=(2, 8, 8)),
            hidden_layer,
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    widget.bounding_box_checkbox.setChecked(True)

    assert viewer.layers["ch1"].bounding_box.visible is True
    assert viewer.layers["hidden"].bounding_box.visible is False


def test_widget_bounding_box_follows_napari_ndisplay_change(qtbot):
    viewer = _Viewer(
        [
            _image("ch1", shape=(2, 8, 8)),
            _image("ch2", shape=(2, 8, 8)),
        ],
        ndisplay=2,
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    widget.bounding_box_checkbox.setChecked(True)

    assert viewer.layers["ch1"].bounding_box.visible is False
    assert viewer.layers["ch2"].bounding_box.visible is False

    viewer.dims.ndisplay = 3
    viewer.dims.events.ndisplay.emit()

    assert viewer.layers["ch1"].bounding_box.visible is True
    assert viewer.layers["ch2"].bounding_box.visible is True


def test_widget_colormap_dropdown_updates_scatter_and_result_layer(qtbot):
    viewer = _Viewer(
        [
            _image("ch1"),
            _image("ch2"),
            _image(
                "RACC: ch1 x ch2",
                metadata={
                    "napari_racc_kind": "result",
                    "racc_input_1": "ch1",
                    "racc_input_2": "ch2",
                },
            ),
            _image(
                "RACC: ch1 x ch2 MIP",
                metadata={"napari_racc_kind": "mip", "racc_mip_kind": "racc"},
            ),
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    widget.colormap_combo.setCurrentText("viridis")

    assert widget.scatter_widget._colormap_name == "viridis"
    assert (
        viewer.layers["RACC: ch1 x ch2"].colormap.name
        == "racc-viridis-transparent"
    )
    assert (
        viewer.layers["RACC: ch1 x ch2 MIP"].colormap.name
        == "racc-viridis-transparent"
    )
    assert viewer.layers["RACC: ch1 x ch2"].colormap.map([0.0])[0, 3] == 0.0


def test_widget_overlay_alpha_updates_selected_channels(qtbot):
    viewer = _Viewer(
        [
            _image("ch1"),
            _image("ch2"),
            _image(
                "RACC overlay red: ch1 x ch2",
                metadata={
                    "napari_racc_kind": "overlay",
                    "racc_overlay_kind": "side_by_side",
                    "racc_overlay_channel": "red",
                },
            ),
            _image(
                "RACC overlay green: ch1 x ch2",
                metadata={
                    "napari_racc_kind": "overlay",
                    "racc_overlay_kind": "side_by_side",
                    "racc_overlay_channel": "green",
                },
            ),
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    widget.overlay_alpha_spin.setValue(38)

    assert viewer.layers["ch1"].opacity == 1.0
    assert viewer.layers["ch2"].opacity == 1.0
    assert np.isclose(viewer.layers["ch1"].colormap.map([1.0])[0, 3], 0.38)
    assert np.isclose(viewer.layers["ch2"].colormap.map([1.0])[0, 3], 0.38)
    red_overlay = viewer.layers["RACC overlay red: ch1 x ch2"]
    green_overlay = viewer.layers["RACC overlay green: ch1 x ch2"]
    expected_volume_alpha = (
        OVERLAY_VOLUME_ALPHA_MAX * 0.38**OVERLAY_VOLUME_ALPHA_EXPONENT
    )
    assert red_overlay.opacity == 1.0
    assert green_overlay.opacity == 1.0
    assert np.isclose(red_overlay.colormap.map([1.0])[0, 3], expected_volume_alpha)
    assert np.isclose(green_overlay.colormap.map([1.0])[0, 3], expected_volume_alpha)


def test_widget_overlay_alpha_is_percentage(qtbot):
    viewer = _Viewer([_image("ch1"), _image("ch2")])
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    widget.overlay_alpha_spin.setValue(180)

    assert widget.overlay_alpha_spin.value() == 100
    assert widget.overlay_alpha_slider.value() == 100
    assert viewer.layers["ch1"].opacity == 1.0
    assert viewer.layers["ch2"].opacity == 1.0
    assert viewer.layers["ch1"].colormap.name == "racc-overlay-red-1.00-gain"
    assert viewer.layers["ch2"].colormap.name == "racc-overlay-green-1.00-gain"
    assert np.isclose(viewer.layers["ch1"].colormap.map([1.0])[0, 3], 1.0)


def test_widget_overlay_alpha_updates_raw_overlay_volume(qtbot):
    viewer = _Viewer([_image("ch1", shape=(3, 4, 5)), _image("ch2", shape=(3, 4, 5))])
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    widget._show_overlay()
    overlay = viewer.layers["RACC overlay volume: ch1 x ch2"]
    initial_alpha = float(overlay.data[..., 3].max())

    widget.overlay_alpha_spin.setValue(100)

    assert overlay.visible is True
    assert overlay.rgb is True
    assert overlay.data.shape == (3, 4, 5, 4)
    assert overlay.data[..., 3].max() >= initial_alpha
    assert overlay.data[..., 3].max() <= 1.0


def test_widget_overlay_alpha_skips_hidden_source_channels(qtbot):
    ch1 = _image("ch1")
    ch2 = _image("ch2")
    ch1.visible = False
    ch2.visible = False
    viewer = _Viewer(
        [
            ch1,
            ch2,
            _image(
                "RACC overlay red: ch1 x ch2",
                metadata={
                    "napari_racc_kind": "overlay",
                    "racc_overlay_kind": "side_by_side",
                    "racc_overlay_channel": "red",
                },
            ),
            _image(
                "RACC overlay green: ch1 x ch2",
                metadata={
                    "napari_racc_kind": "overlay",
                    "racc_overlay_kind": "side_by_side",
                    "racc_overlay_channel": "green",
                },
            ),
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    widget.overlay_alpha_spin.setValue(14)

    assert viewer.layers["ch1"].opacity == 1.0
    assert viewer.layers["ch2"].opacity == 1.0
    assert np.isclose(
        viewer.layers["RACC overlay red: ch1 x ch2"].colormap.map([1.0])[0, 3],
        OVERLAY_VOLUME_ALPHA_MAX * 0.14**OVERLAY_VOLUME_ALPHA_EXPONENT,
    )


def test_widget_scatter_fill_checkbox_updates_scatter_option(qtbot):
    widget = RaccWidget(_Viewer())
    qtbot.addWidget(widget)

    assert widget.scatter_widget._show_percentile_fill is False

    widget.percentile_fill_checkbox.setChecked(True)

    assert widget.scatter_widget._show_percentile_fill is True


def test_widget_volume_alpha_updates_3d_racc_colormap(qtbot):
    viewer = _Viewer(
        [
            _image("ch1", shape=(2, 8, 8)),
            _image("ch2", shape=(2, 8, 8)),
            _image(
                "RACC: ch1 x ch2",
                shape=(2, 8, 8),
                metadata={
                    "napari_racc_kind": "result",
                    "racc_input_1": "ch1",
                    "racc_input_2": "ch2",
                },
            ),
            _image(
                "RACC: ch1 x ch2 MIP",
                metadata={"napari_racc_kind": "mip", "racc_mip_kind": "racc"},
            ),
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    widget.volume_alpha_spin.setValue(24)

    mapped = viewer.layers["RACC: ch1 x ch2"].colormap.map([0.0, 1.0])
    mip_mapped = viewer.layers["RACC: ch1 x ch2 MIP"].colormap.map([0.0, 1.0])
    assert mapped[0, 3] == 0.0
    assert np.isclose(mapped[1, 3], 0.24)
    assert mip_mapped[0, 3] == 0.0
    assert mip_mapped[1, 3] == 1.0


def test_widget_volume_alpha_handles_napari_thumbnail_failure(qtbot, monkeypatch):
    result = _image(
        "RACC: ch1 x ch2",
        shape=(2, 8, 8),
        metadata={
            "napari_racc_kind": "result",
            "racc_input_1": "ch1",
            "racc_input_2": "ch2",
        },
    )
    viewer = _Viewer(
        [
            _image("ch1", shape=(2, 8, 8)),
            _image("ch2", shape=(2, 8, 8)),
            result,
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    thumbnail_calls = 0

    def fail_thumbnail_update():
        nonlocal thumbnail_calls
        thumbnail_calls += 1
        if thumbnail_calls > 2:
            raise RuntimeError("sequence argument must have length equal to input rank")

    monkeypatch.setattr(result, "_update_thumbnail", fail_thumbnail_update)

    widget.volume_alpha_spin.setValue(94)

    assert result.colormap.name == "racc-magma-volume-0.94"


def test_widget_exports_racc_result_as_tiff(qtbot, tmp_path):
    result_data = np.zeros((2, 3, 4), dtype=np.float32)
    result_data[0] = 0.25
    result_data[1] = 0.75
    result = Image(
        result_data,
        name="RACC: ch1 x ch2",
        metadata={
            "napari_racc_kind": "result",
            "racc_input_1": "ch1",
            "racc_input_2": "ch2",
            "threshold_1": 5,
            "threshold_2": 6,
            "theta_degrees": 45,
        },
    )
    viewer = _Viewer(
        [
            _image("ch1", shape=(2, 3, 4)),
            _image("ch2", shape=(2, 3, 4)),
            result,
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    written_path = widget._export_racc_tiff_to_path(tmp_path / "racc_export", result)

    assert written_path == tmp_path / "racc_export.tif"
    np.testing.assert_allclose(tifffile.imread(written_path), result_data)
    with tifffile.TiffFile(written_path) as tiff:
        assert '"axes": "ZYX"' in tiff.pages[0].description
        assert '"input_1": "ch1"' in tiff.pages[0].description


def test_widget_result_update_handles_contrast_thumbnail_failure(qtbot, monkeypatch):
    result = _image(
        "RACC: ch1 x ch2",
        shape=(2, 8, 8),
        metadata={
            "napari_racc_kind": "result",
            "racc_input_1": "ch1",
            "racc_input_2": "ch2",
        },
    )
    viewer = _Viewer(
        [
            _image("ch1", shape=(2, 8, 8)),
            _image("ch2", shape=(2, 8, 8)),
            result,
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    thumbnail_calls = 0

    def fail_thumbnail_update():
        nonlocal thumbnail_calls
        thumbnail_calls += 1
        if thumbnail_calls > 2:
            raise RuntimeError("sequence argument must have length equal to input rank")

    monkeypatch.setattr(result, "_update_thumbnail", fail_thumbnail_update)
    widget._latest_job_id = 1
    widget._pending_jobs[1] = ("ch1", "ch2")

    widget._handle_result(
        (
            1,
            SimpleNamespace(
                index=np.zeros((2, 8, 8), dtype=np.float32),
                scatter_histogram=np.zeros((256, 256), dtype=np.float32),
                parameters=_ResultParameters(),
                warnings=[],
            ),
        )
    )

    assert result.contrast_limits == [1e-06, 1.0]
    assert result.rendering == "translucent"
    assert result._calculate_value_from_ray(np.array([], dtype=np.float32)) is None


def test_widget_live_result_refreshes_existing_mips(qtbot):
    result_layer = Image(
        np.zeros((2, 3, 4), dtype=np.float32),
        name="RACC: ch1 x ch2",
        metadata={
            "napari_racc_kind": "result",
            "racc_input_1": "ch1",
            "racc_input_2": "ch2",
        },
    )
    viewer = _Viewer(
        [
            _image("ch1", shape=(2, 3, 4)),
            _image("ch2", shape=(2, 3, 4)),
            result_layer,
            Image(
                np.zeros((3, 4, 3), dtype=np.float32),
                name="RACC overlay MIP: ch1 x ch2",
                rgb=True,
                metadata={
                    "napari_racc_kind": "mip",
                    "racc_mip_kind": "raw_overlay",
                },
            ),
            Image(
                np.zeros((3, 4), dtype=np.float32),
                name="RACC: ch1 x ch2 MIP",
                metadata={"napari_racc_kind": "mip", "racc_mip_kind": "racc"},
            ),
        ]
    )
    widget = RaccWidget(viewer)
    qtbot.addWidget(widget)

    new_index = np.zeros((2, 3, 4), dtype=np.float32)
    new_index[0] = 0.2
    new_index[1] = 0.8
    widget._latest_job_id = 1
    widget._pending_jobs[1] = ("ch1", "ch2")

    widget._handle_result(
        (
            1,
            SimpleNamespace(
                index=new_index,
                scatter_histogram=np.zeros((256, 256), dtype=np.float32),
                parameters=_ResultParameters(),
                warnings=[],
            ),
        )
    )

    np.testing.assert_allclose(
        viewer.layers["RACC: ch1 x ch2 MIP"].data,
        new_index.max(axis=0),
    )
