from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_racc import _sample_data
from napari_racc._racc import compute_racc


@pytest.mark.parametrize("dimensions", [2, 3])
def test_demo_data_is_portable_complete_and_computable(monkeypatch, dimensions):
    def unexpected_legacy_lookup(*args):
        raise AssertionError("The demo must not depend on local legacy files")

    monkeypatch.setattr(_sample_data, "_has_legacy_data", unexpected_legacy_lookup)
    maker = getattr(_sample_data, f"make_demo_data_{dimensions}d")
    layers = maker()
    repeated = maker()

    assert len(layers) == 2
    assert layers[0][0].shape == layers[1][0].shape
    assert layers[0][0].ndim == dimensions
    if dimensions == 3:
        assert layers[0][0].shape == (64, 128, 128)
    for (data, kwargs, layer_type), (again, _, _) in zip(
        layers, repeated, strict=True
    ):
        np.testing.assert_array_equal(data, again)
        assert layer_type == "image"
        assert data.dtype == np.float32
        assert data.min() == 0
        assert data.max() == 255
        assert "synthetic" in kwargs["name"]
        assert kwargs["metadata"]["sample_type"] == "synthetic"
        assert "scale" not in kwargs
        # A complete shape has empty boundary planes along every axis.
        for axis in range(dimensions):
            assert not np.any(np.take(data, [0, -1], axis=axis))

    result = compute_racc(layers[0][0], layers[1][0])
    assert result.index.shape == layers[0][0].shape
    assert np.all(np.isfinite(result.index))
    assert result.index.max() == 1
    assert result.parameters.overlap_voxels > 0


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[3] / "scripts" / "launch_racc_examples.py"
    spec = importlib.util.spec_from_file_location("racc_demo_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_launcher_environment(monkeypatch, launcher):
    state = SimpleNamespace(
        viewers=[], widgets=[], timers=[], runs=0, sources=[]
    )

    class Signal:
        def __init__(self):
            self.callbacks = []

        def connect(self, callback):
            self.callbacks.append(callback)

        def disconnect(self, callback):
            self.callbacks.remove(callback)

        def emit(self, value):
            for callback in tuple(self.callbacks):
                callback(value)

    class Viewer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.layers = []
            self.window = SimpleNamespace(add_dock_widget=lambda *a, **k: None)
            state.viewers.append(self)

        def add_image(self, data, **kwargs):
            self.layers.append((data, kwargs))

    class Widget:
        def __init__(self, viewer):
            self.result_ready = Signal()
            self.run_count = 0
            self.compare_count = 0
            self.tab_index = 0
            self.settings_tabs = SimpleNamespace(setCurrentIndex=self.set_tab)
            state.widgets.append(self)

        def set_tab(self, index):
            self.tab_index = index

        def run_racc(self):
            self.run_count += 1

        def _show_side_by_side(self):
            self.compare_count += 1

    def source(name):
        def make_source():
            state.sources.append(name)
            return [(np.zeros((2, 2)), {"name": name}, "image")]

        return make_source

    def event_loop():
        state.runs += 1
        for _delay, callback in tuple(state.timers):
            callback()

    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(launcher.napari, "Viewer", Viewer)
    monkeypatch.setattr(launcher.napari, "run", event_loop)
    monkeypatch.setattr(launcher, "RaccWidget", Widget)
    monkeypatch.setattr(
        launcher,
        "QTimer",
        SimpleNamespace(singleShot=lambda delay, cb: state.timers.append((delay, cb))),
    )
    for kind in ("sample", "demo"):
        for dims in ("2d", "3d"):
            name = f"{kind}_{dims}"
            monkeypatch.setattr(launcher, f"make_{kind}_data_{dims}", source(name))
    return state


@pytest.mark.parametrize("example", ["2d", "3d"])
def test_demo_waits_for_result_before_showing_comparison(
    monkeypatch, launcher, example
):
    state = _fake_launcher_environment(monkeypatch, launcher)
    launcher.main(["--example", example, "--demo"])
    widget = state.widgets[0]

    assert state.sources == [f"demo_{example}"]
    assert "synthetic" in state.viewers[0].kwargs["title"]
    assert state.runs == 1
    assert len(state.timers) == 1
    assert state.timers[0][0] == 0
    assert widget.run_count == 1
    assert widget.compare_count == 0

    widget.result_ready.emit(object())
    assert widget.compare_count == 1
    assert widget.tab_index == 1
    widget.result_ready.emit(object())
    assert widget.compare_count == 1


def test_ordinary_mixed_example_launch_is_unchanged(monkeypatch, launcher):
    state = _fake_launcher_environment(monkeypatch, launcher)
    launcher.main(["--example", "both"])

    assert state.sources == ["sample_2d", "sample_3d"]
    assert state.viewers[0].kwargs == {}
    assert state.viewers[0].layers[1][1]["visible"] is False
    assert state.widgets[0].run_count == 0
    assert not state.timers


def test_demo_rejects_mixed_dimensions_before_startup(launcher):
    with pytest.raises(SystemExit, match="2"):
        launcher.main(["--example", "both", "--demo"])
