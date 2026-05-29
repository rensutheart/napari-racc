from __future__ import annotations

from napari_racc._sample_data import make_sample_data_2d, make_sample_data_3d


def test_sample_data_2d_loads():
    layers = make_sample_data_2d()

    assert len(layers) == 2
    assert layers[0][0].ndim == 2
    assert layers[1][0].shape == layers[0][0].shape
    assert layers[0][1]["colormap"] == "red"
    assert layers[1][1]["colormap"] == "green"


def test_sample_data_3d_loads():
    layers = make_sample_data_3d()

    assert len(layers) == 2
    assert layers[0][0].ndim == 3
    assert layers[1][0].shape == layers[0][0].shape
    assert layers[0][1]["colormap"] == "red"
    assert layers[1][1]["colormap"] == "green"
    assert layers[0][1]["rendering"] == "translucent"
    assert layers[1][1]["depiction"] == "volume"
