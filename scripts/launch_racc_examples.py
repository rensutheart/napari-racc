from __future__ import annotations

import argparse
import os
import sys

import napari
from qtpy.QtCore import QTimer

from napari_racc._sample_data import (
    make_demo_data_2d,
    make_demo_data_3d,
    make_sample_data_2d,
    make_sample_data_3d,
)
from napari_racc._widget import RaccWidget


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example",
        choices=("2d", "3d", "both"),
        default="3d",
        help=(
            "Example data to load. Use separate 2D and 3D sessions when possible; "
            "'both' is only for mixed-dimension debugging."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Use compact synthetic data, calculate RACC, and open a prepared "
            "side-by-side comparison."
        ),
    )
    args = parser.parse_args(argv)
    if args.demo and args.example == "both":
        parser.error("--demo requires one dimensionality: --example 2d or 3d")

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        sys.exit(
            "Refusing to launch napari with QT_QPA_PLATFORM=offscreen. "
            "Napari needs a real Qt/OpenGL context for the viewer."
        )

    viewer = (
        napari.Viewer(title=f"RACC | synthetic {args.example.upper()} demo")
        if args.demo
        else napari.Viewer()
    )
    if args.example in {"2d", "both"}:
        sample_2d = make_demo_data_2d if args.demo else make_sample_data_2d
        for data, kwargs, layer_type in sample_2d():
            if layer_type == "image":
                viewer.add_image(data, **kwargs)
    if args.example in {"3d", "both"}:
        sample_3d = make_demo_data_3d if args.demo else make_sample_data_3d
        for data, kwargs, layer_type in sample_3d():
            if layer_type == "image":
                kwargs = dict(kwargs)
                if args.example == "both":
                    kwargs["visible"] = False
                viewer.add_image(data, **kwargs)

    widget = RaccWidget(viewer)
    viewer.window.add_dock_widget(widget, name="RACC", area="right")
    if args.demo:
        def show_demo_result(_result_layer):
            widget.result_ready.disconnect(show_demo_result)
            widget._show_side_by_side()
            widget.settings_tabs.setCurrentIndex(1)

        widget.result_ready.connect(show_demo_result)
        QTimer.singleShot(0, widget.run_racc)
    napari.run()


if __name__ == "__main__":
    main()
