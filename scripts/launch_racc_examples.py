from __future__ import annotations

import argparse
import os
import sys

import napari

from napari_racc._sample_data import make_sample_data_2d, make_sample_data_3d
from napari_racc._widget import RaccWidget


def main() -> None:
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
    args = parser.parse_args()

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        sys.exit(
            "Refusing to launch napari with QT_QPA_PLATFORM=offscreen. "
            "Napari needs a real Qt/OpenGL context for the viewer."
        )

    viewer = napari.Viewer()
    if args.example in {"2d", "both"}:
        for data, kwargs, layer_type in make_sample_data_2d():
            if layer_type == "image":
                viewer.add_image(data, **kwargs)
    if args.example in {"3d", "both"}:
        for data, kwargs, layer_type in make_sample_data_3d():
            if layer_type == "image":
                kwargs = dict(kwargs)
                if args.example == "both":
                    kwargs["visible"] = False
                viewer.add_image(data, **kwargs)

    widget = RaccWidget(viewer)
    viewer.window.add_dock_widget(widget, name="RACC", area="right")
    napari.run()


if __name__ == "__main__":
    main()
