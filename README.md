# napari-racc

`napari-racc` is a napari plugin for Regression Adjusted Colocalisation Colour
Mapping (RACC), a qualitative visualization method for 2D and 3D fluorescence
microscopy data.

The plugin takes two image layers, computes the RACC index in 3D whenever the
inputs are volumes, and adds interactive overlay, RACC, side-by-side, MIP, and
scatter-plot views to the napari viewer.

## Screenshots

![RACC 3D side-by-side volume view](https://raw.githubusercontent.com/rensutheart/napari-racc/main/docs/images/racc-side-by-side-volume.png)

3D side-by-side view with the thresholded channel overlay on the left and the
RACC volume on the right.

![RACC side-by-side MIP view](https://raw.githubusercontent.com/rensutheart/napari-racc/main/docs/images/racc-side-by-side-mip.png)

3D-derived maximum-intensity projection view.

<img src="https://raw.githubusercontent.com/rensutheart/napari-racc/main/docs/images/racc-widget-controls.png" alt="RACC widget controls" width="360">

Scrollable RACC controls with manual thresholds, Costes thresholding, display
scale controls, probe colors, scatter diagnostics, result export, and view
switching.

## Features

- two-channel RACC calculation from napari `Image` layers
- live threshold, theta, percentile, and Costes threshold controls
- transparent zero-valued RACC voxels for clean volume rendering
- thresholded RGB overlay volume with selectable probe colors
- side-by-side overlay/RACC and 3D-derived MIP views
- scatter histogram with visible axes, regression, threshold, and percentile-band overlays
- XY and Z display scale controls for metadata-light TIFF stacks
- scrollable control panel with expandable input layer selectors
- export of the numeric RACC result stack as TIFF

## Installation

Install from PyPI:

```bash
pip install napari-racc
```

For local development:

```bash
git clone https://github.com/rensutheart/napari-racc.git
cd napari-racc
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Fish shell users should activate the environment with:

```fish
source .venv/bin/activate.fish
```

## Usage

1. Open napari.
2. Open two image stacks or use `File > Open Sample > RACC`.
3. Start the widget from `Plugins > RACC (napari-racc)`.
4. Select channel 1 and channel 2.
5. Adjust thresholds manually or press `Costes thresholds`.
6. Press `Run RACC`.
7. Use `Overlay`, `RACC`, `3D side by side`, and `MIPs` to switch views.
8. Press `Export RACC TIFF` to save the numeric RACC result stack for use in other software.

RACC is calculated over the full 3D volume when 3D inputs are used. The MIP view
is derived from the 3D calculation; it is not a 2D recalculation.

## Development

```bash
python -m npe2 validate src/napari_racc/napari.yaml
python -m ruff check src
python -m pytest
python -m build
```

Launch one example dimensionality at a time:

```bash
python scripts/launch_racc_examples.py --example 3d
python scripts/launch_racc_examples.py --example 2d
```

Do not launch the napari viewer with `QT_QPA_PLATFORM=offscreen`; napari needs a
real Qt/OpenGL context for the viewer on macOS.

## Citation

If you use this plugin or the RACC method in research, cite:

Theart RP, Loos B, Niesler TR. Regression adjusted colocalisation colour mapping
(RACC): A novel biological visual analysis method for qualitative
colocalisation analysis of 3D fluorescence micrographs. PLOS ONE 14(11):
e0225141. <https://doi.org/10.1371/journal.pone.0225141>

## License And Patent Notice

This software is licensed under the PolyForm Noncommercial License 1.0.0. It is
source-available for noncommercial research, education, and evaluation use, but
it is not an OSI open-source license.

Use of the RACC method may be covered by patent rights, including US patent
application US20220189129A1 and related patent family members. Commercial,
clinical, diagnostic, or for-profit service use requires a separate license from
the rights holder. See `LICENSE`, `NOTICE`, and `PATENTS.md`.
