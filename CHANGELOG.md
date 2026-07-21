# Changelog

## 0.3.0 - 2026-07-21

- Added linked or independent display cutoffs for the two intensity channels.
- Added a display-only RACC minimum that preserves the fixed 0..1 color scale
  and leaves exported numeric data unchanged.
- Added synchronized translucent, maximum-intensity, and additive 3D renderers.
- Added separate 0–100 intensity and RACC opacity controls alongside background
  suppression, so strong retained signal can be fully opaque without restoring
  low-intensity haze.
- Applied display cutoffs before creating the fixed 2D Z-MIP views.

## 0.2.1 - 2026-06-01

- Fixed napari hover-status errors in 3D translucent volume views when the cursor ray samples no voxels.

## 0.2.0 - 2026-06-01

- Added TIFF export for the numeric RACC result stack.
- Reorganized widget actions into result, single-view, and side-by-side sections.
- Added a scrollable widget panel for smaller dock heights.
- Improved input layer dropdown sizing and long-name tooltips.
- Added visible scatter-plot axes and endpoint intensity values.
- Added documentation screenshots for GitHub and PyPI.
- Updated the plugin display name to avoid `RACC (RACC)` in napari's widget title.

## 0.1.0 - 2026-05-29

- Initial napari plugin release for RACC visualization.
- Added two-channel 2D/3D RACC calculation with live parameter controls.
- Added Costes threshold calculation.
- Added overlay, RACC-only, side-by-side, MIP, and scatter-plot views.
- Added 3D scale controls, bounding-box controls, and selectable probe/RACC colors.
- Added noncommercial license, citation, patent notice, and CI configuration.
