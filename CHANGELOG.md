# Changelog

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
