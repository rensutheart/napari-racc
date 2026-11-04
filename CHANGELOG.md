# Changelog

## 0.5.0 - 2026-11-04

- Fixed clipped regression lines and percentile bands in the scatter plot.
- Added logarithmic intensity axes and a toggle for the fitted guides.
- Added a focused histogram of voxels that meet both thresholds, with improved
  visibility for sparse bins.
- Made the square plot grow with the width of the dock.

## 0.4.0 - 2026-11-04

- Improved scatter histogram contrast for images with large empty backgrounds,
  without altering histogram counts or the RACC calculation.
- Added individual reset buttons beside the XY and Z scale fields to restore
  the default display scale of 1.00.
- Fixed bounding boxes retaining an old grid-cell clipping filter when switching
  between single and side-by-side views.
- Labelled view actions with their destination dimensionality and explained
  fixed Z maximum-intensity projections beside the projection action.
- Enlarged the Analysis/Appearance tabs, clarified view labels, and added
  theme-aware spacing, expandable rows, and a more prominent calculation action.
- Separated Analysis and Appearance settings, with optional method controls,
  diagnostics, visibility filters, colors, and geometry in expandable sections.
- Kept calculation, status, comparison, and export actions visible while
  scrolling, and disabled result views until the selected channels have a result.
- Clarified working-scale threshold units and the meaning of the overlap
  voxel fraction, and prevented previous-input workers updating current results.
- Added a prepared `--demo` launch with complete, deterministic synthetic
  spheres and a five-minute presentation guide.

## 0.3.1 - 2026-08-15

- Made XY/Z scale changes transform-only, avoiding volume reconstruction and
  GPU texture re-upload on every slider step.
- Moved expensive cutoff/color remapping to a debounced background worker and
  reduced generated overlay volumes from float RGBA to compact uint8 RGBA.
- Made renderer, opacity, suppression, RACC-floor, and colormap edits update
  display state without replacing voxel data.
- Preserved the brighter per-voxel opacity accumulation for translucent and
  additive 3D rendering through shader/colormap transfer settings.
- Separated 3D volume opacity/suppression from single-slice and fixed 2D Z-MIP
  views, which now render retained pixels at full strength.

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
