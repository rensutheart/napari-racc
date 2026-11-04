# RACC demonstration guide

Lead with the image comparison and a single question: **Where do the two
fluorescence patterns agree in 3D?** The initial demonstration can take
about five minutes; leave the method details available for questions.

## Start a prepared example

From the project folder, with the development environment active:

```bash
python scripts/launch_racc_examples.py --example 3d --demo
```

The demo loads compact, deterministic synthetic spheres, calculates RACC, and
opens the intensity/RACC comparison with Appearance selected. The title and
channel names identify the example as synthetic. It uses complete spheres at
their generated scale, without stretching the geometry. No external images or
network connection are needed once the environment is installed.

For a separate 2D session:

```bash
python scripts/launch_racc_examples.py --example 2d --demo
```

The ordinary launch without `--demo` keeps the existing example-loader
behavior. It may use the larger legacy example files if they exist beside the
checkout. `--demo` always uses the same small synthetic source and supports one
dimensionality per session.

## Five-minute walkthrough

1. Start with the prepared 3D comparison. The intensity view shows the two
   synthetic input channels, and the RACC view shows their spatial agreement.
2. Rotate the 3D view and try changing the opacity or background suppression
   under Appearance. Switch between the channels, RACC map, side by side, and
   2D Z max projection views to compare different ways of displaying the same
   inputs and result.
3. Open Analysis and adjust the two channel thresholds. Run RACC again to see
   how the changed thresholds affect the result.
4. Use Export RACC TIFF to save the numeric result. When trying your own data,
   check the channel pairing and voxel scale and record the analysis settings.

## Answers to likely questions

**What changes the calculation?** The selected channels, channel thresholds,
theta, and included percentile affect the result. Thresholds use the plugin's
0–255 intensity scale. Costes provides an automatic threshold estimate that
still needs inspection for the dataset.

**What changes only the picture?** Display cutoffs, the displayed RACC minimum,
colours, rendering method, opacity, and background suppression control how the
data are shown. Display cutoffs can be linked to the analysis thresholds, so
an analysis threshold change also updates that appearance setting.

**Why do slices and fixed MIPs look stronger?** Single slices and fixed 2D
Z-projections (maximum-intensity projections, or MIPs) ignore the 3D volume
opacity and suppression controls. They still apply the chosen display cutoffs
and RACC display minimum.

**What does the bright colour mean?** It represents a higher value in this
qualitative RACC colour mapping. It is not a probability of molecular
interaction. RACC should be interpreted with acquisition quality, background,
channel registration, and the biological question in mind.

**Is a MIP a new analysis?** No. The fixed RACC Z-MIP is a maximum-intensity
projection of the full 3D RACC calculation. It combines depth and hides where
along a viewing ray the maximum occurs. Compare it with slices or the
translucent view when depth matters. The maximum-intensity 3D renderer also
selects maxima along viewing rays; it can make an internal bright region
appear on an object's silhouette.

**Can a single overlap percentage replace the image?** The status fraction
counts voxels passing both channel thresholds. It is not a universal
colocalisation coefficient or a substitute for interpreting the spatial map.
