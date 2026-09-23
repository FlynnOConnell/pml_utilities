# MESc files

One HDF5 file, one unit per recording, all from a Femtonics AOD two-photon
microscope. The MESc tab lists them, one row each.

| session | holds |
|---|---|
| `MSession_0` | what the operator ran |
| `MSession_1` | what MEScan saved beside each run, on its own |

Units number from 0 in every session, so `MSession_0/MUnit_3` and
`MSession_1/MUnit_3` are different recordings.

## Kinds of recording

| modality | the laser | what you get |
|---|---|---|
| `timeseries` | rasters the whole field | one frame per timepoint; a single frame is a **picture** |
| `zstack` | rasters at each depth | one frame per slice, no time axis |
| `linescan` | runs along each drawn line | one row of samples per line per cycle: a kymograph |
| `chessboard` | rasters each drawn square | each patch is a small movie |
| `ribbon` | sweeps a bent strip along a dendrite | each strip is a patch |
| `multicube` | scans small volumes slice by slice | Z is real depth |

The last four are AOD scans: the operator drew lines or patches on a picture,
and the laser jumps only between those spots, thousands of times a second,
never recording the rest of the field. Each line or patch is one step of the
viewer's ROI slider.

## What MEScan saves beside a scan

| in the table | what it is |
|---|---|
| the **picture** cell | the picture the lines were drawn on: its own unit, role `background`, named by the scan's `BackgroundImagePath` |
| behind the **RTMC** cell | the region RTMC re-scanned each cycle: its own unit, role `motionCorrection`, named by `MotionCorrectionImagePath` |
| the MC plot under the traces | the X, Y, Z microns RTMC moved the scan, once per cycle: curves on the scan unit itself |

**companion units as rows** gives the two units rows of their own.

## Columns

| column | means |
|---|---|
| session, unit | which recording; hover a name for what it is |
| modality | how the laser moved |
| layout | how the reader unpacks the raw pixels |
| ROIs | lines or patches the operator drew |
| picture | the picture they were drawn on; click to see them on it |
| RTMC | real-time motion correction was on (`yes`) or off (`no`) |
| T C Z Y X | the 5D shape: timepoints, colors, slices or ROIs, rows, columns |
| fs, duration | timepoints per second; recorded length, short if stopped early |
| start, comment | acquisition time (UTC-05:00) and the comment typed in MESc |

Hover a header for the long version. Right-click one to show or hide columns.

## Where the lines were drawn

Every picture and Z-stack records the micron position of its top-left pixel
and its size; every AOD scan records the micron coordinates of each line's
ends or each patch's corners. The **picture** button draws them back on:

| image | shows |
|---|---|
| the picture | every line, whatever its depth, because that is where they were drawn |
| a Z-stack | only the lines scanned inside it, in field **and** depth range; the caption says how many of the scan's lines those are |

The ROI slider's line is drawn thicker; click a line to move the slider to it.
The popup's **display** button puts that image in the viewer, a Z-stack at the
slice the lines sit on. That is the only place a Z-stack is offered.

## Clicks

| do | gets |
|---|---|
| click a row | displays that recording; its ROIs, runs and traces are parked and come back when you return |
| click the picture cell | the reference image |
| right-click a header | show or hide columns |
