# MESc files

A `.mesc` is one HDF5 file holding every recording of a session at a Femtonics AOD two-photon microscope. The MESc tab lists them. This page says what each kind of recording is and how the columns read.

## What the microscope did, in order

- **Took one picture of the tissue**: an ordinary raster frame, the whole field (512 x 512 px, both colors). MEScan saves it as its own unit with the role `background`. The table calls it the **picture**.
- **The operator drew on that picture where to record**: straight lines or small rectangles (patches). The laser then jumps only between those spots, hundreds or thousands of times a second, and never records the rest of the field. That is an AOD scan.
- **Line scan**: each line is one pixel wide and a few tens of pixels long. Every cycle the laser runs along each line once and records one row of samples per line. Over time each line becomes a (time, samples) image, a kymograph. The viewer puts the lines on its ROI slider.
- **Chessboard**: each patch is a small square, say 20 x 20 px. Every cycle the laser rasters each square once, so each square is a small movie. On disk MESc writes the squares side by side in one wide row per timepoint; the reader cuts them apart onto the ROI slider.
- **Ribbon**: a bent strip drawn along a dendrite, swept every cycle; each strip is a patch.
- **While recording, the microscope tracked drift**: that is RTMC, real-time motion correction. Every cycle it also scans a small reference region, compares it with the first cycle and shifts the scan to follow the tissue.
- **Z-stack**: ordinary raster frames taken at each depth, a micron or so apart, with no time axis. The stack is a picture of the whole volume the scans were placed in.

## What MEScan saves beside each scan

- **The picture**: the scan's `BackgroundImagePath` attribute names it. The table folds it into the scan's row as the **picture** cell.
- **The RTMC reference pixels**: the small region re-scanned every cycle, saved as its own unit with the role `motionCorrection`; the scan's `MotionCorrectionImagePath` names it. For a line scan it is two reference lines written as rows, for a chessboard one reference square per cycle. It is the reference target, not a movie of the field, so it is rarely worth displaying. The table folds it into the scan's row behind the **RTMC** cell.
- **The RTMC shifts**: how far the scan was moved, in microns, X, Y and Z, once per cycle. These are curves on the scan unit itself and are what the Traces tab's MC plot draws.
- Sessions: `MSession_0` holds what the operator ran; `MSession_1` holds what MEScan saved beside each run. Units are numbered from 0 in every session, so `MSession_0/MUnit_3` and `MSession_1/MUnit_3` are different recordings.

## The RTMC cell

- **yes**: real-time motion correction was on for this recording. Hover the cell: if the tissue moved, the scan was shifted to follow it and the X, Y, Z shifts in microns are the MC plot under the traces; if it never moved there is nothing to plot.
- **no**: it was off.
- The reference region RTMC re-scanned is its own unit, folded into the row; **companion units as rows** lists it.

## Where the lines were drawn

Every picture and Z-stack records the micron position of its top-left pixel and its width and height in microns. Every AOD scan records the micron coordinates of each line's two ends, or of each patch's four corners. From those the tab draws the lines and patches back on the picture:

- **Reference image** (the button, or the **picture** and **Z-stack** cells' popups) opens a popup with the picture the scan's lines or patches were drawn on and the max projection over depth of every Z-stack whose field holds them, with them drawn in MESc's colours, the one the ROI slider is on thicker. Every line is drawn whatever its depth. Click a line there to move the ROI slider to it.
- **Z-stack**: for a scan, the Z-stack whose field covers its lines or patches. For a Z-stack, how many scans it covers. MESc records no such link; it is worked out from the micron positions.

## Rows and clicks

- One row per recording. A scan's picture and RTMC reference pixels are cells of its row; **companion units as rows** gives them rows of their own.
- Click a row to display that recording. Its ROIs, runs and traces are parked when you switch away and come back when you return.
- Right-click a column header to show or hide columns; hover one for what it means.
