# mbo_utilities.annotation

The GUI-free half of the manual ROI + labeling tool. `gui/manual_roi.py` is
the imgui/pygfx shell; everything here imports numpy/zarr only.

## Modules

- `events.py` — `Observable` / `ModelEvent`: the event model the rest of the
  package uses, after fastplotlib's `GraphicFeature` (`add_event_handler(fn,
  "rois")`, `block_events` around bulk edits). Handlers run inside the mutation.
- `store.py` — `RoiLabelStore`: one `RoiRecord` per ROI (its flat `pixels`
  `y * X + x` and `weights` on one `plane`, class index, free-text note, the
  `polygon` as drawn), so ROIs may overlap; one plane per combination of the
  data's scrolling dims (`plane_axes`, z last; depthless data gets one plane;
  T never keys a plane), the user-defined class-label set, and the display
  palettes (`CLASS_COLORS` is the same tab10 set as masknmf's classification GUI,
  so a shared label set looks the same in both tools). `label_image(plane)` is
  the `(Y, X)` picture derived on demand (ROI `i` paints `i + 1`, the smallest
  on top), `mask(i)` / `footprint(i)` one ROI. The store owns the plane
  arithmetic: `plane_of(pos)`, `plane_pos(plane)`, `plane_label(plane)`,
  `roi_z(i)` / `roi_c(i)`. Mutations emit `rois` events; `snapshot()` /
  `restore()` are the undo primitive. Each record also carries a persistent
  `uid` (never reused; `next_uid` round-trips through the zarr) and a `source`
  string (`""` = drawn by hand).
- `traces.py` — `RoiTrace` / `RoiTraceTable`: one row per measurement, keyed by
  `(uid, z, c, engine)` for a drawn ROI (re-running replaces the row) or
  `(source, member)` for rows that stand for no drawn ROI. `ENGINES` names the
  extraction engines.
- `model.py` — `RoiModel`: the store, the trace table and the slider position in
  one observable object; `targets(indices, z=, c=)` says where a run reads each
  ROI, `traced(index, ...)` what it has been measured with.
- `imagej.py` — the exchange with Fiji and plain files: `export_roiset` /
  `import_roiset` (`RoiSet.zip` via `roifile`, polygons on their plane),
  `export_label_tiff` (ImageJ hyperstack), `export_traces` (CSV, one column
  per `RoiTrace.name`), `export_footprints_npz` (masknmf's
  `spatial_footprints`).
- `ngff.py` — `RoiFile`: the rois zarr, `<stem>.rois.zarr`, one zarr v3
  group written whole (`.tmp` sibling swapped into place). `rois/` is the
  truth: ragged membership (`offsets`, `member`, `weight`) plus a column per
  ROI (`uid`, `plane`, `class`, `color`, `polygon`), notes and sources in
  attrs. `traces/` keeps the trace rows of `SAVED_SOURCES` (quick traces and
  pixel averages). `labels/drawn` is a derived OME-NGFF 0.5 label image
  (`write_label_image`: `multiscales` axes and scale, `image-label` colors,
  properties and source) for napari; nothing here reads it back. The root
  attrs hold the shape, `plane_axes`, label names, `next_uid`, the source
  block and the `runs` list (loaded runs with their curation).
  `write_label_image` is the one NGFF label writer in the package;
  `arrays/suite2p.py::_add_suite2p_labels` uses it too.

## Abstraction seam with masknmf-toolbox

The shared imgui widgets that used to come from `masknmf.visualization.imgui`
(label set, ROI table, stroke capture, panels, summary popup) now live in
`mbo_utilities/gui/imgui/`, so the viewer does not track a masknmf branch. This package is
the counterpart on the mbo side, and the split here is drawn so a future
shared package is a file move, not a refactor:

- **shared-shape today** (duplicated by design, aligned APIs): the label-set
  model (`label_names` + `add_label_name` + tab10 colors + hotkey-per-class)
  mirrors masknmf `ClassificationVis`; autosave-on-mutation with the error
  captured into a status line mirrors `CurationVis._autosave`.
- **stays per-package**: what a "component" is (a drawn mask here, a demixed
  footprint there) and the persistence substrate (rois zarr here, results
  hdf5 there).
- **when extracting**: `store.py`'s label-set/palette block and the
  imgui label-button row in `gui/roi/curation.py` (`_draw_labels`) are
  the pieces both packages would import; keep them free of mbo/masknmf
  imports. Third-party imports the package makes today: `cmap` (`model.py`,
  color by), `scipy.ndimage` (`display.py`, the dF/F baseline), `roifile`,
  `tifffile` and `cv2` (`imagej.py`), `zarr` (`ngff.py`, function-local); the GUI draws the
  ROI tool with masknmf's imgui widgets (`TracePlot`, `draw_roi_table`), which
  this package never imports.
