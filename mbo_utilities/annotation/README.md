# mbo_utilities.annotation

The GUI-free half of the manual ROI + labeling tool. `gui/manual_roi.py` is
the imgui/pygfx shell; everything here imports numpy/zarr only.

## Modules

- `events.py` — `Observable` / `ModelEvent`: the event model the rest of the
  package uses, after fastplotlib's `GraphicFeature` (`add_event_handler(fn,
  "rois")`, `block_events` around bulk edits). Handlers run inside the mutation.
- `store.py` — `RoiLabelStore`: `(P, Y, X)` uint16 label volume, one plane per
  combination of the data's scrolling dims (`plane_axes`, z last; depthless data
  gets one plane; T never keys a plane), per-ROI records (`plane`, area, class
  index, free-text note), the user-defined class-label set, and the display
  palettes (`CLASS_COLORS` is the same tab10 set as masknmf's classification GUI,
  so a shared label set looks the same in both tools). The store owns the plane
  arithmetic: `plane_of(pos)`, `plane_pos(plane)`, `plane_label(plane)`,
  `roi_z(i)` / `roi_c(i)`. Mutations emit `rois` events and track `dirty_planes`
  for incremental saving. Each record also carries a persistent `uid` (never
  reused; `next_uid` round-trips through the zarr) and a `source` string (`""` =
  drawn by hand).
- `traces.py` — `RoiTrace` / `RoiTraceTable`: one row per measurement, keyed by
  `(uid, z, c, engine)` for a drawn ROI (re-running replaces the row) or
  `(source, member)` for rows that stand for no drawn ROI. `ENGINES` names the
  extraction engines.
- `model.py` — `RoiModel`: the store, the trace table and the slider position in
  one observable object; `targets(indices, z=, c=)` says where a run reads each
  ROI, `traced(index, ...)` what it has been measured with.
- `ngff.py` — `LabelsZarr`: OME-NGFF-style labels zarr, layout matched to
  `arrays/suite2p.py::_add_suite2p_labels` (`{"version": "0.5", "labels":
  ["0"]}` root attrs, `0` = `(Z, Y, X)` uint32 with `image-label` attrs).
  Per-ROI class/note/plane round-trip through `image-label.properties`;
  colors through `image-label.colors`; the label-name set and source-image
  path through a root `mbo` attr. One-plane chunks make autosave a
  per-stroke plane write. Foreign labels zarrs (no `properties`) load too —
  records are derived from the volume, unclassified.

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
  footprint there) and the persistence substrate (labels zarr here, results
  hdf5 there).
- **when extracting**: `store.py`'s label-set/palette block and the
  imgui label-button row in `gui/manual_roi.py` (`_draw_label_buttons`) are
  the pieces both packages would import; keep them free of mbo/masknmf
  imports.
