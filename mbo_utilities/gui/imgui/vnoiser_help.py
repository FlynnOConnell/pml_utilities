"""The vnoiser guide (h): the voltage pipeline from a scan's lines to curated events.

The steps, what each stage does to a trace, the Voltage window and its domain
table, the curation window and its four rules, and the files, as diagrams and
tables, with the curation window's keybinds behind a button.
``python -m mbo_utilities.gui.imgui.vnoiser_help`` opens it in a window of its own.
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp
from imgui_bundle import icons_fontawesome_6 as fa

from mbo_utilities.gui._theme import close_button, popup
from mbo_utilities.gui.imgui.panels import draw_keybinds_popup
from mbo_utilities.preferences import get_mbo_dirs

ACCENT = imgui.ImVec4(0.40, 0.68, 1.00, 1.0)
KEY = imgui.ImVec4(1.00, 0.80, 0.20, 1.0)
DIM = imgui.ImVec4(0.62, 0.62, 0.65, 1.0)
TEXT = imgui.ImVec4(0.92, 0.92, 0.94, 1.0)
CARD = imgui.ImVec4(0.17, 0.18, 0.21, 1.0)
EDGE = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)
PLOT = imgui.ImVec4(0.06, 0.06, 0.08, 1.0)
# the Run Voltage button's green
RUN = imgui.ImVec4(0.18, 0.65, 0.18, 1.0)
# one trace on its way through the pipeline: counts, z-score, denoised
RAW = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)
ZSCORE = imgui.ImVec4(0.35, 0.65, 0.95, 1.0)
DENOISED = imgui.ImVec4(0.30, 0.85, 0.40, 1.0)
# the curation window's lines: the candidate threshold, the auto-pass peak, the PC1 line, the cosine
THRESHOLD = imgui.ImVec4(0.84, 0.15, 0.24, 1.0)
AUTO_PASS = imgui.ImVec4(0.16, 0.62, 0.56, 1.0)
PC1 = imgui.ImVec4(0.62, 0.45, 0.90, 1.0)
COSINE = imgui.ImVec4(0.95, 0.68, 0.25, 1.0)
SNIPPET = imgui.ImVec4(0.30, 0.47, 0.66, 1.0)
# a candidate's colour by its label, as vnoiser's curation draws them
LABELS = {
    "yes": imgui.ImVec4(0.106, 0.600, 0.545, 1.0),
    "no": imgui.ImVec4(0.843, 0.149, 0.239, 1.0),
    "auto_yes": imgui.ImVec4(0.569, 0.843, 0.788, 1.0),
    "auto_no": imgui.ImVec4(0.953, 0.627, 0.667, 1.0),
    "unlabeled": imgui.ImVec4(0.322, 0.345, 0.400, 1.0),
}
# the domains of the made-up scan, in the trace plot's first group colours
GROUP = (
    imgui.ImVec4(1.00, 0.55, 0.10, 1.0),
    imgui.ImVec4(0.25, 0.85, 0.35, 1.0),
    imgui.ImVec4(0.95, 0.35, 0.90, 1.0),
)
WIDTH_EM = 44
WINDOW_SIZE = (760, 1000)

TITLE = "vnoiser"
TOOLTIP = (
    "the steps, what each stage does to a trace, the Voltage window and its domains, the curation "
    "window and its rules, the files, the keybinds"
)
# the curation window's keys; its handler and its popup read this table
KEYBINDS = (
    ("y", "label the focused candidate yes"),
    ("n", "label the focused candidate no"),
    ("backspace", "clear its label"),
    ("← / →", "previous / next candidate in view"),
    ("↑ / ↓", "previous / next recording"),
    ("click", "focus a candidate on the trace or the PCA plot"),
    (
        "Box accept / reject",
        "box mode: right-drag a box on the trace or the PCA, then drag its edges",
    ),
    ("enter", "apply the box"),
    ("esc", "leave box mode"),
    (
        "drag line",
        "move the threshold (red) / auto-pass (teal) line on the trace, or the PC1 (purple) line on the PCA",
    ),
    ("scroll", "zoom (shift: x only, alt: y only); drag pans; double-click fits"),
    ("h", "the vnoiser guide"),
    ("k", "this list"),
)
STEPS = (
    (fa.ICON_FA_WAVE_SQUARE, "Scan", "a .mesc scan: its lines or patches"),
    (fa.ICON_FA_OBJECT_GROUP, "Domains", "group ROIs: a soma, a branch"),
    (fa.ICON_FA_PLAY, "Run", "one ROI and channel, or all"),
    (fa.ICON_FA_CHECK_DOUBLE, "Curate", "yes / no every candidate"),
    (fa.ICON_FA_FLOPPY_DISK, "Results", "one .voltage.zarr per run"),
)
# the stages in order, each in the colour of the trace it leaves
FLOW = (
    ("ROI means", RAW),
    ("domain", RAW),
    ("dF/F", RAW),
    ("z-score", ZSCORE),
    ("wavelets", ACCENT),
    ("mask", ACCENT),
    ("denoised", DENOISED),
    ("peaks", KEY),
)
STAGES = (
    ("stage", "does"),
    (
        "ROI means",
        "each line's or patch's mean brightness per frame, in one channel: the counts as MESc stored them",
        (
            "one ROI is one line of a line scan, or one patch of a chessboard or ribbon scan",
            "Apply the file's conversion offset (Pipeline Settings) makes zero mean no photons",
        ),
    ),
    (
        "domain",
        "the ROIs of one domain averaged into one trace, each weighted by its pixel count",
    ),
    (
        "dF/F",
        "the change over a slow baseline, a Gaussian 1500 samples wide (1.4 s at 1075 Hz); the first "
        "1000 samples, where that filter starts up, are replaced by the mean of the rest",
    ),
    (
        "z-score",
        "a slower baseline (5000 samples) taken off, divided by the trace's SD, and the sign flipped: "
        "JEDI darkens when the cell depolarises, so a spike now points up",
    ),
    (
        "wavelets",
        "the trace split by time scale: 100 scales, log-spaced from 1 to 1000 samples, grouped into 10 "
        "frequency bands",
        (
            "a continuous wavelet transform with a complex Morlet wavelet",
            "the scales reduced to 30 principal components, the first 10 clustered into the bands",
            "the slowest stage: about 4 s for a five-minute scan at 1587 Hz",
        ),
    ),
    (
        "mask",
        "per band, an event window opens where the band rises above 2.0 SD (bands under 30 Hz) or 2.5 SD "
        "(30 Hz and up); outside the windows a band is turned down to 0.7, 0.5, 0.2 and 0.01 of itself "
        "(under 5, 5-30, 30-80, above 80 Hz)",
    ),
    (
        "denoised",
        "the masked bands summed: the trace the curation window shows; the 1 Hz baseline is saved beside "
        "it, not added",
    ),
    (
        "peaks",
        "local maxima of the trace band-passed 2-400 Hz, above its mean + 3.5 SD there and + 4.0 SD on the "
        "trace itself, on a stretch of 5 ms or more, 3 samples apart",
    ),
    (
        "frame rate",
        "the settings are written for 1075 Hz line scans; everything counted in samples is scaled to the "
        "scan's own rate, and the band-pass is capped below half of it",
    ),
)
WINDOW = (
    ("block", "holds"),
    (
        "Voltage on MUnit_n",
        "the MESc tab's button: opens this window set to the scan on screen and to the ROI and channel its "
        "sliders are on",
    ),
    ("Current dataset", "the file, the scan on screen and its frame rate"),
    (
        "Output folder",
        "where the results file goes: beside the .mesc unless you change it",
    ),
    (
        "Set slice",
        "frames (one unbroken window), ROIs and one channel: only the chosen ROIs are read, and every "
        "domain is cut down to them",
    ),
    (
        "Scans",
        "the file's recordings that have lines or patches; the ticked ones run together with the same "
        "domains, one frame rate per run",
        (
            "the scan on screen is ticked to start",
            "On screen only and All set the ticks",
            "New environment: the first scan after the animal changed arena or track; a note for later "
            "analysis, the pipeline does not use it",
        ),
    ),
    ("Domains", "which ROIs make each trace (below)"),
    (
        "Pipeline Settings",
        "every number of the stages above; a changed one turns orange and is listed under Modified "
        "parameters, Defaults puts them back",
    ),
    (
        "Run Voltage",
        "starts the run in the background; the process console follows it, every ROI read and every domain "
        "denoised",
    ),
    (
        "Load into Traces",
        "the finished run's denoised and line traces in the Traces tab, over the recording",
    ),
    (
        "Curate",
        "the curation window on the finished run; before any run, on the file's lines as recorded, each "
        "denoised when you click it",
    ),
)
DOMAIN_RULES = (
    ("domains", "rule"),
    (
        "a domain",
        "a named group of ROIs averaged into one trace: the lines over a soma, a branch, one cell's patch",
    ),
    (
        "ROIs",
        "0-based, in the order they were drawn: 0,1,2 or 0:2 (both ends included)",
    ),
    ("to start", "one domain per ROI of the scan on screen: roi0, roi1, ..."),
    (
        "another scan",
        "a table you loaded or edited stays while it names only ROIs the new scan has; else it starts over",
    ),
    (
        "ROI on screen",
        "opened from the button, a table that leaves that ROI out gets it as a domain of its own, so the "
        "run is never empty",
    ),
    (
        "previous run",
        "its table comes back when the scan on screen has as many ROIs as the scans that run processed",
    ),
    ("Add, x", "a new row; remove a row"),
    (
        "Load, Save",
        "domains.json beside the file: the domains, the ticked scans and the new-environment ticks; "
        "mbo voltage reads the same file",
    ),
)
PANELS = (
    ("panel", "shows"),
    (
        "A. trace",
        "the denoised trace of one domain of one scan, each candidate a dot in its label's colour; click "
        "one to focus it, the arrows above flip to the next trace",
        (
            "the red line is A1, the teal line A2: drag either",
            "MC puts the scan's motion correction over the trace, on the same time axis",
            "scroll zooms (shift: time only, alt: height only), drag pans, double-click fits",
        ),
    ),
    (
        "A1 - A4",
        "the four rules as sliders, each with how many candidates it passes; they apply on release",
    ),
    (
        "B. template",
        "the average shape of the events it is built from, their snippets faint behind it",
    ),
    (
        "C. candidate",
        "the focused candidate over the template; the title carries how alike they are (cosine, 1 = the "
        "same shape)",
    ),
    (
        "D. PCA",
        "every candidate's 400 ms snippet as one dot on its two main axes of variation: alike shapes sit "
        "together; the purple line is A3",
    ),
    (
        "Decision",
        "Yes (y), No (n), Clear (backspace); Box accept and Box reject label everything inside a box "
        "right-dragged on the trace or the PCA; Clear all",
    ),
    (
        "Recordings",
        "the mode, the data path and every scan / domain with its candidate and yes / no counts: click a "
        "row to load it",
    ),
)
RULES = (
    ("rule", "decides"),
    (
        "A1 thr",
        "what a candidate is: every local maximum of the trace above the red line",
    ),
    (
        "A2 peak",
        "a candidate peaking at or above the teal line passes whatever its shape",
    ),
    (
        "A3 PC1",
        "every candidate on the passing side of the purple line on the PCA passes; the arrow flips the side",
    ),
    (
        "A4 cos",
        "likeness to the seed template: at or above it a candidate passes, below it is rejected; at the "
        "bottom nothing is rejected",
    ),
    (
        "your label",
        "a Yes or No always wins over the rules; Clear hands the candidate back to them",
    ),
    ("saved", "labels and sliders, per trace and per mode, as you go"),
)
MODES = (
    ("mode", "candidates and template"),
    (
        "fast",
        "candidates from the denoised trace; the seed template is the quarter of them with the largest "
        "amplitude",
    ),
    (
        "slow",
        "candidates from its low-pass view (under 40 Hz, editable), seeded the same way",
    ),
    (
        "manual",
        "no seed template and no rules: only your Yes events build the template",
    ),
)
FILES = (
    ("in the run", "holds"),
    ("scan<n>/traces", "denoised, dff and zscore, one row per domain"),
    ("scan<n>/members", "raw: each line's or patch's own mean counts"),
    ("scan<n>/events", "the detected peaks, per domain"),
    (
        "voltage/",
        "the run's own record",
        (
            "pipeline.json: the source file and scans, the channel, frames and ROIs, the domain table, "
            "every parameter, versions",
            "timings.json: seconds, CPU and memory of every step",
            "traces/: the same traces as .npy and .csv, and two figures per scan",
            "test.h5: dF/F and z per domain",
        ),
    ),
    (
        ".curation/",
        "<mode>_template_curation.json: your labels and sliders, keyed scan=<n>/domain=<name>",
    ),
    (
        "pkl",
        "Output format pkl writes the older PF folder of pickles instead; the curation window opens either",
    ),
    (
        "terminal",
        "mbo voltage file.mesc runs it, mbo curate file.mesc opens the curation window",
    ),
)

# the sketches' made-up recording: six spikes as (time, height) over SPAN time steps
SPAN = 60.0
SPIKES = ((7.0, 1.0), (15.0, 0.6), (19.5, 0.9), (31.0, 0.35), (44.0, 1.0), (52.5, 0.7))
SAMPLES = 240
N_BANDS = 10
COLS = 120
# a band's event window opens above this; outside it the band keeps this fraction, slow bands first
OPEN = 0.3
SOFT = (0.7, 0.7, 0.5, 0.5, 0.5, 0.2, 0.2, 0.01, 0.01, 0.01)


def hash01(a: float, b: float) -> float:
    """A deterministic value in [0, 1) for the pair: the same noise every run."""
    return math.sin(12.9898 * a + 78.233 * b) * 43758.5453 % 1.0


def bumps(t: float, width: float) -> float:
    """The spikes at time t, each a bump ``width`` time steps wide."""
    return sum(h * math.exp(-(((t - s) / width) ** 2)) for s, h in SPIKES)


TIMES = tuple(SPAN * i / SAMPLES for i in range(SAMPLES + 1))
# counts dip at a spike and bleach; the z-score is flat and flipped; the denoised trace keeps the spikes
TRACES = {
    "raw": tuple(
        0.82 - 0.004 * t - 0.5 * bumps(t, 0.5) + 0.14 * (hash01(t, 1.0) - 0.5)
        for t in TIMES
    ),
    "z": tuple(
        min(1.0, 0.14 + 0.72 * bumps(t, 0.5) + 0.2 * (hash01(t, 1.0) - 0.5))
        for t in TIMES
    ),
    "denoised": tuple(0.08 + 0.8 * bumps(t, 0.5) for t in TIMES),
}
# each band's response to the spikes, slow (wide) bands first, then with the noise the fast bands carry
RESPONSE = tuple(
    tuple(bumps(SPAN * c / COLS, 4.0 * 0.79**b) for c in range(COLS))
    for b in range(N_BANDS)
)
BANDS = tuple(
    tuple(min(1.0, r + (0.06 + 0.05 * b) * hash01(c, b)) for c, r in enumerate(row))
    for b, row in enumerate(RESPONSE)
)
MASKED = tuple(
    tuple(v if r > OPEN else v * SOFT[b] for v, r in zip(BANDS[b], row, strict=True))
    for b, row in enumerate(RESPONSE)
)
# each band's event windows as (first column, one past the last)
WINDOWS = tuple(
    tuple(
        zip(
            [
                c
                for c in range(COLS)
                if row[c] > OPEN and (c == 0 or row[c - 1] <= OPEN)
            ],
            [
                c + 1
                for c in range(COLS)
                if row[c] > OPEN and (c == COLS - 1 or row[c + 1] <= OPEN)
            ],
            strict=True,
        )
    )
    for row in RESPONSE
)

# the Voltage window sketch: the file's scans as (unit, length, ticked), the domain table as
# (name, text, ROIs) and the seven lines on their picture as (x0, y0, x1, y1) fractions of the card
SCANS = (
    ("MUnit_30 (on screen)", "300 s", True),
    ("MUnit_32", "10 s", False),
    ("MUnit_27", "30 s", False),
)
DOMAINS = (
    ("soma", "0,1,2", (0, 1, 2)),
    ("dend", "3:5", (3, 4, 5)),
    ("roi6", "6", (6,)),
)
LINES = (
    (0.10, 0.62, 0.24, 0.50),
    (0.16, 0.76, 0.31, 0.66),
    (0.26, 0.86, 0.40, 0.80),
    (0.44, 0.46, 0.58, 0.34),
    (0.60, 0.32, 0.72, 0.20),
    (0.74, 0.20, 0.88, 0.14),
    (0.70, 0.74, 0.86, 0.66),
)
OWNER = tuple(
    next(i for i, (_name, _text, rois) in enumerate(DOMAINS) if k in rois)
    for k in range(len(LINES))
)

# the curation sketch: each spike is a candidate with a label, a likeness to the template, a spot on
# the PCA and a snippet as wide as given; the lines sit at these heights of the trace
CALLS = ("yes", "auto_yes", "yes", "auto_no", "auto_yes", "no")
COSINES = (0.97, 0.88, 0.95, 0.41, 0.98, 0.52)
PCA = (
    (0.78, 0.60),
    (0.55, 0.30),
    (0.70, 0.42),
    (0.18, 0.70),
    (0.84, 0.35),
    (0.30, 0.25),
)
SNIPPET_WIDTHS = (0.20, 0.24, 0.21, 0.60, 0.19, 0.50)
THRESHOLD_AT = 0.2
AUTO_PASS_AT = 0.74
PC1_AT = 0.42
SNIPPETS = tuple(
    tuple(0.1 + 0.8 * h * math.exp(-(((u - 20) / 20 / width) ** 2)) for u in range(41))
    for (_s, h), width in zip(SPIKES, SNIPPET_WIDTHS, strict=True)
)
# the template: the mean snippet of the events it is built from
TEMPLATE_SOURCE = (0, 1, 2, 4)
TEMPLATE = tuple(np.mean([SNIPPETS[k] for k in TEMPLATE_SOURCE], axis=0).tolist())
# the A1 to A4 sliders as (name, colour, where the grab sits, the count under it)
SLIDERS = (
    ("thr", THRESHOLD, THRESHOLD_AT, "6 found"),
    ("peak", AUTO_PASS, AUTO_PASS_AT, "3/6"),
    ("PC1", PC1, PC1_AT, "4/6"),
    ("cos", COSINE, 0.6, "4/6"),
)


def u32(color: imgui.ImVec4, alpha: float = 1.0) -> int:
    return imgui.color_convert_float4_to_u32(
        imgui.ImVec4(color.x, color.y, color.z, alpha)
    )


def box(
    dl,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    color: imgui.ImVec4,
    fill_alpha: float = 0.0,
) -> None:
    """A rounded box with its label centred; fill_alpha tints it with the label colour over the card."""
    a, b = imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h)
    dl.add_rect_filled(a, b, u32(CARD), 4.0)
    if fill_alpha:
        dl.add_rect_filled(a, b, u32(color, fill_alpha), 4.0)
    dl.add_rect(a, b, u32(color, 0.6), 4.0)
    size = imgui.calc_text_size(label)
    dl.add_text(
        imgui.ImVec2(x + (w - size.x) / 2, y + (h - size.y) / 2), u32(color), label
    )


def arrow(dl, x0: float, x1: float, y: float) -> None:
    col = u32(DIM)
    dl.add_line(imgui.ImVec2(x0, y), imgui.ImVec2(x1 - 5, y), col, 1.5)
    dl.add_triangle_filled(
        imgui.ImVec2(x1, y),
        imgui.ImVec2(x1 - 7, y - 4),
        imgui.ImVec2(x1 - 7, y + 4),
        col,
    )


def noted_arrow(
    dl, x: float, y: float, w: float, h: float, over: str, under: str
) -> None:
    """An arrow across a gap between two boxes of height h, what happens over it and a note under it."""
    em = imgui.get_font_size()
    arrow(dl, x + 0.4 * em, x + w - 0.4 * em, y + h / 2)
    for text, color, dy in ((over, ACCENT, -1.3 * em), (under, DIM, 0.3 * em)):
        size = imgui.calc_text_size(text)
        dl.add_text(
            imgui.ImVec2(x + (w - size.x) / 2, y + h / 2 + dy), u32(color), text
        )


def polyline(
    dl, x: float, y: float, w: float, h: float, values, color: int, weight: float = 1.5
) -> None:
    """Values in 0..1, 1 at the top, as one line across the rectangle."""
    step = w / (len(values) - 1)
    points = [
        imgui.ImVec2(x + i * step, y + h * (1.0 - v)) for i, v in enumerate(values)
    ]
    dl.add_polyline(points, color, weight, 0)


def card(
    dl,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    tint: imgui.ImVec4 = PLOT,
) -> float:
    """A panel's card: its title in a strip on top, the plot area below in tint. Returns the area's top."""
    em = imgui.get_font_size()
    a, b = imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h)
    dl.add_rect_filled(a, b, u32(CARD), 4.0)
    area = imgui.ImVec2(x, y + 1.2 * em)
    dl.add_rect_filled(area, b, u32(tint), 4.0, imgui.ImDrawFlags_.round_corners_bottom)
    dl.add_rect(a, b, u32(EDGE), 4.0)
    size = imgui.calc_text_size(title)
    dl.add_text(
        imgui.ImVec2(x + (w - size.x) / 2, y + (1.2 * em - size.y) / 2),
        u32(TEXT),
        title,
    )
    return y + 1.2 * em


def heading(icon: str, text: str) -> None:
    em = imgui.get_font_size()
    imgui.dummy(imgui.ImVec2(0, 0.7 * em))
    imgui.text_colored(ACCENT, f"{icon}  {text}")
    imgui.dummy(imgui.ImVec2(0, 0.1 * em))


def table(name: str, rows: tuple) -> None:
    """Two columns: the first row dim as the header, the rest a name and its wrapped meaning; a (?)
    after the name lists a row's third element as bullets in its tooltip.
    """
    em = imgui.get_font_size()
    flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
    if not imgui.begin_table(f"##{name}", 2, flags, imgui.ImVec2(WIDTH_EM * em, 0)):
        return
    imgui.table_setup_column("name", imgui.TableColumnFlags_.width_fixed, 11 * em)
    imgui.table_setup_column("meaning")
    for i, row in enumerate(rows):
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(DIM if i == 0 else KEY, row[0])
        if len(row) == 3:
            imgui.same_line(0, 0.4 * em)
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                for line in row[2]:
                    imgui.bullet_text(line)
                imgui.end_tooltip()
        imgui.table_next_column()
        if i == 0:
            imgui.text_colored(DIM, row[1])
        else:
            imgui.text_wrapped(row[1])
    imgui.end_table()


def draw_vnoiser_help(is_open: bool, keys_open: bool) -> tuple[bool, bool]:
    """The page as a centred window with its keybinds button, whose popup the caller draws; returns
    (page open, popup open).
    """
    if not is_open:
        return False, keys_open
    em = imgui.get_font_size()
    w = WIDTH_EM * em
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_size_constraints(
        imgui.ImVec2(0, 0), imgui.ImVec2(viewport.size.x, 0.94 * viewport.size.y)
    )
    # the sketches are drawn for a dark backdrop: a host with see-through windows would show through them
    imgui.set_next_window_bg_alpha(1.0)
    opened, is_open = popup(f"{TITLE} guide", is_open)
    if not opened:
        imgui.end()
        return is_open, keys_open
    dl = imgui.get_window_draw_list()
    # the cursor sweeps the sketches every 7.5 s; the ROI on screen and the focused candidate step every 1.5 s
    now = imgui.get_time()
    cursor = now * 8.0 % SPAN
    roi = int(now / 1.5) % len(LINES)
    focus = int(now / 1.5) % len(SPIKES)
    imgui.push_text_wrap_pos(w)
    imgui.text_colored(ACCENT, f"{fa.ICON_FA_CIRCLE_QUESTION}  {TITLE}")
    keys = f"{fa.ICON_FA_KEYBOARD}  keybinds"
    imgui.same_line(
        w - imgui.calc_text_size(keys).x - 2 * imgui.get_style().frame_padding.x
    )
    if imgui.small_button(keys):
        keys_open = not keys_open
    imgui.separator()
    imgui.dummy(imgui.ImVec2(0, 0.5 * em))

    gap = 1.4 * em
    card_w = (w - 4 * gap) / 5
    card_h = 6.8 * em
    imgui.push_style_color(imgui.Col_.child_bg, CARD)
    imgui.push_style_color(imgui.Col_.border, EDGE)
    imgui.push_style_var(imgui.StyleVar_.child_rounding, 5.0)
    imgui.push_style_var(
        imgui.StyleVar_.window_padding, imgui.ImVec2(0.5 * em, 0.6 * em)
    )
    for i, (icon, name, hint) in enumerate(STEPS):
        if i:
            right = imgui.get_item_rect_max()
            mid = imgui.get_item_rect_min().y + card_h / 2
            arrow(dl, right.x + 0.25 * em, right.x + gap - 0.25 * em, mid)
            imgui.same_line(0, gap)
        imgui.begin_child(
            f"##step{i}",
            imgui.ImVec2(card_w, card_h),
            child_flags=imgui.ChildFlags_.borders,
            window_flags=imgui.WindowFlags_.no_scrollbar,
        )
        imgui.push_font(None, 1.8 * em)
        imgui.set_cursor_pos_x((card_w - imgui.calc_text_size(icon).x) / 2)
        imgui.text_colored(ACCENT, icon)
        imgui.pop_font()
        imgui.dummy(imgui.ImVec2(0, 0.2 * em))
        imgui.set_cursor_pos_x(
            (card_w - imgui.calc_text_size(f"{i + 1}  {name}").x) / 2
        )
        imgui.text_colored(DIM, f"{i + 1}")
        imgui.same_line(0, 0.5 * em)
        imgui.text_colored(KEY, name)
        imgui.push_text_wrap_pos(card_w - 0.5 * em)
        imgui.text_colored(DIM, hint)
        imgui.pop_text_wrap_pos()
        imgui.end_child()
    imgui.pop_style_var(2)
    imgui.pop_style_color(2)

    heading(fa.ICON_FA_DIAGRAM_PROJECT, "The pipeline: one domain, stage by stage")
    p = imgui.get_cursor_screen_pos()
    bh = 1.9 * em
    gap = 1.0 * em
    bw = (w - (len(FLOW) - 1) * gap) / len(FLOW)
    for i, (label, color) in enumerate(FLOW):
        x = p.x + i * (bw + gap)
        if i:
            arrow(dl, x - gap + 0.15 * em, x - 0.15 * em, p.y + bh / 2)
        box(dl, x, p.y, bw, bh, label, color, 0.18)
    imgui.dummy(imgui.ImVec2(w, bh))
    imgui.dummy(imgui.ImVec2(0, 0.1 * em))
    # the trace after three of the stages, named in a gutter on the left, the detected peaks as dots
    p = imgui.get_cursor_screen_pos()
    ph = 8.4 * em
    top = card(dl, p.x, p.y, w, 1.2 * em + ph, "a made-up domain's trace")
    gutter = 5.5 * em
    x0, pw = p.x + gutter, w - gutter - 0.5 * em
    slot = ph / 3
    for i, (name, kind, color) in enumerate(
        (
            ("counts", "raw", RAW),
            ("z-score", "z", ZSCORE),
            ("denoised", "denoised", DENOISED),
        )
    ):
        y = top + i * slot
        dl.add_text(imgui.ImVec2(p.x + 0.6 * em, y + (slot - em) / 2), u32(color), name)
        polyline(dl, x0, y + 0.12 * slot, pw, 0.76 * slot, TRACES[kind], u32(color))
    y = top + 2 * slot + 0.12 * slot
    for s, h in SPIKES:
        # the weakest spike is under the peak detector's height; the curation window still offers it
        if h > 0.5:
            at = imgui.ImVec2(
                x0 + s / SPAN * pw, y + 0.76 * slot * (1.0 - (0.08 + 0.8 * h))
            )
            dl.add_circle_filled(at, 0.28 * em, u32(KEY))
    cx = x0 + cursor / SPAN * pw
    dl.add_line(
        imgui.ImVec2(cx, top + 0.3 * em),
        imgui.ImVec2(cx, top + ph - 0.3 * em),
        u32(TEXT, 0.7),
        1.5,
    )
    imgui.dummy(imgui.ImVec2(w, 1.2 * em + ph))
    imgui.dummy(imgui.ImVec2(0, 0.1 * em))
    # the bands before and after the mask, the event windows outlined on the second
    p = imgui.get_cursor_screen_pos()
    gap = 0.4 * em
    cw, ih = (w - gap) / 2, 6.0 * em
    heat = u32(ACCENT) & 0xFFFFFF
    cell_w, cell_h = cw / COLS, ih / N_BANDS
    for j, (title, grid) in enumerate(
        (
            ("wavelet bands: slow at the bottom, fast on top", BANDS),
            ("after the mask: kept in the event windows", MASKED),
        )
    ):
        x = p.x + j * (cw + gap)
        top = card(dl, x, p.y, cw, 1.2 * em + ih, title)
        for b, row in enumerate(grid):
            y = top + ih - (b + 1) * cell_h
            for c, v in enumerate(row):
                if v > 0.03:
                    dl.add_rect_filled(
                        imgui.ImVec2(x + c * cell_w, y),
                        imgui.ImVec2(x + (c + 1) * cell_w, y + cell_h),
                        heat | int(255 * v) << 24,
                    )
            for c0, c1 in WINDOWS[b] if j else ():
                dl.add_rect(
                    imgui.ImVec2(x + c0 * cell_w, y),
                    imgui.ImVec2(x + c1 * cell_w, y + cell_h),
                    u32(KEY, 0.8),
                )
        cx = x + cursor / SPAN * cw
        dl.add_line(
            imgui.ImVec2(cx, top), imgui.ImVec2(cx, top + ih), u32(TEXT, 0.7), 1.5
        )
    imgui.dummy(imgui.ImVec2(w, 1.2 * em + ih))
    imgui.text_colored(
        DIM,
        "a spike is in every band at once, noise mostly in the fast ones: inside an event window (yellow) a "
        "band is kept whole, outside it is turned down, the fast bands almost to nothing",
    )
    table("stages", STAGES)

    heading(fa.ICON_FA_WINDOW_RESTORE, "The Voltage window")
    held = OWNER[roi]
    p = imgui.get_cursor_screen_pos()
    ch = 7.2 * em
    sw, dw = 17.0 * em, 10.5 * em
    lw = w - sw - dw - 2 * gap
    row_h = 1.3 * em
    top = card(dl, p.x, p.y, sw, ch, "Scans")
    for text, dx in (("Unit", 0.5), ("ROIs", 11.9), ("Length", 14.0)):
        dl.add_text(imgui.ImVec2(p.x + dx * em, top + 0.3 * em), u32(DIM), text)
    for i, (unit, length, ticked) in enumerate(SCANS):
        ry = top + 0.3 * em + (i + 1) * row_h
        a = imgui.ImVec2(p.x + 0.5 * em, ry + 0.05 * em)
        dl.add_rect(a, imgui.ImVec2(a.x + 0.95 * em, a.y + 0.95 * em), u32(DIM), 2.0)
        if ticked:
            dl.add_rect_filled(
                imgui.ImVec2(a.x + 0.2 * em, a.y + 0.2 * em),
                imgui.ImVec2(a.x + 0.75 * em, a.y + 0.75 * em),
                u32(ACCENT),
                1.0,
            )
        for text, dx in ((unit, 1.9), (str(len(LINES)), 11.9), (length, 14.0)):
            dl.add_text(
                imgui.ImVec2(p.x + dx * em, ry), u32(TEXT if ticked else DIM), text
            )
    x = p.x + sw + gap
    top = card(dl, x, p.y, dw, ch, "Domains")
    for text, dx in (("Domain", 0.5), ("ROIs", 6.4)):
        dl.add_text(imgui.ImVec2(x + dx * em, top + 0.3 * em), u32(DIM), text)
    for i, (name, text, _rois) in enumerate(DOMAINS):
        ry = top + 0.3 * em + (i + 1) * row_h
        if i == held:
            dl.add_rect_filled(
                imgui.ImVec2(x + 1, ry - 0.1 * em),
                imgui.ImVec2(x + dw - 1, ry + row_h - 0.15 * em),
                u32(GROUP[i], 0.22),
            )
        dl.add_rect_filled(
            imgui.ImVec2(x + 0.5 * em, ry + 0.2 * em),
            imgui.ImVec2(x + 1.2 * em, ry + 0.9 * em),
            u32(GROUP[i]),
            2.0,
        )
        dl.add_text(imgui.ImVec2(x + 1.7 * em, ry), u32(TEXT), name)
        dl.add_text(imgui.ImVec2(x + 6.4 * em, ry), u32(TEXT), text)
    # the scan's lines where they were drawn, the one the ROI slider is on lit
    x += dw + gap
    top = card(dl, x, p.y, lw, ch, f"MUnit_30    ROI {roi + 1} / {len(LINES)}    ch 1")
    ih = ch - 1.2 * em
    dl.add_rect_filled_multi_color(
        imgui.ImVec2(x, top),
        imgui.ImVec2(x + lw, top + ih),
        u32(TEXT, 0.04),
        u32(TEXT, 0.20),
        u32(TEXT, 0.12),
        u32(TEXT, 0.03),
    )
    for k, (u0, v0, u1, v1) in enumerate(LINES):
        a = imgui.ImVec2(x + u0 * lw, top + v0 * ih)
        b = imgui.ImVec2(x + u1 * lw, top + v1 * ih)
        if k == roi:
            dl.add_line(a, b, u32(TEXT), 0.5 * em)
            dl.add_line(a, b, u32(GROUP[OWNER[k]]), 0.26 * em)
        else:
            dl.add_line(a, b, u32(GROUP[OWNER[k]], 0.55), 2.0)
    # the slice line and the run under them: the domain holding the ROI, cut down to it
    y = p.y + ch + gap
    run_w = 9.0 * em
    strip = (
        f"Set slice     47620 frames  ·  1 / {len(LINES)} ROIs  ·  ch 1          "
        f"runs   {DOMAINS[held][0]}  [{roi}]"
    )
    box(dl, p.x, y, w - run_w - gap, 1.6 * em, strip, DIM)
    box(dl, p.x + w - run_w, y, run_w, 1.6 * em, "Run Voltage", RUN, 0.3)
    imgui.dummy(imgui.ImVec2(w, ch + gap + 1.6 * em))
    imgui.text_colored(
        DIM,
        "opened from the MESc tab's Voltage on MUnit_30 button, the window is set to the scan on screen and "
        "the ROI and channel its sliders are on: only that ROI is read, and the domain holding it is cut "
        "down to it. Set slice back to every ROI and the whole table runs.",
    )
    table("window", WINDOW)

    heading(fa.ICON_FA_OBJECT_GROUP, "Domains")
    table("domains", DOMAIN_RULES)

    heading(fa.ICON_FA_CHECK_DOUBLE, "The curation window")
    # the trace row: A with its two lines, the candidates in their label colours, the sliders beside it
    p = imgui.get_cursor_screen_pos()
    col = 2.7 * em
    sliders_w = len(SLIDERS) * col + 0.8 * em
    aw, ah = w - sliders_w - gap, 7.0 * em
    n = len(SPIKES)
    top = card(
        dl, p.x, p.y, aw, 1.2 * em + ah, f"A. fast candidates: {n} ({n} in view)"
    )
    x0, pw = p.x + 0.5 * em, aw - em
    y0, plot_h = top + 0.9 * em, ah - 1.3 * em
    polyline(dl, x0, y0, pw, plot_h, TRACES["denoised"], u32(RAW), 1.2)
    for level, color in ((THRESHOLD_AT, THRESHOLD), (AUTO_PASS_AT, AUTO_PASS)):
        y = y0 + plot_h * (1.0 - level)
        dl.add_line(imgui.ImVec2(x0, y), imgui.ImVec2(x0 + pw, y), u32(color), 1.5)
    fx = x0 + SPIKES[focus][0] / SPAN * pw
    dl.add_line(
        imgui.ImVec2(fx, y0), imgui.ImVec2(fx, y0 + plot_h), u32(THRESHOLD, 0.8), 1.0
    )
    for k, ((s, h), call) in enumerate(zip(SPIKES, CALLS, strict=True)):
        at = imgui.ImVec2(x0 + s / SPAN * pw, y0 + plot_h * (1.0 - (0.08 + 0.8 * h)))
        dl.add_circle_filled(at, 0.34 * em, u32(LABELS[call]))
        if k == focus:
            dl.add_circle(at, 0.58 * em, u32(TEXT), 0, 1.5)
    sx = p.x + aw + gap
    top = card(dl, sx, p.y, sliders_w, 1.2 * em + ah, "A1 - A4")
    t0, t1 = top + 0.5 * em, top + ah - 2.9 * em
    for i, (name, color, at, count) in enumerate(SLIDERS):
        cx = sx + 0.4 * em + (i + 0.5) * col
        dl.add_rect_filled(
            imgui.ImVec2(cx - 0.35 * em, t0),
            imgui.ImVec2(cx + 0.35 * em, t1),
            u32(TEXT, 0.08),
            3.0,
        )
        gy = t1 - at * (t1 - t0)
        dl.add_rect_filled(
            imgui.ImVec2(cx - 0.65 * em, gy - 0.3 * em),
            imgui.ImVec2(cx + 0.65 * em, gy + 0.3 * em),
            u32(color),
            2.0,
        )
        for j, (text, tint) in enumerate(((name, color), (count, DIM))):
            size = imgui.calc_text_size(text)
            dl.add_text(
                imgui.ImVec2(cx - size.x / 2, t1 + 0.3 * em + j * 1.2 * em),
                u32(tint),
                text,
            )
    # the candidate row: the template, the focused candidate over it, every candidate on the PCA
    y = p.y + 1.2 * em + ah + gap
    bw, bh = (w - 2 * gap) / 3, 6.2 * em
    top = card(dl, p.x, y, bw, 1.2 * em + bh, "B. Current template")
    for k in TEMPLATE_SOURCE:
        polyline(
            dl,
            p.x + 0.5 * em,
            top + 0.4 * em,
            bw - em,
            bh - 0.8 * em,
            SNIPPETS[k],
            u32(SNIPPET, 0.5),
            1.0,
        )
    polyline(
        dl,
        p.x + 0.5 * em,
        top + 0.4 * em,
        bw - em,
        bh - 0.8 * em,
        TEMPLATE,
        u32(TEXT),
        2.5,
    )
    x = p.x + bw + gap
    top = card(
        dl,
        x,
        y,
        bw,
        1.2 * em + bh,
        f"C. Focused candidate (cosine {COSINES[focus]:.2f})",
    )
    polyline(
        dl,
        x + 0.5 * em,
        top + 0.4 * em,
        bw - em,
        bh - 0.8 * em,
        SNIPPETS[focus],
        u32(SNIPPET),
        1.5,
    )
    polyline(
        dl,
        x + 0.5 * em,
        top + 0.4 * em,
        bw - em,
        bh - 0.8 * em,
        TEMPLATE,
        u32(TEXT),
        2.0,
    )
    for dash in range(0, int(bh - 0.8 * em), 6):
        dl.add_line(
            imgui.ImVec2(x + bw / 2, top + 0.4 * em + dash),
            imgui.ImVec2(x + bw / 2, top + 0.4 * em + dash + 3),
            u32(DIM),
            1.0,
        )
    x += bw + gap
    top = card(dl, x, y, bw, 1.2 * em + bh, "D. Candidate PCA (400 ms)")
    lx = x + PC1_AT * bw
    dl.add_line(imgui.ImVec2(lx, top), imgui.ImVec2(lx, top + bh), u32(PC1), 1.5)
    dl.add_text(imgui.ImVec2(lx + 0.4 * em, top + 0.2 * em), u32(PC1), "pass >")
    for k, ((u, v), call) in enumerate(zip(PCA, CALLS, strict=True)):
        at = imgui.ImVec2(x + u * bw, top + v * bh)
        dl.add_circle_filled(at, 0.34 * em, u32(LABELS[call]))
        if k == focus:
            dl.add_circle(at, 0.58 * em, u32(TEXT), 0, 1.5)
    imgui.dummy(imgui.ImVec2(w, y + 1.2 * em + bh - p.y))
    # the label colours, as the dots carry them
    p = imgui.get_cursor_screen_pos()
    x = p.x
    for name, color in LABELS.items():
        text = name.replace("_", " ")
        dl.add_circle_filled(
            imgui.ImVec2(x + 0.45 * em, p.y + 0.6 * em), 0.36 * em, u32(color)
        )
        dl.add_text(imgui.ImVec2(x + 1.2 * em, p.y), u32(TEXT), text)
        x += 1.2 * em + imgui.calc_text_size(text).x + 1.6 * em
    imgui.dummy(imgui.ImVec2(w, 1.3 * em))
    imgui.text_colored(
        DIM,
        "Curate opens it in a window of its own (mbo curate): one trace at a time, every candidate labelled "
        "yes or no by you or, pale, by the rules; the ring is the focused candidate",
    )
    table("panels", PANELS)
    imgui.dummy(imgui.ImVec2(0, 0.3 * em))
    table("rules", RULES)
    imgui.dummy(imgui.ImVec2(0, 0.3 * em))
    table("modes", MODES)

    heading(fa.ICON_FA_FLOPPY_DISK, "Files")
    p = imgui.get_cursor_screen_pos()
    bh = 1.9 * em
    y = p.y + 1.4 * em
    x = p.x
    box(dl, x, y, 8 * em, bh, "session1.mesc", TEXT)
    x += 8 * em
    noted_arrow(dl, x, y, 8 * em, bh, "Run Voltage", "a new file per run")
    x += 8 * em
    box(dl, x, y, 15 * em, bh, "session1.<time>.voltage.zarr", KEY)
    x += 15 * em
    noted_arrow(dl, x, y, 7.5 * em, bh, "curate", "saved as you go")
    x += 7.5 * em
    box(dl, x, y, 5.5 * em, bh, ".curation/", KEY)
    imgui.dummy(imgui.ImVec2(w, 1.4 * em + bh + 1.4 * em))
    imgui.text_colored(
        DIM,
        "the .mesc is never written; every run is a new file beside it, named after it and the time, and "
        "belongs to that file alone; your labels live inside the run they were made on",
    )
    table("files", FILES)

    imgui.pop_text_wrap_pos()
    if close_button():
        is_open = False
    imgui.end()
    return is_open, keys_open


def draw_window(state: dict) -> None:
    """One frame of the standalone window: the ? as it sits in the Voltage window, then the page."""
    state["frames"] += 1
    if state["max_frames"] is not None and state["frames"] >= state["max_frames"]:
        hello_imgui.get_runner_params().app_shall_exit = True
    imgui.text_disabled(TITLE)
    imgui.same_line(0, 12)
    if imgui.small_button(f"{fa.ICON_FA_CIRCLE_QUESTION} vnoiser guide"):
        state["open"] = True
    if imgui.is_item_hovered():
        imgui.set_tooltip(TOOLTIP)
    state["open"], state["keys"] = draw_vnoiser_help(state["open"], state["keys"])
    state["keys"] = draw_keybinds_popup(KEYBINDS, state["keys"], "Curation keybinds")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--frames", type=int, default=None, help="exit after this many frames"
    )
    ap.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="with --frames: save the last frame",
    )
    args = ap.parse_args(argv)
    state = {"open": True, "keys": False, "frames": 0, "max_frames": args.frames}
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = f"{TITLE} guide"
    params.app_window_params.window_geometry.size = WINDOW_SIZE
    params.app_window_params.window_geometry.size_auto = False
    params.ini_filename = str(get_mbo_dirs()["imgui"] / "vnoiser_help.ini")
    params.imgui_window_params.tweaked_theme.theme = (
        hello_imgui.ImGuiTheme_.darcula_darker
    )
    params.callbacks.default_icon_font = hello_imgui.DefaultIconFont.font_awesome6
    params.callbacks.show_gui = partial(draw_window, state)
    immapp.run(runner_params=params)
    if args.screenshot is not None:
        iio.imwrite(
            str(args.screenshot), np.asarray(hello_imgui.final_app_window_screenshot())
        )
        print(f"screenshot -> {args.screenshot}", flush=True)


if __name__ == "__main__":
    main()
