# STYLE.md

Prose rules for code in pml_utilities. Anything ruff can check is not repeated here:
it lives in `pyproject.toml` under `[tool.ruff]`, and `.github/workflows/format.yml`
applies `ruff format` and `ruff check --fix` on every push to `main`. You do not need
to run ruff on a branch.

## Dependencies

- Support exactly one version of each dependency: pin it in `pyproject.toml` and
  import it at the top of the module.
- Never write lazy-import wrappers with version guards. If an older version cannot do
  the job, the pin handles it.
- The one sanctioned function-local import is for optional or heavy packages so
  `import mbo_utilities` stays cheap: `torch`, `suite2p`, `lbm_suite2p_python`,
  `masknmf`, `vnoiser`, `cupy`, `imgui_bundle`, `fastplotlib`, `pygfx`, and for
  breaking a genuine import cycle (`_writers` ↔ `arrays`).

## Functions

- No nested functions or classes. No functions defined inside tests.
- Do not split code into helpers unless the same code is needed in two places;
  inline it.
- Type-hint public signatures. Internal helpers may omit hints when obvious.
- `from __future__ import annotations` in modules that use `X | None` or `list[X]`
  in class bodies or dataclass fields.

## Comments

- Single line, lowercase, only when the code cannot say it.
- No banner or separator comments, no section headers.
- No `TODO` without a tracking issue.

## Logging

- Loggers come from `mbo_utilities.log`: `logger = log.get("arrays.zarr")`. Never
  `logging.getLogger(...)`; it forms a second tree the GUI console and
  `set_global_level` cannot see.
- No `print()` in library code (AGENTS.md §8).

## Errors

- Raise specific exceptions.
- In the GUI, catch at the draw boundary and show the message; in workers, let it
  propagate after `monitor.fail(...)`.
- No defensive `try`/`except`, no speculative edge cases, no extra abstraction
  layers unless asked.

## Scope of a change

- Do not add annotations, comments, or docstrings to code you are not otherwise
  changing.
- No backwards-compatibility shims for names that were never released.
- Keep notebook cells clean; code comments instead of markdown cells.

## Indexing

Public selections (`planes`, `timepoints`, `channels`, `roi`, `unit`) are 1-based.
Everything internal is 0-based. The conversion happens in
`arrays/features/_selection.py` and `_slicing.parse_selection`, nowhere else.

## Docstrings

Numpy style.

- Module: one-line summary; a paragraph only when the behavior is non-obvious.
- Class: what one instance represents and the shape it reports.
- Function: required when the signature does not make intent obvious or there are
  side effects (writes to disk, mutates metadata). One line when that is enough.
  No `Parameters`/`Returns` sections that restate the signature.
- Skip docstrings on trivially named one-liners.

```python
class MescArray(...):
    """One Femtonics MESc measurement unit as (T, C, Z, Y, X).

    ``MethodType`` decides what axis 0 of ``Channel_N`` means; AOD ROIs land on Z.
    """
```
