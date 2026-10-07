"""Figure I/O helper + auto monkey-patch.

**Automatic hook (activated simply by importing this module)**:
    Wraps `matplotlib.figure.Figure.savefig` so that every call saving under
    figures/ also writes a `.pkl` (pickle dump) and a `.source.md`
    (traceability metadata) alongside the image.

    Existing `fig.savefig(...)` calls therefore need no change; adding the single
    line `import scripts._figure_io` (or `import _figure_io` with PYTHONPATH=scripts)
    is enough to keep all four artifacts (png/pdf + pkl + source.md) together.

(Extra helper) `save_figure_with_metadata(fig, name, data_dict, ...)` saves the
four artifacts plus the CSV/JSON raw data in one call.

Every figure-generating script saves the following four artifacts together with
a single `save_figure_with_metadata(...)` call:

  1. figures/<name>.png or .pdf          (image)
  2. data/<name>.csv                     (raw data — list of dict)
  3. figures/<name>.pkl                  (matplotlib figure pickle)
  4. figures/<name>.source.md            (traceability metadata)

Usage example:

  from scripts._figure_io import save_figure_with_metadata

  save_figure_with_metadata(
      fig,
      name="cmame_b2_kg_residual",
      data_dict={
          "Won2013_residual_per_G": [{"G": 1e6, "residual": 1.3e-9}, ...],
          "Amichi2010_residual_per_G": [{"G": 1e6, "residual": 9e-20}, ...],
      },
      source_script=__file__,
      used_in=["main Fig 6 (a) — Won 2013 vs Amichi K(G) affine residual"],
      key_numbers={"K_residual_Won": "1.3e-9", "K_residual_Amichi": "9e-20"},
  )
"""
from __future__ import annotations

import csv
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.figure as mpl_figure

# ---------------------------------------------------------------------------
# Resolve project layout
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
PKL_SUBDIR = FIGURES_DIR / "_pkl"


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    PKL_SUBDIR.mkdir(exist_ok=True)


def _save_data(name: str, data_dict: dict[str, Any]) -> tuple[Path, Path | None]:
    """Save data_dict as JSON (always) and CSV (if all values are list of dict).

    Returns (json_path, csv_path).  csv_path is None if data is not tabular.
    """
    json_path = DATA_DIR / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(data_dict, f, indent=2, default=str)

    csv_path: Path | None = None
    # Heuristic: if every value is a list of dicts, also emit one wide CSV
    if (
        data_dict
        and all(isinstance(v, list) and v and isinstance(v[0], dict) for v in data_dict.values())
    ):
        csv_path = DATA_DIR / f"{name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer: csv.DictWriter | None = None
            for series_name, rows in data_dict.items():
                for row in rows:
                    out = {"series": series_name, **row}
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(out.keys()))
                        writer.writeheader()
                    writer.writerow(out)

    return json_path, csv_path


def _save_pkl(fig: mpl_figure.Figure, name: str) -> Path:
    pkl_path = PKL_SUBDIR / f"{name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(fig, f)
    return pkl_path


def _write_source_md(
    name: str,
    img_path: Path,
    json_path: Path,
    csv_path: Path | None,
    pkl_path: Path,
    source_script: str,
    used_in: Iterable[str] | None,
    key_numbers: dict[str, Any] | None,
    data_inputs: Iterable[str] | None,
) -> Path:
    md_path = FIGURES_DIR / f"{name}.source.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# {name}",
        "",
        f"- **Generator script**: `{Path(source_script).relative_to(ROOT) if Path(source_script).is_absolute() else source_script}`",
        f"- **Image output**: `{img_path.relative_to(ROOT)}`",
        f"- **Data (JSON)**: `{json_path.relative_to(ROOT)}`",
    ]
    if csv_path is not None:
        lines.append(f"- **Data (CSV)**: `{csv_path.relative_to(ROOT)}`")
    lines.extend(
        [
            f"- **Matplotlib pickle**: `{pkl_path.relative_to(ROOT)}`",
            f"- **Last regenerated**: {timestamp}",
        ]
    )

    if data_inputs:
        lines.append("- **Data inputs**:")
        for src in data_inputs:
            lines.append(f"  - {src}")

    if key_numbers:
        lines.append("- **Key numbers**:")
        for k, v in key_numbers.items():
            lines.append(f"  - `{k}` = {v}")

    if used_in:
        lines.append("- **Used in**:")
        for ref in used_in:
            lines.append(f"  - {ref}")

    lines.append("")

    md_path.write_text("\n".join(lines))
    return md_path


def save_figure_with_metadata(
    fig: mpl_figure.Figure,
    name: str,
    data_dict: dict[str, Any],
    source_script: str,
    *,
    used_in: Iterable[str] | None = None,
    key_numbers: dict[str, Any] | None = None,
    data_inputs: Iterable[str] | None = None,
    img_format: str = "png",
    dpi: int = 140,
) -> dict[str, Path]:
    """Save fig + data + pkl + source.md in one call.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Base name without extension.  Example: "cmame_b2_kg_residual".
    data_dict : dict
        Series-name to plot data (list of dict rows) or scalar metadata.
    source_script : str
        Usually `__file__` from the calling script.
    used_in, key_numbers, data_inputs : iterable / dict
        Free-form annotations recorded in source.md.
    img_format : "png" or "pdf"
    dpi : int

    Returns
    -------
    dict mapping artifact name to path: img, json, csv (optional), pkl, source_md.
    """
    _ensure_dirs()

    img_path = FIGURES_DIR / f"{name}.{img_format}"
    fig.savefig(img_path, dpi=dpi, bbox_inches="tight")

    json_path, csv_path = _save_data(name, data_dict)
    pkl_path = _save_pkl(fig, name)
    md_path = _write_source_md(
        name=name,
        img_path=img_path,
        json_path=json_path,
        csv_path=csv_path,
        pkl_path=pkl_path,
        source_script=source_script,
        used_in=used_in,
        key_numbers=key_numbers,
        data_inputs=data_inputs,
    )

    paths = {
        "img": img_path,
        "json": json_path,
        "pkl": pkl_path,
        "source_md": md_path,
    }
    if csv_path is not None:
        paths["csv"] = csv_path

    print(f"[saved] {img_path}")
    print(f"[saved] {json_path}")
    if csv_path is not None:
        print(f"[saved] {csv_path}")
    print(f"[saved] {pkl_path}")
    print(f"[saved] {md_path}")
    return paths


# ---------------------------------------------------------------------------
# Auto monkey-patch: Figure.savefig -> also save .pkl + .source.md siblings
# ---------------------------------------------------------------------------
import os
import sys
import traceback

import matplotlib.figure as _mpl_figure

_orig_savefig = _mpl_figure.Figure.savefig


def _resolve_caller_script() -> str:
    """Return path of the user script that called savefig (skip matplotlib + this module)."""
    try:
        stack = traceback.extract_stack()
        this_file = os.path.abspath(__file__)
        for frame in reversed(stack[:-1]):
            fname = os.path.abspath(frame.filename)
            if fname == this_file:
                continue
            if "matplotlib" in fname.split(os.sep):
                continue
            if fname.endswith("savefig.py"):
                continue
            return fname
    except Exception:
        pass
    return "unknown"


def _auto_save_pkl_source(fig: "_mpl_figure.Figure", fname: object) -> None:
    """Best-effort side-channel saver. Never raises on caller path."""
    try:
        from pathlib import Path as _P

        if not isinstance(fname, (str, os.PathLike)):
            return  # bytes / file-like: skip
        fp = _P(os.fspath(fname)).resolve()
        # Only mirror outputs under figures/ (PNG/PDF) — skip test temp paths, etc.
        if "figures" not in fp.parts:
            return
        name = fp.stem

        pkl_dir = FIGURES_DIR / "_pkl"
        pkl_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = pkl_dir / f"{name}.pkl"
        try:
            with open(pkl_path, "wb") as _f:
                pickle.dump(fig, _f)
        except Exception as e:  # pickle may fail on lambdas / C-closures
            sys.stderr.write(f"[_figure_io] pickle skip ({name}): {e}\n")
            pkl_path = None

        caller = _resolve_caller_script()
        md_path = FIGURES_DIR / f"{name}.source.md"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# {name}",
            "",
            f"- **Image output**: `{fp.relative_to(ROOT) if ROOT in fp.parents or fp == ROOT else fp.name}`",
        ]
        try:
            cp = _P(caller)
            rel = cp.relative_to(ROOT) if ROOT in cp.parents else cp.name
            lines.append(f"- **Generator script**: `{rel}`")
        except Exception:
            lines.append(f"- **Generator script**: `{caller}`")
        if pkl_path is not None:
            lines.append(
                f"- **Matplotlib pickle**: `{pkl_path.relative_to(ROOT)}` "
                f"(re-open via `pickle.load(open(path, 'rb'))`, no Python re-run needed)"
            )
        else:
            lines.append("- **Matplotlib pickle**: skipped (unpicklable artist)")
        lines.append(f"- **Last regenerated**: {ts}")
        lines.append("")
        md_path.write_text("\n".join(lines))
    except Exception as e:
        sys.stderr.write(f"[_figure_io] auto-save side-channel failed: {e}\n")


def _patched_savefig(self, fname, *args, **kwargs):
    result = _orig_savefig(self, fname, *args, **kwargs)
    _auto_save_pkl_source(self, fname)
    return result


# Idempotent patch (don't double-wrap on re-import)
if not getattr(_mpl_figure.Figure.savefig, "_figure_io_patched", False):
    _patched_savefig._figure_io_patched = True  # type: ignore[attr-defined]
    _mpl_figure.Figure.savefig = _patched_savefig  # type: ignore[assignment]

