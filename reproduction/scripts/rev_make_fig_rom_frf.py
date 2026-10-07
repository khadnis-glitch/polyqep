#!/usr/bin/env python3
"""Fig. 5 (seven-method FRF comparison, ``rev_fig_rom_frf.pdf``) re-plotted from
the archived sweep ``data/rev_bench_rom_frf.npz`` written by ``rev_bench_rom.py``.

This script only draws the archived curves (no benchmark re-run, no timing
involved) with the labels and the style of the paper.

Oracle: the maximum relative errors printed here must equal the fit-bounded
column of Table 3 (``data/rev_bench_rom.json``): PW-R 0.023 %, PW-RG < 1e-9 %,
XS 7.7e-5 %, XA 3.3e-5 %, PW SOAR 5.34 %.
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _figure_io import save_figure_with_metadata  # noqa: E402

ROOT = HERE.parent
DATA, FIG = ROOT / "data", ROOT / "figures"
OUT_TEX = ROOT.parent / "latex" / "_revision_amm_R1" / "figures" / "rev_fig_rom_frf.pdf"
FLOOR_PCT = 1e-6   # lower axis limit of the error panel (stated in the caption)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.2, "axes.grid": True, "grid.alpha": 0.3,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})

# key in npz, legend label, colour, line style, width  (styles follow the caption)
METHODS = [
    ("D_dense_numpy",   "Direct (dense LU)",            "k",       "-",  1.5),
    ("D_sparse_native", "Direct (sparse LU)",           "0.55",    "--", 0.9),
    ("PW",              "PW SOAR (poly-$G$)",           "#1f77b4", "--", 1.2),
    ("PW_R",            "PW-R (raw-$G$)",               "#1a7f1a", "-",  1.2),
    ("PW_RG",           "PW-RG (raw-$G$, global basis)", "m",      ":",  1.5),
    ("XS",              "XS (Xie et al., $m=100$)",     "#ff7f0e", "-.", 1.2),
    ("XA",              "XA (Xie et al., adaptive)",    "#5fc45f", ":",  1.5),
]


def main() -> None:
    z = np.load(DATA / "rev_bench_rom_frf.npz")
    f_hz = z["omegas"] / (2 * np.pi)
    ref = z["D_dense_numpy"]
    ref_db = 20 * np.log10(np.abs(ref) / np.abs(ref).max())

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.5, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1]})
    rows, key_numbers = [], {}
    for key, label, color, ls, lw in METHODS:
        y = z[key]
        a0.plot(f_hz, 20 * np.log10(np.abs(y) / np.abs(ref).max()), color=color, ls=ls, lw=lw, label=label)
        if key == "D_dense_numpy":
            continue
        err = np.abs(y - ref) / np.abs(ref) * 100
        key_numbers[f"{label} max err [%]"] = float(err.max())
        if key == "D_sparse_native":
            continue                      # error at the round-off floor; omitted from the error panel
        a1.semilogy(f_hz, np.maximum(err, FLOOR_PCT), color=color, ls=ls, lw=lw, label=label)
        rows.append((key, err))

    a0.set_ylabel(r"$|Y(\omega)|$ [dB re max]")
    a0.set_ylim(-60, 3)
    a0.legend(ncol=2, loc="upper right")
    a1.set_ylabel("Relative error vs dense LU [%]")
    a1.set_xlabel("Frequency [Hz]")
    a1.set_ylim(FLOOR_PCT * 0.5, 3e3)          # headroom so the legend covers no curve (max error 5.3 %)
    a1.axhline(1.0, color="0.4", lw=0.8, ls=":")
    a1.legend(ncol=3, loc="upper center")
    a1.set_xlim(0, 505)
    fig.tight_layout()

    data = {
        "frequency_hz": f_hz.tolist(),
        "ref_db": ref_db.tolist(),
        "relative_error_pct": {k: e.tolist() for k, e in rows},
        "floor_pct": FLOOR_PCT,
    }
    paths = save_figure_with_metadata(
        fig, "rev_fig_rom_frf", data, __file__,
        used_in=["main_en.tex Fig. 5 (fig:xie2018-frf)"],
        key_numbers=key_numbers,
        data_inputs=["data/rev_bench_rom_frf.npz", "data/rev_bench_rom.json"],
        img_format="pdf", dpi=300,
    )
    plt.close(fig)

    # ---- oracle against Table 3 --------------------------------------------
    tab = json.load(open(DATA / "rev_bench_rom.json"))["errors"]
    for key, label, *_ in METHODS[2:]:
        tkey = {"PW": "PW", "PW_R": "PW-R", "PW_RG": "PW-RG", "XS": "XS", "XA": "XA"}[key]
        want = tab[tkey]["fit_bounded"]["max_pct"]
        got = key_numbers[f"{label} max err [%]"]
        print(f"{tkey:6s} max err {got:.3e} %   Table 3 {want:.3e} %")
        assert abs(got - want) <= 1e-9 * max(1.0, abs(want)) + 1e-12, (tkey, got, want)

    if OUT_TEX.parent.exists():           # manuscript tree present (not in the public repository)
        if OUT_TEX.exists() and not OUT_TEX.with_suffix(".pdf.bak_pre_relabel").exists():
            shutil.copy2(OUT_TEX, OUT_TEX.with_suffix(".pdf.bak_pre_relabel"))
        shutil.copy2(paths["img"], OUT_TEX)
        print(f"copied -> {OUT_TEX}")


if __name__ == "__main__":
    main()
