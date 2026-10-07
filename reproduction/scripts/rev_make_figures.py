"""Additional studies — figures regenerated from the revision data files.

Main text
    rev_fig_size_accuracy.pdf      method trade-off (Fig. 6)
    rev_fig_frf_augmentation.pdf   FRF comparison with GHM/ADF (Fig. 7)
Supplementary material
    rev_figS1_modulus_fits.pdf     GHM/ADF fits, causal bound, order study (Fig. S1)
    rev_figS2_partition.pdf        partition sensitivity (partition guide of the Discussion; not in the Supplementary Material)
    rev_figS3_noise.pdf            noisy-data Monte Carlo study (Fig. S2)
    rev_figS3_measured.pdf         digitized DMA curves (Fig. S3)
    rev_figS4_scaling.pdf          mesh scaling and offline cost (Fig. S4)

Every figure is drawn from JSON/NPZ files written by the ``rev_*`` scripts;
nothing is timed here.

Run::

    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_make_figures.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import _figure_io  # noqa: F401  (pickle + source stub next to every figure)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

from rev_causality_check import prony_blocks
from rev_modulus_data import digitized, digitized_table, representative
from baselines import adf_augmentation as adf
from baselines import ghm_augmentation as ghm
from polyqep.fitting import fit_joint_spline

_ROOT = Path(__file__).resolve().parent.parent
DATA, FIG = _ROOT / "data", _ROOT / "figures"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.2, "axes.grid": True, "grid.alpha": 0.3,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})
C_POLY, C_GHM, C_ADF, C_R, C_RG, C_PRONY = (
    "#1f77b4", "#d62728", "#2ca02c", "#17becf", "#9467bd", "#ff7f0e")
BASELINE_HZ = np.array([1.0, 50.0, 150.0, 350.0, 700.0, 1500.0])


def load(name):
    with open(DATA / name) as fh:
        return json.load(fh)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"saved figures/{name}")


def ghm_curve(p, f_hz):
    params = ghm.GHMParameters(p["G_inf_Pa"], p["alpha"],
                               2 * np.pi * np.array(p["what_hz"]), p["zeta"])
    return ghm.ghm_modulus(2 * np.pi * f_hz, params)


def adf_curve(p, f_hz):
    params = adf.ADFParameters(p["G_e_Pa"], p["Gv_Pa"],
                               2 * np.pi * np.array(p["Omega_hz"]))
    return adf.adf_modulus(2 * np.pi * f_hz, params)


def quadratic_fit(f_samples, G_samples, edges_hz, f_eval):
    spl_re, spl_im = fit_joint_spline(2 * np.pi * f_samples, G_samples.real,
                                      G_samples.imag, 2 * np.pi * edges_hz)
    om = 2 * np.pi * f_eval
    return np.asarray(spl_re(om)) + 1j * np.asarray(spl_im(om))


# ----------------------------------------------------------------------------
def fig_size_accuracy():
    rows = load("rev_bench_augmentation.json")["rows"]
    style = {
        "GHM N=3": (C_GHM, "s"), "GHM N=5": (C_GHM, "^"),
        "ADF N=2": (C_ADF, "s"), "ADF N=5": (C_ADF, "^"),
        "PolyQEP modal expansion": (C_POLY, "*"),
        "PW-R": (C_R, "D"), "PW-RG": (C_RG, "o"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for name, (color, marker) in style.items():
        r = rows[name]
        err = r["frf"]["max_pct"]
        label = name.replace("N=", "$N=$").replace("$N=$", "$N$=")
        for ax, x in zip(axes, (r["size"], r["per_freq_ms"])):
            ax.scatter(x, err, s=80 if marker == "*" else 40, c=color,
                       marker=marker, edgecolors="k", linewidths=0.4, zorder=3,
                       label=label)
    axes[0].axvline(rows["Direct"]["size"], color="k", ls=":", lw=0.8)
    axes[1].axvline(rows["Direct"]["per_freq_ms"], color="k", ls=":", lw=0.8,
                    label="direct solve")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-11, 1e3)
    axes[0].set_xlabel("System size solved per frequency")
    axes[1].set_xlabel("Online time per frequency [ms]")
    axes[0].set_ylabel("Peak-band max relative FRF error [%]")
    fig.tight_layout()
    save(fig, "rev_fig_size_accuracy.pdf")


def fig_frf_augmentation():
    z = np.load(DATA / "rev_bench_augmentation_frf.npz")
    f = z["f_hz"]
    curves = [
        ("Direct", "Direct", "k", "-", 1.5),
        ("GHM_N3", "GHM $N=3$", C_GHM, "--", 1.0),
        ("GHM_N5", "GHM $N=5$", C_GHM, ":", 1.2),
        ("ADF_N2", "ADF $N=2$", C_ADF, "--", 1.0),
        ("ADF_N5", "ADF $N=5$", C_ADF, ":", 1.2),
        ("PolyQEP_modal_expansion", "PolyQEP modal expansion", C_POLY, "-", 1.2),
    ]
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.2, 5.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
    ref = z["Direct"]
    for key, label, color, ls, lw in curves:
        a0.semilogy(f, np.abs(z[key]), color=color, ls=ls, lw=lw, label=label)
        if key != "Direct":
            a1.semilogy(f, np.abs(z[key] - ref) / np.abs(ref) * 100.0,
                        color=color, ls=ls, lw=lw)
    a1.axhline(1.0, color="gray", ls=":", lw=0.8)
    a0.set_ylabel(r"$|Y(\omega)|$ [m/N]")
    a1.set_ylabel("Relative error [%]")
    a1.set_xlabel("Frequency [Hz]")
    a0.legend(ncol=2, loc="upper right")
    save(fig, "rev_fig_frf_augmentation.pdf")


# ----------------------------------------------------------------------------
def fig_s1():
    diag = load("rev_ghm_adf_diagnosis.json")["order_study"]
    A = [r for r in diag if r["sampling"] == "A_uniform_1_1500"]
    pick = {(r["method"], r["order"]): r for r in A}
    f = np.geomspace(1.0, 1500.0, 500)
    G = representative(f)
    f_fit = np.linspace(1.0, 1500.0, 200)
    G_poly = quadratic_fit(f_fit, representative(f_fit), BASELINE_HZ, f)

    # storage-only causal fit (convex): loss implied by the dispersion
    pole_hz = np.logspace(-4, np.log10(1500.0) + 4, 58)
    A_re, A_im = prony_blocks(f_fit, pole_hz)
    G_fit = representative(f_fit)
    w = 1.0 / np.abs(G_fit)
    x, _ = nnls(w[:, None] * A_re, w * G_fit.real, maxiter=5000)
    B_re, B_im = prony_blocks(f, pole_hz)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for ax, part, lab in ((axes[0, 0], np.real, r"$\mathrm{Re}\,G$ [MPa]"),
                          (axes[0, 1], np.imag, r"$\mathrm{Im}\,G$ [MPa]")):
        ax.semilogx(f, part(G) / 1e6, "k-", lw=1.6, label="representative profile")
        ax.semilogx(f, part(G_poly) / 1e6, color=C_POLY, ls="--",
                    label="piecewise quadratic (5 seg.)")
        ax.semilogx(f, part(ghm_curve(pick[("GHM", 5)]["parameters"], f)) / 1e6,
                    color=C_GHM, ls=":", label="GHM $N=5$")
        ax.semilogx(f, part(adf_curve(pick[("ADF", 5)]["parameters"], f)) / 1e6,
                    color=C_ADF, ls="-.", label="ADF $N=5$")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(lab)
    axes[0, 1].semilogx(f, (B_im @ x) / 1e6, color=C_PRONY, ls="-", lw=1.0,
                        label="loss implied by Re $G$\n(causal relaxation model)")
    axes[0, 0].legend(loc="upper left", fontsize=6.8)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    axes[0, 1].legend(handles[-1:], labels[-1:], loc="upper left", fontsize=6.8)
    axes[0, 0].set_title("(a) storage modulus")
    axes[0, 1].set_title("(b) loss modulus")

    orders = sorted({r["order"] for r in A})
    for method, color in (("GHM", C_GHM), ("ADF", C_ADF)):
        rows = [pick[(method, n)] for n in orders]
        axes[1, 0].plot(orders, [r["objective_rms_rel_pct"] for r in rows],
                        color=color, marker="o", ms=4, label=f"{method}")
        axes[1, 1].plot(orders, [r["fit_grid_errors"]["re_max"] for r in rows],
                        color=color, marker="o", ms=4, ls="-",
                        label=f"{method}, Re $G$")
        axes[1, 1].plot(orders, [r["fit_grid_errors"]["im_max"] for r in rows],
                        color=color, marker="s", ms=4, ls="--",
                        label=f"{method}, Im $G$")
    floor = load("rev_causality_check.json")["results"][0]["joint_fit"]
    axes[1, 0].axhline(floor["rms_relative_residual_pct"], color=C_PRONY, ls="--",
                       label="convex bound, relaxation models")
    axes[1, 0].set_ylabel("Optimal objective: r.m.s. relative residual [%]")
    axes[1, 1].set_ylabel("Component-wise max relative error [%]")
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 1].set_ylim(0, 110)
    for ax in axes[1]:
        ax.set_xlabel("Model order $N$")
        ax.legend()
    axes[1, 0].set_title("(c) minimized objective")
    axes[1, 1].set_title("(d) tabulated metric")
    fig.tight_layout()
    save(fig, "rev_figS1_modulus_fits.pdf")


def fig_s2():
    rows = [r for r in load("rev_partition_sensitivity.json")["rows"]
            if "skipped" not in r and r["sampling"] == "log2000"]
    groups = {
        "baseline": ("k", "*", 90), "log-uniform": (C_POLY, "o", 22),
        "uniform": (C_ADF, "s", 22), "baseline - 1": (C_GHM, "v", 22),
        "baseline + 1": (C_GHM, "^", 22), "baseline + 2": (C_GHM, "D", 18),
        "shifted": (C_PRONY, "x", 26), "random": ("0.55", ".", 28),
    }
    labels = {"baseline": "baseline (50/150/350/700 Hz)",
              "log-uniform": "log-uniform", "uniform": "uniform",
              "baseline - 1": "baseline, one breakpoint removed",
              "baseline + 1": "baseline, one segment split",
              "baseline + 2": "baseline, two segments split",
              "shifted": "breakpoints scaled by 0.8 to 1.2",
              "random": r"breakpoints moved at random, $\pm$20%"}
    keys = (("frf_fitted_full_order", "(a) fitted-modulus solve"),
            ("frf_pw_r", "(b) PW-R"), ("frf_pw_rg", "(c) PW-RG"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for ax, (key, title) in zip(axes, keys):
        for g, (color, marker, size) in groups.items():
            sel = [r for r in rows if r["group"] == g]
            ax.scatter([r["n_segments"] for r in sel],
                       [r[key]["max_pct"] for r in sel], s=size, c=color,
                       marker=marker, label=labels[g], zorder=3,
                       linewidths=0.8 if marker == "x" else 0.3)
        ax.set_yscale("log")
        ax.set_xlabel("Number of segments")
        ax.set_title(title)
        ax.set_xticks([2, 5, 10, 15, 20])
    axes[0].set_ylabel("Max relative FRF error [%]")
    axes[0].set_ylim(1, 12)
    axes[0].set_yticks([1, 2, 5, 10])
    axes[0].set_yticklabels(["1", "2", "5", "10"])
    axes[2].set_ylim(1e-11, 1e-4)
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "rev_figS2_partition.pdf")


def fig_s3_noise():
    study = load("rev_noise_study.json")
    ex = np.load(DATA / "rev_noise_example.npz")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    f = ex["f_plot"]
    for ax, part, lab in ((axes[0], np.real, r"$\mathrm{Re}\,G$ [MPa]"),
                          (axes[1], np.imag, r"$\mathrm{Im}\,G$ [MPa]")):
        ax.semilogx(f, part(ex["G_clean"]) / 1e6, color="0.6", lw=1.0,
                    label="noise-free profile")
        ax.semilogx(f, part(ex["G_noisy"]) / 1e6, "k-", lw=1.3,
                    label=r"noisy profile, $\sigma=5\%$")
        ax.semilogx(f, part(ex["G_fit"]) / 1e6, color=C_POLY, ls="--",
                    label="piecewise-quadratic fit")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(lab)
    axes[0].legend(loc="upper left")
    axes[0].set_title("(a) one draw, storage")
    axes[1].set_title("(b) one draw, loss")
    sig = [lv["sigma"] * 100 for lv in study["levels"]]
    series = (("e_d", "data propagation $e_d$", "k", "o"),
              ("e_m_fitted", "fitted-modulus solve $e_m$", C_POLY, "s"),
              ("e_m_pw_r", "PW-R $e_m$", C_R, "D"),
              ("e_m_pw_rg", "PW-RG $e_m$", C_RG, "^"))
    for key, label, color, marker in series:
        mean = [lv[key]["mean"] for lv in study["levels"]]
        p95 = [lv[key]["p95"] for lv in study["levels"]]
        axes[2].plot(sig, mean, color=color, marker=marker, ms=4, label=label)
        axes[2].plot(sig, p95, color=color, marker=marker, ms=3, ls=":", lw=0.9)
    axes[2].set_yscale("log")
    axes[2].set_xticks(sig)
    axes[2].set_xlabel(r"Noise level $\sigma$ [%]")
    axes[2].set_ylabel("Max relative FRF error [%]")
    axes[2].set_title("(c) 50 draws: mean (solid), 95th pct. (dotted)")
    axes[2].legend(loc="center right")
    fig.tight_layout()
    save(fig, "rev_figS3_noise.pdf")


def fig_s3_measured():
    study = load("rev_measured_dma_study.json")["rows"]
    f_d, _, _ = digitized_table()
    G_d = digitized(f_d)
    f = np.linspace(f_d[0], f_d[-1], 400)
    G_poly = quadratic_fit(f_d, G_d, np.linspace(f_d[0], f_d[-1], 6), f)
    om_d = 2 * np.pi * f_d
    rng = np.random.default_rng([20260509, 0, 5])
    p_ghm, _ = ghm.fit_ghm_parameters(om_d, G_d.real, G_d.imag, n_mini=5,
                                      n_starts=30, rng=rng)
    rng = np.random.default_rng([20260509, 1, 2])
    p_adf, _ = adf.fit_adf_parameters(om_d, G_d.real, G_d.imag, n_adf=2,
                                      n_starts=30, rng=rng)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for ax, part, lab in ((axes[0], np.real, r"$\mathrm{Re}\,G$ [MPa]"),
                          (axes[1], np.imag, r"$\mathrm{Im}\,G$ [MPa]")):
        ax.plot(f_d, part(G_d) / 1e6, "k.", ms=2.5, label="digitized DMA curves")
        ax.plot(f, part(G_poly) / 1e6, color=C_POLY, ls="--",
                label="piecewise quadratic (5 seg.)")
        ax.plot(f, part(ghm.ghm_modulus(2 * np.pi * f, p_ghm)) / 1e6,
                color=C_GHM, ls=":", label="GHM $N=5$")
        ax.plot(f, part(adf.adf_modulus(2 * np.pi * f, p_adf)) / 1e6,
                color=C_ADF, ls="-.", label="ADF $N=2$")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(lab)
    axes[0].legend(loc="lower right", fontsize=6.5)
    axes[0].set_title("(a) storage modulus")
    axes[1].set_title("(b) loss modulus")
    for method, setting, color, marker, label in (
            ("PolyQEP", "uniform partition", C_POLY, "o", "PolyQEP, uniform partition"),
            ("PolyQEP", "log partition", C_POLY, "s", "PolyQEP, log partition"),
            ("GHM", "30-start bounded NLLS", C_GHM, "^", "GHM"),
            ("ADF", "30-start bounded NLLS", C_ADF, "v", "ADF")):
        sel = [r for r in study if r["method"] == method and r["setting"] == setting]
        axes[2].plot([r["order"] for r in sel],
                     [r["frf"]["peak_band_max_pct"] for r in sel], color=color,
                     marker=marker, ms=4, ls="-" if marker in "o^v" else "--",
                     label=label)
    floor = [r for r in study if r["method"] == "Prony floor"][0]
    axes[2].axhline(floor["frf"]["peak_band_max_pct"], color=C_PRONY, ls="--",
                    lw=0.9, label="convex bound, relaxation models")
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.3, 30)
    axes[2].set_xlabel("Segments or model order $N$")
    axes[2].set_ylabel("Peak-band max rel. FRF error [%]")
    axes[2].set_title("(c) response error")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.16), fontsize=7.5)
    fig.tight_layout()
    save(fig, "rev_figS3_measured.pdf")


def fig_s4():
    dense = [json.load(open(p)) for p in sorted(glob.glob(str(DATA / "rev_bench_scaling_nx*.json")))]
    sparse = {d["nx"]: d for d in
              (json.load(open(p)) for p in
               sorted(glob.glob(str(DATA / "rev_bench_offline_sparse_nx*.json"))))}
    n = [d["n_free_dof"] for d in dense]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].loglog(n, [d["per_freq_ms"]["D_dense_numpy"] for d in dense], "k-o",
                   ms=4, label="direct, dense LU")
    axes[0].loglog(n, [d["per_freq_ms"]["D_sparse"] for d in dense], "k--s",
                   ms=4, label="direct, sparse LU")
    axes[0].loglog(n, [d["summary"]["PW-RG"]["per_freq_ms"] for d in dense],
                   color=C_RG, marker="^", ms=4, label="PW-RG")
    axes[0].loglog(n, [d["summary"]["PW-R"]["per_freq_ms"] for d in dense],
                   color=C_R, marker="D", ms=4, label="PW-R")
    axes[0].set_xlabel("Free DOFs $n$")
    axes[0].set_ylabel("Online time per frequency [ms]")
    axes[0].set_title("(a) per-frequency cost")
    axes[0].legend()
    axes[1].loglog(n, [d["summary"]["PW-RG"]["offline_total_ms"] / 1e3 for d in dense],
                   color=C_RG, marker="^", ms=4, label="PW-RG build, dense factorization")
    axes[1].loglog(n, [sparse[d["nx"]]["offline_total_ms"]["PW-RG"] / 1e3 for d in dense],
                   color=C_RG, marker="^", ms=4, ls="--",
                   label="PW-RG build, sparse factorization")
    axes[1].loglog(n, [d["per_freq_ms"]["D_sparse"] / 1e3 for d in dense], "k--s",
                   ms=4, label="one sparse direct solve")
    axes[1].set_xlabel("Free DOFs $n$")
    axes[1].set_ylabel("Time [s]")
    axes[1].set_title("(b) offline cost")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "rev_figS4_scaling.pdf")


def main():
    FIG.mkdir(exist_ok=True)
    fig_size_accuracy()
    fig_frf_augmentation()
    fig_s1()
    fig_s2()
    fig_s3_noise()
    fig_s3_measured()
    fig_s4()


if __name__ == "__main__":
    main()
