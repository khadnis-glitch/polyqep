# === [auto-resized: figsize x1.5] ===
"""Frequency-response comparison: GHM, ADF, PolyQEP+ROM.

Goal
----
Apples-to-apples comparison of five frequency-domain methods on the
Amichi 8x8 simply-supported case (n = 471, 5–500 Hz, 200 evaluation
frequencies).  This is the data generator for the method-comparison table of the paper.

Methods
-------
M0  Direct (per-freq assembly + LU)         reference
M1  GHM N=3 mini-oscillators                augmented (M, C, K)
M2  GHM N=5 mini-oscillators
M3  ADF N_adf=2 fields                      first-order state-space
M4  ADF N_adf=5 fields
M5  PolyQEP modal expansion (full segment spectrum)   proposed method

Each method records: build/setup CPU, per-freq solve CPU, peak-band
max/mean relative error vs Direct.

Output
------
stdout : 6-row comparison table
data/cmame_c1_frf_comparison.csv : per-method aggregated metrics
data/cmame_c1_frf_curves.csv     : per-method full FRF (real/imag)
figures/cmame_c1_frf_comparison.png : top: |Y(omega)| overlay
                                        bottom: relative error vs Direct

Run from project root:
    PYTHONPATH=. python scripts/cmame_c1_ghm_adf_comparison.py
"""
# @MX:NOTE: 5-method FRF benchmark on Amichi 8x8 SS.
# @MX:WARN: when the ADF/GHM augmented system is large, the dense LU makes this somewhat heavy.
# @MX:REASON: augmented size ~n + N·rank(K_shear).  K_shear is nearly full rank under the SS BC.
from __future__ import annotations

import csv
import time
from pathlib import Path

# === [auto-inserted: enable _figure_io monkey-patch for .pkl + .source.md] ===
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _figure_io  # noqa: F401
# === [/auto-inserted] ===

import matplotlib

# === [font scaling: reduced to fit figure cleanly] ===
import matplotlib as _mpl  # noqa: E402
_mpl.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
})
# === [/auto-inserted] ===


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla

from baselines.adf_augmentation import (
    adf_frequency_response,
    assemble_adf_state_space,
    fit_adf_parameters,
)
from baselines.ghm_augmentation import (
    assemble_ghm_augmented,
    fit_ghm_parameters,
    ghm_frequency_response,
)
from polyqep.models.sandwich_plate_amichi2010 import (
    apply_boundary_conditions,
    assemble_K_base_and_shear,
    assemble_global,
)
from polyqep.qep_solver import solve_qep

from scripts.phase3_amichi_polynomial_qep import (
    G_complex_at_freq,
    assemble_complex_per_interval_amichi,
    fit_G_polynomial,
    modal_response_per_interval,
    paper_5_1_params,
    sample_G_omega,
    solve_qep_per_interval,
)


# --------------------------------------------------------------------------- #
# 1. Plate setup (Amichi 8x8 SS)                                              #
# --------------------------------------------------------------------------- #
def build_amichi_8x8_ss():
    """Returns (K_base, K_shear, M, F_red, obs_dof, params)."""
    params = paper_5_1_params(nx=8, ny=8)
    K_base_full, K_shear_full = assemble_K_base_and_shear(params)
    M_full, _ = assemble_global(params, complex(1.0, 0.0))

    F_full = np.zeros(M_full.shape[0], dtype=complex)
    cx, cy = params.nx // 2, params.ny // 2
    n_nodes_x = params.nx + 1
    cw_full = 7 * (cx + cy * n_nodes_x) + 4
    F_full[cw_full] = 1.0
    M_red, _, F_red, free_dofs = apply_boundary_conditions(
        M_full, K_base_full.astype(complex), F_full, params, bc="simply_supported_w",
    )
    free_idx = np.asarray(free_dofs)
    K_base = K_base_full[np.ix_(free_idx, free_idx)]
    K_shear = K_shear_full[np.ix_(free_idx, free_idx)]
    M = (M_red.real + M_red.real.T) / 2.0  # symmetrise
    F = F_red.astype(float).real
    obs = int(np.where(free_idx == cw_full)[0][0])
    return K_base, K_shear, M, F, obs, params


# --------------------------------------------------------------------------- #
# 2. Direct reference                                                         #
# --------------------------------------------------------------------------- #
def direct_response_ss(
    K_base: np.ndarray, K_shear: np.ndarray, M: np.ndarray,
    F: np.ndarray, obs: int, omega_array: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Per-frequency direct method.  G(omega) from Amichi DMA anchors."""
    K_base_c = K_base.astype(complex)
    K_shear_c = K_shear.astype(complex)
    M_c = M.astype(complex)
    F_c = F.astype(complex)
    Y = np.zeros(omega_array.size, dtype=complex)
    t0 = time.perf_counter()
    for k, om in enumerate(omega_array):
        G = G_complex_at_freq(om / (2.0 * np.pi))
        D = K_base_c + G * K_shear_c - om * om * M_c
        Y[k] = spla.solve(D, F_c)[obs]
    return Y, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# 3. GHM runner                                                                #
# --------------------------------------------------------------------------- #
def run_ghm(
    K_base: np.ndarray, K_shear: np.ndarray, M: np.ndarray,
    F: np.ndarray, obs: int, omega_array: np.ndarray,
    omega_sample: np.ndarray, G_re: np.ndarray, G_im: np.ndarray,
    n_mini: int, n_starts: int, rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Fit GHM, assemble augmented (M_aug, C_aug, K_aug), solve per omega."""
    t_fit0 = time.perf_counter()
    params, info_fit = fit_ghm_parameters(
        omega_sample, G_re, G_im, n_mini=n_mini, n_starts=n_starts, rng=rng,
    )
    t_fit = time.perf_counter() - t_fit0

    t_build0 = time.perf_counter()
    aug = assemble_ghm_augmented(K_base, K_shear, M, F, params)
    t_build = time.perf_counter() - t_build0

    t_solve0 = time.perf_counter()
    Y = ghm_frequency_response(aug, omega_array, obs)
    t_solve = time.perf_counter() - t_solve0
    return Y, dict(
        method=f"GHM_N{n_mini}",
        n_aug=aug.M_aug.shape[0],
        n_internal=aug.n_internal,
        t_fit=t_fit, t_build=t_build, t_solve=t_solve,
        fit_re_max=info_fit["rel_re_max"], fit_re_mean=info_fit["rel_re_mean"],
        fit_im_max=info_fit["rel_im_max"], fit_im_mean=info_fit["rel_im_mean"],
    )


# --------------------------------------------------------------------------- #
# 4. ADF runner                                                                #
# --------------------------------------------------------------------------- #
def run_adf(
    K_base: np.ndarray, K_shear: np.ndarray, M: np.ndarray,
    F: np.ndarray, obs: int, omega_array: np.ndarray,
    omega_sample: np.ndarray, G_re: np.ndarray, G_im: np.ndarray,
    n_adf: int, n_starts: int, rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Fit ADF, assemble first-order state-space (A, B), solve per omega."""
    t_fit0 = time.perf_counter()
    params, info_fit = fit_adf_parameters(
        omega_sample, G_re, G_im, n_adf=n_adf, n_starts=n_starts, rng=rng,
    )
    t_fit = time.perf_counter() - t_fit0

    t_build0 = time.perf_counter()
    ss = assemble_adf_state_space(K_base, K_shear, M, F, params)
    t_build = time.perf_counter() - t_build0

    t_solve0 = time.perf_counter()
    Y = adf_frequency_response(ss, F, omega_array, obs)
    t_solve = time.perf_counter() - t_solve0
    return Y, dict(
        method=f"ADF_N{n_adf}",
        n_aug=ss.A.shape[0],
        n_internal=ss.n_internal,
        t_fit=t_fit, t_build=t_build, t_solve=t_solve,
        fit_re_max=info_fit["rel_re_max"], fit_re_mean=info_fit["rel_re_mean"],
        fit_im_max=info_fit["rel_im_max"], fit_im_mean=info_fit["rel_im_mean"],
    )


# --------------------------------------------------------------------------- #
# 5. PolyQEP + ROM runner                                                     #
# --------------------------------------------------------------------------- #
def run_polyqep_modal(
    K_base: np.ndarray, K_shear: np.ndarray, M: np.ndarray,
    F: np.ndarray, obs: int, omega_array: np.ndarray,
    omega_sample: np.ndarray, G_re: np.ndarray, G_im: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """5-segment polynomial-QEP + per-interval modal superposition.

    For SS BC this method is *algorithmically exact* — the paper reports
    0.0000% max error vs Direct on segment 0 over 5–50 Hz. This script
    measures it across the full 5–500 Hz band by stitching the modal
    sums on five segments.
    """
    t_fit0 = time.perf_counter()
    spline_re, spline_im, _ = fit_G_polynomial(omega_sample, G_re, G_im, n_segments=5)
    t_fit = time.perf_counter() - t_fit0

    t_build0 = time.perf_counter()
    interval_mats = assemble_complex_per_interval_amichi(
        spline_re, spline_im, M, K_base.astype(complex), K_shear.astype(complex),
    )
    interval_sols = solve_qep_per_interval(interval_mats, damping_tol=1e-6)
    t_build = time.perf_counter() - t_build0

    # Use the modal_response_per_interval helper — same routine the
    # paper uses for the SS case (full per-segment modal sum, no truncation).
    F_c = F.astype(complex)
    t_solve0 = time.perf_counter()
    Y = modal_response_per_interval(omega_array, interval_sols, F_c, obs)
    t_solve = time.perf_counter() - t_solve0

    # Effective augmented size = total active modes summed across segments.
    n_modes = int(sum(s["lams_full_with_conj"].size for s in interval_sols))
    info = dict(
        method="PolyQEP_modal_sup",
        n_aug=n_modes, n_internal=0,
        t_fit=t_fit, t_build=t_build, t_solve=t_solve,
        fit_re_max=np.nan, fit_re_mean=np.nan,
        fit_im_max=np.nan, fit_im_mean=np.nan,
    )
    return Y, info


# --------------------------------------------------------------------------- #
# 6. Aggregation                                                              #
# --------------------------------------------------------------------------- #
def relative_error(Y_pred: np.ndarray, Y_ref: np.ndarray, mask: np.ndarray
                   ) -> tuple[float, float]:
    valid = np.isfinite(Y_pred) & np.isfinite(Y_ref) & mask
    if not valid.any():
        return float("nan"), float("nan")
    rel = np.abs(Y_pred[valid] - Y_ref[valid]) / np.maximum(np.abs(Y_ref[valid]), 1e-30)
    return float(rel.max() * 100.0), float(rel.mean() * 100.0)


# --------------------------------------------------------------------------- #
# 7. Main                                                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    np.set_printoptions(linewidth=140, precision=4)
    print("=" * 78)
    print("FRF comparison: GHM, ADF, PolyQEP")
    print("=" * 78)

    # 1. Setup
    K_base, K_shear, M, F, obs, params = build_amichi_8x8_ss()
    n = K_base.shape[0]
    print(f"\n[setup] Amichi 8x8 SS: n = {n}, obs DOF = {obs}, "
          f"force DOF = {obs} (mid-plate w)")

    f_eval = np.linspace(5.0, 500.0, 200)
    omega_eval = 2.0 * np.pi * f_eval
    print(f"[grid]  evaluation: {len(f_eval)} freqs in {f_eval[0]:.0f}–{f_eval[-1]:.0f} Hz")

    f_min, f_max = 1.0, 1500.0
    omega_sample, G_re, G_im = sample_G_omega(f_min, f_max, 200)
    print(f"[DMA]   sample: {len(omega_sample)} uniformly spaced points in {f_min}–{f_max} Hz")

    # 2. Direct reference
    print("\n[M0] Direct (per-omega LU, reference)")
    Y_direct, t_direct = direct_response_ss(K_base, K_shear, M, F, obs, omega_eval)
    print(f"     CPU = {t_direct*1000:.1f} ms total, "
          f"{t_direct*1000/len(omega_eval):.3f} ms/freq")
    print(f"     |Y|_max = {np.abs(Y_direct).max():.3e}, "
          f"f_peak = {f_eval[int(np.argmax(np.abs(Y_direct)))]:.1f} Hz")

    mask = np.abs(Y_direct) > 0.01 * np.abs(Y_direct).max()

    # 3. GHM N=3, N=5
    rng = np.random.default_rng(20260509)
    Y_curves: dict[str, np.ndarray] = {"Direct (M0)": Y_direct}
    metrics: list[dict] = [dict(
        method="Direct", n_aug=n, n_internal=0,
        t_fit=0.0, t_build=0.0, t_solve=t_direct,
        fit_re_max=float("nan"), fit_re_mean=float("nan"),
        fit_im_max=float("nan"), fit_im_mean=float("nan"),
        rel_max=0.0, rel_mean=0.0,
    )]

    for label, N in [("GHM N=3 (M1)", 3), ("GHM N=5 (M2)", 5)]:
        print(f"\n[{label}]")
        Y, info = run_ghm(K_base, K_shear, M, F, obs, omega_eval,
                          omega_sample, G_re, G_im, n_mini=N, n_starts=30, rng=rng)
        rmax, rmean = relative_error(Y, Y_direct, mask)
        info.update(rel_max=rmax, rel_mean=rmean)
        metrics.append(info)
        Y_curves[label] = Y
        print(f"     n_aug = {info['n_aug']} (n + {N}*r, r = {info['n_internal']})")
        print(f"     CPU: fit={info['t_fit']*1000:.0f} ms, "
              f"build={info['t_build']*1000:.0f} ms, "
              f"solve={info['t_solve']*1000:.0f} ms "
              f"({info['t_solve']*1000/len(omega_eval):.2f} ms/freq)")
        print(f"     fit Re max={info['fit_re_max']:.2f}%, mean={info['fit_re_mean']:.2f}%")
        print(f"     response rel error: max={rmax:.2f}%, mean={rmean:.2f}%")

    # 4. ADF N_adf=2, N_adf=5
    for label, N in [("ADF N_adf=2 (M3)", 2), ("ADF N_adf=5 (M4)", 5)]:
        print(f"\n[{label}]")
        Y, info = run_adf(K_base, K_shear, M, F, obs, omega_eval,
                          omega_sample, G_re, G_im, n_adf=N, n_starts=30, rng=rng)
        rmax, rmean = relative_error(Y, Y_direct, mask)
        info.update(rel_max=rmax, rel_mean=rmean)
        metrics.append(info)
        Y_curves[label] = Y
        print(f"     n_aug = {info['n_aug']} (2n + {N}*r, r = {info['n_internal']})")
        print(f"     CPU: fit={info['t_fit']*1000:.0f} ms, "
              f"build={info['t_build']*1000:.0f} ms, "
              f"solve={info['t_solve']*1000:.0f} ms "
              f"({info['t_solve']*1000/len(omega_eval):.2f} ms/freq)")
        print(f"     fit Re max={info['fit_re_max']:.2f}%, mean={info['fit_re_mean']:.2f}%")
        print(f"     response rel error: max={rmax:.2f}%, mean={rmean:.2f}%")

    # 5. PolyQEP + per-interval modal superposition (proposed method)
    print("\n[PolyQEP modal superposition (M5, proposed method)]")
    Y, info = run_polyqep_modal(K_base, K_shear, M, F, obs, omega_eval,
                                omega_sample, G_re, G_im)
    rmax, rmean = relative_error(Y, Y_direct, mask)
    info.update(rel_max=rmax, rel_mean=rmean)
    metrics.append(info)
    Y_curves["PolyQEP modal-sup (M5)"] = Y
    print(f"     active modes (sum across 5 segments) = {info['n_aug']}")
    print(f"     CPU: fit={info['t_fit']*1000:.0f} ms, "
          f"build={info['t_build']*1000:.0f} ms, "
          f"solve={info['t_solve']*1000:.0f} ms "
          f"({info['t_solve']*1000/len(omega_eval):.3f} ms/freq)")
    print(f"     response rel error: max={rmax:.2f}%, mean={rmean:.2f}%")

    # 6. Save CSVs
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    summ_path = data_dir / "cmame_c1_frf_comparison.csv"
    with summ_path.open("w", newline="") as fh:
        fields = list(metrics[0].keys())
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in metrics:
            w.writerow(r)
    print(f"\n[saved] {summ_path}")

    curves_path = data_dir / "cmame_c1_frf_curves.csv"
    with curves_path.open("w", newline="") as fh:
        labels = list(Y_curves.keys())
        head = ["f_Hz"] + sum([[f"Re_{lab}", f"Im_{lab}"] for lab in labels], [])
        w = csv.writer(fh)
        w.writerow(head)
        for k, f in enumerate(f_eval):
            row = [f]
            for lab in labels:
                Y = Y_curves[lab]
                row += [float(Y[k].real), float(Y[k].imag)]
            w.writerow(row)
    print(f"[saved] {curves_path}")

    # 7. Figure
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 10.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
    ax_top, ax_bot = axes

    style_map = {
        "Direct (M0)":             dict(color="black",  ls="-",  lw=1.6),
        "GHM N=3 (M1)":            dict(color="C3",     ls="--", lw=1.0),
        "GHM N=5 (M2)":            dict(color="C3",     ls=":",  lw=1.2),
        "ADF N_adf=2 (M3)":        dict(color="C2",     ls="--", lw=1.0),
        "ADF N_adf=5 (M4)":        dict(color="C2",     ls=":",  lw=1.2),
        "PolyQEP modal-sup (M5)":   dict(color="C0",     ls="-",  lw=1.4),
    }
    for label, Y in Y_curves.items():
        st = style_map.get(label, dict(lw=1.0))
        ax_top.semilogy(f_eval, np.abs(Y), label=label, **st)

    ax_top.set_ylabel(r"$|Y(\omega)|$  [m/N]")
    ax_top.grid(True, which="both", alpha=0.3)
    ax_top.legend(fontsize=12, loc="best", ncols=2)
    ax_top.set_title(
        f"FRF comparison on Amichi 8x8 SS\n"
        f"(n={n}, 5–500 Hz, mid-plate drive)")

    for label, Y in Y_curves.items():
        if label == "Direct (M0)":
            continue
        st = style_map.get(label, dict(lw=1.0))
        rel = np.abs(Y - Y_direct) / np.maximum(np.abs(Y_direct), 1e-30) * 100.0
        ax_bot.semilogy(f_eval, rel, label=label, **st)
    ax_bot.set_xlabel("Frequency [Hz]")
    ax_bot.set_ylabel("Rel. error [%]")
    ax_bot.grid(True, which="both", alpha=0.3)
    ax_bot.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)

    fig.tight_layout()
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    fig_path = fig_dir / "cmame_c1_frf_comparison.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {fig_path}")

    # 8. Summary table
    print("\n" + "=" * 78)
    print("Summary (peak-band threshold |Y| > 1% max(|Y|), 5–500 Hz)")
    print("=" * 78)
    head = (f"  {'method':<22}{'n_aug':>7}{'fit Re%':>9}{'fit Im%':>9}"
            f"{'CPU/freq[ms]':>14}{'rel max%':>11}{'rel mean%':>11}")
    print(head)
    for r in metrics:
        cpf = r["t_solve"] * 1000.0 / len(omega_eval)
        fr = r.get("fit_re_max", np.nan)
        fi = r.get("fit_im_max", np.nan)
        fr_s = f"{fr:.2f}" if np.isfinite(fr) else "  n/a"
        fi_s = f"{fi:.2f}" if np.isfinite(fi) else "  n/a"
        print(f"  {r['method']:<22}{r['n_aug']:>7}{fr_s:>9}{fi_s:>9}"
              f"{cpf:>14.3f}{r['rel_max']:>11.2f}{r['rel_mean']:>11.2f}")

    print("\n" + "=" * 78)
    print("Comparison complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
