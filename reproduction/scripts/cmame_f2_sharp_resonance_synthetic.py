# === [auto-resized: figsize x1.5] ===
"""CMAME F2 — Sharp-resonance synthetic DMA stress test.

Background
----------
In the GHM/ADF comparison of the paper, the PolyQEP < GHM < ADF ordering
depends on the fact that, for the smooth, monotone Amichi DMA data, GHM/ADF
have no sharp peak to resolve and are therefore over-parameterized. Running
PolyQEP vs GHM/ADF once more on a synthetic G*(ω) that mimics a glassy-rubbery
transition largely removes this fairness concern.

This script compares PW, PW-RG, GHM(N=1, 2) and ADF(N=1, 2) on a
sharp-resonance synthetic modulus: 2-relaxation Maxwell + glassy plateau.

Synthetic G*(s)
---------------
    G*(s) = G_∞ + Σ_{k=1..2} ΔG_k · (s τ_k) / (1 + s τ_k),   s = jω

Parameters (typical glassy-rubbery transition):
    G_∞     = 10 MPa  (rubbery plateau, low ω)
    G_glass = 2 GPa   (glassy plateau, high ω)
    ΔG_1 = ΔG_2 = (G_glass - G_∞) / 2 ≈ 995 MPa
    τ_1 = 1e-3 s  (ω_peak,1 = 1000 rad/s ≈ 159 Hz)  ← sharp peak inside the 5-500 Hz analysis band
    τ_2 = 1e-1 s  (ω_peak,2 = 10 rad/s ≈ 1.6 Hz)    ← outside the analysis band (low frequency)

Im G(ω) peak at 159 Hz, FWHM ~ decade in ω.

4-method comparison scope
-------------------------
    PW      — Piecewise SOAR (polynomial G online)
    PW-RG   — PW-Refined Global (cross-segment union-SVD, raw G online)
    GHM N=1 — 1 mini-oscillator (can represent a 1-relaxation Maxwell model exactly)
    GHM N=2 — 2 mini-oscillators (1:1 match with the synthetic model itself)
    ADF N=1 — 1 Prony term
    ADF N=2 — 2 Prony terms (1:1 match with the synthetic model itself)

→ GHM N=2 and ADF N=2 have the same order as the synthetic modulus, so a fit
   residual of 0 is theoretically possible. PW is piecewise-quadratic, so a fit
   error appears near the sharp peak. PW-RG evaluates the raw G(j ω) online,
   which removes the fit error.

This experiment places the PolyQEP framework in the \"natural habitat\" of
GHM/ADF as a fair comparison, and checks whether PW-RG can match or exceed the
accuracy of GHM/ADF while retaining modal info.

Output
------
    data/cmame_f2_sharp_resonance.json
    figures/cmame_f2_sharp_resonance.pdf
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# === [auto-inserted: enable _figure_io monkey-patch for .pkl + .source.md] ===
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _figure_io  # noqa: F401
# === [/auto-inserted] ===

import matplotlib

# === [font scaling: reduced to fit stacked subplots cleanly] ===
import matplotlib as _mpl  # noqa: E402
_mpl.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
})
# === [/auto-inserted] ===


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from phase3_amichi_polynomial_qep import (  # type: ignore
    fit_G_polynomial,
    assemble_complex_per_interval_amichi,
    solve_qep_per_interval,
)
from cmame_e2_segmentwise_rom import (  # type: ignore
    setup_amichi_8x8,
    per_segment_soar_rom_response,
)
from polyqep.pw_refined import (
    build_pw_refined_roms_profiled,
    pw_refined_global_response,
)
from baselines.ghm_augmentation import (
    fit_ghm_parameters,
    assemble_ghm_augmented,
    ghm_frequency_response,
)
from baselines.adf_augmentation import (
    fit_adf_parameters,
    assemble_adf_state_space,
    adf_frequency_response,
)


# =============================================================================
# Synthetic G*(s) — 2-relaxation Maxwell + glassy plateau
# =============================================================================

G_INF = 10.0e6       # 10 MPa  rubbery plateau
G_GLASS = 2.0e9      # 2 GPa   glassy plateau
DELTA_G_1 = 0.5 * (G_GLASS - G_INF)
DELTA_G_2 = 0.5 * (G_GLASS - G_INF)
TAU_1 = 1.0e-3       # peak at ω ≈ 1000 rad/s ≈ 159 Hz
TAU_2 = 1.0e-1       # peak at ω ≈ 10 rad/s ≈ 1.6 Hz


def synthetic_G_of_s(s):
    """G*(s) = G_∞ + Σ ΔG_k · (sτ_k)/(1 + sτ_k).  s = jω (or complex)."""
    term1 = DELTA_G_1 * (s * TAU_1) / (1.0 + s * TAU_1)
    term2 = DELTA_G_2 * (s * TAU_2) / (1.0 + s * TAU_2)
    return G_INF + term1 + term2


def synthetic_G_complex_at_omega(omega):
    """Convenience: G(jω) returning complex."""
    return complex(synthetic_G_of_s(1j * omega))


def synthetic_G_at_freq(f_hz):
    """G(jω(f)) as complex(re, im).  Matches G_complex_at_freq signature."""
    omega = 2.0 * np.pi * f_hz
    return synthetic_G_complex_at_omega(omega)


def synthetic_G_re_im_arrays(omega_array):
    """Sample Re G(ω), Im G(ω) on omega array (no Hilbert reconstruction needed)."""
    G_vals = np.array([synthetic_G_complex_at_omega(om) for om in omega_array])
    return G_vals.real, G_vals.imag


# =============================================================================
# Constants
# =============================================================================
F_MIN_HZ = 5.0
F_MAX_HZ = 500.0
N_FREQ = 496
N_SEG_PW = 5
M_PER_SEG_PW = 20
M_SOAR_HYBRID = 15
N_MODAL_AUGMENT = 5


def make_omega_array() -> np.ndarray:
    return 2.0 * np.pi * np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)


def make_G_func_for_xie(G_freq_fn):
    """Same interface as make_G_func — passed to pw_refined_global_response."""
    def G_func(s):
        omega = abs(s.imag) if s.imag != 0.0 else abs(s.real)
        f_hz = omega / (2.0 * np.pi)
        return G_freq_fn(f_hz)
    return G_func


def relative_error_pct(Y_ref: np.ndarray, Y_test: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(Y_ref), 1e-30 * max(np.abs(Y_ref).max(), 1e-30))
    return np.abs(Y_ref - Y_test) / denom * 100.0


def err_stats(err: np.ndarray) -> dict:
    return {
        "max_pct": float(np.max(err)),
        "mean_pct": float(np.mean(err)),
        "p95_pct": float(np.percentile(err, 95)),
        "p99_pct": float(np.percentile(err, 99)),
    }


# =============================================================================
# Direct reference (synthetic G)
# =============================================================================

def run_direct_synthetic(K_base, K_shear, M, F, obs, omegas):
    """Per-frequency direct dense LU with synthetic G."""
    K_base_c = K_base.astype(complex)
    K_shear_c = K_shear.astype(complex)
    M_c = M.astype(complex)
    F_c = F.astype(complex)
    Y = np.zeros(omegas.size, dtype=complex)
    t0 = time.perf_counter()
    for k, om in enumerate(omegas):
        G = synthetic_G_complex_at_omega(om)
        D = K_base_c + G * K_shear_c - om * om * M_c
        Y[k] = spla.solve(D, F_c)[obs]
    return Y, time.perf_counter() - t0


# =============================================================================
# PW (polynomial G online) — modal superposition
# =============================================================================

def run_pw_polynomial(K_base, K_shear, M, F, obs, omegas,
                     omega_sample, G_re, G_im, m_per_seg=M_PER_SEG_PW):
    """PW SOAR with polynomial G online."""
    t_fit_0 = time.perf_counter()
    spline_re, spline_im, _ = fit_G_polynomial(
        omega_sample, G_re, G_im, n_segments=N_SEG_PW,
    )
    interval_mats = assemble_complex_per_interval_amichi(
        spline_re, spline_im, M, K_base.astype(complex), K_shear.astype(complex),
    )
    t_fit_assemble = time.perf_counter() - t_fit_0

    t_qep_0 = time.perf_counter()
    interval_sols = solve_qep_per_interval(interval_mats, damping_tol=1e-6)
    t_qep = time.perf_counter() - t_qep_0

    t_soar_0 = time.perf_counter()
    Y, t_soar_build_inner = per_segment_soar_rom_response(
        omegas, interval_sols, F, obs, m_per_seg,
    )
    t_solve = time.perf_counter() - t_soar_0 - t_soar_build_inner
    t_build = t_fit_assemble + t_qep + t_soar_build_inner

    # Fit error diagnostics (Re/Im)
    g_re_fit = np.zeros_like(omega_sample)
    g_im_fit = np.zeros_like(omega_sample)
    for seg_re, seg_im in zip(spline_re.segments, spline_im.segments):
        mask = (omega_sample >= seg_re.omega_lower) & (omega_sample < seg_re.omega_upper)
        if seg_re is spline_re.segments[-1]:
            mask = (omega_sample >= seg_re.omega_lower) & (omega_sample <= seg_re.omega_upper)
        g_re_fit[mask] = seg_re.a * omega_sample[mask] ** 2 + seg_re.b * omega_sample[mask] + seg_re.c
        g_im_fit[mask] = seg_im.a * omega_sample[mask] ** 2 + seg_im.b * omega_sample[mask] + seg_im.c
    err_re = np.abs(G_re - g_re_fit) / np.maximum(np.abs(G_re), 1e-30) * 100
    err_im = np.abs(G_im - g_im_fit) / np.maximum(np.abs(G_im), 1e-30) * 100
    fit_info = {
        "fit_err_re_max": float(err_re.max()),
        "fit_err_re_mean": float(err_re.mean()),
        "fit_err_im_max": float(err_im.max()),
        "fit_err_im_mean": float(err_im.mean()),
    }
    return Y, t_build, t_solve, interval_sols, fit_info


# =============================================================================
# PW-RG (cross-segment union-SVD, raw G online)
# =============================================================================

def run_pw_rg(K_base, K_shear, M, F, obs, omegas,
              interval_sols, G_func,
              m_soar=M_SOAR_HYBRID, n_modal=N_MODAL_AUGMENT):
    """PW-Refined Global."""
    t0 = time.perf_counter()
    rom_g, profile = build_pw_refined_roms_profiled(
        K_base, K_shear, M, F, interval_sols,
        variant="global",
        m_soar_per_seg=m_soar,
        n_modal_augment=n_modal,
        global_tol_sv=1e-10,
    )
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    Y = pw_refined_global_response(omegas, rom_g, G_func, obs)
    t_solve = time.perf_counter() - t1
    return Y, t_build, t_solve, profile


# =============================================================================
# GHM / ADF runners
# =============================================================================

def run_ghm(K_base, K_shear, M, F, obs, omegas,
            omega_sample, G_re, G_im, n_mini, n_starts=10, seed=0):
    rng = np.random.default_rng(seed)
    t_fit0 = time.perf_counter()
    params, info_fit = fit_ghm_parameters(
        omega_sample, G_re, G_im, n_mini=n_mini, n_starts=n_starts, rng=rng,
    )
    t_fit = time.perf_counter() - t_fit0

    t_build0 = time.perf_counter()
    aug = assemble_ghm_augmented(K_base, K_shear, M, F, params)
    t_build = time.perf_counter() - t_build0

    t_solve0 = time.perf_counter()
    Y = ghm_frequency_response(aug, omegas, obs)
    t_solve = time.perf_counter() - t_solve0
    return Y, dict(
        method=f"GHM_N{n_mini}",
        n_aug=int(aug.M_aug.shape[0]),
        n_internal=int(aug.n_internal),
        t_fit=float(t_fit), t_build=float(t_build), t_solve=float(t_solve),
        fit_re_max=float(info_fit["rel_re_max"]),
        fit_re_mean=float(info_fit["rel_re_mean"]),
        fit_im_max=float(info_fit["rel_im_max"]),
        fit_im_mean=float(info_fit["rel_im_mean"]),
    )


def run_adf(K_base, K_shear, M, F, obs, omegas,
            omega_sample, G_re, G_im, n_adf, n_starts=10, seed=0):
    rng = np.random.default_rng(seed)
    t_fit0 = time.perf_counter()
    params, info_fit = fit_adf_parameters(
        omega_sample, G_re, G_im, n_adf=n_adf, n_starts=n_starts, rng=rng,
    )
    t_fit = time.perf_counter() - t_fit0

    t_build0 = time.perf_counter()
    ss = assemble_adf_state_space(K_base, K_shear, M, F, params)
    t_build = time.perf_counter() - t_build0

    t_solve0 = time.perf_counter()
    Y = adf_frequency_response(ss, F, omegas, obs)
    t_solve = time.perf_counter() - t_solve0
    return Y, dict(
        method=f"ADF_N{n_adf}",
        n_aug=int(ss.A.shape[0]),
        n_internal=int(ss.n_internal),
        t_fit=float(t_fit), t_build=float(t_build), t_solve=float(t_solve),
        fit_re_max=float(info_fit["rel_re_max"]),
        fit_re_mean=float(info_fit["rel_re_mean"]),
        fit_im_max=float(info_fit["rel_im_max"]),
        fit_im_mean=float(info_fit["rel_im_mean"]),
    )


# =============================================================================
# Figure
# =============================================================================

def plot_results(out_path, f_hz, Y_d, curves: dict, errs: dict,
                 G_sample_omega, G_sample_complex):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(15.8, 16.5), sharex=False)
    fig.suptitle(
        "Sharp-resonance synthetic DMA stress test\n"
        f"(G_∞={G_INF/1e6:.0f} MPa, G_glass={G_GLASS/1e9:.1f} GPa, "
        f"τ_1={TAU_1*1e3:.1f} ms, τ_2={TAU_2*1e3:.0f} ms)",
        fontsize=15,
    )

    # Panel 1: G*(ω)
    ax0 = axes[0]
    f_sample = G_sample_omega / (2.0 * np.pi)
    ax0.semilogx(f_sample, np.abs(G_sample_complex.real) / 1e6, "k-", lw=1.5,
                 label="|Re G(ω)| [MPa]")
    ax0.semilogx(f_sample, np.abs(G_sample_complex.imag) / 1e6, "r--", lw=1.5,
                 label="|Im G(ω)| [MPa]")
    ax0.axvspan(F_MIN_HZ, F_MAX_HZ, alpha=0.1, color="blue",
                label=f"analysis band {F_MIN_HZ:.0f}–{F_MAX_HZ:.0f} Hz")
    ax0.set_xlabel("Frequency [Hz]")
    ax0.set_ylabel("|G*| [MPa]")
    ax0.legend(fontsize=12)
    ax0.grid(True, alpha=0.3, which="both")
    ax0.set_title("Synthetic 2-relaxation Maxwell modulus (sharp peak at ~159 Hz)",
                  fontsize=13)

    ref = np.abs(Y_d).max()
    db = lambda Y: 20.0 * np.log10(np.maximum(np.abs(Y) / ref, 1e-15))
    style = {
        "Direct":  dict(color="k",        lw=1.5, ls="-"),
        "PW":      dict(color="#2196F3",  lw=1.2, ls="--"),
        "PW-RG":   dict(color="#D81B60",  lw=1.4, ls="-"),
        "GHM N=1": dict(color="#FF9800",  lw=1.0, ls=":"),
        "GHM N=2": dict(color="#F57C00",  lw=1.2, ls="-."),
        "ADF N=1": dict(color="#4CAF50",  lw=1.0, ls=":"),
        "ADF N=2": dict(color="#2E7D32",  lw=1.2, ls="-."),
    }

    # Panel 2: FRF magnitude overlay
    ax1 = axes[1]
    ax1.plot(f_hz, db(Y_d), color="k", lw=1.5, label="Direct (synthetic G)", zorder=10)
    for key, Y in curves.items():
        s = style.get(key, dict(color="gray", lw=1.0, ls="-"))
        ax1.plot(f_hz, db(Y), color=s["color"], lw=s["lw"], ls=s["ls"],
                 label=key, zorder=3)
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Amplitude [dB]")
    ax1.set_title("FRF magnitude (synthetic G)", fontsize=13)
    ax1.legend(fontsize=12, loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=-95)

    # Panel 3: relative error
    ax2 = axes[2]
    for key, err in errs.items():
        s = style.get(key, dict(color="gray", lw=1.0, ls="-"))
        ax2.semilogy(f_hz, np.maximum(err, 1e-6),
                     color=s["color"], lw=s["lw"], ls=s["ls"],
                     label=key)
    ax2.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(0.01, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Relative error vs Direct [%]")
    ax2.legend(fontsize=12, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title("Relative error against direct (synthetic G, dense LU)", fontsize=13)
    ax2.set_ylim(5e-7, 1.5e2)  # show everything up to the GHM peaks (22-25%)

    # h_pad gives extra vertical room so each subplot's y-axis label/title does not
    # collide with the neighbouring subplot; rect leaves headroom for the suptitle.
    fig.tight_layout(h_pad=2.5, rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    np.set_printoptions(linewidth=120, precision=4)
    print("=" * 80)
    print("CMAME F2 — Sharp-resonance synthetic DMA stress test")
    print("=" * 80)

    # [1] Mesh
    print("\n[1] Amichi 8×8 mesh (simply_supported_w)...")
    params, M_red, K_base_red, K_shear_red, F_red, cw_red = setup_amichi_8x8(
        bc="simply_supported_w",
    )
    n_dof = M_red.shape[0]
    print(f"    n_dof = {n_dof}")
    omegas = make_omega_array()
    f_hz = omegas / (2.0 * np.pi)

    # [2] Sample synthetic G
    print("\n[2] Synthetic G*(ω) sampling (1–1500 Hz, 400 pts)...")
    omega_sample = 2.0 * np.pi * np.linspace(1.0, 1500.0, 400)
    G_re_arr, G_im_arr = synthetic_G_re_im_arrays(omega_sample)
    G_sample_complex = G_re_arr + 1j * G_im_arr
    print(f"    G_∞      = {G_INF/1e6:.1f} MPa")
    print(f"    G_glass  = {G_GLASS/1e9:.2f} GPa")
    print(f"    τ_1, τ_2 = {TAU_1*1e3:.2f} ms, {TAU_2*1e3:.0f} ms")
    print(f"    Re G range : {G_re_arr.min()/1e6:.1f} – {G_re_arr.max()/1e6:.1f} MPa")
    print(f"    Im G range : {G_im_arr.min()/1e6:.1f} – {G_im_arr.max()/1e6:.1f} MPa")
    # Peak in 5-500Hz?
    in_band = (omega_sample >= 2*np.pi*F_MIN_HZ) & (omega_sample <= 2*np.pi*F_MAX_HZ)
    f_peak = omega_sample[in_band][np.argmax(G_im_arr[in_band])] / (2*np.pi)
    print(f"    Im G peak in band: {f_peak:.1f} Hz")

    # [3] Direct (synthetic G)
    print("\n[3] Direct dense LU (synthetic G)...")
    Y_d, t_d = run_direct_synthetic(
        K_base_red, K_shear_red, M_red, F_red, cw_red, omegas,
    )
    print(f"    solve {t_d*1e3:.0f} ms ({t_d*1e3/N_FREQ:.3f} ms/freq)")

    # [4] PW (polynomial G online)
    print("\n[4] PW — Piecewise SOAR (polynomial G online)...")
    Y_pw, t_pw_build, t_pw_solve, interval_sols, fit_info = run_pw_polynomial(
        K_base_red, K_shear_red, M_red, F_red, cw_red, omegas,
        omega_sample, G_re_arr, G_im_arr, m_per_seg=M_PER_SEG_PW,
    )
    err_pw = relative_error_pct(Y_d, Y_pw)
    print(f"    G fit: Re mean {fit_info['fit_err_re_mean']:.2f}% / "
          f"max {fit_info['fit_err_re_max']:.2f}%,  "
          f"Im mean {fit_info['fit_err_im_mean']:.2f}% / "
          f"max {fit_info['fit_err_im_max']:.2f}%")
    print(f"    build {t_pw_build*1e3:.0f} ms, solve {t_pw_solve*1e3:.1f} ms")
    print(f"    FRF err: max {err_pw.max():.3f}%, mean {err_pw.mean():.3f}%")

    # [5] PW-RG (raw G online)
    print("\n[5] PW-RG — cross-segment union-SVD + raw G online...")
    G_func = make_G_func_for_xie(synthetic_G_at_freq)
    # pure SOAR-20 recipe
    Y_pwrg, t_pwrg_build, t_pwrg_solve, pwrg_profile = run_pw_rg(
        K_base_red, K_shear_red, M_red, F_red, cw_red, omegas,
        interval_sols, G_func,
        m_soar=20, n_modal=0,
    )
    err_pwrg = relative_error_pct(Y_d, Y_pwrg)
    print(f"    m_global = {pwrg_profile['m_reduced_global']}, "
          f"build {t_pwrg_build*1e3:.0f} ms, solve {t_pwrg_solve*1e3:.1f} ms")
    print(f"    FRF err: max {err_pwrg.max():.4f}%, mean {err_pwrg.mean():.4f}%")

    # [6] GHM
    methods_records = []
    curves = {"PW": Y_pw, "PW-RG": Y_pwrg}
    errs = {"PW": err_pw, "PW-RG": err_pwrg}

    for n_mini in (1, 2):
        print(f"\n[6.{n_mini}] GHM N={n_mini} fit + augmented response...")
        Y_ghm, info_ghm = run_ghm(
            K_base_red, K_shear_red, M_red, F_red, cw_red, omegas,
            omega_sample, G_re_arr, G_im_arr,
            n_mini=n_mini, n_starts=30, seed=42,
        )
        err_ghm = relative_error_pct(Y_d, Y_ghm)
        info_ghm["err_max_pct"] = float(err_ghm.max())
        info_ghm["err_mean_pct"] = float(err_ghm.mean())
        print(f"    n_aug={info_ghm['n_aug']}, fit Re max {info_ghm['fit_re_max']:.2f}% / "
              f"mean {info_ghm['fit_re_mean']:.2f}%; "
              f"Im max {info_ghm['fit_im_max']:.2f}% / mean {info_ghm['fit_im_mean']:.2f}%")

        print(f"    FRF err: max {err_ghm.max():.3f}%, mean {err_ghm.mean():.3f}%")
        methods_records.append(info_ghm)
        curves[f"GHM N={n_mini}"] = Y_ghm
        errs[f"GHM N={n_mini}"] = err_ghm

    # [7] ADF
    for n_adf in (1, 2):
        print(f"\n[7.{n_adf}] ADF N_adf={n_adf} fit + state-space response...")
        Y_adf, info_adf = run_adf(
            K_base_red, K_shear_red, M_red, F_red, cw_red, omegas,
            omega_sample, G_re_arr, G_im_arr,
            n_adf=n_adf, n_starts=30, seed=42,
        )
        err_adf = relative_error_pct(Y_d, Y_adf)
        info_adf["err_max_pct"] = float(err_adf.max())
        info_adf["err_mean_pct"] = float(err_adf.mean())
        print(f"    n_aug={info_adf['n_aug']}, fit Re max {info_adf['fit_re_max']:.2f}% / "
              f"mean {info_adf['fit_re_mean']:.2f}%; "
              f"Im max {info_adf['fit_im_max']:.2f}% / mean {info_adf['fit_im_mean']:.2f}%")
        print(f"    FRF err: max {err_adf.max():.3f}%, mean {err_adf.mean():.3f}%")
        methods_records.append(info_adf)
        curves[f"ADF N={n_adf}"] = Y_adf
        errs[f"ADF N={n_adf}"] = err_adf

    # [8] Summary table
    print("\n" + "=" * 90)
    print(f"{'method':<14} {'n_aug':>6} {'fit Re mean%':>12} {'fit Im mean%':>12} "
          f"{'err_max%':>10} {'err_mean%':>10}")
    print("-" * 90)
    print(f"{'PW':<14} {N_SEG_PW*M_PER_SEG_PW:>6} "
          f"{fit_info['fit_err_re_mean']:>12.2f} "
          f"{fit_info['fit_err_im_mean']:>12.2f} "
          f"{err_pw.max():>10.3f} {err_pw.mean():>10.3f}")
    print(f"{'PW-RG':<14} {pwrg_profile['m_reduced_global']:>6} "
          f"{'(raw G)':>12} {'(raw G)':>12} "
          f"{err_pwrg.max():>10.4f} {err_pwrg.mean():>10.4f}")
    for r in methods_records:
        print(f"{r['method']:<14} {r['n_aug']:>6} "
              f"{r['fit_re_mean']:>12.2f} {r['fit_im_mean']:>12.2f} "
              f"{r['err_max_pct']:>10.3f} {r['err_mean_pct']:>10.3f}")
    print("=" * 90)

    # [9] Save JSON
    summary = {
        "test_setup": {
            "synthetic_G_model": "2-relaxation Maxwell + glassy plateau",
            "G_inf_Pa": G_INF,
            "G_glass_Pa": G_GLASS,
            "delta_G_1_Pa": DELTA_G_1,
            "delta_G_2_Pa": DELTA_G_2,
            "tau_1_s": TAU_1,
            "tau_2_s": TAU_2,
            "tau1_peak_omega_rad_s": 1.0 / TAU_1,
            "tau1_peak_f_hz": 1.0 / (2.0 * np.pi * TAU_1),
            "analysis_range_hz": [F_MIN_HZ, F_MAX_HZ],
            "n_freq": N_FREQ,
            "mesh": "Amichi 8×8 simply_supported_w",
            "n_dof_free": int(n_dof),
        },
        "methods": {
            "Direct": {
                "label": "Per-frequency dense LU (synthetic G)",
                "t_solve_s": float(t_d),
                "n_dof": int(n_dof),
            },
            "PW": {
                "label": "Piecewise SOAR (polynomial G online)",
                "m_total": N_SEG_PW * M_PER_SEG_PW,
                "n_segments": N_SEG_PW,
                "t_build_s": float(t_pw_build),
                "t_solve_s": float(t_pw_solve),
                "g_fit": fit_info,
                "error": err_stats(err_pw),
                "online_G_evaluation": "polynomial fit",
                "modal_info": "yes",
            },
            "PW-RG": {
                "label": "PW-Refined Global (raw G online)",
                "m_global": int(pwrg_profile["m_reduced_global"]),
                "source_dims_per_segment": pwrg_profile["source_dims_per_segment"],
                "t_build_s": float(t_pwrg_build),
                "t_solve_s": float(t_pwrg_solve),
                "error": err_stats(err_pwrg),
                "online_G_evaluation": "raw G(jω)",
                "modal_info": "yes (basis-embedded)",
            },
        },
        "ghm_adf_methods": methods_records,
        "interpretation": {
            "claim": (
                "For the sharp-resonance synthetic G (2-relaxation Maxwell), GHM N=2 and "
                "ADF N=2 have the same order as the modulus itself → fit residual ~0. This "
                "checks how the PolyQEP framework performs under that condition."
            ),
            "expected_outcomes": {
                "GHM_N=2": "fit residual ~0, very small FRF err (represents its own model)",
                "ADF_N=2": "fit residual ~0, very small FRF err",
                "PW": "piecewise-quadratic fit oscillates near the sharp peak",
                "PW-RG": "raw G online → independent of the fit, FRF err near machine precision",
            },
        },
    }

    out_dir_data = _ROOT / "data"
    out_dir_data.mkdir(exist_ok=True)
    out_json = out_dir_data / "cmame_f2_sharp_resonance.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\n  JSON: {out_json}")

    # [10] Figure
    out_fig = _ROOT / "figures" / "cmame_f2_sharp_resonance.pdf"
    plot_results(out_fig, f_hz, Y_d, curves, errs, omega_sample, G_sample_complex)

    print("\nDone.")


if __name__ == "__main__":
    main()
