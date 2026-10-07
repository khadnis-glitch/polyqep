# === [auto-resized: figsize x1.5] ===
"""CMAME F1 v3 — Fair comparison + PW-Refined hybrid.

Background (from the v2 results)
--------------------------------
The real reason XA reached err_max = 0.0015% in v2 (4000x better than PW at
6.16%) is that xie2018_rom._adaptive_frf_from_reduced **uses the original
G(jω) directly during online evaluation**. The PW path uses the
piecewise-polynomial fit both offline and online, so the fit residual becomes
the floor of the response error.

This v3 quantifies the following:
    (i)  fair-comparison metrics: ROM dim m, online G representation, modal-info availability
    (ii) PW-Refined (PW-R)  — SOAR basis as in PW, online uses raw G(jω)
    (iii) PW-Refined Hybrid (PW-RH) — SOAR + segment effective eigenvecs combined

7-method comparison
-------------------
    D_dense  — Direct dense LU baseline
    D_sparse — Direct sparse LU baseline
    PW       — Piecewise SOAR (polynomial G online)            [unchanged from v2]
    PW-R     — Piecewise SOAR basis + raw G(jω) online         [NEW]
    PW-RH    — PW-R + modal eigenvec augment                   [NEW]
    XS       — Xie 2018 single-point Taylor + R2               [from v2]
    XA       — Xie 2018 adaptive (raw G online via Krylov)     [from v2]

4-Axis Metric
-------------
    Accuracy   : err_max [%], err_mean [%], p95, p99 vs Direct dense
    ROM dim m  : reduced system size (segment-summed)
    Per-freq   : online solve cost [ms/freq]
    Modal info : yes / partial / none

Output
------
    data/cmame_f1_xie2018_summary_v3.json
    figures/cmame_f1_frf_comparison_v3.pdf
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

# === [auto-inserted: 2x font scaling for all figures] ===
import matplotlib as _mpl  # noqa: E402
_mpl.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
})
# === [/auto-inserted] ===


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla
import scipy.sparse as spsparse
import scipy.sparse.linalg as spspla

# path setup
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from phase3_amichi_polynomial_qep import (  # type: ignore
    G_complex_at_freq,
    direct_response_per_freq,
    fit_G_polynomial,
    sample_G_omega,
    assemble_complex_per_interval_amichi,
    solve_qep_per_interval,
)
from cmame_e2_segmentwise_rom import (  # type: ignore
    setup_amichi_8x8,
    per_segment_soar_rom_response,
)
from baselines.xie2018_rom import (
    build_xie_rom,
    xie_rom_response,
    xie_adaptive_response,
)
from polyqep.pw_refined import (
    build_pw_refined_roms_profiled,
    pw_refined_response,
    pw_refined_global_response,
)


# =============================================================================
# Constants
# =============================================================================
F_MIN_HZ = 5.0
F_MAX_HZ = 500.0
N_FREQ = 496
N_SEG = 5
M_PER_SEG = 20            # PW / PW-R default
M_SOAR_HYBRID = 15        # SOAR part of PW-RH
N_MODAL_AUGMENT = 5       # modal part of PW-RH (per segment)


def make_omega_array() -> np.ndarray:
    return 2.0 * np.pi * np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)


def make_G_func(G_fn):
    def G_func(s: complex) -> complex:
        omega = abs(s.imag) if s.imag != 0.0 else abs(s.real)
        f_hz = omega / (2.0 * np.pi)
        return G_fn(f_hz)
    return G_func


def relative_error_pct(Y_ref: np.ndarray, Y_test: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(Y_ref), 1e-30 * max(np.abs(Y_ref).max(), 1e-30))
    return np.abs(Y_ref - Y_test) / denom * 100.0


# =============================================================================
# Method runners — Direct
# =============================================================================

def run_direct_dense(params, M_red, K_base_red, K_shear_red, F_red, cw_red, omegas):
    t0 = time.perf_counter()
    Y = direct_response_per_freq(
        params, K_base_red, K_shear_red, M_red, omegas, F_red, cw_red,
    )
    t_solve = time.perf_counter() - t0
    return Y, 0.0, t_solve


def run_direct_sparse(K_base_red, K_shear_red, M_red, F_red, cw_red, omegas):
    K_base_c = K_base_red.astype(complex)
    K_shear_c = K_shear_red.astype(complex)
    M_c = M_red.astype(complex)
    Y = np.zeros(len(omegas), dtype=complex)
    t0 = time.perf_counter()
    for i, omega in enumerate(omegas):
        f_hz = omega / (2.0 * np.pi)
        G = complex(G_complex_at_freq(f_hz))
        K = K_base_c + G * K_shear_c
        D = K - (omega ** 2) * M_c
        D_sp = spsparse.csc_matrix(D)
        lu = spspla.splu(D_sp)
        u = lu.solve(F_red.astype(complex))
        Y[i] = u[cw_red]
    t_solve = time.perf_counter() - t0
    return Y, 0.0, t_solve


# =============================================================================
# Method runners — PW (same polynomial G online as v2)
# =============================================================================

def run_piecewise_soar(M_red, K_base_red, K_shear_red, F_red, cw_red, omegas,
                       interval_sols, m_per_seg=M_PER_SEG):
    """Original PW path: per_segment_soar_rom_response uses the reduced
    M*_r, C*_r, K*_r built from the polynomial G as they are (polynomial G online)."""
    t0 = time.perf_counter()
    Y, t_soar_build = per_segment_soar_rom_response(
        omegas, interval_sols, F_red, cw_red, m_per_seg,
    )
    t_total = time.perf_counter() - t0
    t_solve = t_total - t_soar_build
    return Y, t_soar_build, t_solve


# =============================================================================
# Method runners — PW-R, PW-RH (raw G online)
# =============================================================================

def run_pw_refined(M_red, K_base_red, K_shear_red, F_red, cw_red, omegas,
                   interval_sols, G_func, m_per_seg=M_PER_SEG):
    """PW-Refined: SOAR basis (polynomial G), online uses raw G(jω)."""
    t0 = time.perf_counter()
    roms, profile = build_pw_refined_roms_profiled(
        K_base_red, K_shear_red, M_red, F_red, interval_sols,
        variant="soar", m_per_seg=m_per_seg,
    )
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    Y = pw_refined_response(omegas, roms, G_func, cw_red)
    t_solve = time.perf_counter() - t1
    return Y, t_build, t_solve, profile


def run_pw_refined_hybrid(M_red, K_base_red, K_shear_red, F_red, cw_red, omegas,
                          interval_sols, G_func,
                          m_soar=M_SOAR_HYBRID, n_modal=N_MODAL_AUGMENT):
    """PW-Refined Hybrid: SOAR + modal eigenvec augmented basis, raw G online."""
    t0 = time.perf_counter()
    roms, profile = build_pw_refined_roms_profiled(
        K_base_red, K_shear_red, M_red, F_red, interval_sols,
        variant="hybrid",
        m_soar_per_seg=m_soar,
        n_modal_augment=n_modal,
    )
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    Y = pw_refined_response(omegas, roms, G_func, cw_red)
    t_solve = time.perf_counter() - t1
    return Y, t_build, t_solve, profile


def run_pw_refined_global(M_red, K_base_red, K_shear_red, F_red, cw_red, omegas,
                          interval_sols, G_func,
                          m_soar=M_SOAR_HYBRID, n_modal=N_MODAL_AUGMENT,
                          global_tol_sv=1e-10):
    """PW-Refined Global: cross-segment union-SVD → single broadband basis, raw G online."""
    t0 = time.perf_counter()
    rom_g, profile = build_pw_refined_roms_profiled(
        K_base_red, K_shear_red, M_red, F_red, interval_sols,
        variant="global",
        m_soar_per_seg=m_soar,
        n_modal_augment=n_modal,
        global_tol_sv=global_tol_sv,
    )
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    Y = pw_refined_global_response(omegas, rom_g, G_func, cw_red)
    t_solve = time.perf_counter() - t1
    return Y, t_build, t_solve, profile


# =============================================================================
# Method runners — XS, XA
# =============================================================================

def run_xie_single(K_base_red, K_shear_red, M_red, G_func, F_red, cw_red, omegas,
                   f0_hz=50.0, m=100):
    n_dof = M_red.shape[0]
    L_vec = np.zeros(n_dof, dtype=complex)
    L_vec[cw_red] = 1.0
    t0 = time.perf_counter()
    rom = build_xie_rom(K_base_red, K_shear_red, M_red, G_func,
                        s0=1j * 2.0 * np.pi * f0_hz, m=m, force_vector=F_red)
    t_build = time.perf_counter() - t0
    t1 = time.perf_counter()
    Y = xie_rom_response(rom, omegas, F_red, L_vec, include_remainder=True)
    t_solve = time.perf_counter() - t1
    return Y, t_build, t_solve, m


def run_xie_adaptive(K_base_red, K_shear_red, M_red, G_func, F_red, cw_red, omegas,
                     m_per_point=20, tol=1e-2, max_points=10):
    n_dof = M_red.shape[0]
    L_vec = np.zeros(n_dof, dtype=complex)
    L_vec[cw_red] = 1.0
    t0 = time.perf_counter()
    Y, adaptive_rom = xie_adaptive_response(
        K_base_red, K_shear_red, M_red, G_func,
        omegas, F_red, L_vec,
        m_per_point=m_per_point, tol=tol, max_points=max_points,
    )
    t_build = time.perf_counter() - t0
    m_final = adaptive_rom.m_reduced if adaptive_rom is not None else 0
    return Y, t_build, 0.0, adaptive_rom, m_final


# =============================================================================
# Metric packaging
# =============================================================================

def err_stats(err: np.ndarray) -> dict:
    return {
        "max_pct": float(np.max(err)),
        "mean_pct": float(np.mean(err)),
        "p95_pct": float(np.percentile(err, 95)),
        "p99_pct": float(np.percentile(err, 99)),
    }


def method_record(
    label: str,
    t_build: float,
    t_solve: float,
    n_freq: int,
    m_reduced: int,
    g_representation_online: str,
    modal_info: str,
    err: np.ndarray | None,
    t_d_dense_solve: float,
    extras: dict | None = None,
) -> dict:
    per_freq_ms = (
        (t_solve / n_freq * 1e3) if t_solve > 0
        else ((t_build + t_solve) / n_freq * 1e3)
    )
    t_d_per_freq_ms = t_d_dense_solve / n_freq * 1e3
    rec = {
        "label": label,
        "rom_dim_m": int(m_reduced),
        "online_G_evaluation": g_representation_online,
        "modal_info_available": modal_info,
        "t_build_s": float(t_build),
        "t_solve_s": float(t_solve),
        "t_total_s": float(t_build + t_solve),
        "per_freq_ms": float(per_freq_ms),
        "speedup_solve_only": float(t_d_per_freq_ms / max(per_freq_ms, 1e-9)),
        "speedup_total_wall": float(t_d_dense_solve / max(t_build + t_solve, 1e-9)),
    }
    if err is not None:
        rec["error"] = err_stats(err)
    if extras:
        rec.update(extras)
    return rec


# =============================================================================
# Figure
# =============================================================================

def plot_frf_v3(output_path: Path, omegas: np.ndarray, Y_d: np.ndarray,
                method_curves: dict, method_errs: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f_hz = omegas / (2.0 * np.pi)
    ref = np.abs(Y_d).max()
    db = lambda Y: 20.0 * np.log10(np.maximum(np.abs(Y) / ref, 1e-15))

    style = {
        "D_dense":  dict(color="k",        lw=1.5, ls="-",  label="D_dense — Direct (dense LU)"),
        "D_sparse": dict(color="#607D8B",  lw=1.0, ls=(0, (3, 3)), label="D_sparse — Direct (sparse LU)"),
        "PW":       dict(color="#2196F3",  lw=1.2, ls="--", label="PW SOAR — PolyQEP+SOAR (polynomial G online)"),
        "PW-R":     dict(color="#1B5E20",  lw=1.2, ls="-",  label="PW-R — PolyQEP SOAR basis + raw G online [NEW]"),
        "PW-RH":    dict(color="#9C27B0",  lw=1.4, ls=(0, (5, 2, 1, 2)), label="PW-RH — PW-R + modal augment [NEW]"),
        "PW-RG":    dict(color="#D81B60",  lw=1.6, ls=(0, (1, 1)), label="PW-RG — cross-segment union-SVD of SOAR bases [NEW]"),
        "XS":       dict(color="#FF5722",  lw=1.2, ls="-.", label="XS — Xie Single-point (m=100)"),
        "XA":       dict(color="#4CAF50",  lw=1.2, ls=":",  label="XA — Xie Adaptive (raw G via Krylov)"),
    }

    fig, axes = plt.subplots(2, 1, figsize=(15.8, 12), sharex=True)
    fig.suptitle("Amichi §5.1 8×8 simply-supported (n=471): 7-method FRF comparison (5–500 Hz)",
                 fontsize=22)

    ax0 = axes[0]
    ax0.plot(f_hz, db(Y_d), color="k", lw=1.5, label=style["D_dense"]["label"], zorder=10)
    for key, Y in method_curves.items():
        if key == "D_dense":
            continue
        s = style.get(key, {"color": "gray", "lw": 1, "ls": "-", "label": key})
        ax0.plot(f_hz, db(Y), color=s["color"], lw=s["lw"], ls=s["ls"],
                 label=s["label"], zorder=3)
    ax0.set_ylabel("Amplitude [dB re max|Y_D|]", fontsize=18)
    ax0.legend(fontsize=15, loc="upper right", ncol=1)
    ax0.grid(True, alpha=0.3)
    ax0.set_ylim(bottom=-60)

    ax1 = axes[1]
    for key, err in method_errs.items():
        s = style.get(key, {"color": "gray", "lw": 1, "ls": "-", "label": key})
        short = s["label"].split(" —")[0]  # short display name (e.g. "PW SOAR")
        ax1.semilogy(f_hz, np.maximum(err, 1e-6),
                     color=s["color"], lw=s["lw"], ls=s["ls"],
                     label=f"{short} err")
    ax1.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax1.axhline(0.01, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax1.set_xlabel("Frequency [Hz]", fontsize=18)
    ax1.set_ylabel("Relative error vs D_dense [%]", fontsize=18)
    ax1.legend(fontsize=15, loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {output_path}")


# =============================================================================
# Table
# =============================================================================

def print_table_v3(records: dict, n_freq: int):
    print()
    print("=" * 120)
    hdr = (f"{'method':<10} {'m':>5} {'online_G':<12} {'modal':<8} "
           f"{'build[ms]':>10} {'solve[ms]':>10} {'pf[ms]':>9} "
           f"{'sp_tot':>8} {'err_max%':>10} {'err_mean%':>10}")
    print(hdr)
    print("-" * 120)
    for key, r in records.items():
        err = r.get("error", {})
        print(f"{key:<10} "
              f"{r['rom_dim_m']:>5} "
              f"{r['online_G_evaluation']:<12} "
              f"{r['modal_info_available']:<8} "
              f"{r['t_build_s']*1e3:>10.1f} "
              f"{r['t_solve_s']*1e3:>10.1f} "
              f"{r['per_freq_ms']:>9.4f} "
              f"{r['speedup_total_wall']:>7.2f}× "
              f"{err.get('max_pct', float('nan')):>10.4f} "
              f"{err.get('mean_pct', float('nan')):>10.4f}")
    print("=" * 120)


# =============================================================================
# Main
# =============================================================================

def main():
    np.set_printoptions(linewidth=120, precision=4)

    print("=" * 80)
    print("CMAME F1 v3 — Fair comparison + PW-Refined")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Amichi 8×8 mesh (simply_supported_w → 471 DOF)
    # ------------------------------------------------------------------
    print("\n[1] Assembling Amichi 8×8 mesh (simply_supported_w BC)...")
    params, M_red, K_base_red, K_shear_red, F_red, cw_red = setup_amichi_8x8(
        bc="simply_supported_w",
    )
    n_dof = M_red.shape[0]
    print(f"    DOF: {n_dof}")
    omegas = make_omega_array()
    G_func = make_G_func(G_complex_at_freq)
    f_hz = omegas / (2.0 * np.pi)
    print(f"    frequency: {f_hz[0]:.1f}–{f_hz[-1]:.1f} Hz, {N_FREQ} points")

    # ------------------------------------------------------------------
    # 2. Segment split + per-segment QEP solve (offline, shared by PW / PW-R / PW-RH)
    # ------------------------------------------------------------------
    print("\n[2] Segment split + per-segment QEP solve (offline) ...")
    t_fit_0 = time.perf_counter()
    omega_fit, G_re, G_im = sample_G_omega(1.0, 1500.0, 400)
    spline_re, spline_im, _ = fit_G_polynomial(
        omega_fit, G_re, G_im, n_segments=N_SEG,
    )
    interval_mats = assemble_complex_per_interval_amichi(
        spline_re, spline_im, M_red, K_base_red, K_shear_red,
    )
    t_fit_assemble = time.perf_counter() - t_fit_0
    t_qep_0 = time.perf_counter()
    interval_sols = solve_qep_per_interval(interval_mats, damping_tol=1e-6)
    t_qep = time.perf_counter() - t_qep_0
    eff_counts = [s["lams_effective"].size for s in interval_sols]
    print(f"    fit+assemble: {t_fit_assemble*1e3:.1f} ms,  "
          f"QEP solve (5 seg): {t_qep*1e3:.0f} ms")
    print(f"    Per-segment effective eigenvalue counts: {eff_counts}")

    # ------------------------------------------------------------------
    # 3. D_dense baseline
    # ------------------------------------------------------------------
    print("\n[3] D_dense — Dense LU baseline...")
    Y_d, t_dd_build, t_dd_solve = run_direct_dense(
        params, M_red, K_base_red, K_shear_red, F_red, cw_red, omegas,
    )
    print(f"    solve: {t_dd_solve*1e3:.1f} ms ({t_dd_solve*1e3/N_FREQ:.4f} ms/freq)")

    # ------------------------------------------------------------------
    # 4. D_sparse
    # ------------------------------------------------------------------
    print("\n[4] D_sparse — Sparse LU baseline...")
    Y_ds, t_ds_build, t_ds_solve = run_direct_sparse(
        K_base_red, K_shear_red, M_red, F_red, cw_red, omegas,
    )
    err_ds = relative_error_pct(Y_d, Y_ds)
    print(f"    solve: {t_ds_solve*1e3:.1f} ms,  vs dense err: max {err_ds.max():.4e}%")

    # ------------------------------------------------------------------
    # 5. PW (original)
    # ------------------------------------------------------------------
    print("\n[5] PW — Piecewise SOAR (polynomial G online)...")
    Y_pw, t_pw_build_soar, t_pw_solve = run_piecewise_soar(
        M_red, K_base_red, K_shear_red, F_red, cw_red, omegas,
        interval_sols, m_per_seg=M_PER_SEG,
    )
    err_pw = relative_error_pct(Y_d, Y_pw)
    # PW full build = fit_assemble + QEP + SOAR
    t_pw_build = t_fit_assemble + t_qep + t_pw_build_soar
    print(f"    build (incl. fit+assemble+QEP+SOAR): {t_pw_build*1e3:.0f} ms, "
          f"solve: {t_pw_solve*1e3:.1f} ms")
    print(f"    err: max {err_pw.max():.3f}%, mean {err_pw.mean():.3f}%")

    # ------------------------------------------------------------------
    # 6. PW-R (NEW)
    # ------------------------------------------------------------------
    print("\n[6] PW-R — PolyQEP SOAR basis + raw G(jω) online [NEW]...")
    Y_pwr, t_pwr_basis, t_pwr_solve, pwr_profile = run_pw_refined(
        M_red, K_base_red, K_shear_red, F_red, cw_red, omegas,
        interval_sols, G_func, m_per_seg=M_PER_SEG,
    )
    err_pwr = relative_error_pct(Y_d, Y_pwr)
    t_pwr_build = t_fit_assemble + t_qep + t_pwr_basis
    m_pwr = pwr_profile["m_reduced_sum"]
    print(f"    build (incl. fit+assemble+QEP+basis+project): "
          f"{t_pwr_build*1e3:.0f} ms,  solve: {t_pwr_solve*1e3:.1f} ms")
    print(f"    m_reduced per segment: {pwr_profile['m_reduced_per_segment']}  "
          f"(sum={m_pwr})")
    print(f"    err: max {err_pwr.max():.4f}%, mean {err_pwr.mean():.4f}%")

    # ------------------------------------------------------------------
    # 7. PW-RH (NEW)
    # ------------------------------------------------------------------
    print("\n[7] PW-RH — PW-R + modal eigenvec augment [NEW]...")
    Y_pwrh, t_pwrh_basis, t_pwrh_solve, pwrh_profile = run_pw_refined_hybrid(
        M_red, K_base_red, K_shear_red, F_red, cw_red, omegas,
        interval_sols, G_func,
        m_soar=M_SOAR_HYBRID, n_modal=N_MODAL_AUGMENT,
    )
    err_pwrh = relative_error_pct(Y_d, Y_pwrh)
    t_pwrh_build = t_fit_assemble + t_qep + t_pwrh_basis
    m_pwrh = pwrh_profile["m_reduced_sum"]
    print(f"    build (incl. fit+assemble+QEP+basis+project): "
          f"{t_pwrh_build*1e3:.0f} ms,  solve: {t_pwrh_solve*1e3:.1f} ms")
    print(f"    m_reduced per segment: {pwrh_profile['m_reduced_per_segment']}  "
          f"(sum={m_pwrh})  [SOAR={M_SOAR_HYBRID} + modal={N_MODAL_AUGMENT}]")
    print(f"    err: max {err_pwrh.max():.4f}%, mean {err_pwrh.mean():.4f}%")

    # ------------------------------------------------------------------
    # 7b. PW-RG (NEW: cross-segment union-SVD)
    # ------------------------------------------------------------------
    print("\n[7b] PW-RG — PW-Refined Global (cross-segment union-SVD) [NEW]...")
    # Recipe: pure SOAR-20 source bases (modal augmentation is counterproductive
    # for the global variant; pure SOAR-20 reaches the same floor with m_global=81)
    Y_pwrg, t_pwrg_basis, t_pwrg_solve, pwrg_profile = run_pw_refined_global(
        M_red, K_base_red, K_shear_red, F_red, cw_red, omegas,
        interval_sols, G_func,
        m_soar=M_PER_SEG, n_modal=0,
        global_tol_sv=1e-10,
    )
    err_pwrg = relative_error_pct(Y_d, Y_pwrg)
    t_pwrg_build = t_fit_assemble + t_qep + t_pwrg_basis
    m_pwrg = pwrg_profile["m_reduced_global"]
    print(f"    build (incl. fit+assemble+QEP+basis+SVD+project): "
          f"{t_pwrg_build*1e3:.0f} ms,  solve: {t_pwrg_solve*1e3:.1f} ms")
    print(f"    m_global = {m_pwrg}  "
          f"(source per-segment: {pwrg_profile['source_dims_per_segment']}, "
          f"sum={pwrg_profile['source_dims_sum']})")
    print(f"    err: max {err_pwrg.max():.4f}%, mean {err_pwrg.mean():.4f}%")

    # ------------------------------------------------------------------
    # 8. XS
    # ------------------------------------------------------------------
    print("\n[8] XS — Xie Single-point (s0=j·2π·50, m=100)...")
    Y_xs, t_xs_build, t_xs_solve, m_xs = run_xie_single(
        K_base_red, K_shear_red, M_red, G_func, F_red, cw_red, omegas,
        f0_hz=50.0, m=100,
    )
    err_xs = relative_error_pct(Y_d, Y_xs)
    print(f"    build: {t_xs_build*1e3:.1f} ms,  solve: {t_xs_solve*1e3:.1f} ms")
    print(f"    err: max {err_xs.max():.3f}%, mean {err_xs.mean():.3f}%")

    # ------------------------------------------------------------------
    # 9. XA
    # ------------------------------------------------------------------
    print("\n[9] XA — Xie Adaptive (ε=1e-2, m_per_point=20)...")
    Y_xa, t_xa_build, t_xa_solve, adaptive_rom, m_xa = run_xie_adaptive(
        K_base_red, K_shear_red, M_red, G_func, F_red, cw_red, omegas,
        m_per_point=20, tol=1e-2, max_points=10,
    )
    err_xa = relative_error_pct(Y_d, Y_xa)
    n_pts = len(adaptive_rom.expansion_points) if adaptive_rom else 0
    print(f"    build+solve: {t_xa_build*1e3:.1f} ms,  expansion points: {n_pts},  "
          f"m_final: {m_xa}")
    print(f"    err: max {err_xa.max():.4f}%, mean {err_xa.mean():.4f}%")

    # ------------------------------------------------------------------
    # 10. Metric packaging
    # ------------------------------------------------------------------
    records = {
        "D_dense": method_record(
            "Direct dense LU (baseline)", t_dd_build, t_dd_solve,
            N_FREQ, n_dof, "raw G(jω)", "yes (full)", None, t_dd_solve,
        ),
        "D_sparse": method_record(
            "Direct sparse LU", t_ds_build, t_ds_solve,
            N_FREQ, n_dof, "raw G(jω)", "yes (full)", err_ds, t_dd_solve,
        ),
        "PW": method_record(
            "PolyQEP + per-segment SOAR (polynomial G online)",
            t_pw_build, t_pw_solve,
            N_FREQ, M_PER_SEG * N_SEG, "polynomial fit", "yes",
            err_pw, t_dd_solve,
            extras={"m_per_seg": M_PER_SEG, "n_segments": N_SEG},
        ),
        "PW-R": method_record(
            "PW-Refined: SOAR basis + raw G online [NEW]",
            t_pwr_build, t_pwr_solve,
            N_FREQ, m_pwr, "raw G(jω)", "yes (from QEP offline)",
            err_pwr, t_dd_solve,
            extras={
                "m_per_seg": M_PER_SEG,
                "n_segments": N_SEG,
                "m_reduced_per_segment": pwr_profile["m_reduced_per_segment"],
            },
        ),
        "PW-RH": method_record(
            "PW-Refined Hybrid: SOAR + modal eigenvec augment [NEW]",
            t_pwrh_build, t_pwrh_solve,
            N_FREQ, m_pwrh, "raw G(jω)", "yes (basis-embedded)",
            err_pwrh, t_dd_solve,
            extras={
                "m_soar_per_seg": M_SOAR_HYBRID,
                "n_modal_augment": N_MODAL_AUGMENT,
                "n_segments": N_SEG,
                "m_reduced_per_segment": pwrh_profile["m_reduced_per_segment"],
            },
        ),
        "PW-RG": method_record(
            "PW-Refined Global: cross-segment union-SVD [NEW]",
            t_pwrg_build, t_pwrg_solve,
            N_FREQ, m_pwrg, "raw G(jω)", "yes (basis-embedded, global)",
            err_pwrg, t_dd_solve,
            extras={
                "m_soar_per_seg": M_PER_SEG,   # pure SOAR-20 recipe
                "n_modal_augment": 0,
                "n_segments": N_SEG,
                "source_dims_per_segment": pwrg_profile["source_dims_per_segment"],
                "source_dims_sum": pwrg_profile["source_dims_sum"],
            },
        ),
        "XS": method_record(
            "Xie 2018 single-point (Taylor + R2)",
            t_xs_build, t_xs_solve,
            N_FREQ, m_xs, "Taylor + R2", "no",
            err_xs, t_dd_solve,
        ),
        "XA": method_record(
            "Xie 2018 adaptive (Taylor-Krylov, raw G online)",
            t_xa_build, t_xa_solve,
            N_FREQ, m_xa, "raw G(jω)", "no",
            err_xa, t_dd_solve,
            extras={
                "n_expansion_points": n_pts,
            },
        ),
    }

    # 4-axis diagnostic output
    print_table_v3(records, N_FREQ)

    # ------------------------------------------------------------------
    # 11. Key reframe — do PW-R / PW-RH reach the target?
    # ------------------------------------------------------------------
    print("\n[Reframe] Accuracy floor by online G representation:")
    print(f"  - PW    (polynomial G online): err_max = {err_pw.max():.3f}%   ← fit-residual floor")
    print(f"  - PW-R  (raw G online):        err_max = {err_pwr.max():.4f}%  ← fit floor removed")
    print(f"  - PW-RH (raw G + modal aug):   err_max = {err_pwrh.max():.4f}%  ← further improvement")
    print(f"  - PW-RG (raw G + cross-seg union-SVD): err_max = {err_pwrg.max():.4f}%  ← broadband merge")
    print(f"  - XA    (raw G via Krylov):    err_max = {err_xa.max():.4f}%   ← reference target")
    pwr_vs_pw_ratio = (err_pw.max() / max(err_pwr.max(), 1e-9))
    pwr_vs_xa_ratio = (err_pwr.max() / max(err_xa.max(), 1e-12))
    pwrh_vs_xa_ratio = (err_pwrh.max() / max(err_xa.max(), 1e-12))
    pwrg_vs_xa_ratio = (err_pwrg.max() / max(err_xa.max(), 1e-12))
    print(f"\n  PW-R accuracy gain: {pwr_vs_pw_ratio:.0f}× improvement over PW")
    print(f"  PW-R vs XA       : {pwr_vs_xa_ratio:.2f}× (relative to XA)  "
          f"{'(PW-R better)' if pwr_vs_xa_ratio < 1 else '(XA better)'}")
    print(f"  PW-RH vs XA      : {pwrh_vs_xa_ratio:.2f}× (relative to XA)  "
          f"{'(PW-RH better)' if pwrh_vs_xa_ratio < 1 else '(XA better)'}")
    print(f"  PW-RG vs XA      : {pwrg_vs_xa_ratio:.2f}× (relative to XA)  "
          f"{'(PW-RG better)' if pwrg_vs_xa_ratio < 1 else '(XA better)'}")
    print(f"\n  ROM dim comparison: PW={M_PER_SEG*N_SEG}, PW-R={m_pwr}, PW-RH={m_pwrh}, "
          f"PW-RG={m_pwrg}, XS={m_xs}, XA={m_xa}")
    print(f"  Modal info: PW/PW-R/PW-RH/PW-RG = yes,  XS/XA = no")

    # ------------------------------------------------------------------
    # 12. Save
    # ------------------------------------------------------------------
    base = _ROOT
    json_path = base / "data" / "cmame_f1_xie2018_summary_v3.json"
    fig_path = base / "figures" / "cmame_f1_frf_comparison_v3.pdf"

    summary = {
        "benchmark": {
            "mesh": "Amichi 2010 §5.1, 8×8, simply_supported_w (471 free DOF)",
            "freq_range_hz": [F_MIN_HZ, F_MAX_HZ],
            "n_freq": N_FREQ,
            "n_segments": N_SEG,
            "m_per_seg_baseline": M_PER_SEG,
        },
        "shared_offline_costs_ms": {
            "g_fit_and_segment_assemble": t_fit_assemble * 1e3,
            "qep_per_segment_solve": t_qep * 1e3,
            "effective_eigenvalue_counts": eff_counts,
        },
        "methods": records,
        "reframe_summary": {
            "claim": (
                "The real reason XA came out 4000x more accurate than PW in v2 is "
                "that xie2018_rom._adaptive_frf_from_reduced uses the original "
                "G(jω) directly during online evaluation. This is a plain Krylov "
                "subspace projection, not the model order reduction (Taylor truncation) of Xie 2018."
            ),
            "fair_metric_axes": [
                "Accuracy (err_max/mean vs Direct dense)",
                "ROM dim m (segment-summed)",
                "Per-freq online cost [ms]",
                "Modal information availability (yes/partial/no)",
            ],
            "new_methods_pw_r_family": {
                "PW-R": (
                    "Piecewise SOAR basis (built with polynomial G) + raw G(jω) online. "
                    "Q^H K_base Q, Q^H K_shear Q, Q^H M Q reduced operator pieces are "
                    "precomputed. Modal info preserved."
                ),
                "PW-RH": (
                    "PW-R + per-segment effective eigenvectors augmented via "
                    "thin SVD orthogonalization. Total dim ≈ m_soar + n_modal."
                ),
                "PW-RG": (
                    "Cross-segment union-SVD of per-segment hybrid bases → single broadband "
                    "basis Q_global. No online segment dispatch needed. A variant of the union-SVD "
                    "technique of Xie XA applied to SOAR+modal hybrid source bases."
                ),
            },
            "pw_r_vs_pw_ratio": float(pwr_vs_pw_ratio),
            "pw_r_vs_xa_ratio": float(pwr_vs_xa_ratio),
            "pw_rh_vs_xa_ratio": float(pwrh_vs_xa_ratio),
            "pw_rg_vs_xa_ratio": float(pwrg_vs_xa_ratio),
            "goal_achieved": bool(
                err_pwr.max() < err_xa.max()
                or err_pwrh.max() < err_xa.max()
                or err_pwrg.max() < err_xa.max()
            ),
        },
        "xa_diagnostic": {
            "source": "baselines/xie2018_rom.py::_adaptive_frf_from_reduced",
            "key_line": "G_val = complex(G_func(s))   # raw G(jω) used in online operator",
            "implication": (
                "XA uses the Taylor polynomial only to generate the SOAR basis. The actual reduced "
                "operator is Q^H[K_e + G_raw(jω) K_v - ω² M]Q with raw G → no fit-residual floor."
            ),
        },
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\n  JSON v3 saved: {json_path}")

    # Figure
    method_curves = {
        "D_dense": Y_d,
        "D_sparse": Y_ds,
        "PW": Y_pw,
        "PW-R": Y_pwr,
        "PW-RG": Y_pwrg,
        "XS": Y_xs,
        "XA": Y_xa,
    }
    method_errs = {
        "PW": err_pw,
        "PW-R": err_pwr,
        "PW-RG": err_pwrg,
        "XS": err_xs,
        "XA": err_xa,
    }
    plot_frf_v3(fig_path, omegas, Y_d, method_curves, method_errs)

    print("\nDone.")


if __name__ == "__main__":
    main()
