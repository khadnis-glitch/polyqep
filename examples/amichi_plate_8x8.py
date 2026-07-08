"""End-to-end PolyQEP / PW-R / PW-RG pipeline on the Amichi (2010) sandwich
plate (8x8 mesh, free boundary), the strict-affine benchmark of Section 5.1
of the paper.

Pipeline
--------
1. Assemble the plate and split the stiffness into the affine decomposition
   K(w) = K_base + G(w) K_shear.
2. Sample the complex core shear modulus G(f) from DMA-type anchors and fit
   Re[G], Im[G] with a C0/C1-continuous piecewise-quadratic spline on shared
   segment boundaries.
3. Assemble the omega-independent complex triplet (M*, C*, K*) per segment
   and solve one complex QEP per segment; select the effective eigenvalues.
4. Build the PW-R (per-segment SOAR) and PW-RG (global hybrid-basis) ROMs
   and evaluate the FRF online from the raw G(jw) via the affine identity.
5. Compare against the dense direct solution (per-frequency LU with raw G).

Run from the repository root after ``pip install -e .``:

    python examples/amichi_plate_8x8.py
"""
from __future__ import annotations

import time

import numpy as np
import scipy.linalg as spla

from polyqep import (
    assemble_complex_per_interval,
    solve_qep_per_interval,
    build_pw_refined_roms_profiled,
    pw_refined_response,
    pw_refined_global_response,
)
from polyqep.fitting import QuadraticSpline, fit_joint_spline
from polyqep.models import (
    SandwichPlateParameters,
    assemble_global,
    assemble_K_base_and_shear,
    apply_boundary_conditions,
)


# ---------------------------------------------------------------------------
# Model: Amichi & Atalla (2010) symmetric sandwich plate, paper Section 5.1
# ---------------------------------------------------------------------------
def paper_5_1_params(nx: int = 8, ny: int = 8) -> SandwichPlateParameters:
    return SandwichPlateParameters(
        Lx=0.3048, Ly=0.3048,
        h1=0.475e-3, h2=0.035e-3, h3=0.475e-3,
        E1=2.1e11, E3=2.1e11,
        nu1=0.26, nu3=0.26,
        rho1=7780, rho3=7780,
        rho2=1134, nu2=0.45,
        E2_in_plane=1.0e7,
        nx=nx, ny=ny,
    )


# ---------------------------------------------------------------------------
# Core modulus: smooth log-frequency interpolation of DMA-type anchors
# ---------------------------------------------------------------------------
F_ANCHORS = np.array([10.0, 50.0, 100.0, 200.0, 400.0, 700.0, 1000.0])
E2_ANCHORS = np.array([4.5e6, 5.0e6, 6.0e6, 8.0e6, 1.4e7, 1.8e7, 2.0e7])
ETA_ANCHORS = np.array([1.40, 1.36, 1.34, 1.30, 1.22, 1.18, 1.15])
NU2 = 0.45


def G_complex_at_freq(freq_hz: float) -> complex:
    """Complex core shear modulus G(f) from the DMA-type anchor table."""
    log_f = np.log(np.clip(freq_hz, 1.0, 2000.0))
    log_fa = np.log(F_ANCHORS)
    E2 = np.interp(log_f, log_fa, E2_ANCHORS)
    eta = np.interp(log_f, log_fa, ETA_ANCHORS)
    G = E2 / (2.0 * (1.0 + NU2))
    return G * (1.0 + 1j * eta)


def G_of_s(s: complex) -> complex:
    """Raw G(s) at s = jw, as required by the PW-Refined online stage."""
    freq_hz = abs(s.imag) / (2.0 * np.pi)
    return G_complex_at_freq(freq_hz)


def fit_G_polynomial(
    omega: np.ndarray,
    G_re: np.ndarray,
    G_im: np.ndarray,
) -> tuple[QuadraticSpline, QuadraticSpline, np.ndarray]:
    """Fit Re[G] and Im[G] with shared 5-segment breakpoints (paper setup)."""
    f_brk = np.array([1.0, 50.0, 150.0, 350.0, 700.0, 1500.0])
    boundaries = 2.0 * np.pi * f_brk[(f_brk >= omega[0] / (2 * np.pi))
                                     & (f_brk <= omega[-1] / (2 * np.pi))]
    boundaries = np.unique(np.concatenate([[omega[0]], boundaries, [omega[-1]]]))
    spline_re, spline_im = fit_joint_spline(omega, G_re, G_im, boundaries)
    return spline_re, spline_im, boundaries


def main() -> None:
    # -- 1. model assembly and affine split ---------------------------------
    params = paper_5_1_params(nx=8, ny=8)
    K_base, K_shear = assemble_K_base_and_shear(params)
    M_full, _ = assemble_global(params, complex(1.0, 0.0))  # M independent of G

    # unit transverse load at the plate centre node (w DOF)
    F_full = np.zeros(M_full.shape[0], dtype=complex)
    cx, cy = params.nx // 2, params.ny // 2
    cw_full = 7 * (cx + cy * (params.nx + 1)) + 4
    F_full[cw_full] = 1.0

    M_red, _, F_red, free_dofs = apply_boundary_conditions(
        M_full, K_base.astype(complex), F_full, params, bc="free"
    )
    free_idx = np.asarray(free_dofs)
    idx_obs = int(np.where(free_idx == cw_full)[0][0])
    K_base_red = K_base[np.ix_(free_idx, free_idx)]
    K_shear_red = K_shear[np.ix_(free_idx, free_idx)]
    M_red = (M_red.real + M_red.real.T) / 2.0  # M is real symmetric
    n = M_red.shape[0]
    print(f"[model] Amichi 8x8 plate, free BC: n = {n} DOFs, "
          f"observation DOF (reduced) = {idx_obs}")

    # -- 2. modulus sampling and piecewise-quadratic fit ---------------------
    f_hz = np.linspace(1.0, 1500.0, 400)
    omega_s = 2.0 * np.pi * f_hz
    G_samples = np.array([G_complex_at_freq(f) for f in f_hz])
    spline_re, spline_im, boundaries = fit_G_polynomial(
        omega_s, G_samples.real, G_samples.imag)
    print(f"[fit]   {len(spline_re.segments)} segments, boundaries at "
          f"{np.round(boundaries / (2 * np.pi), 1)} Hz")

    # -- 3. per-segment complex QEP ------------------------------------------
    interval_mats = assemble_complex_per_interval(
        spline_re, spline_im, M_red, K_base_red, K_shear_red)
    t0 = time.perf_counter()
    interval_sols = solve_qep_per_interval(interval_mats, damping_tol=1e-6)
    t_qep = time.perf_counter() - t0
    n_eff_total = sum(s["lams_effective"].size for s in interval_sols)
    print(f"[qep]   {len(interval_sols)} segment QEPs solved in "
          f"{t_qep:.2f} s, {n_eff_total} effective eigenvalues")
    for s in interval_sols:
        f_lo = s["omega_lower"] / (2 * np.pi)
        f_hi = s["omega_upper"] / (2 * np.pi)
        print(f"        segment {s['interval_idx']}: "
              f"[{f_lo:7.1f}, {f_hi:7.1f}] Hz -> "
              f"{s['lams_effective'].size} effective")

    # -- 4. PW-R and PW-RG reduced-order models ------------------------------
    omegas = 2.0 * np.pi * np.linspace(5.0, 500.0, 200)

    # np.errstate: the SOAR builder probes for basis breakdown internally,
    # which emits benign overflow/invalid warnings on the discarded vectors.
    with np.errstate(all="ignore"):
        roms_pwr, prof_pwr = build_pw_refined_roms_profiled(
            K_base_red, K_shear_red, M_red, F_red, interval_sols,
            variant="soar", m_per_seg=20)
    with np.errstate(all="ignore"):
        Y_pwr = pw_refined_response(omegas, roms_pwr, G_of_s, idx_obs)
    print(f"[pw-r]  per-segment SOAR: m = {prof_pwr['m_reduced_per_segment']}, "
          f"build {prof_pwr['build_total_s']:.2f} s")

    with np.errstate(all="ignore"):
        rom_pwrg, prof_pwrg = build_pw_refined_roms_profiled(
            K_base_red, K_shear_red, M_red, F_red, interval_sols,
            variant="global", m_soar_per_seg=15, n_modal_augment=5)
    with np.errstate(all="ignore"):
        Y_pwrg = pw_refined_global_response(omegas, rom_pwrg, G_of_s, idx_obs)
    print(f"[pw-rg] global hybrid basis: m = {rom_pwrg.m_reduced}, "
          f"build {prof_pwrg['build_total_s']:.2f} s")

    # -- 5. dense direct reference (raw G, per-frequency LU) ------------------
    t0 = time.perf_counter()
    Y_ref = np.zeros_like(Y_pwr)
    for i, w in enumerate(omegas):
        G_raw = G_of_s(1j * w)
        D = K_base_red + G_raw * K_shear_red - (w ** 2) * M_red
        Y_ref[i] = spla.solve(D, F_red)[idx_obs]
    t_direct = time.perf_counter() - t0

    err_pwr = np.max(np.abs(Y_pwr - Y_ref) / np.max(np.abs(Y_ref)))
    err_pwrg = np.max(np.abs(Y_pwrg - Y_ref) / np.max(np.abs(Y_ref)))
    print(f"[check] dense direct reference: {t_direct:.2f} s "
          f"for {omegas.size} frequencies (n = {n})")
    print(f"[check] max normalized FRF error, 5-500 Hz: "
          f"PW-R = {err_pwr:.2e}, PW-RG = {err_pwrg:.2e}")

    assert err_pwr < 1e-4 and err_pwrg < 1e-9, "ROM accuracy check failed"
    print("[ok]    both ROMs reproduce the dense direct FRF")

    # -- optional plot --------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    f_plot = omegas / (2 * np.pi)
    ax.semilogy(f_plot, np.abs(Y_ref), "k-", lw=2, label="Direct (dense LU)")
    ax.semilogy(f_plot, np.abs(Y_pwr), "--", lw=1.2, label="PW-R")
    ax.semilogy(f_plot, np.abs(Y_pwrg), ":", lw=1.6, label="PW-RG")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|FRF| [m/N]")
    ax.legend()
    fig.tight_layout()
    fig.savefig("examples/amichi_plate_8x8_frf.png", dpi=150)
    print("[plot]  examples/amichi_plate_8x8_frf.png written")


if __name__ == "__main__":
    main()
