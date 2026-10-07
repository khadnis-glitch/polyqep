"""Additional studies — full-order claims of the manuscript, regenerated in one place.

1. Affine decomposition residual
       ||K(G) - K_base - G K_shear||_inf / ||K(G)||_inf
   on the 28-DOF element (1x1 mesh) and on the 8x8 mesh, five moduli.
2. Piecewise-quadratic fit residuals of G (baseline partition, the submitted
   400-point uniform sampling of 1--1500 Hz).
3. Free-free 8x8 plate (567 DOF): fitted-modulus full-order solve against the
   raw-modulus direct solve (200 frequencies, 5--500 Hz, peak band), the
   per-segment effective-eigenvalue counts, and the lowest effective mode.
4. Simply supported 8x8 plate (471 DOF), first segment: eigenpairs passing the
   relative-residual filter 1e-3 and the modal sum built from them against the
   fitted-modulus direct solve (80 frequencies, 5--50 Hz), with its run time.

Run::

    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_fullorder_checks.py

Output: ``data/rev_fullorder_checks.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rev_bench_rom import FIT_BAND_HZ, N_FIT_SAMPLES, N_SEG, direct_fitted
from rev_env import environment_record, timed_repeats
from rev_modulus_data import representative

from cmame_e2_segmentwise_rom import setup_amichi_8x8
from phase3_amichi_polynomial_qep import (
    assemble_complex_per_interval_amichi,
    evaluate_spline_at,
    fit_G_polynomial,
    paper_5_1_params,
    sample_G_omega,
    solve_qep_per_interval,
)
from phase3_f4_residual_and_ss import modal_sum_filtered, solve_qep_with_residual
from polyqep.models.sandwich_plate_amichi2010 import (
    assemble_K_base_and_shear,
    assemble_global,
)

_ROOT = Path(__file__).resolve().parent.parent
SEGMENT_EDGES_HZ = (1.0, 50.0, 150.0, 350.0, 700.0, 1500.0)


def affine_residuals(nx):
    params = paper_5_1_params(nx=nx, ny=nx)
    K_base, K_shear = assemble_K_base_and_shear(params)
    out = []
    for f in (10.0, 50.0, 200.0, 500.0, 1000.0):
        G = complex(representative(f)[0])
        K = assemble_global(params, G)[1]
        num = np.linalg.norm(K - K_base - G * K_shear, ord=np.inf)
        out.append(float(num / np.linalg.norm(K, ord=np.inf)))
    return dict(n_dof=int(K_base.shape[0]), residual_min=min(out),
                residual_max=max(out), residuals=out)


def fit_residuals():
    omega, G_re, G_im = sample_G_omega(*FIT_BAND_HZ, N_FIT_SAMPLES)
    spl_re, spl_im, _ = fit_G_polynomial(omega, G_re, G_im, n_segments=N_SEG)
    err_re = np.abs(G_re - evaluate_spline_at(spl_re, omega)) / np.abs(G_re) * 100.0
    err_im = np.abs(G_im - evaluate_spline_at(spl_im, omega)) / np.abs(G_im) * 100.0
    f = omega / (2.0 * np.pi)
    per_segment = []
    for lo, hi in zip(SEGMENT_EDGES_HZ[:-1], SEGMENT_EDGES_HZ[1:]):
        m = (f >= lo) & (f <= hi)
        per_segment.append(dict(band_hz=[lo, hi], n_samples=int(m.sum()),
                                re_max=float(err_re[m].max()),
                                im_max=float(err_im[m].max())))
    band = (f >= 5.0) & (f <= 500.0)
    return (spl_re, spl_im), dict(
        n_samples=int(omega.size), sampling="uniform in frequency",
        re_max=float(err_re.max()), re_mean=float(err_re.mean()),
        im_max=float(err_im.max()), im_mean=float(err_im.mean()),
        re_max_5_500=float(err_re[band].max()), im_max_5_500=float(err_im[band].max()),
        re_mean_5_500=float(err_re[band].mean()), im_mean_5_500=float(err_im[band].mean()),
        per_segment=per_segment)


def raw_direct(M, K_base, K_shear, F, obs, f_hz):
    K_base_c, K_shear_c, F_c = (K_base.astype(complex), K_shear.astype(complex),
                                F.astype(complex))
    return np.array([np.linalg.solve(
        K_base_c + g * K_shear_c - (2 * np.pi * f) ** 2 * M, F_c)[obs]
        for f, g in zip(f_hz, representative(f_hz))])


def peak_band(Y, Y_ref):
    mask = np.abs(Y_ref) > 0.01 * np.abs(Y_ref).max()
    rel = np.abs(Y - Y_ref) / np.abs(Y_ref) * 100.0
    return dict(peak_band_max_pct=float(rel[mask].max()),
                peak_band_mean_pct=float(rel[mask].mean()),
                all_max_pct=float(rel.max()), all_mean_pct=float(rel.mean()))


def free_free(splines):
    _, M, K_base, K_shear, F, obs = setup_amichi_8x8(bc="free")
    f_eval = np.linspace(5.0, 500.0, 200)
    mats = assemble_complex_per_interval_amichi(*splines, M, K_base, K_shear)
    Y_ref = raw_direct(M, K_base, K_shear, F, obs, f_eval)
    Y_fit = direct_fitted(mats, F.astype(complex), obs, 2 * np.pi * f_eval)
    sols = solve_qep_per_interval(mats, damping_tol=1e-6)
    counts = [int(s["lams_effective"].size) for s in sols]
    eff_all = np.concatenate([s["lams_effective"] for s in sols])
    return dict(n_free_dof=int(M.shape[0]),
                eigenvalues_per_segment=int(sols[0]["lams_all"].size),
                effective_counts=counts,
                lowest_effective_mode_hz=float(np.sort(eff_all.imag)[0] / (2 * np.pi)),
                fitted_vs_raw_direct=peak_band(Y_fit, Y_ref))


def simply_supported(splines):
    _, M, K_base, K_shear, F, obs = setup_amichi_8x8(bc="simply_supported_w")
    F_c = F.astype(complex)
    mats = assemble_complex_per_interval_amichi(*splines, M, K_base, K_shear)
    f_eval = np.linspace(5.0, 500.0, 200)
    Y_ref = raw_direct(M, K_base, K_shear, F, obs, f_eval)
    Y_fit = direct_fitted(mats, F_c, obs, 2 * np.pi * f_eval)

    seg = mats[0]
    M_s, C_s, K_s = seg["M_star"], seg["C_star"], seg["K_star"]
    omega_80 = 2.0 * np.pi * np.linspace(5.0, 50.0, 80)
    Y_seg = np.array([np.linalg.solve(K_s + 1j * om * C_s - om * om * M_s, F_c)[obs]
                      for om in omega_80])
    eigvals, v_part, residuals = solve_qep_with_residual(M_s, C_s, K_s)
    finite = np.isfinite(residuals)
    (Y_modal, n_used), t_modal = timed_repeats(
        lambda: modal_sum_filtered(omega_80, eigvals, v_part, residuals,
                                   M_s, C_s, F_c, obs, abs_lam_min=0.1,
                                   residual_max=1e-3), 10)
    rel = np.abs(Y_modal - Y_seg) / np.abs(Y_seg)
    return dict(
        n_free_dof=int(M.shape[0]),
        fitted_vs_raw_direct_200freq=peak_band(Y_fit, Y_ref),
        segment1=dict(
            n_eigenvalues=int(eigvals.size),
            n_finite=int(finite.sum()),
            residual_filter=1e-3,
            n_passing_residual_filter=int((residuals[finite] < 1e-3).sum()),
            n_modes_used=int(n_used),
            modal_vs_fitted_direct_max_rel=float(rel.max()),
            modal_vs_fitted_direct_mean_rel=float(rel.mean()),
            modal_sum_time_s=t_modal["median_s"],
            modal_sum_includes="weights and 80-frequency pole sum"))


def main():
    env = environment_record()
    result = dict(environment=env)
    result["affine_residual_element"] = affine_residuals(1)
    result["affine_residual_8x8"] = affine_residuals(8)
    print("affine residual (element):", result["affine_residual_element"])
    print("affine residual (8x8)    :", {k: v for k, v in
                                         result["affine_residual_8x8"].items()
                                         if k != "residuals"})
    splines, result["fit_residuals"] = fit_residuals()
    print("fit residuals:", {k: v for k, v in result["fit_residuals"].items()
                             if k != "per_segment"})
    for seg in result["fit_residuals"]["per_segment"]:
        print("   ", seg)
    result["simply_supported"] = simply_supported(splines)
    print("simply supported:", result["simply_supported"])
    result["free_free"] = free_free(splines)
    print("free-free:", result["free_free"])
    out = _ROOT / "data" / "rev_fullorder_checks.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
