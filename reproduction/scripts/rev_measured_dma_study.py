"""Additional studies — the comparison repeated on the digitized DMA curves.

The manuscript benchmark uses a seven-anchor representative modulus. This
script repeats the modulus-fit and FRF comparison on the pixel-digitized
curves of Fig. 7 of Amichi and Atalla (2010): 142 points, 75--639 Hz, which
carry the digitization scatter of a measured data set.

For each method the fitted modulus replaces G in the full-order solve
[K_base + G_fit(w) K_shear - w^2 M] x = F (identical to the augmented GHM/ADF
systems), and the result is compared with the direct solve on the digitized
modulus itself.

    PolyQEP     C0/C1 piecewise-quadratic fit, 1..6 segments (uniform and
                log-uniform partitions of the measured band)
    GHM, ADF    orders 1..6, the submitted 30-start bounded fit settings
    Prony floor convex non-negative least squares on a dense pole grid
    PW-R, PW-RG raw-modulus ROMs built on the fitted segment matrices

Run::

    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_measured_dma_study.py

Output: ``data/rev_measured_dma_study.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rev_causality_check import prony_blocks
from rev_env import environment_record
from rev_modulus_data import digitized, digitized_table
from scipy.optimize import nnls

from cmame_c1_ghm_adf_comparison import build_amichi_8x8_ss
from phase3_amichi_polynomial_qep import assemble_complex_per_interval_amichi
from baselines import adf_augmentation as adf
from baselines import ghm_augmentation as ghm
from polyqep.pw_refined import (
    PWRefinedSegmentROM,
    _orthonormalize_columns_svd,
    _project_operator_pieces,
)
from polyqep.rom import build_soar_rom
from polyqep.fitting import fit_joint_spline

_ROOT = Path(__file__).resolve().parent.parent
SEED, N_STARTS = 20260509, 30
ORDERS = (1, 2, 3, 4, 5, 6)
SEGMENTS = (1, 2, 3, 4, 5, 6)
M_PER_SEG, SVD_TOL = 20, 1e-10
F_EVAL = np.linspace(75.0, 500.0, 200)     # inside the measured band


def component_errors(G_fit, G):
    rel_re = np.abs(G_fit.real - G.real) / np.abs(G.real) * 100.0
    rel_im = np.abs(G_fit.imag - G.imag) / np.abs(G.imag) * 100.0
    return dict(re_max=float(rel_re.max()), re_mean=float(rel_re.mean()),
                im_max=float(rel_im.max()), im_mean=float(rel_im.mean()))


def frf_full_order(system, omega, G_values):
    K_base_c, K_shear_c, M_c, F_c, obs = system
    Y = np.zeros(omega.size, dtype=complex)
    for i, (om, G) in enumerate(zip(omega, G_values)):
        Y[i] = np.linalg.solve(K_base_c + G * K_shear_c - om * om * M_c, F_c)[obs]
    return Y


def frf_errors(Y, Y_ref):
    mask = np.abs(Y_ref) > 0.01 * np.abs(Y_ref).max()
    rel = np.abs(Y - Y_ref) / np.abs(Y_ref) * 100.0
    return dict(peak_band_max_pct=float(rel[mask].max()),
                peak_band_mean_pct=float(rel[mask].mean()),
                all_max_pct=float(rel.max()), all_mean_pct=float(rel.mean()))


def spline_values(spline_re, spline_im, omega):
    return np.asarray(spline_re(omega)) + 1j * np.asarray(spline_im(omega))


def main():
    env = environment_record()
    f_data, _, _ = digitized_table()
    G_data = digitized(f_data)
    omega_data = 2.0 * np.pi * f_data
    omega_eval = 2.0 * np.pi * F_EVAL
    G_eval = digitized(F_EVAL)

    K_base, K_shear, M, F, obs, _ = build_amichi_8x8_ss()
    system = (K_base.astype(complex), K_shear.astype(complex), M.astype(complex),
              F.astype(complex), obs)
    Y_ref = frf_full_order(system, omega_eval, G_eval)
    print(f"digitized DMA: {f_data.size} points {f_data[0]:.0f}-{f_data[-1]:.0f} Hz; "
          f"FRF grid {F_EVAL.size} points {F_EVAL[0]:.0f}-{F_EVAL[-1]:.0f} Hz; "
          f"n = {K_base.shape[0]}")

    rows = []

    # ---- PolyQEP piecewise-quadratic fits -----------------------------------
    for partition in ("uniform", "log"):
        for n_seg in SEGMENTS:
            edges = (np.linspace(f_data[0], f_data[-1], n_seg + 1)
                     if partition == "uniform"
                     else np.geomspace(f_data[0], f_data[-1], n_seg + 1))
            spl_re, spl_im = fit_joint_spline(omega_data, G_data.real, G_data.imag,
                                              2.0 * np.pi * edges)
            row = dict(method="PolyQEP", setting=f"{partition} partition",
                       order=n_seg, n_free_params=2 * (n_seg + 2),
                       fit=component_errors(
                           spline_values(spl_re, spl_im, omega_data), G_data),
                       frf=frf_errors(frf_full_order(
                           system, omega_eval,
                           spline_values(spl_re, spl_im, omega_eval)), Y_ref))
            rows.append(row)
            print(f"  PolyQEP {partition:<7} N_seg={n_seg}: fit Re/Im max "
                  f"{row['fit']['re_max']:.2f}/{row['fit']['im_max']:.2f}% mean "
                  f"{row['fit']['re_mean']:.2f}/{row['fit']['im_mean']:.2f}%  "
                  f"FRF {row['frf']['peak_band_max_pct']:.2f}/"
                  f"{row['frf']['peak_band_mean_pct']:.2f}%")

    # ---- GHM / ADF, submitted fit settings ----------------------------------
    for method in ("GHM", "ADF"):
        for n in ORDERS:
            rng = np.random.default_rng([SEED, 0 if method == "GHM" else 1, n])
            if method == "GHM":
                params, info = ghm.fit_ghm_parameters(
                    omega_data, G_data.real, G_data.imag, n_mini=n,
                    n_starts=N_STARTS, rng=rng)
                G_fit_data = ghm.ghm_modulus(omega_data, params)
                G_fit_eval = ghm.ghm_modulus(omega_eval, params)
                n_params = 3 * n + 1
            else:
                params, info = adf.fit_adf_parameters(
                    omega_data, G_data.real, G_data.imag, n_adf=n,
                    n_starts=N_STARTS, rng=rng)
                G_fit_data = adf.adf_modulus(omega_data, params)
                G_fit_eval = adf.adf_modulus(omega_eval, params)
                n_params = 2 * n + 1
            row = dict(method=method, setting="30-start bounded NLLS", order=n,
                       n_free_params=n_params,
                       objective=float(info["best_residual_norm"]),
                       objective_rms_rel_pct=float(
                           info["best_residual_norm"]
                           / np.sqrt(2 * omega_data.size) * 100.0),
                       fit=component_errors(G_fit_data, G_data),
                       frf=frf_errors(frf_full_order(system, omega_eval, G_fit_eval),
                                      Y_ref))
            rows.append(row)
            print(f"  {method} N={n}: rms {row['objective_rms_rel_pct']:.2f}% fit "
                  f"Re/Im max {row['fit']['re_max']:.2f}/{row['fit']['im_max']:.2f}% "
                  f"mean {row['fit']['re_mean']:.2f}/{row['fit']['im_mean']:.2f}%  "
                  f"FRF {row['frf']['peak_band_max_pct']:.2f}/"
                  f"{row['frf']['peak_band_mean_pct']:.2f}%")

    # ---- Prony-family floor (convex) ----------------------------------------
    pole_hz = np.logspace(np.log10(f_data[0]) - 4, np.log10(f_data[-1]) + 4, 73)
    A_re, A_im = prony_blocks(f_data, pole_hz)
    wgt = 1.0 / np.abs(G_data)
    x, _ = nnls(np.vstack([wgt[:, None] * A_re, wgt[:, None] * A_im]),
                np.concatenate([wgt * G_data.real, wgt * G_data.imag]),
                maxiter=50 * (pole_hz.size + 1))
    B_re, B_im = prony_blocks(F_EVAL, pole_hz)
    G_floor_data = A_re @ x + 1j * (A_im @ x)
    G_floor_eval = B_re @ x + 1j * (B_im @ x)
    res = np.concatenate([wgt * (G_floor_data.real - G_data.real),
                          wgt * (G_floor_data.imag - G_data.imag)])
    row = dict(method="Prony floor", setting="NNLS, 73 poles over +-4 decades",
               order=int(np.sum(x[1:] > 0)), n_free_params=None,
               objective_rms_rel_pct=float(
                   np.linalg.norm(res) / np.sqrt(res.size) * 100.0),
               fit=component_errors(G_floor_data, G_data),
               frf=frf_errors(frf_full_order(system, omega_eval, G_floor_eval), Y_ref))
    rows.append(row)
    print(f"  Prony floor: rms {row['objective_rms_rel_pct']:.2f}% fit Re/Im max "
          f"{row['fit']['re_max']:.2f}/{row['fit']['im_max']:.2f}%  FRF "
          f"{row['frf']['peak_band_max_pct']:.2f}/{row['frf']['peak_band_mean_pct']:.2f}%")

    # ---- raw-modulus ROMs on the measured data ------------------------------
    for n_seg in (3, 5):
        edges = np.linspace(f_data[0], f_data[-1], n_seg + 1)
        spl_re, spl_im = fit_joint_spline(omega_data, G_data.real, G_data.imag,
                                          2.0 * np.pi * edges)
        mats = assemble_complex_per_interval_amichi(spl_re, spl_im, M, K_base, K_shear)
        soar = [build_soar_rom(m["M_star"], m["C_star"], m["K_star"],
                               0.5 * (m["omega_lower"] + m["omega_upper"]),
                               M_PER_SEG, F) for m in mats]
        lowers = np.array([m["omega_lower"] for m in mats])
        roms = []
        for m, s in zip(mats, soar):
            p = _project_operator_pieces(s.Q, K_base, K_shear, M, F)
            roms.append(PWRefinedSegmentROM(m["omega_lower"], m["omega_upper"],
                                            s.Q, p[0], p[1], p[2], p[3]))
        Y_r = np.zeros(omega_eval.size, dtype=complex)
        for i, (om, G) in enumerate(zip(omega_eval, G_eval)):
            r = roms[int(np.clip(np.searchsorted(lowers, om, side="right") - 1,
                                 0, len(roms) - 1))]
            Y_r[i] = (r.Q @ np.linalg.solve(
                r.K_base_r + G * r.K_shear_r - om * om * r.M_r, r.F_r))[obs]
        Q_g = _orthonormalize_columns_svd(np.column_stack([s.Q for s in soar]),
                                          tol_sv=SVD_TOL)
        Kb, Ks, Mr, Fr = _project_operator_pieces(Q_g, K_base, K_shear, M, F)
        Y_g = np.array([(Q_g @ np.linalg.solve(Kb + G * Ks - om * om * Mr, Fr))[obs]
                        for om, G in zip(omega_eval, G_eval)])
        for name, Y, m_dim in (("PW-R", Y_r, M_PER_SEG), ("PW-RG", Y_g, Q_g.shape[1])):
            row = dict(method=name, setting=f"uniform partition, {n_seg} segments",
                       order=n_seg, m_online=int(m_dim), frf=frf_errors(Y, Y_ref))
            rows.append(row)
            print(f"  {name} ({n_seg} seg, m={m_dim}): FRF "
                  f"{row['frf']['peak_band_max_pct']:.3e}/"
                  f"{row['frf']['peak_band_mean_pct']:.3e}%")

    out = _ROOT / "data" / "rev_measured_dma_study.json"
    with open(out, "w") as fh:
        json.dump(dict(environment=env, data_band_hz=[float(f_data[0]), float(f_data[-1])],
                       n_data=int(f_data.size),
                       frf_grid_hz=[float(F_EVAL[0]), float(F_EVAL[-1]), int(F_EVAL.size)],
                       n_starts=N_STARTS, seed=SEED, rows=rows), fh, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
