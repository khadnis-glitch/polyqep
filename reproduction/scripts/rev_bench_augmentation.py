"""Additional studies — comparison with the GHM and ADF augmentations (Table 4).

Benchmark: Amichi 8x8, hard simple support (n = 471), 200 frequencies in
5--500 Hz, representative core modulus; GHM (N = 3, 5) and ADF (N = 2, 5) are
fitted with the submitted settings (200 uniform samples of 1--1500 Hz,
30-start bounded NLLS, shared random stream).

Regenerated on the corrected element with the timing protocol of
``rev_bench_rom.py``: single thread, median over repeats, and one LAPACK
driver (``numpy.linalg.solve``) for every per-frequency solve. The rows are

    Direct              full-order solve with the raw modulus (reference)
    GHM / ADF           unreduced augmented systems
    PolyQEP modal       closed-form modal expansion over the complete spectrum
                        of each segment QEP (fitted modulus)
    PW-R, PW-RG         raw-modulus ROMs

Offline work is reported separately for every row: parameter fit and
augmented assembly (GHM/ADF); fit, segment QEP solves and residue weights
(modal expansion); fit, SOAR bases, projections, SVD compression (ROMs).

Run::

    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_bench_augmentation.py

Output: ``data/rev_bench_augmentation.json`` and ``..._frf.npz``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rev_bench_rom import (
    M_PER_SEG, N_SEG, SVD_TOL, direct_fitted, online_affine_global, online_pw_r,
)
from rev_env import environment_record, require_single_thread, timed_repeats
from rev_modulus_data import representative

from cmame_c1_ghm_adf_comparison import build_amichi_8x8_ss
from cmame_f1_xie2018_comparison_v3 import make_G_func
from phase3_amichi_polynomial_qep import (
    G_complex_at_freq,
    assemble_complex_per_interval_amichi,
    fit_G_polynomial,
    modal_weights_blockwise,
    sample_G_omega,
    solve_qep_per_interval,
)
from baselines import adf_augmentation as adf
from baselines import ghm_augmentation as ghm
from polyqep.pw_refined import (
    PWRefinedSegmentROM,
    _orthonormalize_columns_svd,
    _project_operator_pieces,
)
from polyqep.rom import build_soar_rom

_ROOT = Path(__file__).resolve().parent.parent
SEED, N_STARTS = 20260509, 30
F_EVAL = np.linspace(5.0, 500.0, 200)
FIT_SAMPLES = (1.0, 1500.0, 200)


def peak_band(Y, Y_ref):
    mask = np.abs(Y_ref) > 0.01 * np.abs(Y_ref).max()
    rel = np.abs(Y - Y_ref) / np.abs(Y_ref) * 100.0
    return dict(max_pct=float(rel[mask].max()), mean_pct=float(rel[mask].mean()))


def sweep_second_order(K, C, M, F, obs, omegas):
    Y = np.zeros(omegas.size, dtype=complex)
    for i, om in enumerate(omegas):
        Y[i] = np.linalg.solve(K - om * om * M + 1j * om * C, F)[obs]
    return Y


def sweep_first_order(A, drive, obs, omegas):
    eye = np.eye(A.shape[0], dtype=complex)
    Y = np.zeros(omegas.size, dtype=complex)
    for i, om in enumerate(omegas):
        Y[i] = np.linalg.solve(1j * om * eye - A, drive)[obs]
    return Y


def main():
    parser = argparse.ArgumentParser(description="GHM/ADF comparison benchmark")
    parser.add_argument("--n-aug", type=int, default=3,
                        help="measured repeats of the augmented sweeps")
    parser.add_argument("--n-online", type=int, default=10)
    args = parser.parse_args()

    require_single_thread()
    env = environment_record()
    K_base, K_shear, M, F, obs, _ = build_amichi_8x8_ss()
    n = K_base.shape[0]
    omegas = 2.0 * np.pi * F_EVAL
    K_base_c, K_shear_c, M_c = (K_base.astype(complex), K_shear.astype(complex),
                                M.astype(complex))
    F_c = F.astype(complex)
    G_eval = representative(F_EVAL)
    rows, curves = {}, {}

    def direct():
        return np.array([np.linalg.solve(K_base_c + g * K_shear_c - om * om * M_c,
                                         F_c)[obs] for om, g in zip(omegas, G_eval)])

    Y_ref, t = timed_repeats(direct, args.n_online)
    rows["Direct"] = dict(size=int(n), size_kind="full order",
                          per_freq_ms=t["median_s"] / omegas.size * 1e3,
                          offline_ms=0.0)
    curves["Direct"] = Y_ref
    print(f"Direct: {rows['Direct']['per_freq_ms']:.3f} ms/freq", flush=True)

    # ---- GHM / ADF -----------------------------------------------------------
    omega_fit, G_re, G_im = sample_G_omega(*FIT_SAMPLES)
    rng = np.random.default_rng(SEED)
    for method, order in (("GHM", 3), ("GHM", 5), ("ADF", 2), ("ADF", 5)):
        t0 = time.perf_counter()
        if method == "GHM":
            params, info = ghm.fit_ghm_parameters(
                omega_fit, G_re, G_im, n_mini=order, n_starts=N_STARTS, rng=rng)
            t_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            aug = ghm.assemble_ghm_augmented(K_base, K_shear, M, F, params)
            t_build = time.perf_counter() - t0
            K_a, C_a, M_a = (aug.K_aug.astype(complex), aug.C_aug.astype(complex),
                             aug.M_aug.astype(complex))
            F_a = np.asarray(aug.F_aug, dtype=complex)
            Y, ts = timed_repeats(
                lambda: sweep_second_order(K_a, C_a, M_a, F_a, obs, omegas),
                args.n_aug, n_warmup=0)
            size, n_params = int(M_a.shape[0]), 3 * order + 1
        else:
            params, info = adf.fit_adf_parameters(
                omega_fit, G_re, G_im, n_adf=order, n_starts=N_STARTS, rng=rng)
            t_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            ss = adf.assemble_adf_state_space(K_base, K_shear, M, F, params)
            t_build = time.perf_counter() - t0
            A_c = ss.A.astype(complex)
            drive = ss.B_in @ F_c
            Y, ts = timed_repeats(
                lambda: sweep_first_order(A_c, drive, ss.n_phys + obs, omegas),
                args.n_aug, n_warmup=0)
            size, n_params = int(A_c.shape[0]), 2 * order + 1
        name = f"{method} N={order}"
        rows[name] = dict(
            size=size, size_kind="augmented system", n_fit_params=n_params,
            fit=dict(re_max=info["rel_re_max"], re_mean=info["rel_re_mean"],
                     im_max=info["rel_im_max"], im_mean=info["rel_im_mean"]),
            per_freq_ms=ts["median_s"] / omegas.size * 1e3,
            offline_ms=(t_fit + t_build) * 1e3,
            offline_fit_ms=t_fit * 1e3, offline_assembly_ms=t_build * 1e3,
            frf=peak_band(Y, Y_ref))
        curves[name] = Y
        print(f"{name}: size {size} {rows[name]['per_freq_ms']:.1f} ms/freq "
              f"fit {info['rel_re_max']:.1f}/{info['rel_im_max']:.1f}% "
              f"FRF {rows[name]['frf']['max_pct']:.2f}/"
              f"{rows[name]['frf']['mean_pct']:.2f}% "
              f"offline {rows[name]['offline_ms']:.0f} ms", flush=True)

    # ---- PolyQEP: modal expansion over the complete segment spectrum ---------
    t0 = time.perf_counter()
    spl_re, spl_im, _ = fit_G_polynomial(omega_fit, G_re, G_im, n_segments=N_SEG)
    mats = assemble_complex_per_interval_amichi(spl_re, spl_im, M, K_base, K_shear)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    sols = solve_qep_per_interval(mats, damping_tol=1e-6)
    t_qep = time.perf_counter() - t0
    t0 = time.perf_counter()
    poles = []
    for s in sols:
        ok = np.isfinite(s["lams_all"])
        poles.append(modal_weights_blockwise(
            s["lams_all"][ok], s["vecs_all"][:, ok], s["matrices"]["M_star"],
            s["matrices"]["C_star"], F_c, obs))
    t_weights = time.perf_counter() - t0
    lowers = np.array([m["omega_lower"] for m in mats])
    eff_counts = [int(s["lams_effective"].size) for s in sols]
    del sols

    def modal():
        Y = np.zeros(omegas.size, dtype=complex)
        for i, om in enumerate(omegas):
            lam, w = poles[int(np.clip(np.searchsorted(lowers, om, side="right") - 1,
                                       0, len(poles) - 1))]
            Y[i] = np.sum(w / (1j * om - lam))
        return Y

    Y_modal, t = timed_repeats(modal, args.n_online)
    Y_fit = direct_fitted(mats, F_c, obs, omegas)
    f_fit = omega_fit / (2.0 * np.pi)
    G_fit_err = np.asarray(spl_re(omega_fit)) + 1j * np.asarray(spl_im(omega_fit))
    rows["PolyQEP modal expansion"] = dict(
        size=int(max(p[0].size for p in poles)),
        size_kind="poles per segment (2n eigenvalues before clustering)",
        poles_per_segment=[int(p[0].size) for p in poles],
        effective_counts=eff_counts, n_fit_params=2 * (N_SEG + 2),
        n_stored_coefficients=6 * N_SEG,
        fit=dict(
            re_max=float(np.max(np.abs(G_fit_err.real - G_re) / np.abs(G_re)) * 100),
            re_mean=float(np.mean(np.abs(G_fit_err.real - G_re) / np.abs(G_re)) * 100),
            im_max=float(np.max(np.abs(G_fit_err.imag - G_im) / np.abs(G_im)) * 100),
            im_mean=float(np.mean(np.abs(G_fit_err.imag - G_im) / np.abs(G_im)) * 100),
            evaluated_on=f"the {f_fit.size} fit samples, "
                         f"{f_fit[0]:.0f}-{f_fit[-1]:.0f} Hz"),
        per_freq_ms=t["median_s"] / omegas.size * 1e3,
        offline_ms=(t_fit + t_qep + t_weights) * 1e3,
        offline_fit_ms=t_fit * 1e3, offline_qep_ms=t_qep * 1e3,
        offline_weights_ms=t_weights * 1e3,
        frf=peak_band(Y_modal, Y_ref),
        frf_fitted_full_order=peak_band(Y_fit, Y_ref),
        modal_vs_fitted_full_order=peak_band(Y_modal, Y_fit))
    curves["PolyQEP modal expansion"] = Y_modal
    r = rows["PolyQEP modal expansion"]
    print(f"PolyQEP modal: poles/segment {r['poles_per_segment']} "
          f"{r['per_freq_ms']:.4f} ms/freq FRF {r['frf']['max_pct']:.2f}/"
          f"{r['frf']['mean_pct']:.2f}% (fitted full-order "
          f"{r['frf_fitted_full_order']['max_pct']:.2f}/"
          f"{r['frf_fitted_full_order']['mean_pct']:.2f}%) offline "
          f"{r['offline_ms']:.0f} ms", flush=True)

    # ---- raw-modulus ROMs ----------------------------------------------------
    G_func = make_G_func(G_complex_at_freq)
    t0 = time.perf_counter()
    soar = [build_soar_rom(m["M_star"], m["C_star"], m["K_star"],
                           0.5 * (m["omega_lower"] + m["omega_upper"]),
                           M_PER_SEG, F) for m in mats]
    roms = []
    for m, s in zip(mats, soar):
        p = _project_operator_pieces(s.Q, K_base, K_shear, M, F)
        roms.append(PWRefinedSegmentROM(m["omega_lower"], m["omega_upper"],
                                        s.Q, p[0], p[1], p[2], p[3]))
    t_r = time.perf_counter() - t0
    t0 = time.perf_counter()
    Q_g = _orthonormalize_columns_svd(np.column_stack([s.Q for s in soar]),
                                      tol_sv=SVD_TOL)
    pg = _project_operator_pieces(Q_g, K_base, K_shear, M, F)
    t_g = time.perf_counter() - t0
    Y_r, ts_r = timed_repeats(
        lambda: online_pw_r(roms, lowers, G_func, obs, omegas), args.n_online)
    Y_g, ts_g = timed_repeats(
        lambda: online_affine_global(Q_g, pg[0], pg[1], pg[2], pg[3], G_func,
                                     obs, omegas), args.n_online)
    rows["PW-R"] = dict(size=M_PER_SEG, size_kind="online basis per segment",
                        per_freq_ms=ts_r["median_s"] / omegas.size * 1e3,
                        offline_ms=(t_fit + t_r) * 1e3, frf=peak_band(Y_r, Y_ref))
    rows["PW-RG"] = dict(size=int(Q_g.shape[1]), size_kind="online basis",
                         per_freq_ms=ts_g["median_s"] / omegas.size * 1e3,
                         offline_ms=(t_fit + t_r + t_g) * 1e3,
                         frf=peak_band(Y_g, Y_ref))
    curves["PW-R"], curves["PW-RG"] = Y_r, Y_g
    for name in ("PW-R", "PW-RG"):
        print(f"{name}: m={rows[name]['size']} {rows[name]['per_freq_ms']:.4f} ms/freq "
              f"FRF {rows[name]['frf']['max_pct']:.2e}/{rows[name]['frf']['mean_pct']:.2e}% "
              f"offline {rows[name]['offline_ms']:.0f} ms", flush=True)

    out = _ROOT / "data" / "rev_bench_augmentation.json"
    with open(out, "w") as fh:
        json.dump(dict(environment=env, n_free_dof=int(n),
                       freq_grid_hz=[5.0, 500.0, int(F_EVAL.size)],
                       fit_samples_hz=list(FIT_SAMPLES), n_starts=N_STARTS, seed=SEED,
                       timing_protocol=dict(statistic="median",
                                            augmented_repeats=args.n_aug,
                                            online_repeats=args.n_online),
                       rows=rows), fh, indent=2)
    np.savez(_ROOT / "data" / "rev_bench_augmentation_frf.npz", f_hz=F_EVAL,
             **{k.replace(" ", "_").replace("=", "").replace("-", "_"): v
                for k, v in curves.items()})
    print(f"saved {out}")


if __name__ == "__main__":
    main()
