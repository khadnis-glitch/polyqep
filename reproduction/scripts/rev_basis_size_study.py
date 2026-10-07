"""Additional studies — PW-RG accuracy and online time against the basis size.

The PW-RG basis stacks the SOAR bases of the segments. This study varies the
per-segment SOAR dimension and the set of stacked segments (all five, or only
the four that intersect the 5--500 Hz reporting band) on the benchmark of the
seven-method comparison, and reports the basis size, the maximum relative FRF
error, and the online time, next to the adaptive Krylov reference (XA).

Output: ``data/rev_basis_size_study.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rev_bench_rom import (
    F_MAX_HZ, F_MIN_HZ, FIT_BAND_HZ, N_FIT_SAMPLES, N_FREQ, N_SEG, SVD_TOL,
    direct_dense, err_stats, online_affine_global,
)
from rev_env import environment_record, require_single_thread, timed_repeats

from cmame_e2_segmentwise_rom import setup_amichi_8x8
from cmame_f1_xie2018_comparison_v3 import make_G_func
from phase3_amichi_polynomial_qep import (
    G_complex_at_freq, assemble_complex_per_interval_amichi, fit_G_polynomial,
    sample_G_omega,
)
from polyqep.pw_refined import _orthonormalize_columns_svd, _project_operator_pieces
from polyqep.rom import build_soar_rom
from baselines.xie2018_rom import xie_adaptive_response

_ROOT = Path(__file__).resolve().parent.parent


def main():
    require_single_thread()
    env = environment_record()
    _, M, K_base, K_shear, F, obs = setup_amichi_8x8(bc="simply_supported_w")
    omegas = 2.0 * np.pi * np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)
    G_func = make_G_func(G_complex_at_freq)
    Y_ref = direct_dense(K_base.astype(complex), K_shear.astype(complex), M, F,
                         obs, omegas, np.linalg.solve)
    omega_fit, G_re, G_im = sample_G_omega(*FIT_BAND_HZ, N_FIT_SAMPLES)
    spl_re, spl_im, _ = fit_G_polynomial(omega_fit, G_re, G_im, n_segments=N_SEG)
    mats = assemble_complex_per_interval_amichi(spl_re, spl_im, M, K_base, K_shear)

    rows = []
    for label, seg_ids in (("all five segments", range(5)),
                           ("four in-band segments", range(4))):
        for m_seg in (6, 8, 10, 12, 15, 20, 25):
            Qs = [build_soar_rom(mats[i]["M_star"], mats[i]["C_star"], mats[i]["K_star"],
                                 0.5 * (mats[i]["omega_lower"] + mats[i]["omega_upper"]),
                                 m_seg, F).Q for i in seg_ids]
            Q = _orthonormalize_columns_svd(np.column_stack(Qs), tol_sv=SVD_TOL)
            p = _project_operator_pieces(Q, K_base, K_shear, M, F)
            Y, t = timed_repeats(lambda: online_affine_global(
                Q, p[0], p[1], p[2], p[3], G_func, obs, omegas), 5)
            row = dict(segments=label, m_per_segment=m_seg, m_global=int(Q.shape[1]),
                       per_freq_ms=t["median_s"] / N_FREQ * 1e3, **err_stats(Y_ref, Y))
            rows.append(row)
            print(f"{label:<24} m_seg={m_seg:<3} m_global={row['m_global']:<4} "
                  f"{row['per_freq_ms']:.4f} ms/freq  max err {row['max_pct']:.2e}%")

    L = np.zeros(M.shape[0], dtype=complex)
    L[obs] = 1.0
    for tol in (1e-2, 1e-4, 1e-6):
        for m_pt in (10, 15, 20, 30):
            Y, rom = xie_adaptive_response(K_base, K_shear, M, G_func, omegas, F, L,
                                           m_per_point=m_pt, tol=tol, max_points=10)
            Fr = rom.Q.conj().T @ F.astype(complex)
            Yx, t = timed_repeats(lambda: online_affine_global(
                rom.Q, rom.Ker, rom.Kvr, rom.Mr, Fr, G_func, obs, omegas), 5)
            row = dict(segments="XA", m_per_segment=m_pt, tolerance=tol,
                       m_global=int(rom.m_reduced),
                       n_points=len(rom.expansion_points),
                       per_freq_ms=t["median_s"] / N_FREQ * 1e3,
                       **err_stats(Y_ref, Yx))
            rows.append(row)
            print(f"XA tol={tol:.0e} m_per_point={m_pt:<3} points={row['n_points']} "
                  f"m={row['m_global']:<4} {row['per_freq_ms']:.4f} ms/freq  "
                  f"max err {row['max_pct']:.2e}%")

    with open(_ROOT / "data" / "rev_basis_size_study.json", "w") as fh:
        json.dump(dict(environment=env, rows=rows), fh, indent=2)


if __name__ == "__main__":
    main()
