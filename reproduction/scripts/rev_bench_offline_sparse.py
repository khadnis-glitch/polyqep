"""Additional studies — offline cost of the ROM build with sparse factorizations.

Companion of ``rev_bench_scaling.py``. The released SOAR routine factorizes
the shifted segment matrix with dense LU and guards it with a dense 2-norm
condition number; both are O(n^3). This script rebuilds the PW-R and PW-RG
operators with ``rev_soar_sparse`` (sparse LU, no condition-number SVD,
sparse projections) and reports the offline time and the resulting accuracy
against the sparse direct solve.

Run::

    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_bench_offline_sparse.py --nx 8

Output: ``data/rev_bench_offline_sparse_nx<NN>.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rev_bench_rom import (
    F_MAX_HZ, F_MIN_HZ, FIT_BAND_HZ, M_PER_SEG, N_FIT_SAMPLES, N_SEG, SVD_TOL,
    direct_sparse_native, err_stats, online_affine_global, online_pw_r,
)
from rev_bench_scaling import N_FREQ, setup_plate
from rev_env import environment_record, require_single_thread, timed_repeats
from rev_soar_sparse import (
    as_sparse, project_pieces_sparse, segment_matrices_sparse, soar_basis_sparse,
)

from cmame_f1_xie2018_comparison_v3 import make_G_func
from phase3_amichi_polynomial_qep import (
    G_complex_at_freq, fit_G_polynomial, sample_G_omega,
)
from polyqep.pw_refined import PWRefinedSegmentROM, _orthonormalize_columns_svd

_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="sparse offline build")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--n-offline", type=int, default=3)
    args = parser.parse_args()

    require_single_thread()
    env = environment_record()
    M, K_base, K_shear, F, obs, _ = setup_plate(args.nx)
    n = M.shape[0]
    Kb_s, Ks_s, M_s = as_sparse(K_base, K_shear, M)
    del K_base, K_shear
    omegas = 2.0 * np.pi * np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)
    G_func = make_G_func(G_complex_at_freq)
    F_c = F.astype(complex)
    Y_ref = direct_sparse_native(Kb_s, Ks_s, M_s, F_c, obs, omegas)

    stages = {}

    def stage_fit():
        omega_fit, G_re, G_im = sample_G_omega(*FIT_BAND_HZ, N_FIT_SAMPLES)
        return fit_G_polynomial(omega_fit, G_re, G_im, n_segments=N_SEG)

    (spl_re, spl_im, _), stages["g_fit"] = timed_repeats(
        stage_fit, args.n_offline, n_warmup=0)
    mats, stages["segment_assembly"] = timed_repeats(
        lambda: segment_matrices_sparse(spl_re, spl_im, M_s, Kb_s, Ks_s),
        args.n_offline, n_warmup=0)
    lowers = np.array([m["omega_lower"] for m in mats])
    Qs, stages["soar_bases"] = timed_repeats(
        lambda: [soar_basis_sparse(m["M_star"], m["C_star"], m["K_star"],
                                   0.5 * (m["omega_lower"] + m["omega_upper"]),
                                   M_PER_SEG, F) for m in mats],
        args.n_offline, n_warmup=0)

    def stage_project_r():
        out = []
        for m, Q in zip(mats, Qs):
            p = project_pieces_sparse(Q, Kb_s, Ks_s, M_s, F)
            out.append(PWRefinedSegmentROM(m["omega_lower"], m["omega_upper"],
                                           Q, p[0], p[1], p[2], p[3]))
        return out

    roms, stages["pw_r_projection"] = timed_repeats(
        stage_project_r, args.n_offline, n_warmup=0)
    Q_g, stages["pw_rg_svd_compression"] = timed_repeats(
        lambda: _orthonormalize_columns_svd(np.column_stack(Qs), tol_sv=SVD_TOL),
        args.n_offline, n_warmup=0)
    pieces, stages["pw_rg_projection"] = timed_repeats(
        lambda: project_pieces_sparse(Q_g, Kb_s, Ks_s, M_s, F),
        args.n_offline, n_warmup=0)

    Y_r = online_pw_r(roms, lowers, G_func, obs, omegas)
    Y_g = online_affine_global(Q_g, pieces[0], pieces[1], pieces[2], pieces[3],
                               G_func, obs, omegas)
    off_ms = {k: v["median_s"] * 1e3 for k, v in stages.items()}
    shared = off_ms["g_fit"] + off_ms["segment_assembly"] + off_ms["soar_bases"]
    result = dict(
        environment=env, nx=args.nx, n_free_dof=int(n), n_freq=N_FREQ,
        offline_repeats=args.n_offline, offline_stage_ms=off_ms,
        offline_total_ms={
            "PW-R": shared + off_ms["pw_r_projection"],
            "PW-RG": shared + off_ms["pw_rg_svd_compression"]
            + off_ms["pw_rg_projection"]},
        m_per_segment=[int(Q.shape[1]) for Q in Qs], m_pw_rg=int(Q_g.shape[1]),
        error_vs_sparse_direct={"PW-R": err_stats(Y_ref, Y_r),
                                "PW-RG": err_stats(Y_ref, Y_g)},
    )
    out = _ROOT / "data" / f"rev_bench_offline_sparse_nx{args.nx:02d}.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"nx={args.nx} n={n}: offline PW-R {result['offline_total_ms']['PW-R']:.1f} ms, "
          f"PW-RG {result['offline_total_ms']['PW-RG']:.1f} ms (m={result['m_pw_rg']}); "
          f"err PW-R {result['error_vs_sparse_direct']['PW-R']['max_pct']:.2e}%, "
          f"PW-RG {result['error_vs_sparse_direct']['PW-RG']['max_pct']:.2e}%; "
          f"stages {({k: round(v, 1) for k, v in off_ms.items()})}")


if __name__ == "__main__":
    main()
