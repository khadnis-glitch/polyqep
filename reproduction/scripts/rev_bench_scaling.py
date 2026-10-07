"""Additional studies — mesh-scaling benchmark with dense and sparse direct baselines.

Regenerates the scalability statement of the manuscript (per-frequency
speedup of the raw-modulus ROMs as the mesh is refined) on the corrected
Amichi element, single thread, with

* two direct baselines: dense LU (``numpy.linalg.solve``) and sparse LU
  (SuperLU via ``scipy.sparse.linalg.splu``) on matrices that are kept sparse
  throughout (no dense-to-sparse conversion inside the frequency loop);
* the PW-R and PW-RG ROMs with the offline stages timed separately;
* the same online definition as ``rev_bench_rom.py`` (one sweep of
  per-frequency solves, operator assembly and full-field recovery included).

Each mesh runs in its own process so that memory is released between meshes::

    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_bench_scaling.py --nx 8

Output: ``data/rev_bench_scaling_nx<NN>.json`` per mesh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as spsparse

from rev_bench_rom import (
    F_MAX_HZ, F_MIN_HZ, FIT_BAND_HZ, M_PER_SEG, N_FIT_SAMPLES, N_SEG, SVD_TOL,
    direct_dense, direct_sparse_native, err_stats, online_affine_global,
    online_pw_r,
)
from rev_env import environment_record, require_single_thread, timed_repeats

from phase3_amichi_polynomial_qep import (
    G_complex_at_freq,
    assemble_complex_per_interval_amichi,
    fit_G_polynomial,
    paper_5_1_params,
    sample_G_omega,
)
from cmame_f1_xie2018_comparison_v3 import make_G_func
from polyqep.pw_refined import (
    PWRefinedSegmentROM,
    _orthonormalize_columns_svd,
    _project_operator_pieces,
)
from polyqep.rom import build_soar_rom
from polyqep.models.sandwich_plate_amichi2010 import (
    apply_boundary_conditions,
    assemble_K_base_and_shear,
    assemble_global,
)

_ROOT = Path(__file__).resolve().parent.parent
N_FREQ = 200


def setup_plate(nx: int, bc: str = "simply_supported_w"):
    """Amichi section-5.1 plate on an nx-by-nx mesh, centre excitation."""
    params = paper_5_1_params(nx=nx, ny=nx)
    K_base, K_shear = assemble_K_base_and_shear(params)
    M_full, _ = assemble_global(params, complex(1.0, 0.0))
    F_full = np.zeros(M_full.shape[0], dtype=complex)
    centre = 7 * (params.nx // 2 + (params.ny // 2) * (params.nx + 1)) + 4
    F_full[centre] = 1.0
    M_red, _, F_red, free = apply_boundary_conditions(
        M_full, K_base.astype(complex), F_full, params, bc=bc)
    free = np.asarray(free)
    M = (M_red.real + M_red.real.T) / 2.0
    return (M, K_base[np.ix_(free, free)], K_shear[np.ix_(free, free)], F_red,
            int(np.where(free == centre)[0][0]), int(M_full.shape[0]))


def main():
    parser = argparse.ArgumentParser(description="mesh-scaling benchmark")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--n-direct", type=int, default=1,
                        help="measured repeats of the direct sweeps")
    parser.add_argument("--n-online", type=int, default=10)
    parser.add_argument("--n-offline", type=int, default=1)
    parser.add_argument("--warmup-direct", type=int, default=0)
    args = parser.parse_args()

    require_single_thread()
    env = environment_record()
    M, K_base, K_shear, F, obs, n_full = setup_plate(args.nx)
    n = M.shape[0]
    omegas = 2.0 * np.pi * np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)
    G_func = make_G_func(G_complex_at_freq)
    F_c = F.astype(complex)
    print(f"mesh {args.nx}x{args.nx}: n_full = {n_full}, n_free = {n}", flush=True)

    Kb_s = spsparse.csc_matrix(K_base).astype(complex)
    Ks_s = spsparse.csc_matrix(K_shear).astype(complex)
    M_s = spsparse.csc_matrix(M).astype(complex)
    nnz = int((abs(Kb_s) + abs(Ks_s) + abs(M_s)).nnz)

    online, offline = {}, {}
    Y_sparse, online["D_sparse"] = timed_repeats(
        lambda: direct_sparse_native(Kb_s, Ks_s, M_s, F_c, obs, omegas),
        args.n_direct, n_warmup=args.warmup_direct)
    print(f"  sparse LU  {online['D_sparse']['median_s']/N_FREQ*1e3:.3f} ms/freq",
          flush=True)
    K_base_c, K_shear_c = K_base.astype(complex), K_shear.astype(complex)
    Y_dense, online["D_dense_numpy"] = timed_repeats(
        lambda: direct_dense(K_base_c, K_shear_c, M, F, obs, omegas,
                             np.linalg.solve),
        args.n_direct, n_warmup=args.warmup_direct)
    print(f"  dense LU   {online['D_dense_numpy']['median_s']/N_FREQ*1e3:.3f} ms/freq",
          flush=True)

    # ---- offline stages ------------------------------------------------------
    def stage_fit():
        omega_fit, G_re, G_im = sample_G_omega(*FIT_BAND_HZ, N_FIT_SAMPLES)
        return fit_G_polynomial(omega_fit, G_re, G_im, n_segments=N_SEG)

    (spline_re, spline_im, _), offline["g_fit"] = timed_repeats(
        stage_fit, args.n_offline, n_warmup=0)
    mats, offline["segment_assembly"] = timed_repeats(
        lambda: assemble_complex_per_interval_amichi(
            spline_re, spline_im, M, K_base, K_shear),
        args.n_offline, n_warmup=0)
    lowers = np.array([m["omega_lower"] for m in mats])
    soar, offline["soar_bases"] = timed_repeats(
        lambda: [build_soar_rom(m["M_star"], m["C_star"], m["K_star"],
                                0.5 * (m["omega_lower"] + m["omega_upper"]),
                                M_PER_SEG, F) for m in mats],
        args.n_offline, n_warmup=0)

    def stage_project_r():
        out = []
        for m, s in zip(mats, soar):
            p = _project_operator_pieces(s.Q, K_base, K_shear, M, F)
            out.append(PWRefinedSegmentROM(m["omega_lower"], m["omega_upper"],
                                           s.Q, p[0], p[1], p[2], p[3]))
        return out

    roms_r, offline["pw_r_projection"] = timed_repeats(
        stage_project_r, args.n_offline, n_warmup=0)
    Q_g, offline["pw_rg_svd_compression"] = timed_repeats(
        lambda: _orthonormalize_columns_svd(
            np.column_stack([s.Q for s in soar]), tol_sv=SVD_TOL),
        args.n_offline, n_warmup=0)
    pieces_g, offline["pw_rg_projection"] = timed_repeats(
        lambda: _project_operator_pieces(Q_g, K_base, K_shear, M, F),
        args.n_offline, n_warmup=0)
    del mats

    # ---- online ROM sweeps ---------------------------------------------------
    Y_r, online["PW-R"] = timed_repeats(
        lambda: online_pw_r(roms_r, lowers, G_func, obs, omegas), args.n_online)
    Y_g, online["PW-RG"] = timed_repeats(
        lambda: online_affine_global(Q_g, pieces_g[0], pieces_g[1], pieces_g[2],
                                     pieces_g[3], G_func, obs, omegas),
        args.n_online)

    per_freq = {k: v["median_s"] / N_FREQ * 1e3 for k, v in online.items()}
    off_ms = {k: v["median_s"] * 1e3 for k, v in offline.items()}
    shared = off_ms["g_fit"] + off_ms["segment_assembly"] + off_ms["soar_bases"]
    off_total = {
        "PW-R": shared + off_ms["pw_r_projection"],
        "PW-RG": shared + off_ms["pw_rg_svd_compression"] + off_ms["pw_rg_projection"],
    }
    summary = {}
    for name, Y in (("PW-R", Y_r), ("PW-RG", Y_g)):
        t = per_freq[name]
        summary[name] = dict(
            m_online=int(M_PER_SEG if name == "PW-R" else Q_g.shape[1]),
            per_freq_ms=t,
            speedup_vs_dense=per_freq["D_dense_numpy"] / t,
            speedup_vs_sparse=per_freq["D_sparse"] / t,
            offline_total_ms=off_total[name],
            break_even_n_freq_vs_dense=off_total[name] / (per_freq["D_dense_numpy"] - t),
            break_even_n_freq_vs_sparse=off_total[name] / (per_freq["D_sparse"] - t),
            error_vs_dense=err_stats(Y_dense, Y),
        )
    result = dict(
        environment=env, nx=args.nx, n_full_dof=n_full, n_free_dof=int(n),
        nnz_union_pattern=nnz, n_freq=N_FREQ, freq_band_hz=[F_MIN_HZ, F_MAX_HZ],
        timing_protocol=dict(statistic="median", direct_repeats=args.n_direct,
                             direct_warmup=args.warmup_direct,
                             online_repeats=args.n_online, online_warmup=1,
                             offline_repeats=args.n_offline),
        per_freq_ms=per_freq, offline_stage_ms=off_ms, online=online,
        offline_stages=offline, summary=summary,
        sparse_vs_dense_reference=err_stats(Y_dense, Y_sparse),
    )
    out = _ROOT / "data" / f"rev_bench_scaling_nx{args.nx:02d}.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    for name, s in summary.items():
        print(f"  {name}: m={s['m_online']} {s['per_freq_ms']:.4f} ms/freq "
              f"x{s['speedup_vs_dense']:.0f} dense x{s['speedup_vs_sparse']:.0f} sparse "
              f"offline {s['offline_total_ms']:.0f} ms "
              f"N* {s['break_even_n_freq_vs_dense']:.1f}/"
              f"{s['break_even_n_freq_vs_sparse']:.1f} "
              f"max err {s['error_vs_dense']['max_pct']:.2e}%", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
