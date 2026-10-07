"""Additional studies — seven-method ROM benchmark with a uniform offline/online split.

Regenerates the reduced-order comparison (manuscript Table 3) on the corrected
Amichi element in a single recorded environment. Compared with the original
generator (``cmame_f1_xie2018_comparison_v3.py``) this script

* applies the same offline/online boundary to every method: all basis
  construction (SOAR, SVD compression, the Xie single-point build, and the
  whole Xie adaptive procedure) is offline, and the online time is one sweep
  of per-frequency reduced solves;
* times each offline stage separately (fit, segment assembly, SOAR bases,
  projections, SVD compression, segment QEP solves);
* repeats every measurement and reports the median;
* solves the direct and the reduced systems with the same LAPACK driver
  (``numpy.linalg.solve``) and also records the ``scipy.linalg.solve`` baseline
  used at submission;
* stores the execution environment with the numbers.

Run (single thread is enforced)::

    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_bench_rom.py

Output: ``data/rev_bench_rom.json`` and ``data/rev_bench_rom_frf.npz``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg as spla
import scipy.sparse as spsparse
import scipy.sparse.linalg as spspla

from rev_env import environment_record, require_single_thread, timed_repeats

from phase3_amichi_polynomial_qep import (
    G_complex_at_freq,
    assemble_complex_per_interval_amichi,
    fit_G_polynomial,
    sample_G_omega,
    solve_qep_per_interval,
)
from cmame_e2_segmentwise_rom import setup_amichi_8x8
from cmame_f1_xie2018_comparison_v3 import make_G_func, relative_error_pct
from polyqep.rom import build_soar_rom
from polyqep.pw_refined import (
    PWRefinedGlobalROM,
    PWRefinedSegmentROM,
    _orthonormalize_columns_svd,
    _project_operator_pieces,
    pw_refined_global_response,
    pw_refined_response,
)
from baselines.xie2018_rom import (
    build_xie_rom,
    xie_adaptive_response,
    xie_rom_response,
)

_ROOT = Path(__file__).resolve().parent.parent

F_MIN_HZ, F_MAX_HZ, N_FREQ = 5.0, 500.0, 496
FIT_BAND_HZ, N_FIT_SAMPLES = (1.0, 1500.0), 400
N_SEG, M_PER_SEG = 5, 20
SVD_TOL = 1e-10
XS_F0_HZ, XS_M = 50.0, 100
XA_M_PER_POINT, XA_TOL, XA_MAX_POINTS = 20, 1e-2, 10


# ----------------------------------------------------------------------------
# Direct per-frequency solvers (online only; no offline stage)
# ----------------------------------------------------------------------------
def direct_dense(K_base_c, K_shear_c, M, F, obs, omegas, solve):
    Y = np.zeros(omegas.size, dtype=complex)
    for i, omega in enumerate(omegas):
        G = G_complex_at_freq(omega / (2.0 * np.pi))
        D = K_base_c + G * K_shear_c - (omega**2) * M
        Y[i] = solve(D, F)[obs]
    return Y


def direct_sparse(K_base_c, K_shear_c, M_c, F, obs, omegas):
    Y = np.zeros(omegas.size, dtype=complex)
    for i, omega in enumerate(omegas):
        G = complex(G_complex_at_freq(omega / (2.0 * np.pi)))
        D = K_base_c + G * K_shear_c - (omega**2) * M_c
        Y[i] = spspla.splu(spsparse.csc_matrix(D)).solve(F)[obs]
    return Y


def direct_sparse_native(Kb_s, Ks_s, M_s, F, obs, omegas):
    """Sparse LU on operators that stay sparse (CSC) in the frequency loop."""
    Y = np.zeros(omegas.size, dtype=complex)
    for i, omega in enumerate(omegas):
        G = complex(G_complex_at_freq(omega / (2.0 * np.pi)))
        D = (Kb_s + G * Ks_s - (omega**2) * M_s).tocsc()
        Y[i] = spspla.splu(D).solve(F)[obs]
    return Y


def direct_fitted(interval_mats, F, obs, omegas):
    """Full-order solve with the fitted segment matrices (PW consistent FOM)."""
    lowers = np.array([m["omega_lower"] for m in interval_mats])
    Y = np.zeros(omegas.size, dtype=complex)
    for i, omega in enumerate(omegas):
        j = int(np.clip(np.searchsorted(lowers, omega, side="right") - 1,
                        0, len(interval_mats) - 1))
        mat = interval_mats[j]
        D = mat["K_star"] + 1j * omega * mat["C_star"] - (omega**2) * mat["M_star"]
        Y[i] = np.linalg.solve(D, F)[obs]
    return Y


# ----------------------------------------------------------------------------
# Online sweeps. ``full_field=True`` recovers the full displacement vector
# (Q @ y_r, the O(m n) term of the complexity table) before reading the
# observation DOF; ``False`` projects onto the observation row only.
# ----------------------------------------------------------------------------
def _readout(Q, obs, y_r, full_field):
    return (Q @ y_r)[obs] if full_field else Q[obs, :] @ y_r


def online_pw_poly(soar_roms, lowers, obs, omegas, full_field=True):
    Y = np.zeros(omegas.size, dtype=complex)
    for i, omega in enumerate(omegas):
        j = int(np.clip(np.searchsorted(lowers, omega, side="right") - 1,
                        0, len(soar_roms) - 1))
        rom = soar_roms[j]
        D_r = rom.K_r + 1j * omega * rom.C_r - (omega**2) * rom.M_r
        Y[i] = _readout(rom.Q, obs, np.linalg.solve(D_r, rom.F_r), full_field)
    return Y


def online_pw_r(roms, lowers, G_func, obs, omegas, full_field=True):
    Y = np.zeros(omegas.size, dtype=complex)
    for i, omega in enumerate(omegas):
        j = int(np.clip(np.searchsorted(lowers, omega, side="right") - 1,
                        0, len(roms) - 1))
        rom = roms[j]
        G = complex(G_func(1j * omega))
        D_r = rom.K_base_r + G * rom.K_shear_r - (omega**2) * rom.M_r
        Y[i] = _readout(rom.Q, obs, np.linalg.solve(D_r, rom.F_r), full_field)
    return Y


def online_affine_global(Q, Kb_r, Ks_r, M_r, F_r, G_func, obs, omegas,
                         full_field=True):
    """Single-basis raw-G ROM sweep (PW-RG and the Xie adaptive ROM)."""
    Y = np.zeros(omegas.size, dtype=complex)
    for i, omega in enumerate(omegas):
        G = complex(G_func(1j * omega))
        D_r = Kb_r + G * Ks_r - (omega**2) * M_r
        Y[i] = _readout(Q, obs, np.linalg.solve(D_r, F_r), full_field)
    return Y


def online_xs(rom, F_r, obs, omegas, full_field=True):
    Y = np.zeros(omegas.size, dtype=complex)
    for i, omega in enumerate(omegas):
        s = 1j * omega
        delta = s - rom.s0
        R2 = complex(rom.G_func(s)) - complex(rom.G_taylor(s))
        D_r = (delta**2) * rom.Mr + delta * rom.Dr + rom.Kr + R2 * rom.Kvr
        Y[i] = _readout(rom.Q, obs, np.linalg.solve(D_r, F_r), full_field)
    return Y


# ----------------------------------------------------------------------------
def err_stats(Y_ref, Y):
    err = relative_error_pct(Y_ref, Y)
    return {"max_pct": float(err.max()), "mean_pct": float(err.mean())}


def main():
    parser = argparse.ArgumentParser(
        description="Seven-method ROM benchmark, uniform offline/online split")
    parser.add_argument("--n-online", type=int, default=10,
                        help="measured repeats of each online sweep")
    parser.add_argument("--n-offline", type=int, default=5,
                        help="measured repeats of each offline stage")
    parser.add_argument("--n-qep", type=int, default=3,
                        help="measured repeats of the segment QEP solves")
    parser.add_argument("--tag", default="", help="suffix for the output files")
    args = parser.parse_args()

    require_single_thread()
    env = environment_record()
    print(json.dumps(env, indent=2))

    _, M, K_base, K_shear, F, obs = setup_amichi_8x8(bc="simply_supported_w")
    n = M.shape[0]
    omegas = 2.0 * np.pi * np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)
    G_func = make_G_func(G_complex_at_freq)
    K_base_c, K_shear_c, M_c = (K_base.astype(complex), K_shear.astype(complex),
                                M.astype(complex))
    F_c = F.astype(complex)
    Kb_s = spsparse.csc_matrix(K_base).astype(complex)
    Ks_s = spsparse.csc_matrix(K_shear).astype(complex)
    M_s = spsparse.csc_matrix(M).astype(complex)
    print(f"\nAmichi 8x8 simply supported: n = {n}, {N_FREQ} frequencies")

    offline, online, errors, dims = {}, {}, {}, {}

    # ---------------- offline: shared PolyQEP stages -------------------------
    def stage_fit():
        omega_fit, G_re, G_im = sample_G_omega(*FIT_BAND_HZ, N_FIT_SAMPLES)
        return fit_G_polynomial(omega_fit, G_re, G_im, n_segments=N_SEG)

    (spline_re, spline_im, _), offline["g_fit"] = timed_repeats(
        stage_fit, args.n_offline)

    interval_mats, offline["segment_assembly"] = timed_repeats(
        lambda: assemble_complex_per_interval_amichi(
            spline_re, spline_im, M, K_base, K_shear),
        args.n_offline)
    lowers = np.array([m["omega_lower"] for m in interval_mats])

    def stage_soar():
        return [
            build_soar_rom(m["M_star"], m["C_star"], m["K_star"],
                           0.5 * (m["omega_lower"] + m["omega_upper"]),
                           M_PER_SEG, F)
            for m in interval_mats
        ]

    soar_roms, offline["soar_bases"] = timed_repeats(stage_soar, args.n_offline)
    dims["PW"] = [int(r.m_reduced) for r in soar_roms]

    def stage_project_r():
        out = []
        for m, rom in zip(interval_mats, soar_roms):
            pieces = _project_operator_pieces(rom.Q, K_base, K_shear, M, F)
            out.append(PWRefinedSegmentROM(
                omega_lower=m["omega_lower"], omega_upper=m["omega_upper"],
                Q=rom.Q, K_base_r=pieces[0], K_shear_r=pieces[1],
                M_r=pieces[2], F_r=pieces[3]))
        return out

    roms_r, offline["pw_r_projection"] = timed_repeats(
        stage_project_r, args.n_offline)
    dims["PW-R"] = [int(r.m_reduced) for r in roms_r]

    Q_global, offline["pw_rg_svd_compression"] = timed_repeats(
        lambda: _orthonormalize_columns_svd(
            np.column_stack([r.Q for r in soar_roms]), tol_sv=SVD_TOL),
        args.n_offline)
    pieces_g, offline["pw_rg_projection"] = timed_repeats(
        lambda: _project_operator_pieces(Q_global, K_base, K_shear, M, F),
        args.n_offline)
    rom_g = PWRefinedGlobalROM(
        omega_min=float(lowers[0]),
        omega_max=float(interval_mats[-1]["omega_upper"]),
        Q=Q_global, K_base_r=pieces_g[0], K_shear_r=pieces_g[1],
        M_r=pieces_g[2], F_r=pieces_g[3], source_dims=dims["PW"])
    dims["PW-RG"] = int(Q_global.shape[1])

    # Segment QEP solves: needed for the modal expansion, not for the ROMs.
    interval_sols, offline["segment_qep_modal_output"] = timed_repeats(
        lambda: solve_qep_per_interval(interval_mats, damping_tol=1e-6),
        args.n_qep, n_warmup=0)
    eff_counts = [int(s["lams_effective"].size) for s in interval_sols]
    del interval_sols

    # ---------------- offline: Xie single-point and adaptive -----------------
    L_vec = np.zeros(n, dtype=complex)
    L_vec[obs] = 1.0
    rom_xs, offline["xs_build"] = timed_repeats(
        lambda: build_xie_rom(K_base, K_shear, M, G_func,
                              s0=1j * 2.0 * np.pi * XS_F0_HZ, m=XS_M,
                              force_vector=F),
        args.n_offline)
    F_r_xs = rom_xs.Q.conj().T @ F_c
    dims["XS"] = int(rom_xs.Q.shape[1])

    (Y_xa_lib, rom_xa), offline["xa_adaptive_build"] = timed_repeats(
        lambda: xie_adaptive_response(
            K_base, K_shear, M, G_func, omegas, F, L_vec,
            m_per_point=XA_M_PER_POINT, tol=XA_TOL, max_points=XA_MAX_POINTS),
        args.n_offline)
    F_r_xa = rom_xa.Q.conj().T @ F_c
    dims["XA"] = int(rom_xa.m_reduced)
    n_xa_points = len(rom_xa.expansion_points)

    # ---------------- online sweeps -----------------------------------------
    sweeps = {
        "D_dense_numpy": lambda: direct_dense(
            K_base_c, K_shear_c, M, F, obs, omegas, np.linalg.solve),
        "D_dense_scipy": lambda: direct_dense(
            K_base_c, K_shear_c, M, F, obs, omegas, spla.solve),
        "D_sparse": lambda: direct_sparse(
            K_base_c, K_shear_c, M_c, F_c, obs, omegas),
        "D_sparse_native": lambda: direct_sparse_native(
            Kb_s, Ks_s, M_s, F_c, obs, omegas),
    }
    for full_field, suffix in ((True, ""), (False, "_single_output")):
        sweeps["PW" + suffix] = lambda ff=full_field: online_pw_poly(
            soar_roms, lowers, obs, omegas, ff)
        sweeps["PW-R" + suffix] = lambda ff=full_field: online_pw_r(
            roms_r, lowers, G_func, obs, omegas, ff)
        sweeps["PW-RG" + suffix] = lambda ff=full_field: online_affine_global(
            rom_g.Q, rom_g.K_base_r, rom_g.K_shear_r, rom_g.M_r, rom_g.F_r,
            G_func, obs, omegas, ff)
        sweeps["XS" + suffix] = lambda ff=full_field: online_xs(
            rom_xs, F_r_xs, obs, omegas, ff)
        sweeps["XA" + suffix] = lambda ff=full_field: online_affine_global(
            rom_xa.Q, rom_xa.Ker, rom_xa.Kvr, rom_xa.Mr, F_r_xa,
            G_func, obs, omegas, ff)

    Y = {}
    for name, fn in sweeps.items():
        Y[name], stats = timed_repeats(fn, args.n_online)
        stats["per_freq_ms"] = stats["median_s"] / N_FREQ * 1e3
        online[name] = stats
        print(f"  online {name:<24s} {stats['per_freq_ms']:.5f} ms/freq")

    # ---------------- consistency with the library implementations ----------
    checks = {
        "PW-R": pw_refined_response(omegas, roms_r, G_func, obs),
        "PW-RG": pw_refined_global_response(omegas, rom_g, G_func, obs),
        "XS": xie_rom_response(rom_xs, omegas, F, L_vec, include_remainder=True),
        "XA": Y_xa_lib,
    }
    library_agreement = {
        k: float(np.max(np.abs(v - Y[k])) / np.max(np.abs(v)))
        for k, v in checks.items()
    }

    # ---------------- errors -------------------------------------------------
    Y_ref = Y["D_dense_numpy"]
    Y_fit_full = direct_fitted(interval_mats, F_c, obs, omegas)
    for name in ("D_dense_scipy", "D_sparse", "D_sparse_native",
                 "PW", "PW-R", "PW-RG", "XS", "XA"):
        errors[name] = {"fit_bounded": err_stats(Y_ref, Y[name])}
    errors["PW"]["rom_only"] = err_stats(Y_fit_full, Y["PW"])
    errors["fitted_full_order_vs_reference"] = err_stats(Y_ref, Y_fit_full)

    # ---------------- derived quantities ------------------------------------
    t_dense = online["D_dense_numpy"]["per_freq_ms"]
    # sparse baseline: operators kept sparse; "D_sparse" (dense-to-sparse
    # conversion at every frequency, as at submission) is kept for the record
    t_sparse = online["D_sparse_native"]["per_freq_ms"]
    off_ms = {k: v["median_s"] * 1e3 for k, v in offline.items()}
    shared = off_ms["g_fit"] + off_ms["segment_assembly"] + off_ms["soar_bases"]
    offline_total_ms = {
        "PW": shared,
        "PW-R": shared + off_ms["pw_r_projection"],
        "PW-RG": shared + off_ms["pw_rg_svd_compression"] + off_ms["pw_rg_projection"],
        "XS": off_ms["xs_build"],
        "XA": off_ms["xa_adaptive_build"],
    }
    summary = {}
    for name in ("PW", "PW-R", "PW-RG", "XS", "XA"):
        t_on = online[name]["per_freq_ms"]
        summary[name] = {
            "m_online": dims[name] if isinstance(dims[name], int) else max(dims[name]),
            "per_freq_ms": t_on,
            "per_freq_ms_single_output": online[name + "_single_output"]["per_freq_ms"],
            "speedup_vs_dense": t_dense / t_on,
            "speedup_vs_sparse": t_sparse / t_on,
            "offline_total_ms": offline_total_ms[name],
            "break_even_n_freq_vs_dense": offline_total_ms[name] / (t_dense - t_on),
            "break_even_n_freq_vs_sparse": offline_total_ms[name] / (t_sparse - t_on),
            "fit_bounded_max_pct": errors[name]["fit_bounded"]["max_pct"],
            "fit_bounded_mean_pct": errors[name]["fit_bounded"]["mean_pct"],
        }

    result = {
        "environment": env,
        "benchmark": {
            "mesh": "Amichi 8x8, hard simple support",
            "n_free_dof": int(n),
            "freq_band_hz": [F_MIN_HZ, F_MAX_HZ],
            "n_freq": N_FREQ,
            "fit_band_hz": list(FIT_BAND_HZ),
            "n_fit_samples": N_FIT_SAMPLES,
            "n_segments": N_SEG,
            "m_per_segment": M_PER_SEG,
            "svd_truncation": SVD_TOL,
            "xs": {"f0_hz": XS_F0_HZ, "m": XS_M},
            "xa": {"m_per_point": XA_M_PER_POINT, "tol": XA_TOL,
                   "max_points": XA_MAX_POINTS, "n_points_used": n_xa_points},
        },
        "timing_protocol": {
            "statistic": "median",
            "online_repeats": args.n_online,
            "offline_repeats": args.n_offline,
            "qep_repeats": args.n_qep,
            "warmup_runs": 1,
            "clock": "time.perf_counter",
            "online_definition": "one sweep of per-frequency solves, operator "
                                 "assembly included, full-field recovery",
        },
        "dims": dims,
        "effective_eigenvalue_counts": eff_counts,
        "offline_stages": offline,
        "offline_stage_ms": off_ms,
        "online": online,
        "errors": errors,
        "library_agreement_rel": library_agreement,
        "summary": summary,
    }

    out_json = _ROOT / "data" / f"rev_bench_rom{args.tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2)
    np.savez(_ROOT / "data" / f"rev_bench_rom_frf{args.tag}.npz",
             omegas=omegas, Y_fit_full=Y_fit_full,
             **{k.replace("-", "_"): v for k, v in Y.items()})

    print("\noffline stages [ms]:")
    for k, v in off_ms.items():
        print(f"  {k:<28s} {v:10.2f}")
    print(f"\ndense {t_dense:.4f} ms/freq, sparse {t_sparse:.4f} ms/freq, "
          f"dense(scipy) {online['D_dense_scipy']['per_freq_ms']:.4f} ms/freq, "
          f"sparse(converted) {online['D_sparse']['per_freq_ms']:.4f} ms/freq")
    print(f"{'method':<7}{'m':>5}{'ms/freq':>10}{'x dense':>9}{'x sparse':>9}"
          f"{'offline ms':>12}{'N* dense':>10}{'max err %':>12}")
    for name, s in summary.items():
        print(f"{name:<7}{s['m_online']:>5}{s['per_freq_ms']:>10.4f}"
              f"{s['speedup_vs_dense']:>9.1f}{s['speedup_vs_sparse']:>9.1f}"
              f"{s['offline_total_ms']:>12.1f}"
              f"{s['break_even_n_freq_vs_dense']:>10.1f}"
              f"{s['fit_bounded_max_pct']:>12.3e}")
    print(f"PW ROM-only max err: {errors['PW']['rom_only']['max_pct']:.3e} %")
    print(f"library agreement (relative): {library_agreement}")
    print(f"effective-eigenvalue counts: {eff_counts}")
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()
