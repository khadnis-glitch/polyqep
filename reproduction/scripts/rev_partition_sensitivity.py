"""Additional studies — sensitivity to the segment partition.

Benchmark: Amichi 8x8 plate, hard simple support (n = 471), 496 frequencies in
5--500 Hz, representative core modulus (the configuration of the seven-method
comparison). For each alternative partition of the 1--1500 Hz fit band the
script reports

* the piecewise-quadratic fit residuals of Re G and Im G;
* the FRF error of the fitted-modulus full-order solve (the fit-bounded error
  that the polynomial-G route inherits);
* the FRF errors of the raw-modulus ROMs PW-R and PW-RG and the PW-RG basis
  size;
* the offline time (fit, segment assembly, SOAR bases, projections, SVD
  compression) and the online time per frequency.

Partitions
    baseline          1 | 50 | 150 | 350 | 700 | 1500 Hz (manuscript)
    log-uniform N     N equal segments in log-frequency
    uniform N         N equal segments in frequency
    baseline - 1      one internal breakpoint removed (4 variants)
    baseline + 1      one segment split at its log-midpoint (5 variants)
    baseline + 2      the two log-widest segments split
    shifted           all internal breakpoints scaled by 0.8, 0.9, 1.1, 1.2
    random            each internal breakpoint scaled independently by a
                      factor drawn uniformly from [0.8, 1.2] (20 draws)

Modulus sampling for the fit: ``uniform400`` is the submitted setting (400
uniformly spaced points); ``log2000`` (2000 log-spaced points) resolves the
low-frequency segments of every partition. A partition is skipped under
``uniform400`` when one of its segments holds fewer than three samples.

Run::

    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_partition_sensitivity.py

Output: ``data/rev_partition_sensitivity.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rev_bench_rom import (
    F_MAX_HZ, F_MIN_HZ, M_PER_SEG, N_FREQ, SVD_TOL,
    direct_dense, direct_fitted, err_stats, online_affine_global, online_pw_r,
)
from rev_env import environment_record, require_single_thread, timed_repeats
from rev_modulus_data import representative

from cmame_e2_segmentwise_rom import setup_amichi_8x8
from cmame_f1_xie2018_comparison_v3 import make_G_func
from phase3_amichi_polynomial_qep import (
    G_complex_at_freq,
    assemble_complex_per_interval_amichi,
)
from polyqep.pw_refined import (
    PWRefinedSegmentROM,
    _orthonormalize_columns_svd,
    _project_operator_pieces,
)
from polyqep.rom import build_soar_rom
from polyqep.fitting import fit_joint_spline

_ROOT = Path(__file__).resolve().parent.parent
BAND_HZ = (1.0, 1500.0)
BASELINE_HZ = np.array([1.0, 50.0, 150.0, 350.0, 700.0, 1500.0])
SAMPLINGS = {
    "uniform400": np.linspace(*BAND_HZ, 400),
    "log2000": np.geomspace(*BAND_HZ, 2000),
}
F_EVAL = np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)
SEED, N_RANDOM = 20261004, 20


def partitions():
    """(group, label, edges_hz) for every alternative partition."""
    out = [("baseline", "50/150/350/700 Hz", BASELINE_HZ)]
    for n in (2, 3, 4, 5, 6, 7, 10, 15):
        out.append(("log-uniform", f"N={n}", np.geomspace(*BAND_HZ, n + 1)))
    for n in (3, 5, 7, 10, 15, 20):
        out.append(("uniform", f"N={n}", np.linspace(*BAND_HZ, n + 1)))
    for k in range(1, len(BASELINE_HZ) - 1):
        out.append(("baseline - 1", f"without {BASELINE_HZ[k]:.0f} Hz",
                    np.delete(BASELINE_HZ, k)))
    log_widths = np.diff(np.log(BASELINE_HZ))
    for k in range(len(BASELINE_HZ) - 1):
        mid = float(np.sqrt(BASELINE_HZ[k] * BASELINE_HZ[k + 1]))
        out.append(("baseline + 1", f"extra {mid:.1f} Hz",
                    np.sort(np.append(BASELINE_HZ, mid))))
    widest = np.argsort(log_widths)[-2:]
    mids = [float(np.sqrt(BASELINE_HZ[k] * BASELINE_HZ[k + 1])) for k in widest]
    out.append(("baseline + 2", "extra " + ", ".join(f"{m:.1f}" for m in sorted(mids))
                + " Hz", np.sort(np.append(BASELINE_HZ, mids))))
    for factor in (0.8, 0.9, 1.1, 1.2):
        edges = BASELINE_HZ.copy()
        edges[1:-1] *= factor
        out.append(("shifted", f"internal breakpoints x {factor}", edges))
    rng = np.random.default_rng(SEED)
    for i in range(N_RANDOM):
        edges = BASELINE_HZ.copy()
        edges[1:-1] = np.sort(edges[1:-1] * rng.uniform(0.8, 1.2, size=4))
        out.append(("random", f"draw {i + 1:02d}", edges))
    return out


def fit_errors(spl_re, spl_im, f_hz):
    G = representative(f_hz)
    om = 2.0 * np.pi * f_hz
    rel_re = np.abs(np.asarray(spl_re(om)) - G.real) / np.abs(G.real) * 100.0
    rel_im = np.abs(np.asarray(spl_im(om)) - G.imag) / np.abs(G.imag) * 100.0
    return dict(re_max=float(rel_re.max()), re_mean=float(rel_re.mean()),
                im_max=float(rel_im.max()), im_mean=float(rel_im.mean()))


def run_case(edges_hz, f_samples, system, Y_ref, G_func, omegas):
    M, K_base, K_shear, F, obs = system
    G_s = representative(f_samples)
    om_s = 2.0 * np.pi * f_samples

    def offline():
        spl_re, spl_im = fit_joint_spline(om_s, G_s.real, G_s.imag,
                                          2.0 * np.pi * edges_hz)
        mats = assemble_complex_per_interval_amichi(spl_re, spl_im, M, K_base, K_shear)
        soar = [build_soar_rom(m["M_star"], m["C_star"], m["K_star"],
                               0.5 * (m["omega_lower"] + m["omega_upper"]),
                               M_PER_SEG, F) for m in mats]
        roms = []
        for m, s in zip(mats, soar):
            p = _project_operator_pieces(s.Q, K_base, K_shear, M, F)
            roms.append(PWRefinedSegmentROM(m["omega_lower"], m["omega_upper"],
                                            s.Q, p[0], p[1], p[2], p[3]))
        Q_g = _orthonormalize_columns_svd(np.column_stack([s.Q for s in soar]),
                                          tol_sv=SVD_TOL)
        pieces = _project_operator_pieces(Q_g, K_base, K_shear, M, F)
        return spl_re, spl_im, mats, roms, Q_g, pieces

    (spl_re, spl_im, mats, roms, Q_g, pieces), t_off = timed_repeats(
        offline, 3, n_warmup=0)
    lowers = np.array([m["omega_lower"] for m in mats])
    Y_fit = direct_fitted(mats, F.astype(complex), obs, omegas)
    Y_r, t_r = timed_repeats(
        lambda: online_pw_r(roms, lowers, G_func, obs, omegas), 5)
    Y_g, t_g = timed_repeats(
        lambda: online_affine_global(Q_g, pieces[0], pieces[1], pieces[2],
                                     pieces[3], G_func, obs, omegas), 5)
    return dict(
        fit_full_band=fit_errors(spl_re, spl_im, np.geomspace(*BAND_HZ, 2000)),
        fit_eval_band=fit_errors(spl_re, spl_im, F_EVAL),
        frf_fitted_full_order=err_stats(Y_ref, Y_fit),
        frf_pw_r=err_stats(Y_ref, Y_r),
        frf_pw_rg=err_stats(Y_ref, Y_g),
        m_pw_r_per_segment=[int(r.m_reduced) for r in roms],
        m_pw_rg=int(Q_g.shape[1]),
        offline_ms=t_off["median_s"] * 1e3,
        pw_r_per_freq_ms=t_r["median_s"] / omegas.size * 1e3,
        pw_rg_per_freq_ms=t_g["median_s"] / omegas.size * 1e3,
    )


def main():
    require_single_thread()
    env = environment_record()
    _, M, K_base, K_shear, F, obs = setup_amichi_8x8(bc="simply_supported_w")
    omegas = 2.0 * np.pi * F_EVAL
    G_func = make_G_func(G_complex_at_freq)
    Y_ref = direct_dense(K_base.astype(complex), K_shear.astype(complex), M, F,
                         obs, omegas, np.linalg.solve)
    system = (M, K_base, K_shear, F, obs)

    rows = []
    for group, label, edges in partitions():
        for s_name, f_samples in SAMPLINGS.items():
            counts = np.histogram(f_samples, bins=edges)[0]
            row = dict(group=group, label=label, sampling=s_name,
                       n_segments=int(edges.size - 1),
                       edges_hz=[float(e) for e in edges],
                       min_samples_per_segment=int(counts.min()))
            if counts.min() < 3:
                row["skipped"] = "a segment holds fewer than three fit samples"
                rows.append(row)
                continue
            row.update(run_case(edges, f_samples, system, Y_ref, G_func, omegas))
            rows.append(row)
            print(f"{group:<13}{label:<28}{s_name:<11} N={row['n_segments']:<3}"
                  f"fit Re/Im max {row['fit_eval_band']['re_max']:5.2f}/"
                  f"{row['fit_eval_band']['im_max']:5.2f}%  FRF fitted "
                  f"{row['frf_fitted_full_order']['max_pct']:6.3f}/"
                  f"{row['frf_fitted_full_order']['mean_pct']:5.3f}%  PW-R "
                  f"{row['frf_pw_r']['max_pct']:.2e}%  PW-RG "
                  f"{row['frf_pw_rg']['max_pct']:.1e}% (m={row['m_pw_rg']})  "
                  f"off {row['offline_ms']:.0f} ms  on "
                  f"{row['pw_r_per_freq_ms']:.4f}/{row['pw_rg_per_freq_ms']:.4f} ms",
                  flush=True)

    out = _ROOT / "data" / "rev_partition_sensitivity.json"
    with open(out, "w") as fh:
        json.dump(dict(environment=env, seed=SEED, n_freq=N_FREQ,
                       eval_band_hz=[F_MIN_HZ, F_MAX_HZ], fit_band_hz=list(BAND_HZ),
                       m_per_segment=M_PER_SEG, svd_truncation=SVD_TOL,
                       rows=rows), fh, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
