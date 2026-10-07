"""Additional studies — a greedy partition rule (tested as a practical criterion).

Rule: start from a single segment over the fit band; repeatedly bisect, at its
log-frequency midpoint, the segment that carries the largest relative fit
residual (Re or Im) inside the analysis band; stop when that residual falls
below the tolerance. The benchmark and the reported quantities are those of
``rev_partition_sensitivity.py`` (log-spaced modulus sampling).

Output: ``data/rev_partition_greedy.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rev_bench_rom import F_MAX_HZ, F_MIN_HZ, N_FREQ, direct_dense
from rev_env import environment_record, require_single_thread
from rev_modulus_data import representative
from rev_partition_sensitivity import BAND_HZ, F_EVAL, SAMPLINGS, run_case

from cmame_e2_segmentwise_rom import setup_amichi_8x8
from cmame_f1_xie2018_comparison_v3 import make_G_func
from phase3_amichi_polynomial_qep import G_complex_at_freq
from polyqep.fitting import fit_joint_spline

_ROOT = Path(__file__).resolve().parent.parent
TOLERANCES_PCT = (5.0, 2.0, 1.0)
MAX_SEGMENTS = 24


def segment_residuals(edges, f_samples, f_check):
    """Largest relative fit residual (Re or Im, %) per segment on f_check."""
    G_s = representative(f_samples)
    spl_re, spl_im = fit_joint_spline(2 * np.pi * f_samples, G_s.real, G_s.imag,
                                      2 * np.pi * edges)
    G = representative(f_check)
    om = 2 * np.pi * f_check
    rel = np.maximum(np.abs(np.asarray(spl_re(om)) - G.real) / np.abs(G.real),
                     np.abs(np.asarray(spl_im(om)) - G.imag) / np.abs(G.imag)) * 100.0
    out = np.zeros(edges.size - 1)
    for k in range(edges.size - 1):
        m = (f_check >= edges[k]) & (f_check <= edges[k + 1])
        out[k] = rel[m].max() if m.any() else 0.0
    return out


def main():
    require_single_thread()
    env = environment_record()
    _, M, K_base, K_shear, F, obs = setup_amichi_8x8(bc="simply_supported_w")
    omegas = 2.0 * np.pi * F_EVAL
    G_func = make_G_func(G_complex_at_freq)
    Y_ref = direct_dense(K_base.astype(complex), K_shear.astype(complex), M, F,
                         obs, omegas, np.linalg.solve)
    system = (M, K_base, K_shear, F, obs)
    f_samples = SAMPLINGS["log2000"]
    f_check = np.geomspace(F_MIN_HZ, F_MAX_HZ, 2000)

    edges = np.array(BAND_HZ, dtype=float)
    steps, reached = [], {}
    while edges.size - 1 <= MAX_SEGMENTS:
        res = segment_residuals(edges, f_samples, f_check)
        row = dict(n_segments=int(edges.size - 1),
                   edges_hz=[float(e) for e in edges],
                   max_fit_residual_pct=float(res.max()))
        row.update(run_case(edges, f_samples, system, Y_ref, G_func, omegas))
        steps.append(row)
        print(f"N={row['n_segments']:<3} max fit residual {res.max():6.2f}%  FRF fitted "
              f"{row['frf_fitted_full_order']['max_pct']:6.3f}/"
              f"{row['frf_fitted_full_order']['mean_pct']:5.3f}%  PW-R "
              f"{row['frf_pw_r']['max_pct']:.2e}%  PW-RG {row['frf_pw_rg']['max_pct']:.1e}% "
              f"(m={row['m_pw_rg']})  edges {np.round(edges, 1).tolist()}", flush=True)
        for tol in TOLERANCES_PCT:
            if res.max() < tol and tol not in reached:
                reached[tol] = row["n_segments"]
        if res.max() < min(TOLERANCES_PCT):
            break
        k = int(res.argmax())
        edges = np.sort(np.append(edges, np.sqrt(edges[k] * edges[k + 1])))

    out = _ROOT / "data" / "rev_partition_greedy.json"
    with open(out, "w") as fh:
        json.dump(dict(environment=env, rule="bisect the segment with the largest "
                       "in-band fit residual at its log-frequency midpoint",
                       tolerances_pct=list(TOLERANCES_PCT),
                       segments_needed={str(k): v for k, v in reached.items()},
                       n_freq=N_FREQ, steps=steps), fh, indent=2)
    print(f"segments needed per tolerance: {reached}\nsaved {out}")


if __name__ == "__main__":
    main()
