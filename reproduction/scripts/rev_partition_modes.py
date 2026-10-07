"""Additional studies — resonance count per segment and PW-R basis size.

Companion of ``rev_partition_sensitivity.py``: the same 49 partitions of the
1--1500 Hz fit band on the same benchmark (Amichi 8x8 plate, hard simple
support, n = 471, 496 frequencies in 5--500 Hz, representative modulus,
2000 log-spaced fit samples). For each partition the script records

* the number of resonances inside every segment: the undamped natural
  frequencies of (K_base + Re G(omega_c) K_shear, M) that fall inside the
  segment, with omega_c the segment centre, i.e. the expansion point of the
  SOAR basis of that segment;
* the maximum PW-R response error for m = 10, 20, and 40 SOAR vectors per
  segment (m = 20 is the setting of the paper);
* the fitted-modulus response error and the maximum fit residual inside the
  reported band, from which the ratio "response error / fit residual" of the
  fitted-modulus route is read.

Only errors are recorded (no timing), so the run is deterministic.

Run::

    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_partition_modes.py

Output: ``data/rev_partition_modes.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from rev_bench_rom import (
    F_MAX_HZ, F_MIN_HZ, N_FREQ, direct_dense, direct_fitted, err_stats, online_pw_r,
)
from rev_env import environment_record, require_single_thread
from rev_modulus_data import representative
from rev_partition_sensitivity import BAND_HZ, SAMPLINGS, fit_errors, partitions

from cmame_e2_segmentwise_rom import setup_amichi_8x8
from cmame_f1_xie2018_comparison_v3 import make_G_func
from phase3_amichi_polynomial_qep import (
    G_complex_at_freq,
    assemble_complex_per_interval_amichi,
)
from polyqep.pw_refined import PWRefinedSegmentROM, _project_operator_pieces
from polyqep.rom import build_soar_rom
from polyqep.fitting import fit_joint_spline

_ROOT = Path(__file__).resolve().parent.parent
M_VALUES = (10, 20, 40)
F_EVAL = np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)


def dense(a):
    return np.asarray(a.todense()) if hasattr(a, "todense") else np.asarray(a)


def resonances_in_segment(M, K_base, K_shear, f_lo, f_hi):
    """Undamped natural frequencies in [f_lo, f_hi) with Re G at the segment centre."""
    f_c = 0.5 * (f_lo + f_hi)
    g = float(np.real(representative(f_c)))
    w2 = sla.eigh(K_base + g * K_shear, M, eigvals_only=True)
    f_n = np.sqrt(w2[w2 > 0.0]) / (2.0 * np.pi)
    return int(np.sum((f_n >= f_lo) & (f_n < f_hi)))


def run_case(edges_hz, f_samples, system, Y_ref, G_func, omegas):
    M, K_base, K_shear, F, obs = system
    G_s = representative(f_samples)
    om_s = 2.0 * np.pi * f_samples
    spl_re, spl_im = fit_joint_spline(om_s, G_s.real, G_s.imag, 2.0 * np.pi * edges_hz)
    mats = assemble_complex_per_interval_amichi(spl_re, spl_im, M, K_base, K_shear)
    lowers = np.array([m["omega_lower"] for m in mats])
    out = dict(
        fit_eval_band=fit_errors(spl_re, spl_im, F_EVAL),
        frf_fitted_full_order=err_stats(Y_ref, direct_fitted(mats, F.astype(complex), obs, omegas)),
        pw_r=dict(),
    )
    for m_seg in M_VALUES:
        roms = []
        for mm in mats:
            s = build_soar_rom(mm["M_star"], mm["C_star"], mm["K_star"],
                               0.5 * (mm["omega_lower"] + mm["omega_upper"]), m_seg, F)
            p = _project_operator_pieces(s.Q, K_base, K_shear, M, F)
            roms.append(PWRefinedSegmentROM(mm["omega_lower"], mm["omega_upper"],
                                            s.Q, p[0], p[1], p[2], p[3]))
        Y_r = online_pw_r(roms, lowers, G_func, obs, omegas)
        out["pw_r"][str(m_seg)] = err_stats(Y_ref, Y_r)
    return out


def main():
    require_single_thread()
    env = environment_record()
    _, M, K_base, K_shear, F, obs = setup_amichi_8x8(bc="simply_supported_w")
    Md, Kbd, Ksd = dense(M), dense(K_base), dense(K_shear)
    omegas = 2.0 * np.pi * F_EVAL
    G_func = make_G_func(G_complex_at_freq)
    Y_ref = direct_dense(K_base.astype(complex), K_shear.astype(complex), M, F,
                         obs, omegas, np.linalg.solve)
    system = (M, K_base, K_shear, F, obs)
    f_samples = SAMPLINGS["log2000"]

    rows = []
    for group, label, edges in partitions():
        segs = list(zip(edges[:-1], edges[1:]))
        modes = [resonances_in_segment(Md, Kbd, Ksd, a, b) for a, b in segs]
        overlaps = [bool(b > F_MIN_HZ and a < F_MAX_HZ) for a, b in segs]
        row = dict(group=group, label=label, n_segments=int(edges.size - 1),
                   edges_hz=[float(e) for e in edges],
                   resonances_per_segment=modes,
                   segment_overlaps_reported_band=overlaps,
                   max_resonances_reported=int(max(m for m, o in zip(modes, overlaps) if o)))
        row.update(run_case(edges, f_samples, system, Y_ref, G_func, omegas))
        fit_max = max(row["fit_eval_band"]["re_max"], row["fit_eval_band"]["im_max"])
        row["fitted_over_fit_residual"] = row["frf_fitted_full_order"]["max_pct"] / fit_max
        rows.append(row)
        pw = "  ".join(f"m={k}: {v['max_pct']:.2e}%" for k, v in row["pw_r"].items())
        print(f"{group:<13}{label:<28}N={row['n_segments']:<3}modes {modes}  "
              f"max(reported) {row['max_resonances_reported']:3d}  PW-R {pw}  "
              f"fitted/residual {row['fitted_over_fit_residual']:.2f}", flush=True)

    # summary: for each m, worst PW-R error among partitions whose busiest
    # reported segment holds at most m resonances, and among the others
    summary = {}
    for m_seg in M_VALUES:
        within = [r["pw_r"][str(m_seg)]["max_pct"] for r in rows
                  if r["max_resonances_reported"] <= m_seg]
        beyond = [r["pw_r"][str(m_seg)]["max_pct"] for r in rows
                  if r["max_resonances_reported"] > m_seg]
        summary[str(m_seg)] = dict(
            n_within=len(within), pw_r_max_within=float(max(within)) if within else None,
            n_beyond=len(beyond), pw_r_min_beyond=float(min(beyond)) if beyond else None,
            pw_r_max_beyond=float(max(beyond)) if beyond else None)
        print(f"m={m_seg}: resonances<=m in {len(within)} partitions, PW-R max "
              f"{summary[str(m_seg)]['pw_r_max_within']}; resonances>m in {len(beyond)}, "
              f"PW-R {summary[str(m_seg)]['pw_r_min_beyond']} .. "
              f"{summary[str(m_seg)]['pw_r_max_beyond']}")
    ratios = [r["fitted_over_fit_residual"] for r in rows]
    summary["fitted_over_fit_residual"] = dict(min=float(min(ratios)),
                                               median=float(np.median(ratios)),
                                               max=float(max(ratios)), n=len(ratios))
    n_total = resonances_in_segment(Md, Kbd, Ksd, BAND_HZ[0], BAND_HZ[1])
    n_reported = resonances_in_segment(Md, Kbd, Ksd, F_MIN_HZ, F_MAX_HZ)

    out = _ROOT / "data" / "rev_partition_modes.json"
    with open(out, "w") as fh:
        json.dump(dict(environment=env, n_freq=N_FREQ,
                       eval_band_hz=[F_MIN_HZ, F_MAX_HZ], fit_band_hz=list(BAND_HZ),
                       sampling="log2000", m_values=list(M_VALUES),
                       resonance_definition=("undamped natural frequencies of "
                                             "(K_base + Re G(omega_c) K_shear, M) inside the "
                                             "segment, omega_c = segment centre"),
                       resonances_in_fit_band=n_total,
                       resonances_in_reported_band=n_reported,
                       summary=summary, rows=rows), fh, indent=2)
    print(f"\nresonances: {n_total} in the fit band, {n_reported} in the reported band")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
