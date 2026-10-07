#!/usr/bin/env python3
"""Additional studies — how much of the response the effective set alone carries.

Backs the §2.5 / §3.2 statements that the response must be evaluated with the
complete segment spectrum and that the effective poles alone leave a residual
"of order one". Simply supported Amichi 8x8 plate (471 DOF), baseline
partition, 200 frequencies in 5--500 Hz, reference = full-order solve with the
fitted segment matrices (so only the modal truncation is measured).

Variants of the partial-fraction sum (residue formula (13)--(14)):
  complete           all 2n poles of Q_i^+ of the segment containing omega
  effective_only     effective poles of that segment only
  effective_conj     effective poles of that segment + their conjugates
  effective_conj_all effective poles + conjugates pooled over all segments
                     (the "342 in total" set of §3.2), used at every omega

Run::
    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_effective_subset_residual.py
Output: data/rev_effective_subset_residual.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from rev_bench_rom import FIT_BAND_HZ, N_FIT_SAMPLES, N_SEG, direct_fitted
from rev_env import environment_record
from cmame_e2_segmentwise_rom import setup_amichi_8x8
from phase3_amichi_polynomial_qep import (assemble_complex_per_interval_amichi, fit_G_polynomial,
                                          sample_G_omega, solve_qep_per_interval)

_ROOT = Path(__file__).resolve().parent.parent


def weights(lams, vecs, M, C, F, obs):
    ok = np.isfinite(lams)
    lams, v = lams[ok], vecs[:, ok]
    a_k = 2 * lams * np.einsum("ij,ij->j", v, M @ v) + np.einsum("ij,ij->j", v, C @ v)
    nz = np.abs(a_k) > 1e-15 * np.abs(a_k).max()
    lams, v, a_k = lams[nz], v[:, nz], a_k[nz]
    return lams, v[obs, :] * (v.T @ F) / a_k


def pole_sum(omegas, lams, w):
    return np.array([np.sum(w / (1j * om - lams)) for om in omegas])


def stats(Y, Y_ref):
    mask = np.abs(Y_ref) > 0.01 * np.abs(Y_ref).max()
    rel = np.abs(Y - Y_ref) / np.abs(Y_ref) * 100.0
    return dict(peak_band_max_pct=float(rel[mask].max()), peak_band_mean_pct=float(rel[mask].mean()),
                peak_band_median_pct=float(np.median(rel[mask])), all_max_pct=float(rel.max()),
                all_mean_pct=float(rel.mean()), n_peak_band=int(mask.sum()))


def main():
    _, M, K_base, K_shear, F, obs = setup_amichi_8x8(bc="simply_supported_w")
    F_c = F.astype(complex)
    omega, G_re, G_im = sample_G_omega(*FIT_BAND_HZ, N_FIT_SAMPLES)
    spl_re, spl_im, _ = fit_G_polynomial(omega, G_re, G_im, n_segments=N_SEG)
    mats = assemble_complex_per_interval_amichi(spl_re, spl_im, M, K_base, K_shear)
    f_eval = np.linspace(5.0, 500.0, 200)
    om_eval = 2 * np.pi * f_eval
    Y_ref = direct_fitted(mats, F_c, obs, om_eval)
    sols = solve_qep_per_interval(mats, damping_tol=1e-6)
    lowers = np.array([s["omega_lower"] for s in sols])
    seg_of = np.clip(np.searchsorted(lowers, om_eval, side="right") - 1, 0, len(sols) - 1)

    per_seg = {}
    for i, s in enumerate(sols):
        mat = s["matrices"]; Ms, Cs = mat["M_star"], mat["C_star"]
        per_seg[i] = dict(
            complete=weights(s["lams_all"], s["vecs_all"], Ms, Cs, F_c, obs),
            effective_only=weights(s["lams_effective"], s["vecs_effective"], Ms, Cs, F_c, obs),
            effective_conj=weights(s["lams_full_with_conj"], s["vecs_full_with_conj"], Ms, Cs, F_c, obs),
        )
    Y = {k: np.zeros(om_eval.size, dtype=complex) for k in ("complete", "effective_only", "effective_conj")}
    for i in range(len(sols)):
        sel = seg_of == i
        for k in Y:
            lams, w = per_seg[i][k]
            Y[k][sel] = pole_sum(om_eval[sel], lams, w) if lams.size else np.nan
    lams_all = np.concatenate([per_seg[i]["effective_conj"][0] for i in range(len(sols))])
    w_all = np.concatenate([per_seg[i]["effective_conj"][1] for i in range(len(sols))])
    Y["effective_conj_all"] = pole_sum(om_eval, lams_all, w_all)

    out = dict(environment=environment_record(), n_free_dof=int(M.shape[0]), n_freq=int(om_eval.size),
               band_hz=[5.0, 500.0], reference="full-order solve with fitted segment matrices",
               effective_counts=[int(s["lams_effective"].size) for s in sols],
               n_effective_conj_all=int(lams_all.size), variants={})
    for k, y in Y.items():
        out["variants"][k] = stats(y, Y_ref)
        print(f"{k:20s}", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in out["variants"][k].items()})
    # error of the pooled set, per segment band, so the reader sees where it sits
    rel_all = np.abs(Y["effective_conj_all"] - Y_ref) / np.abs(Y_ref) * 100.0
    mask = np.abs(Y_ref) > 0.01 * np.abs(Y_ref).max()
    out["effective_conj_all_by_segment"] = [
        dict(band_hz=[float(sols[i]["omega_lower"] / 2 / np.pi), float(sols[i]["omega_upper"] / 2 / np.pi)],
             n_peak_band=int((mask & (seg_of == i)).sum()),
             peak_band_max_pct=float(rel_all[mask & (seg_of == i)].max()) if (mask & (seg_of == i)).any() else None,
             peak_band_mean_pct=float(rel_all[mask & (seg_of == i)].mean()) if (mask & (seg_of == i)).any() else None)
        for i in range(len(sols))]
    for r in out["effective_conj_all_by_segment"]:
        print("   pooled, band", r)
    path = _ROOT / "data" / "rev_effective_subset_residual.json"
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
