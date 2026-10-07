"""Additional studies — segment count against modulus shape (partition guide of the Discussion).

Three core-modulus shapes on the same benchmark (Amichi 8x8 plate, hard simple
support, n = 471, m_per_seg = 20, SVD truncation 1e-10):

``representative``  the seven-anchor profile of the paper: monotone, with a
                    slope change at every anchor (fit band 1--1500 Hz,
                    reported band 5--500 Hz, 496 frequencies, 2000 log-spaced
                    fit samples);
``digitized``       the pixel-digitized DMA curves of Fig. 7 of Amichi and
                    Atalla (2010): smooth, no loss peak (fit band = measured
                    band 75--639 Hz, the 142 digitized points are the fit
                    samples, reported band 75--500 Hz, 200 frequencies, as in
                    Supplementary Section S2);
``peaked``          the two-relaxation Maxwell-plus-glassy synthetic modulus
                    of Supplementary Section S4 with its loss peak at 159 Hz
                    (fit band 1--1500 Hz, reported band 5--500 Hz, 496
                    frequencies, 2000 log-spaced fit samples).

For each shape the number of segments N is varied (uniform in frequency and
in log-frequency); for the peaked modulus a "peak-centred" family places K
segments of equal log-width across the peak decade [f_p/sqrt(10), f_p*sqrt(10)]
with one segment below and one above (N = K + 2). Each case records

* the maximum fit residual of Re G and Im G inside the reported band;
* the response error of the fitted-modulus full-order solve (the error
  inherited by the modal expansion and PW SOAR) and the ratio
  b = (maximum response error) / (maximum fit residual);
* the PW-R and PW-RG errors (raw modulus online) and the PW-RG basis size;
* the largest number of resonances held by a segment that overlaps the
  reported band (undamped natural frequencies with Re G at the segment
  centre, as in ``rev_partition_modes.py``).

Run::

    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_partition_shapes.py

Output: ``data/rev_partition_shapes.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from rev_bench_rom import M_PER_SEG, SVD_TOL, direct_fitted, err_stats, online_affine_global, online_pw_r
from rev_env import environment_record, require_single_thread
from rev_modulus_data import digitized, digitized_table, representative

from cmame_e2_segmentwise_rom import setup_amichi_8x8
from cmame_f2_sharp_resonance_synthetic import TAU_1, synthetic_G_at_freq
from phase3_amichi_polynomial_qep import assemble_complex_per_interval_amichi
from polyqep.pw_refined import (
    PWRefinedSegmentROM,
    _orthonormalize_columns_svd,
    _project_operator_pieces,
)
from polyqep.rom import build_soar_rom
from polyqep.fitting import fit_joint_spline

_ROOT = Path(__file__).resolve().parent.parent
F_PEAK_HZ = 1.0 / (2.0 * np.pi * TAU_1)


def peaked(f_hz) -> np.ndarray:
    return np.array([synthetic_G_at_freq(f) for f in np.atleast_1d(f_hz)])


SHAPES = {
    "representative": dict(
        G=representative, fit_band=(1.0, 1500.0), eval_hz=np.linspace(5.0, 500.0, 496),
        samples=np.geomspace(1.0, 1500.0, 2000)),
    "digitized": dict(
        G=digitized, fit_band=(75.0, 639.0), eval_hz=np.linspace(75.0, 500.0, 200),
        samples=digitized_table()[0]),
    "peaked": dict(
        G=peaked, fit_band=(1.0, 1500.0), eval_hz=np.linspace(5.0, 500.0, 496),
        samples=np.geomspace(1.0, 1500.0, 2000)),
}
N_UNIFORM = {"representative": (1, 2, 3, 4, 5, 6, 8, 10), "digitized": (1, 2, 3, 4, 5, 6),
             "peaked": (1, 2, 3, 4, 5, 6, 8, 10)}
K_PEAK = (1, 2, 3, 4, 6)


def partitions(shape):
    lo, hi = SHAPES[shape]["fit_band"]
    out = []
    if shape == "representative":
        out.append(("baseline", "50/150/350/700 Hz", np.array([1.0, 50.0, 150.0, 350.0, 700.0, 1500.0])))
    for n in N_UNIFORM[shape]:
        if n == 1:
            out.append(("single", "N=1", np.array([lo, hi])))
            continue
        out.append(("log-uniform", f"N={n}", np.geomspace(lo, hi, n + 1)))
        out.append(("uniform", f"N={n}", np.linspace(lo, hi, n + 1)))
    if shape == "peaked":
        a, b = F_PEAK_HZ / np.sqrt(10.0), F_PEAK_HZ * np.sqrt(10.0)
        for k in K_PEAK:
            edges = np.concatenate([[lo], np.geomspace(a, b, k + 1), [hi]])
            out.append(("peak-centred", f"K={k} on the peak decade", edges))
    return out


def dense(a):
    return np.asarray(a.todense()) if hasattr(a, "todense") else np.asarray(a)


def resonances_in_segment(G, M, K_base, K_shear, f_lo, f_hi):
    f_c = 0.5 * (f_lo + f_hi)
    g = float(np.real(G(f_c)[0]))
    w2 = sla.eigh(K_base + g * K_shear, M, eigvals_only=True)
    f_n = np.sqrt(w2[w2 > 0.0]) / (2.0 * np.pi)
    return int(np.sum((f_n >= f_lo) & (f_n < f_hi)))


def direct_reference(G, K_base, K_shear, M, F, obs, omegas):
    Kb, Ks = K_base.astype(complex), K_shear.astype(complex)
    Y = np.zeros(omegas.size, dtype=complex)
    for i, om in enumerate(omegas):
        g = complex(G(om / (2.0 * np.pi))[0])
        Y[i] = np.linalg.solve(Kb + g * Ks - om**2 * M, F)[obs]
    return Y


def fit_errors(G, spl_re, spl_im, f_hz):
    Gv = G(f_hz)
    om = 2.0 * np.pi * f_hz
    rel_re = np.abs(np.asarray(spl_re(om)) - Gv.real) / np.abs(Gv.real) * 100.0
    rel_im = np.abs(np.asarray(spl_im(om)) - Gv.imag) / np.abs(Gv.imag) * 100.0
    return dict(re_max=float(rel_re.max()), re_mean=float(rel_re.mean()),
                im_max=float(rel_im.max()), im_mean=float(rel_im.mean()))


def run_case(shape, edges_hz, system, Y_ref):
    cfg = SHAPES[shape]
    G = cfg["G"]
    M, K_base, K_shear, F, obs = system
    f_s = cfg["samples"]
    G_s = G(f_s)
    spl_re, spl_im = fit_joint_spline(2.0 * np.pi * f_s, G_s.real, G_s.imag, 2.0 * np.pi * edges_hz)
    mats = assemble_complex_per_interval_amichi(spl_re, spl_im, M, K_base, K_shear)
    omegas = 2.0 * np.pi * cfg["eval_hz"]
    lowers = np.array([m["omega_lower"] for m in mats])
    G_func = lambda s: complex(G(abs(s.imag) / (2.0 * np.pi))[0])  # noqa: E731

    soar = [build_soar_rom(m["M_star"], m["C_star"], m["K_star"],
                           0.5 * (m["omega_lower"] + m["omega_upper"]), M_PER_SEG, F) for m in mats]
    roms = []
    for m, s in zip(mats, soar):
        p = _project_operator_pieces(s.Q, K_base, K_shear, M, F)
        roms.append(PWRefinedSegmentROM(m["omega_lower"], m["omega_upper"], s.Q, p[0], p[1], p[2], p[3]))
    Q_g = _orthonormalize_columns_svd(np.column_stack([s.Q for s in soar]), tol_sv=SVD_TOL)
    pg = _project_operator_pieces(Q_g, K_base, K_shear, M, F)

    fit = fit_errors(G, spl_re, spl_im, cfg["eval_hz"])
    fitted = err_stats(Y_ref, direct_fitted(mats, F.astype(complex), obs, omegas))
    fit_max = max(fit["re_max"], fit["im_max"])
    return dict(
        fit_eval_band=fit,
        frf_fitted_full_order=fitted,
        fitted_over_fit_residual=fitted["max_pct"] / fit_max,
        frf_pw_r=err_stats(Y_ref, online_pw_r(roms, lowers, G_func, obs, omegas)),
        frf_pw_rg=err_stats(Y_ref, online_affine_global(Q_g, pg[0], pg[1], pg[2], pg[3], G_func, obs, omegas)),
        m_pw_rg=int(Q_g.shape[1]),
    )


def main():
    require_single_thread()
    env = environment_record()
    _, M, K_base, K_shear, F, obs = setup_amichi_8x8(bc="simply_supported_w")
    Md, Kbd, Ksd = dense(M), dense(K_base), dense(K_shear)
    system = (M, K_base, K_shear, F, obs)

    result = dict(environment=env, m_per_segment=M_PER_SEG, svd_truncation=SVD_TOL,
                  peak_hz=float(F_PEAK_HZ), shapes={})
    for shape, cfg in SHAPES.items():
        eval_hz = cfg["eval_hz"]
        omegas = 2.0 * np.pi * eval_hz
        Y_ref = direct_reference(cfg["G"], Kbd, Ksd, Md, F, obs, omegas)
        lo, hi = float(eval_hz[0]), float(eval_hz[-1])
        rows = []
        print(f"\n[{shape}] fit band {cfg['fit_band']}, reported {lo}-{hi} Hz, "
              f"{eval_hz.size} frequencies, {len(cfg['samples'])} fit samples")
        for group, label, edges in partitions(shape):
            segs = list(zip(edges[:-1], edges[1:]))
            modes = [resonances_in_segment(cfg["G"], Md, Kbd, Ksd, a, b) for a, b in segs]
            rep = [m for m, (a, b) in zip(modes, segs) if b > lo and a < hi]
            row = dict(group=group, label=label, n_segments=int(edges.size - 1),
                       edges_hz=[float(e) for e in edges], resonances_per_segment=modes,
                       max_resonances_reported=int(max(rep)))
            row.update(run_case(shape, edges, system, Y_ref))
            rows.append(row)
            fe = row["fit_eval_band"]
            print(f"  {group:<13}{label:<26}N={row['n_segments']:<3}fit Re/Im "
                  f"{fe['re_max']:6.2f}/{fe['im_max']:6.2f}%  fitted "
                  f"{row['frf_fitted_full_order']['max_pct']:7.3f}%  b={row['fitted_over_fit_residual']:.2f}  "
                  f"PW-R {row['frf_pw_r']['max_pct']:.2e}%  PW-RG {row['frf_pw_rg']['max_pct']:.1e}% "
                  f"(m={row['m_pw_rg']})  res.max {row['max_resonances_reported']}", flush=True)
        result["shapes"][shape] = dict(
            fit_band_hz=list(cfg["fit_band"]), reported_band_hz=[lo, hi],
            n_freq=int(eval_hz.size), n_fit_samples=int(len(cfg["samples"])), rows=rows)

    out = _ROOT / "data" / "rev_partition_shapes.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
