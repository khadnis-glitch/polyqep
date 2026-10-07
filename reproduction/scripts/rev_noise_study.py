"""Additional studies — robustness to noisy modulus data.

A small Monte Carlo study on the benchmark of the seven-method comparison
(Amichi 8x8, hard simple support, n = 471, 496 frequencies in 5--500 Hz).

Noise model (fixed before running)
    The seven anchor values of the storage modulus G' and of the loss modulus
    G'' are multiplied by independent factors (1 + sigma * e), e ~ N(0, 1),
    for sigma = 1 %, 3 %, 5 %; 50 draws per level, one seeded generator.
    The noisy profile is interpolated between the anchors exactly as the
    benchmark profile (linear in log-frequency for E' and for the loss factor).

For every draw the complete procedure is repeated on the noisy profile: the
C0/C1 piecewise-quadratic fit on the baseline partition, the SOAR bases, and
the PW-R and PW-RG operators. Three errors are reported separately, each as
the maximum over the 496 frequencies of the relative FRF error:

    method error      e_m   a method against the direct solve ON THE SAME noisy
                            data: fitted-modulus full-order solve, PW-R, PW-RG
    data propagation  e_d   direct solve on the noisy data against the direct
                            solve on the noise-free data (method-independent)
    total             e_t   fitted-modulus full-order solve on the noisy data
                            against the noise-free direct solve

The direct solve always uses the interpolated (raw) modulus of its data set.

Run::

    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_noise_study.py

Output: ``data/rev_noise_study.json`` and ``data/rev_noise_example.npz``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rev_bench_rom import (
    F_MAX_HZ, F_MIN_HZ, FIT_BAND_HZ, M_PER_SEG, N_FIT_SAMPLES, N_FREQ, SVD_TOL,
    direct_fitted,
)
from rev_env import environment_record
from rev_modulus_data import NU_CORE, representative
from rev_partition_sensitivity import BASELINE_HZ

from cmame_e2_segmentwise_rom import setup_amichi_8x8
from cmame_f1_xie2018_comparison_v3 import relative_error_pct
from phase3_amichi_polynomial_qep import assemble_complex_per_interval_amichi
from polyqep.pw_refined import (
    _orthonormalize_columns_svd,
    _project_operator_pieces,
)
from polyqep.rom import build_soar_rom
from polyqep.fitting import fit_joint_spline

_ROOT = Path(__file__).resolve().parent.parent
F_ANCHORS = np.array([10.0, 50.0, 100.0, 200.0, 400.0, 700.0, 1000.0])
E_ANCHORS = np.array([4.5e6, 5.0e6, 6.0e6, 8.0e6, 1.4e7, 1.8e7, 2.0e7])
ETA_ANCHORS = np.array([1.40, 1.36, 1.34, 1.30, 1.22, 1.18, 1.15])
SIGMAS = (0.01, 0.03, 0.05)
N_DRAWS, SEED = 50, 20261004


def profile(E_anchor, eta_anchor):
    """Complex shear modulus G(f) interpolated as the benchmark profile."""
    log_fa = np.log(F_ANCHORS)

    def G(f_hz):
        log_f = np.log(np.clip(np.asarray(f_hz, dtype=float), 1.0, 2000.0))
        E = np.interp(log_f, log_fa, E_anchor)
        eta = np.interp(log_f, log_fa, eta_anchor)
        return E / (2.0 * (1.0 + NU_CORE)) * (1.0 + 1j * eta)

    return G


def noisy_anchors(rng, sigma):
    """Independent multiplicative noise on the anchors of G' and of G''."""
    g_store = E_ANCHORS * (1.0 + sigma * rng.standard_normal(F_ANCHORS.size))
    g_loss = (E_ANCHORS * ETA_ANCHORS
              * (1.0 + sigma * rng.standard_normal(F_ANCHORS.size)))
    return g_store, g_loss / g_store


def solve_all(G, system, omegas, f_fit):
    """Direct, fitted full-order, PW-R and PW-RG responses for a profile G."""
    M, K_base, K_shear, F, obs = system
    K_base_c, K_shear_c = K_base.astype(complex), K_shear.astype(complex)
    F_c = F.astype(complex)
    f_eval = omegas / (2.0 * np.pi)
    G_eval = G(f_eval)
    Y_dir = np.array([np.linalg.solve(K_base_c + g * K_shear_c - om * om * M, F_c)[obs]
                      for om, g in zip(omegas, G_eval)])

    G_fit = G(f_fit)
    spl_re, spl_im = fit_joint_spline(2.0 * np.pi * f_fit, G_fit.real, G_fit.imag,
                                      2.0 * np.pi * BASELINE_HZ)
    mats = assemble_complex_per_interval_amichi(spl_re, spl_im, M, K_base, K_shear)
    Y_fit = direct_fitted(mats, F_c, obs, omegas)

    soar = [build_soar_rom(m["M_star"], m["C_star"], m["K_star"],
                           0.5 * (m["omega_lower"] + m["omega_upper"]),
                           M_PER_SEG, F) for m in mats]
    lowers = np.array([m["omega_lower"] for m in mats])
    pieces_r = [_project_operator_pieces(s.Q, K_base, K_shear, M, F) for s in soar]
    Y_r = np.zeros(omegas.size, dtype=complex)
    for i, (om, g) in enumerate(zip(omegas, G_eval)):
        j = int(np.clip(np.searchsorted(lowers, om, side="right") - 1,
                        0, len(soar) - 1))
        Kb, Ks, Mr, Fr = pieces_r[j]
        Y_r[i] = (soar[j].Q @ np.linalg.solve(Kb + g * Ks - om * om * Mr, Fr))[obs]
    Q_g = _orthonormalize_columns_svd(np.column_stack([s.Q for s in soar]),
                                      tol_sv=SVD_TOL)
    Kb, Ks, Mr, Fr = _project_operator_pieces(Q_g, K_base, K_shear, M, F)
    Y_g = np.array([(Q_g @ np.linalg.solve(Kb + g * Ks - om * om * Mr, Fr))[obs]
                    for om, g in zip(omegas, G_eval)])
    fit_err = dict(
        re_max=float(np.max(np.abs(spl_re(2 * np.pi * f_eval) - G_eval.real)
                            / np.abs(G_eval.real)) * 100.0),
        im_max=float(np.max(np.abs(spl_im(2 * np.pi * f_eval) - G_eval.imag)
                            / np.abs(G_eval.imag)) * 100.0))
    return dict(direct=Y_dir, fitted=Y_fit, pw_r=Y_r, pw_rg=Y_g,
                m_pw_rg=int(Q_g.shape[1]), fit_err=fit_err,
                spline=(spl_re, spl_im))


def stats(values):
    v = np.asarray(values, dtype=float)
    return dict(mean=float(v.mean()), median=float(np.median(v)),
                p95=float(np.percentile(v, 95)), max=float(v.max()))


def main():
    env = environment_record()
    _, M, K_base, K_shear, F, obs = setup_amichi_8x8(bc="simply_supported_w")
    system = (M, K_base, K_shear, F, obs)
    omegas = 2.0 * np.pi * np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)
    f_fit = np.linspace(*FIT_BAND_HZ, N_FIT_SAMPLES)

    clean_G = profile(E_ANCHORS, ETA_ANCHORS)
    f_check = np.geomspace(1.0, 1500.0, 300)
    assert np.allclose(clean_G(f_check), representative(f_check), rtol=1e-13)
    clean = solve_all(clean_G, system, omegas, f_fit)

    def e_max(Y_ref, Y):
        return float(relative_error_pct(Y_ref, Y).max())

    result = dict(
        environment=env, seed=SEED, n_draws=N_DRAWS,
        noise_model="independent multiplicative Gaussian noise on the seven "
                    "anchor values of G' and G''",
        noise_free=dict(
            e_m_fitted=e_max(clean["direct"], clean["fitted"]),
            e_m_pw_r=e_max(clean["direct"], clean["pw_r"]),
            e_m_pw_rg=e_max(clean["direct"], clean["pw_rg"]),
            m_pw_rg=clean["m_pw_rg"], fit_err=clean["fit_err"]),
        levels=[])
    print(f"noise-free: {result['noise_free']}")

    rng = np.random.default_rng(SEED)
    example = None
    for sigma in SIGMAS:
        rec = dict(e_d=[], e_m_fitted=[], e_m_pw_r=[], e_m_pw_rg=[], e_t_fitted=[],
                   m_pw_rg=[], fit_re_max=[], fit_im_max=[])
        for draw in range(N_DRAWS):
            E_n, eta_n = noisy_anchors(rng, sigma)
            run = solve_all(profile(E_n, eta_n), system, omegas, f_fit)
            rec["e_d"].append(e_max(clean["direct"], run["direct"]))
            rec["e_m_fitted"].append(e_max(run["direct"], run["fitted"]))
            rec["e_m_pw_r"].append(e_max(run["direct"], run["pw_r"]))
            rec["e_m_pw_rg"].append(e_max(run["direct"], run["pw_rg"]))
            rec["e_t_fitted"].append(e_max(clean["direct"], run["fitted"]))
            rec["m_pw_rg"].append(run["m_pw_rg"])
            rec["fit_re_max"].append(run["fit_err"]["re_max"])
            rec["fit_im_max"].append(run["fit_err"]["im_max"])
            if sigma == SIGMAS[-1] and draw == 0:
                f_plot = np.geomspace(1.0, 1500.0, 600)
                spl_re, spl_im = run["spline"]
                example = dict(
                    f_plot=f_plot, G_clean=clean_G(f_plot),
                    G_noisy=profile(E_n, eta_n)(f_plot),
                    G_fit=np.asarray(spl_re(2 * np.pi * f_plot))
                    + 1j * np.asarray(spl_im(2 * np.pi * f_plot)),
                    f_anchors=F_ANCHORS, E_noisy=E_n, eta_noisy=eta_n,
                    omegas=omegas, Y_clean=clean["direct"], Y_noisy=run["direct"],
                    Y_fitted=run["fitted"], Y_pw_r=run["pw_r"], Y_pw_rg=run["pw_rg"])
        level = dict(sigma=sigma,
                     m_pw_rg_range=[int(min(rec["m_pw_rg"])), int(max(rec["m_pw_rg"]))],
                     **{k: stats(v) for k, v in rec.items() if k != "m_pw_rg"})
        result["levels"].append(level)
        print(f"sigma={sigma:.0%}: e_d mean/p95 {level['e_d']['mean']:.2f}/"
              f"{level['e_d']['p95']:.2f}%  e_m fitted {level['e_m_fitted']['mean']:.2f}/"
              f"{level['e_m_fitted']['p95']:.2f}%  PW-R {level['e_m_pw_r']['mean']:.3e}/"
              f"{level['e_m_pw_r']['p95']:.3e}%  PW-RG {level['e_m_pw_rg']['mean']:.1e}/"
              f"{level['e_m_pw_rg']['p95']:.1e}%  e_t fitted "
              f"{level['e_t_fitted']['mean']:.2f}/{level['e_t_fitted']['p95']:.2f}%  "
              f"m_RG {level['m_pw_rg_range']}", flush=True)

    out = _ROOT / "data" / "rev_noise_study.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    if example is not None:
        np.savez(_ROOT / "data" / "rev_noise_example.npz", **example)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
