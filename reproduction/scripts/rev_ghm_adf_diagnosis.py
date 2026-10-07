"""Additional studies — diagnosis of the GHM/ADF order study.

Question: why do the GHM/ADF fit errors of the comparison table not decrease
consistently with the model order?

For every order and every sampling configuration this script records

1. the optimal objective value (weighted joint L2 norm minimised by the
   fit) next to the reported metrics (component-wise max / mean relative
   error), so that "objective decreased, max error increased" is visible;
2. the multi-start statistics: objective spread over the starts, solver
   termination status, and the bounds that are active at the best fit;
3. a nesting check: the order-N optimum embedded in the order-(N+1) model
   (extra term at its lower bound), and a warm start from that point;
4. the frequency at which each max error occurs, and the errors restricted
   to the 5--500 Hz comparison band;
5. the peak-band FRF error produced by each fitted modulus.

Sampling configurations
    A  uniform 1--1500 Hz, 200 points   (the submitted setting)
    B  log-spaced 1--1500 Hz, 200 points
    C  log-spaced 5--500 Hz, 200 points (comparison band only)
    D  uniform 5--500 Hz, 200 points

The FRF error of a fitted modulus is evaluated with the condensed solve
[K_base + G_fit(w) K_shear - w^2 M] x = F, which is algebraically identical to
the augmented GHM/ADF system; the identity is checked against the genuine
augmented solve for the four submitted cases.

Run::

    export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_ghm_adf_diagnosis.py

Output: ``data/rev_ghm_adf_diagnosis.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from rev_env import environment_record

from phase3_amichi_polynomial_qep import G_complex_at_freq
from cmame_c1_ghm_adf_comparison import build_amichi_8x8_ss
from baselines import adf_augmentation as adf
from baselines import ghm_augmentation as ghm

_ROOT = Path(__file__).resolve().parent.parent
SEED = 20260509
N_STARTS = 30
ORDERS = (1, 2, 3, 4, 5, 6)
F_EVAL = np.linspace(5.0, 500.0, 200)          # FRF comparison grid
SAMPLINGS = {
    "A_uniform_1_1500": np.linspace(1.0, 1500.0, 200),
    "B_log_1_1500": np.geomspace(1.0, 1500.0, 200),
    "C_log_5_500": np.geomspace(5.0, 500.0, 200),
    "D_uniform_5_500": np.linspace(5.0, 500.0, 200),
}


def modulus_data(f_hz):
    G = np.array([G_complex_at_freq(f) for f in f_hz])
    return 2.0 * np.pi * np.asarray(f_hz), G.real, G.imag


# ----------------------------------------------------------------------------
# Model adapters: one small record per method keeps the study code shared.
# ----------------------------------------------------------------------------
def _ghm_start(k, n, rng, omega, G_re, box_decades=0.0):
    widen = 3.0 * 10.0**box_decades
    log_lo, log_hi = np.log(omega.min() / widen), np.log(omega.max() * widen)
    G0 = float(np.median(G_re))
    if G0 <= 0:
        G0 = float(max(np.abs(G_re).mean(), 1.0))
    if k == 0:
        return ghm._pack(G0, np.full(n, 0.5),
                         np.exp(np.linspace(log_lo, log_hi, n)), np.full(n, 0.3))
    log_w = np.sort(rng.uniform(log_lo, log_hi, size=n))
    alpha = rng.uniform(0.05, 5.0, size=n)
    zeta = rng.uniform(0.05, 0.6, size=n)
    return ghm._pack(G0 * float(rng.uniform(0.3, 3.0)), alpha, np.exp(log_w), zeta)


def _adf_start(k, n, rng, omega, G_re, box_decades=0.0):
    widen = 3.0 * 10.0**box_decades
    log_lo, log_hi = np.log(omega.min() / widen), np.log(omega.max() * widen)
    G0 = float(max(G_re.min(), 1.0))
    if k == 0:
        return adf._pack_adf(G0, np.full(n, max(G0, 1.0)),
                             np.exp(np.linspace(log_lo, log_hi, n)))
    log_om = np.sort(rng.uniform(log_lo, log_hi, size=n))
    Gv = G0 * rng.uniform(0.1, 5.0, size=n)
    return adf._pack_adf(G0 * float(rng.uniform(0.3, 3.0)), Gv, np.exp(log_om))


def _ghm_eval(x, n, omega):
    G_inf, a, w, z = ghm._unpack(x, n)
    return ghm.ghm_modulus(omega, ghm.GHMParameters(G_inf, a, w, z))


def _adf_eval(x, n, omega):
    G_e, Gv, Om = adf._unpack_adf(x, n)
    return adf.adf_modulus(omega, adf.ADFParameters(G_e, Gv, Om))


def _ghm_names(n):
    return (["G_inf"] + [f"alpha_{i+1}" for i in range(n)]
            + [f"what_{i+1}" for i in range(n)] + [f"zeta_{i+1}" for i in range(n)])


def _adf_names(n):
    return (["G_e"] + [f"Gv_{i+1}" for i in range(n)]
            + [f"Omega_{i+1}" for i in range(n)])


def _ghm_embed(x, n, lb_next):
    """Order-n optimum inside the order-(n+1) box: extra oscillator at its
    lower strength bound, frequency/damping at the box centre."""
    G_inf, a, w, z = ghm._unpack(x, n)
    lb_a = np.exp(lb_next[1])
    w_new = np.exp(0.5 * (np.log(w.min()) + np.log(w.max()))) if n > 1 else w[0]
    return ghm._pack(G_inf, np.append(a, lb_a), np.append(w, w_new),
                     np.append(z, 0.5))


def _adf_embed(x, n, lb_next):
    G_e, Gv, Om = adf._unpack_adf(x, n)
    lb_gv = np.exp(lb_next[1])
    Om_new = np.exp(0.5 * (np.log(Om.min()) + np.log(Om.max()))) if n > 1 else Om[0]
    return adf._pack_adf(G_e, np.append(Gv, lb_gv), np.append(Om, Om_new))


METHODS = {
    "GHM": dict(start=_ghm_start, residual=ghm._residual, bounds=ghm._ghm_bounds,
                evaluate=_ghm_eval, names=_ghm_names, embed=_ghm_embed,
                n_params=lambda n: 3 * n + 1),
    "ADF": dict(start=_adf_start, residual=adf._residual_adf,
                bounds=adf._adf_bounds, evaluate=_adf_eval, names=_adf_names,
                embed=_adf_embed, n_params=lambda n: 2 * n + 1),
}


# ----------------------------------------------------------------------------
def pole_box(method, n, omega, box_decades=0.0):
    """Library bounds; ``box_decades`` widens the pole-frequency box on each
    side (0 = the submitted [w_min/3, 3 w_max])."""
    k = 10.0**box_decades
    return METHODS[method]["bounds"](n, omega.min() / k, omega.max() * k)


def jacobian_condition(jac, x, lb, ub, tol=1e-3):
    """2-norm condition number of the residual Jacobian restricted to the
    parameters that are not at a bound."""
    free = (x - lb > tol) & (ub - x > tol)
    if not free.any():
        return float("nan")
    sv = np.linalg.svd(np.asarray(jac)[:, free], compute_uv=False)
    return float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")


def multistart(method, n, omega, G_re, G_im, rng, n_starts=N_STARTS,
               box_decades=0.0):
    """Same start generation and solver call as the library fit routines,
    with the per-start solver diagnostics kept."""
    m = METHODS[method]
    weight = 1.0 / np.maximum(np.sqrt(G_re**2 + G_im**2), 1.0)
    lb, ub = pole_box(method, n, omega, box_decades)
    args = (omega, G_re, G_im, n, weight)
    runs = []
    for k in range(n_starts):
        x0 = np.clip(m["start"](k, n, rng, omega, G_re, box_decades),
                     lb + 1e-6, ub - 1e-6)
        sol = least_squares(m["residual"], x0, args=args, method="trf",
                            bounds=(lb, ub), max_nfev=4000)
        runs.append(dict(x=sol.x, norm=float(np.linalg.norm(sol.fun)),
                         status=int(sol.status), nfev=int(sol.nfev),
                         optimality=float(sol.optimality),
                         jac_cond_free=jacobian_condition(sol.jac, sol.x, lb, ub)))
    best = min(runs, key=lambda r: r["norm"])
    return best, runs, (lb, ub), args


def active_bounds(x, lb, ub, names, tol=1e-3):
    """Parameters within ``tol`` (optimiser log/logit units) of a bound."""
    out = []
    for xi, lo, hi, name in zip(x, lb, ub, names):
        if xi - lo < tol:
            out.append(f"{name}@lower")
        elif hi - xi < tol:
            out.append(f"{name}@upper")
    return out


def band_errors(G_fit, G_re, G_im, f_hz):
    rel_re = np.abs(G_fit.real - G_re) / np.abs(G_re) * 100.0
    rel_im = np.abs(G_fit.imag - G_im) / np.abs(G_im) * 100.0
    return dict(
        re_max=float(rel_re.max()), re_mean=float(rel_re.mean()),
        im_max=float(rel_im.max()), im_mean=float(rel_im.mean()),
        f_at_re_max_hz=float(f_hz[int(rel_re.argmax())]),
        f_at_im_max_hz=float(f_hz[int(rel_im.argmax())]),
    )


def frf_condensed(K_base_c, K_shear_c, M_c, F_c, obs, omega, G_values):
    Y = np.zeros(omega.size, dtype=complex)
    for i, (om, G) in enumerate(zip(omega, G_values)):
        Y[i] = np.linalg.solve(K_base_c + G * K_shear_c - om * om * M_c, F_c)[obs]
    return Y


def peak_band_error(Y, Y_ref):
    mask = np.abs(Y_ref) > 0.01 * np.abs(Y_ref).max()
    rel = np.abs(Y[mask] - Y_ref[mask]) / np.abs(Y_ref[mask]) * 100.0
    return float(rel.max()), float(rel.mean())


def describe(method, x, n):
    if method == "GHM":
        G_inf, a, w, z = ghm._unpack(x, n)
        order = np.argsort(w)
        return dict(G_inf_Pa=float(G_inf), alpha=a[order].tolist(),
                    what_hz=(w[order] / (2 * np.pi)).tolist(),
                    zeta=z[order].tolist())
    G_e, Gv, Om = adf._unpack_adf(x, n)
    order = np.argsort(Om)
    return dict(G_e_Pa=float(G_e), Gv_Pa=Gv[order].tolist(),
                Omega_hz=(Om[order] / (2 * np.pi)).tolist())


# ----------------------------------------------------------------------------
def study(method, sampling_name, f_fit, system, Y_ref, G_eval_ref,
          box_decades=0.0):
    K_base_c, K_shear_c, M_c, F_c, obs = system
    omega_fit, G_re, G_im = modulus_data(f_fit)
    omega_eval = 2.0 * np.pi * F_EVAL
    m = METHODS[method]
    rows, prev = [], None
    for n in ORDERS:
        rng = np.random.default_rng([SEED, sorted(METHODS).index(method), n])
        best, runs, (lb, ub), args = multistart(method, n, omega_fit, G_re, G_im, rng,
                                                box_decades=box_decades)
        norms = np.array([r["norm"] for r in runs])
        G_fit_own = m["evaluate"](best["x"], n, omega_fit)
        G_fit_eval = m["evaluate"](best["x"], n, omega_eval)
        Y = frf_condensed(K_base_c, K_shear_c, M_c, F_c, obs, omega_eval, G_fit_eval)
        frf_max, frf_mean = peak_band_error(Y, Y_ref)
        row = dict(
            method=method, sampling=sampling_name, order=n,
            n_params=m["n_params"](n),
            objective=best["norm"],
            objective_rms_rel_pct=float(best["norm"] / np.sqrt(2 * omega_fit.size) * 100),
            starts=dict(
                n=len(runs), best=float(norms.min()),
                median=float(np.median(norms)), worst=float(norms.max()),
                n_within_1pct_of_best=int(np.sum(norms <= 1.01 * norms.min())),
                status_counts={str(s): int(np.sum([r["status"] == s for r in runs]))
                               for s in sorted({r["status"] for r in runs})},
                best_status=best["status"], best_nfev=best["nfev"],
                best_optimality=best["optimality"],
                best_jacobian_condition_free=best["jac_cond_free"],
            ),
            pole_box_extra_decades=box_decades,
            active_bounds=active_bounds(best["x"], lb, ub, m["names"](n)),
            parameters=describe(method, best["x"], n),
            fit_grid_errors=band_errors(G_fit_own, G_re, G_im, f_fit),
            comparison_band_errors=band_errors(
                G_fit_eval, G_eval_ref.real, G_eval_ref.imag, F_EVAL),
            frf_peak_band_max_pct=frf_max, frf_peak_band_mean_pct=frf_mean,
        )
        if prev is not None:
            lb_n, ub_n = pole_box(method, n, omega_fit, box_decades)
            x_emb = np.clip(m["embed"](prev["x"], n - 1, lb_n), lb_n, ub_n)
            j_emb = float(np.linalg.norm(m["residual"](x_emb, *args)))
            warm = least_squares(m["residual"],
                                 np.clip(x_emb, lb_n + 1e-6, ub_n - 1e-6),
                                 args=args, method="trf", bounds=(lb_n, ub_n),
                                 max_nfev=4000)
            j_warm = float(np.linalg.norm(warm.fun))
            row["nesting"] = dict(
                objective_prev_order=prev["norm"],
                objective_embedded=j_emb,
                objective_warm_start=j_warm,
                multistart_minus_warm=best["norm"] - j_warm,
                objective_monotone=bool(best["norm"] <= prev["norm"] * (1 + 1e-9)),
            )
        rows.append(row)
        prev = best
        print(f"  {method} N={n} [{sampling_name}] J={best['norm']:.4f} "
              f"fit Re/Im max {row['fit_grid_errors']['re_max']:.1f}/"
              f"{row['fit_grid_errors']['im_max']:.1f}%  band Re/Im max "
              f"{row['comparison_band_errors']['re_max']:.1f}/"
              f"{row['comparison_band_errors']['im_max']:.1f}%  "
              f"FRF {frf_max:.2f}/{frf_mean:.2f}%  active={row['active_bounds']}")
    return rows


def submitted_cases(system_real, Y_ref):
    """The four submitted fits with the shared random stream of the original
    generator, solved with the genuine augmented systems."""
    K_base, K_shear, M, F, obs = system_real
    omega_fit, G_re, G_im = modulus_data(SAMPLINGS["A_uniform_1_1500"])
    omega_eval = 2.0 * np.pi * F_EVAL
    K_base_c, K_shear_c, M_c = (K_base.astype(complex), K_shear.astype(complex),
                                M.astype(complex))
    F_c = F.astype(complex)
    rng = np.random.default_rng(SEED)
    out = []
    for method, n in (("GHM", 3), ("GHM", 5), ("ADF", 2), ("ADF", 5)):
        if method == "GHM":
            params, info = ghm.fit_ghm_parameters(omega_fit, G_re, G_im, n_mini=n,
                                                  n_starts=N_STARTS, rng=rng)
            aug = ghm.assemble_ghm_augmented(K_base, K_shear, M, F, params)
            Y_aug = ghm.ghm_frequency_response(aug, omega_eval, obs)
            n_aug = int(aug.M_aug.shape[0])
            G_eval = ghm.ghm_modulus(omega_eval, params)
        else:
            params, info = adf.fit_adf_parameters(omega_fit, G_re, G_im, n_adf=n,
                                                  n_starts=N_STARTS, rng=rng)
            ss = adf.assemble_adf_state_space(K_base, K_shear, M, F, params)
            Y_aug = adf.adf_frequency_response(ss, F, omega_eval, obs)
            n_aug = int(ss.A.shape[0])
            G_eval = adf.adf_modulus(omega_eval, params)
        Y_cond = frf_condensed(K_base_c, K_shear_c, M_c, F_c, obs, omega_eval, G_eval)
        frf_max, frf_mean = peak_band_error(Y_aug, Y_ref)
        cond_max, cond_mean = peak_band_error(Y_cond, Y_ref)
        row = dict(
            method=method, order=n, n_aug=n_aug,
            fit_re_max=info["rel_re_max"], fit_re_mean=info["rel_re_mean"],
            fit_im_max=info["rel_im_max"], fit_im_mean=info["rel_im_mean"],
            objective=float(info["best_residual_norm"]),
            frf_augmented_max_pct=frf_max, frf_augmented_mean_pct=frf_mean,
            frf_condensed_max_pct=cond_max, frf_condensed_mean_pct=cond_mean,
            augmented_vs_condensed_rel=float(
                np.max(np.abs(Y_aug - Y_cond)) / np.max(np.abs(Y_cond))),
        )
        out.append(row)
        print(f"  submitted {method} N={n}: n_aug={n_aug} fit Re/Im max "
              f"{row['fit_re_max']:.1f}/{row['fit_im_max']:.1f}% "
              f"FRF aug {frf_max:.2f}/{frf_mean:.2f}% "
              f"cond {cond_max:.2f}/{cond_mean:.2f}% "
              f"aug-vs-cond {row['augmented_vs_condensed_rel']:.1e}")
    return out


def main():
    parser = argparse.ArgumentParser(description="GHM/ADF order-study diagnosis")
    parser.add_argument("--skip-submitted", action="store_true",
                        help="skip the augmented solves of the submitted cases")
    args = parser.parse_args()

    env = environment_record()
    K_base, K_shear, M, F, obs, _ = build_amichi_8x8_ss()
    omega_eval = 2.0 * np.pi * F_EVAL
    G_eval_ref = np.array([G_complex_at_freq(f) for f in F_EVAL])
    system = (K_base.astype(complex), K_shear.astype(complex), M.astype(complex),
              F.astype(complex), obs)
    Y_ref = frf_condensed(*system, omega_eval, G_eval_ref)
    print(f"Amichi 8x8 simply supported, n = {K_base.shape[0]}, "
          f"{F_EVAL.size} frequencies 5-500 Hz")

    result = dict(environment=env, seed=SEED, n_starts=N_STARTS,
                  orders=list(ORDERS),
                  samplings={k: [float(v[0]), float(v[-1]), int(v.size)]
                             for k, v in SAMPLINGS.items()})
    if not args.skip_submitted:
        print("\n[submitted cases, augmented solve]")
        result["submitted_cases"] = submitted_cases((K_base, K_shear, M, F, obs), Y_ref)

    result["order_study"] = []
    for sampling_name, f_fit in SAMPLINGS.items():
        for method in ("GHM", "ADF"):
            print(f"\n[{method}, sampling {sampling_name}]")
            result["order_study"] += study(method, sampling_name, f_fit, system,
                                           Y_ref, G_eval_ref)

    for method in ("GHM", "ADF"):
        print(f"\n[{method}, sampling A, pole box widened by one decade per side]")
        result["order_study"] += study(
            method, "A_uniform_1_1500_wide_box", SAMPLINGS["A_uniform_1_1500"],
            system, Y_ref, G_eval_ref, box_decades=1.0)

    out = _ROOT / "data" / "rev_ghm_adf_diagnosis.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
