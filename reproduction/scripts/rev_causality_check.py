"""Additional studies — can a causal relaxation model represent the benchmark modulus?

A generalized-Maxwell (Prony) modulus with non-negative strengths,

    G(w) = G_e + sum_k g_k * (i w) / (i w + W_k),    G_e >= 0,  g_k >= 0,

is the general form of a linear viscoelastic solid with a positive relaxation
spectrum; the ADF series is a finite member of this family and the GHM
mini-oscillator approaches it in the overdamped limit. With the relaxation
frequencies W_k fixed on a dense log grid that extends far beyond the data
band, fitting the strengths is a convex non-negative least-squares problem.
Its global optimum bounds what any finite Prony-type fit can reach, whatever
the starting points, parameter boxes, or model order.

Three fits per data set, all weighted by 1/|G| as in the GHM/ADF routines:

    joint     Re and Im together      -> residual floor of the model family
    Re only   storage modulus alone   -> the loss modulus that the measured
                                         dispersion implies
    Im only   loss modulus alone      -> the dispersion that the measured
                                         loss implies

Run::

    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_causality_check.py

Output: ``data/rev_causality_check.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from rev_modulus_data import digitized, digitized_table, representative

_ROOT = Path(__file__).resolve().parent.parent
POLES_PER_DECADE = 8
BAND_EXTENSION_DECADES = 4


def prony_blocks(f_hz, pole_hz):
    """Real and imaginary design blocks; column 0 is the static modulus G_e."""
    w = 2.0 * np.pi * np.asarray(f_hz)[:, None]
    W = 2.0 * np.pi * np.asarray(pole_hz)[None, :]
    den = w**2 + W**2
    ones, zeros = np.ones((w.shape[0], 1)), np.zeros((w.shape[0], 1))
    return np.hstack([ones, w**2 / den]), np.hstack([zeros, w * W / den])


def component_errors(G_fit, G):
    rel_re = np.abs(G_fit.real - G.real) / np.abs(G.real) * 100.0
    rel_im = np.abs(G_fit.imag - G.imag) / np.abs(G.imag) * 100.0
    return dict(re_max=float(rel_re.max()), re_mean=float(rel_re.mean()),
                im_max=float(rel_im.max()), im_mean=float(rel_im.mean()))


def analyse(name, f_hz, G):
    lo = np.log10(f_hz.min()) - BAND_EXTENSION_DECADES
    hi = np.log10(f_hz.max()) + BAND_EXTENSION_DECADES
    pole_hz = np.logspace(lo, hi, int(round((hi - lo) * POLES_PER_DECADE)) + 1)
    A_re, A_im = prony_blocks(f_hz, pole_hz)
    wgt = 1.0 / np.abs(G)

    def solve(rows, rhs):
        x, _ = nnls(rows, rhs, maxiter=50 * rows.shape[1])
        return x

    x_joint = solve(np.vstack([wgt[:, None] * A_re, wgt[:, None] * A_im]),
                    np.concatenate([wgt * G.real, wgt * G.imag]))
    x_re = solve(wgt[:, None] * A_re, wgt * G.real)
    x_im = solve(wgt[:, None] * A_im, wgt * G.imag)

    def model(x):
        return A_re @ x + 1j * (A_im @ x)

    G_joint, G_from_re, G_from_im = model(x_joint), model(x_re), model(x_im)
    residual = np.concatenate([wgt * (G_joint.real - G.real),
                               wgt * (G_joint.imag - G.imag)])
    slope = np.gradient(np.log(G.real), np.log(f_hz))
    out = dict(
        data_set=name, n_samples=int(f_hz.size),
        band_hz=[float(f_hz.min()), float(f_hz.max())],
        n_poles=int(pole_hz.size),
        pole_range_hz=[float(pole_hz[0]), float(pole_hz[-1])],
        data_loss_factor_range=[float((G.imag / G.real).min()),
                                float((G.imag / G.real).max())],
        data_storage_loglog_slope_range=[float(slope.min()), float(slope.max())],
        joint_fit=dict(
            rms_relative_residual_pct=float(
                np.linalg.norm(residual) / np.sqrt(residual.size) * 100.0),
            n_active_terms=int(np.sum(x_joint[1:] > 0)),
            **component_errors(G_joint, G)),
        storage_only_fit=dict(
            storage_error=component_errors(G_from_re, G)["re_max"],
            implied_loss_over_data_loss_min=float(
                (G_from_re.imag / G.imag).min()),
            implied_loss_over_data_loss_max=float(
                (G_from_re.imag / G.imag).max()),
            implied_loss_factor_range=[
                float((G_from_re.imag / G_from_re.real).min()),
                float((G_from_re.imag / G_from_re.real).max())]),
        loss_only_fit=dict(
            loss_error=component_errors(G_from_im, G)["im_max"],
            implied_storage_over_data_storage_min=float(
                (G_from_im.real / G.real).min()),
            implied_storage_over_data_storage_max=float(
                (G_from_im.real / G.real).max())),
    )
    j, s = out["joint_fit"], out["storage_only_fit"]
    print(f"{name}: eta {out['data_loss_factor_range'][0]:.2f}-"
          f"{out['data_loss_factor_range'][1]:.2f}, "
          f"dlnG'/dlnf {out['data_storage_loglog_slope_range'][0]:.2f}-"
          f"{out['data_storage_loglog_slope_range'][1]:.2f}")
    print(f"   joint NNLS floor: rms {j['rms_relative_residual_pct']:.2f}%  "
          f"Re max/mean {j['re_max']:.1f}/{j['re_mean']:.1f}%  "
          f"Im max/mean {j['im_max']:.1f}/{j['im_mean']:.1f}%  "
          f"({j['n_active_terms']} active terms)")
    print(f"   storage-only fit (Re max err {s['storage_error']:.2f}%): implied "
          f"loss / data loss = {s['implied_loss_over_data_loss_min']:.2f}-"
          f"{s['implied_loss_over_data_loss_max']:.2f}, implied eta "
          f"{s['implied_loss_factor_range'][0]:.2f}-"
          f"{s['implied_loss_factor_range'][1]:.2f}")
    return out


def main():
    f_dig, _, _ = digitized_table()
    cases = [
        ("representative, uniform 1-1500 Hz (submitted fit grid)",
         np.linspace(1.0, 1500.0, 200), representative),
        ("representative, log 5-500 Hz", np.geomspace(5.0, 500.0, 200),
         representative),
        ("representative, log 10-1000 Hz (anchor span)",
         np.geomspace(10.0, 1000.0, 200), representative),
        ("digitized Fig. 7, 75-639 Hz (142 points)", f_dig, digitized),
    ]
    results = [analyse(name, f, fn(f)) for name, f, fn in cases]
    out = _ROOT / "data" / "rev_causality_check.json"
    with open(out, "w") as fh:
        json.dump(dict(poles_per_decade=POLES_PER_DECADE,
                       band_extension_decades=BAND_EXTENSION_DECADES,
                       results=results), fh, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
