"""Per-segment complex-matrix assembly and quadratic eigenvalue solves.

Given a piecewise-quadratic fit of the complex shear modulus
Re[G_i(w)] = a_i w^2 + b_i w + c_i and Im[G_i(w)] = d_i w^2 + e_i w + f_i
on each segment i, the frequency-dependent dynamic stiffness

    D(w) = K_base + G(w) K_shear - w^2 M

is recast segment-wise into the omega-independent complex triplet

    M*_i = M      - (a_i + j d_i) K_shear
    C*_i =          (e_i - j b_i) K_shear
    K*_i = K_base + (c_i + j f_i) K_shear

so that D(w) = K*_i + j w C*_i - w^2 M*_i holds exactly on segment i.
One complex quadratic eigenvalue problem (QEP) per segment then yields the
effective eigenpairs used by the modal expansion and by the PW-Refined
reduced-order models.
"""
from __future__ import annotations

import numpy as np

from polyqep.fitting import QuadraticSpline
from polyqep.qep_solver import solve_qep


def assemble_complex_per_interval(
    spline_re: QuadraticSpline,
    spline_im: QuadraticSpline,
    M: np.ndarray,
    K_base: np.ndarray,
    K_shear: np.ndarray,
) -> list[dict]:
    """Assemble the complex (M*, C*, K*) triplet for every fitted segment.

    Parameters
    ----------
    spline_re, spline_im : QuadraticSpline
        Piecewise-quadratic fits of Re[G(w)] and Im[G(w)] sharing the same
        segment boundaries (output of ``fit_joint_spline``).
    M, K_base, K_shear : ndarray, shape (n, n)
        Boundary-condition-reduced system matrices satisfying
        K(w) = K_base + G(w) K_shear.

    Returns
    -------
    list of dict
        One entry per segment with keys ``M_star``, ``C_star``, ``K_star``,
        ``omega_lower``, ``omega_upper``, ``coeffs_re``, ``coeffs_im``.
    """
    results = []
    M_c = M.astype(complex)
    K_base_c = K_base.astype(complex)
    K_shear_c = K_shear.astype(complex)
    for seg_re, seg_im in zip(spline_re.segments, spline_im.segments):
        a, b, c = seg_re.a, seg_re.b, seg_re.c     # Re[G] coefficients
        d, e, f = seg_im.a, seg_im.b, seg_im.c     # Im[G] coefficients
        M_star = M_c - (a + 1j * d) * K_shear_c
        C_star = (e - 1j * b) * K_shear_c
        K_star = K_base_c + (c + 1j * f) * K_shear_c
        results.append({
            "M_star": M_star,
            "C_star": C_star,
            "K_star": K_star,
            "omega_lower": seg_re.omega_lower,
            "omega_upper": seg_re.omega_upper,
            "coeffs_re": (a, b, c),
            "coeffs_im": (d, e, f),
        })
    return results


def solve_qep_per_interval(
    interval_matrices: list[dict],
    damping_tol: float = 1e-9,
) -> list[dict]:
    """Solve the complex QEP on every segment and select effective eigenpairs.

    An eigenvalue lambda is *effective* on segment i when its imaginary part
    falls inside the segment band [w_lo, w_hi) and its real part is negative
    (stable, damped). The conjugate pairs are appended so that the returned
    set is closed under complex conjugation (conjugate-symmetric completion).

    Parameters
    ----------
    interval_matrices : list of dict
        Output of ``assemble_complex_per_interval``.
    damping_tol : float
        Effective eigenvalues must satisfy Re(lambda) < -damping_tol.

    Returns
    -------
    list of dict
        One entry per segment with the full spectrum (``lams_all``,
        ``vecs_all``), the effective subset (``lams_effective``,
        ``vecs_effective``), the conjugate-completed set
        (``lams_full_with_conj``, ``vecs_full_with_conj``), and the segment
        matrices under ``matrices``.
    """
    out = []
    for i, mat in enumerate(interval_matrices):
        result = solve_qep(mat["M_star"], mat["C_star"], mat["K_star"])
        lams = result.eigenvalues
        vecs = result.eigenvectors  # (n, 2n)
        lo = mat["omega_lower"]
        hi = mat["omega_upper"]
        mask = (
            np.isfinite(lams)
            & (np.imag(lams) >= lo)
            & (np.imag(lams) < hi)
            & (np.real(lams) < -damping_tol)
        )
        lams_eff = lams[mask]
        vecs_eff = vecs[:, mask]
        if lams_eff.size > 0:
            lams_full = np.concatenate([lams_eff, lams_eff.conjugate()])
            vecs_full = np.concatenate([vecs_eff, vecs_eff.conjugate()], axis=1)
        else:
            lams_full = lams_eff
            vecs_full = vecs_eff
        out.append({
            "interval_idx": i,
            "omega_lower": lo,
            "omega_upper": hi,
            "lams_all": lams,
            "vecs_all": vecs,
            "lams_effective": lams_eff,
            "vecs_effective": vecs_eff,
            "lams_full_with_conj": lams_full,
            "vecs_full_with_conj": vecs_full,
            "matrices": mat,
        })
    return out
