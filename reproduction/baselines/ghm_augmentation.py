"""Golla-Hughes-McTavish (GHM) augmentation method.

Reference
---------
Golla, D.F., Hughes, P.C., 1985. Dynamics of viscoelastic structures: A
time-domain finite element formulation. *J. Appl. Mech.* 52(4), 897-906.
McTavish, D.J., Hughes, P.C., 1993. Modeling of linear viscoelastic
space structures. *J. Vib. Acoust.* 115(1), 103-110.
Augmentation form follows Xie et al. 2018 Eq.(3), *Comput. Methods Appl.
Mech. Engrg.* 332, 363-381, doi:10.1016/j.cma.2017.12.023.

Idea
----
Replace the frequency-dependent shear modulus G^*(omega) by a sum of
mini-oscillators in the standard McTavish-Hughes / Xie 2018 form:

    G^*(s) = G_inf [1 + sum_n alpha_n * (s^2 + 2 zeta_n what_n s)
                              / (s^2 + 2 zeta_n what_n s + what_n^2)]

where s = j*omega and (G_inf, alpha_n, what_n, zeta_n) are fitted to DMA
data. Limits: G^*(0) = G_inf (static modulus); G^*(infty) = G_inf*(1+sum
alpha_n) (rubber plateau). Each mini-oscillator introduces an internal
degree of freedom q_n giving an *omega-independent* augmented system.

Algebraic derivation. Using the identity
    (s^2 + 2 zeta_n what_n s) / (s^2 + 2 zeta_n what_n s + what_n^2)
  = 1 - what_n^2 / (s^2 + 2 zeta_n what_n s + what_n^2)
the modulus rewrites as
    G^*(s) = G_inf*(1 + sum_n alpha_n)
             - G_inf * sum_n alpha_n * what_n^2 / D_n(s)
with D_n(s) = s^2 + 2 zeta_n what_n s + what_n^2. Introducing internal
coordinates q_n satisfying D_n(s) q_n = what_n^2 R X (with K_shear =
R^T R) converts the contribution into a block-coupled, omega-independent
augmented system.

For a base equation
    [K_base + G^*(omega) K_shear - omega^2 M] x = F
factor K_shear = R^T R with R in R^{r x n} (r = rank). The augmented
(M, C, K) matrices are:

    M_aug = blkdiag(M, alpha_n*G_inf/what_n^2 * I_r, ...)
    C_aug = blkdiag(0, 2*zeta_n*alpha_n*G_inf/what_n  * I_r, ...)
    K_aug =
      [ K_base + (1+sum_n alpha_n) G_inf K_shear   -alpha_1 G_inf R^T   ...
        -alpha_1 G_inf R                            alpha_1 G_inf I_r    ...
        ...                                                              .. ]

These matrices reproduce the standard-GHM modulus exactly: eliminating
the internal coordinates gives effective stiffness K_base + G^*(s)
K_shear with G^*(s) the standard form above.

Frequency response:
    [(K_aug - omega^2 M_aug) + j*omega C_aug] X_aug = [F; 0; ...]

Module API
----------
- GHMParameters (dataclass): G_inf and per-mode (alpha, what, zeta)
- ghm_modulus(omega, params)              : G^*(omega)
- fit_ghm_parameters(omega, G_re, G_im, n_mini, n_starts=10)
- assemble_ghm_augmented(K_base, K_shear, M, params, F)
- ghm_frequency_response(omega_array, ...) : per-omega LU
- ghm_modal_response(omega_array, ...)     : standard generalized eigvp + sum

The module is intentionally model-agnostic: callers supply (K_base,
K_shear, M, F) for any element family that admits the affine
decomposition K(G) = K_base + G*K_shear (Amichi 2010 plate, the
sandwich beam example, RKU / Mead-Markus / partial-layerwise).
"""
# @MX:NOTE: GHM standard augmentation. Applies only to the K(G)-affine element class.
# @MX:ANCHOR: assemble_ghm_augmented — entry point that builds the full augmented (M, C, K).
# @MX:REASON: generates the GHM comparison data used as a quantitative baseline in the paper.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.linalg as spla
from scipy.optimize import least_squares


# --------------------------------------------------------------------------- #
# 1. Data model                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class GHMParameters:
    """GHM mini-oscillator parameter set (standard McTavish-Hughes / Xie 2018 Eq.3 form).

    Attributes
    ----------
    G_inf : float
        Static (low-frequency) shear modulus [Pa]. G^*(0) = G_inf.
        The high-frequency rubbery plateau is G_inf*(1 + sum_n alpha_n).
    alpha : (N,) ndarray
        Mini-oscillator strengths (dimensionless), alpha_n > 0.
    what : (N,) ndarray
        Mini-oscillator (un-damped) frequencies [rad/s], what_n > 0.
    zeta : (N,) ndarray
        Mini-oscillator damping ratios, 0 < zeta_n < 1.
    """
    G_inf: float
    alpha: np.ndarray
    what: np.ndarray
    zeta: np.ndarray

    def __post_init__(self) -> None:
        self.alpha = np.asarray(self.alpha, dtype=float)
        self.what = np.asarray(self.what, dtype=float)
        self.zeta = np.asarray(self.zeta, dtype=float)
        if not (self.alpha.size == self.what.size == self.zeta.size):
            raise ValueError("alpha, what, zeta must have identical length")
        if self.G_inf <= 0:
            raise ValueError("G_inf must be positive")
        if np.any(self.alpha <= 0) or np.any(self.what <= 0):
            raise ValueError("alpha and what must be strictly positive")
        if np.any(self.zeta <= 0) or np.any(self.zeta >= 1):
            raise ValueError("zeta must lie in (0, 1) — no over-damped modes")

    @property
    def n_mini(self) -> int:
        return int(self.alpha.size)


@dataclass
class GHMAugmented:
    """Augmented GHM system matrices (omega-independent)."""
    M_aug: np.ndarray
    C_aug: np.ndarray
    K_aug: np.ndarray
    F_aug: np.ndarray
    n_phys: int
    n_internal: int  # per mini-oscillator
    n_mini: int


# --------------------------------------------------------------------------- #
# 2. GHM modulus                                                              #
# --------------------------------------------------------------------------- #
def ghm_modulus(omega: np.ndarray | float, params: GHMParameters) -> np.ndarray:
    """Evaluate G^*(omega) for the standard GHM mini-oscillator series.

    Implements the McTavish-Hughes / Xie 2018 Eq.(3) form:

        G^*(s) = G_inf [1 + sum_n alpha_n
                            * (s^2 + 2 zeta_n what_n s)
                            / (s^2 + 2 zeta_n what_n s + what_n^2)]

    with s = j*omega. The numerator (s^2 + 2 zeta_n what_n s) is the
    standard form; do NOT simplify to s^2 alone (that yields a different
    function and breaks the consistency with assemble_ghm_augmented()'s
    augmentation matrices).
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    s = 1j * omega
    accum = np.ones_like(s, dtype=complex)
    for a, w, z in zip(params.alpha, params.what, params.zeta):
        num = a * (s * s + 2.0 * z * w * s)
        den = s * s + 2.0 * z * w * s + w * w
        accum = accum + num / den
    return params.G_inf * accum


# --------------------------------------------------------------------------- #
# 3. Parameter fitting                                                        #
# --------------------------------------------------------------------------- #
def _pack(G_inf: float, alpha: np.ndarray, what: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    # Optimiser variable space uses log of strictly positive parameters.
    return np.concatenate([
        [np.log(G_inf)],
        np.log(alpha),
        np.log(what),
        np.log(zeta / (1.0 - zeta)),  # logit so 0 < zeta < 1
    ])


def _ghm_bounds(n_mini: int, omega_lo: float, omega_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Hard bounds in the optimiser variable (log) space.

    Order: [log G_inf | log alpha (N) | log what (N) | logit zeta (N)].
    """
    lb = np.concatenate([
        [np.log(1.0e2)],                       # G_inf >= 100 Pa
        np.full(n_mini, np.log(1.0e-3)),       # alpha >= 1e-3
        np.full(n_mini, np.log(omega_lo / 3.0)),  # what >= omega_min/3
        np.full(n_mini, np.log(0.01 / (1.0 - 0.01))),  # zeta >= 0.01
    ])
    ub = np.concatenate([
        [np.log(1.0e11)],                      # G_inf <= 1e11 Pa
        np.full(n_mini, np.log(1.0e3)),        # alpha <= 1e3
        np.full(n_mini, np.log(omega_hi * 3.0)),  # what <= omega_max*3
        np.full(n_mini, np.log(0.95 / (1.0 - 0.95))),  # zeta <= 0.95
    ])
    return lb, ub


def _unpack(x: np.ndarray, n_mini: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    G_inf = float(np.exp(x[0]))
    a = np.exp(x[1:1 + n_mini])
    w = np.exp(x[1 + n_mini: 1 + 2 * n_mini])
    z_logit = x[1 + 2 * n_mini: 1 + 3 * n_mini]
    z = 1.0 / (1.0 + np.exp(-z_logit))
    return G_inf, a, w, z


def _residual(x: np.ndarray, omega: np.ndarray, G_re: np.ndarray, G_im: np.ndarray,
              n_mini: int, weight: np.ndarray) -> np.ndarray:
    G_inf, a, w, z = _unpack(x, n_mini)
    p = GHMParameters(G_inf=G_inf, alpha=a, what=w, zeta=z)
    G_pred = ghm_modulus(omega, p)
    res = np.concatenate([
        weight * (G_pred.real - G_re),
        weight * (G_pred.imag - G_im),
    ])
    return res


def fit_ghm_parameters(
    omega: np.ndarray,
    G_re: np.ndarray,
    G_im: np.ndarray,
    n_mini: int,
    n_starts: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[GHMParameters, dict]:
    """Multi-start nonlinear least-squares fit of GHM parameters to G^*(omega).

    Parameters
    ----------
    omega : (M,) ndarray
        Sample angular frequencies [rad/s].
    G_re, G_im : (M,) ndarray
        Re and Im of G^*(omega) at the samples.
    n_mini : int
        Number of mini-oscillators.
    n_starts : int, default 10
        Number of random restarts (multi-start strategy).
    rng : np.random.Generator, optional
        Random source for reproducibility.

    Returns
    -------
    params : GHMParameters
        Best fit (lowest squared residual).
    info : dict
        {n_starts_done, residual_norms, best_residual_norm, rel_re_max,
         rel_re_mean, rel_im_max, rel_im_mean}
    """
    if rng is None:
        rng = np.random.default_rng(seed=20260509)

    omega = np.asarray(omega, dtype=float)
    G_re = np.asarray(G_re, dtype=float)
    G_im = np.asarray(G_im, dtype=float)
    if not (omega.size == G_re.size == G_im.size):
        raise ValueError("omega, G_re, G_im length mismatch")

    # Weight: invert the magnitude scale so Re and Im contribute on
    # comparable footing across frequencies.
    weight = 1.0 / np.maximum(np.sqrt(G_re ** 2 + G_im ** 2), 1.0)

    # Anchor mini-oscillator initial frequencies on a log-spaced subset
    # of the sample band.
    log_w_lo = np.log(omega.min() / 3.0)
    log_w_hi = np.log(omega.max() * 3.0)
    lb, ub = _ghm_bounds(n_mini, omega.min(), omega.max())

    G_inf_init = float(np.median(G_re))
    if G_inf_init <= 0:
        G_inf_init = float(np.maximum(np.abs(G_re).mean(), 1.0))

    best_x: np.ndarray | None = None
    best_norm = np.inf
    norms: list[float] = []

    for k in range(n_starts):
        # First start: deterministic spacing; subsequent: random perturbation.
        if k == 0:
            log_w0 = np.linspace(log_w_lo, log_w_hi, n_mini)
            alpha0 = np.full(n_mini, 0.5)
            zeta0 = np.full(n_mini, 0.3)
            G_inf0 = G_inf_init
        else:
            log_w0 = rng.uniform(log_w_lo, log_w_hi, size=n_mini)
            log_w0.sort()
            alpha0 = rng.uniform(0.05, 5.0, size=n_mini)
            zeta0 = rng.uniform(0.05, 0.6, size=n_mini)
            G_inf0 = G_inf_init * float(rng.uniform(0.3, 3.0))
        x0 = _pack(G_inf0, alpha0, np.exp(log_w0), zeta0)
        x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)  # snap inside bounds for trf init

        try:
            sol = least_squares(
                _residual, x0,
                args=(omega, G_re, G_im, n_mini, weight),
                method="trf", bounds=(lb, ub), max_nfev=4000,
            )
            norm = float(np.linalg.norm(sol.fun))
            norms.append(norm)
            if norm < best_norm:
                best_norm = norm
                best_x = sol.x
        except (ValueError, np.linalg.LinAlgError):
            norms.append(np.inf)

    if best_x is None:
        raise RuntimeError("All GHM fitting starts failed")

    G_inf, alpha, what, zeta = _unpack(best_x, n_mini)
    # Sort mini-oscillators by frequency for deterministic output.
    order = np.argsort(what)
    params = GHMParameters(
        G_inf=float(G_inf), alpha=alpha[order], what=what[order], zeta=zeta[order],
    )

    G_pred = ghm_modulus(omega, params)
    rel_re = np.abs(G_pred.real - G_re) / np.maximum(np.abs(G_re), 1e-30)
    rel_im = np.abs(G_pred.imag - G_im) / np.maximum(np.abs(G_im), 1e-30)
    info = dict(
        n_starts_done=len(norms),
        residual_norms=np.asarray(norms),
        best_residual_norm=best_norm,
        rel_re_max=float(rel_re.max() * 100.0),
        rel_re_mean=float(rel_re.mean() * 100.0),
        rel_im_max=float(rel_im.max() * 100.0),
        rel_im_mean=float(rel_im.mean() * 100.0),
    )
    return params, info


# --------------------------------------------------------------------------- #
# 4. Augmented system assembly                                                #
# --------------------------------------------------------------------------- #
def _kshear_factor(K_shear: np.ndarray, rank_tol: float = 1e-12
                   ) -> tuple[np.ndarray, np.ndarray, int]:
    """Factor K_shear = R^T R with R in R^{r x n}, rank r.

    Uses symmetric eigendecomposition. K_shear is real symmetric
    positive *semi*-definite (rigid-body modes are in its kernel).

    The cutoff is strictly relative to w.max(). An absolute floor
    (max(w.max(), 1.0)) silently truncates genuine stiffness content when
    K_shear carries only the geometry and the modulus scale G ~ 1e9 lives
    elsewhere (Amichi K_shear: w.max() ~ 1.6e-4), causing O(10%) FRF error.
    """
    K_sym = 0.5 * (K_shear + K_shear.T)
    w, V = spla.eigh(K_sym)
    keep = w > rank_tol * w.max()
    r = int(keep.sum())
    if r == 0:
        raise ValueError("K_shear is the zero matrix; nothing to augment")
    sqrt_w = np.sqrt(w[keep])
    R = (V[:, keep] * sqrt_w[None, :]).T  # (r, n)
    Lambda = np.diag(w[keep])              # (r, r)
    return R, Lambda, r


def assemble_ghm_augmented(
    K_base: np.ndarray,
    K_shear: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
    params: GHMParameters,
    rank_tol: float = 1e-10,
) -> GHMAugmented:
    """Build the GHM augmented (M_aug, C_aug, K_aug, F_aug) per Trindade 2000.

    Each mini-oscillator contributes an internal-DOF block of dimension
    r = rank(K_shear). For Amichi plate free-free, r = n - 6 (rigid body
    nullspace of K_shear has dimension 6).
    """
    n = K_base.shape[0]
    if not (K_shear.shape == (n, n) and M.shape == (n, n)):
        raise ValueError("K_base, K_shear, M dimension mismatch")
    if F.shape != (n,):
        raise ValueError("F dimension mismatch")
    R, Lambda, r = _kshear_factor(K_shear, rank_tol=rank_tol)

    n_mini = params.n_mini
    G_inf = params.G_inf
    alpha = params.alpha
    what = params.what
    zeta = params.zeta
    sum_alpha = float(alpha.sum())

    n_aug = n + n_mini * r
    M_aug = np.zeros((n_aug, n_aug), dtype=float)
    C_aug = np.zeros((n_aug, n_aug), dtype=float)
    K_aug = np.zeros((n_aug, n_aug), dtype=float)

    # Physical block (top-left)
    M_aug[:n, :n] = M
    K_aug[:n, :n] = K_base + (1.0 + sum_alpha) * G_inf * K_shear

    # Per-oscillator internal blocks
    Ir = np.eye(r)
    for k in range(n_mini):
        i0 = n + k * r
        i1 = i0 + r
        a_k = alpha[k]
        w_k = what[k]
        z_k = zeta[k]
        M_aug[i0:i1, i0:i1] = (a_k * G_inf / (w_k * w_k)) * Ir
        C_aug[i0:i1, i0:i1] = (2.0 * z_k * a_k * G_inf / w_k) * Ir
        # Coupling: K_aug[phys, internal] = -a_k G_inf R^T
        K_aug[:n, i0:i1] = -a_k * G_inf * R.T
        K_aug[i0:i1, :n] = -a_k * G_inf * R
        K_aug[i0:i1, i0:i1] = a_k * G_inf * Ir

    F_aug = np.zeros(n_aug, dtype=complex)
    F_aug[:n] = F.astype(complex)

    return GHMAugmented(M_aug=M_aug, C_aug=C_aug, K_aug=K_aug, F_aug=F_aug,
                        n_phys=n, n_internal=r, n_mini=n_mini)


# --------------------------------------------------------------------------- #
# 5. Frequency response                                                       #
# --------------------------------------------------------------------------- #
def ghm_frequency_response(
    aug: GHMAugmented,
    omega_array: np.ndarray,
    obs_dof: int,
) -> np.ndarray:
    """Per-frequency LU on the (M_aug, C_aug, K_aug) augmented system.

    For each omega solves
        [K_aug - omega^2 M_aug + j*omega C_aug] X = F_aug
    and returns the physical-DOF observation Y(omega).
    """
    omega_array = np.asarray(omega_array, dtype=float)
    Y = np.zeros(omega_array.size, dtype=complex)
    for k, om in enumerate(omega_array):
        D = (aug.K_aug.astype(complex)
             - om * om * aug.M_aug.astype(complex)
             + 1j * om * aug.C_aug.astype(complex))
        try:
            x = spla.solve(D, aug.F_aug)
            Y[k] = x[obs_dof]
        except np.linalg.LinAlgError:
            Y[k] = np.nan + 0.0j
    return Y


def ghm_modal_response(
    aug: GHMAugmented,
    omega_array: np.ndarray,
    obs_dof: int,
    eig_tol: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """One-shot modal response via state-space linearization of (M, C, K)_aug.

    Linearization to first-order: y' = A y, with
        A = [ 0   I        ]
            [ -M^{-1}K  -M^{-1}C ]
    Right eigenvectors of A give classical modal expansion. This is the
    standard route used in GHM frequency-domain analysis.

    Returns
    -------
    Y : (N_w,) complex array
        Response at obs_dof.
    info : dict
        {n_modes, eig_residual_max}
    """
    n = aug.M_aug.shape[0]
    Iblk = np.eye(n)
    Zblk = np.zeros_like(aug.M_aug)
    # Solve (M C K) → A explicitly. M_aug must be non-singular: rigid
    # bodies in K_aug nullspace are *not* in M_aug since each
    # mini-oscillator block is positive-definite.
    M_inv_C = spla.solve(aug.M_aug, aug.C_aug)
    M_inv_K = spla.solve(aug.M_aug, aug.K_aug)
    A = np.block([[Zblk, Iblk], [-M_inv_K, -M_inv_C]])
    B_rhs = np.concatenate([np.zeros(n), spla.solve(aug.M_aug, aug.F_aug.real)])
    # Eigenproblem of A: lambda_k, right eigenvectors r_k
    eigvals, R = spla.eig(A)
    L = spla.inv(R)
    # Modal coefficients c_k = (L B_rhs)_k / 1   (driving in second block)
    coeffs = L @ B_rhs.astype(complex)

    omega_array = np.asarray(omega_array, dtype=float)
    Y = np.zeros(omega_array.size, dtype=complex)
    obs_index = obs_dof  # in physical block (top-half of state)
    # Use only first n components of right eigenvector as the displacement
    # part of the state.
    R_disp = R[:n, :]
    for k_om, om in enumerate(omega_array):
        denom = 1j * om - eigvals
        # Filter degenerate poles (very small denominator → near omega).
        Y[k_om] = np.sum(R_disp[obs_index, :] * coeffs / denom)

    # Eigenresidual diagnostic
    res = A @ R - R * eigvals[None, :]
    info = dict(
        n_modes=int(eigvals.size),
        eig_residual_max=float(np.linalg.norm(res, axis=0).max()),
    )
    return Y, info


# --------------------------------------------------------------------------- #
# 6. Self-test (constant-G limit recovers reduced direct system)              #
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Self-tests:
    1) alpha → 0 reduces GHM to constant K_base + G_inf*K_shear (smoke).
    2) Modulus-vs-augmentation consistency: the augmented system response
       must equal direct factorization with G(omega) inserted via the
       ghm_modulus formula. Failure here means ghm_modulus() and
       assemble_ghm_augmented() encode different damping models.
    """
    rng = np.random.default_rng(20260509)
    n = 12
    A = rng.standard_normal((n, n))
    K_base = (A.T @ A) + 5.0 * np.eye(n)
    B = rng.standard_normal((n, n))
    K_shear = (B.T @ B)               # SPD
    M = np.eye(n)
    F = rng.standard_normal(n)
    omega = np.linspace(1.0, 100.0, 30)

    # ----- Test 1: constant-G limit (alpha vanishing) -----
    G_inf = 2.5e6
    Y_ref = np.zeros(omega.size, dtype=complex)
    for k, om in enumerate(omega):
        D = K_base + G_inf * K_shear - om * om * M
        Y_ref[k] = spla.solve(D.astype(complex), F.astype(complex))[0]

    p_trivial = GHMParameters(
        G_inf=G_inf,
        alpha=np.array([1e-12]), what=np.array([100.0]), zeta=np.array([0.1]),
    )
    aug = assemble_ghm_augmented(K_base, K_shear, M, F, p_trivial)
    Y_ghm = ghm_frequency_response(aug, omega, obs_dof=0)
    err1 = np.max(np.abs(Y_ghm - Y_ref) / np.maximum(np.abs(Y_ref), 1e-30))
    pass1 = err1 < 1e-6
    print(f"  [1] constant-G limit (alpha→0): max rel err = {err1:.3e}  "
          f"(< 1e-6)  →  {'PASS' if pass1 else 'FAIL'}")

    # ----- Test 2: modulus-vs-augmentation consistency -----
    # With nontrivial mini-oscillators, the augmented system response must
    # match direct factorization that uses ghm_modulus(omega) inline.
    p = GHMParameters(
        G_inf=G_inf,
        alpha=np.array([0.5, 1.2, 0.3]),
        what=np.array([20.0, 80.0, 200.0]),
        zeta=np.array([0.3, 0.25, 0.4]),
    )
    aug2 = assemble_ghm_augmented(K_base, K_shear, M, F, p)
    Y_aug = ghm_frequency_response(aug2, omega, obs_dof=0)
    G_arr = ghm_modulus(omega, p)
    Y_direct = np.zeros(omega.size, dtype=complex)
    for k, om in enumerate(omega):
        D = K_base.astype(complex) + G_arr[k] * K_shear.astype(complex) \
            - om * om * M.astype(complex)
        Y_direct[k] = spla.solve(D, F.astype(complex))[0]
    err2 = np.max(np.abs(Y_aug - Y_direct) / np.maximum(np.abs(Y_direct), 1e-30))
    pass2 = err2 < 1e-9
    print(f"  [2] modulus-vs-augmentation consistency: max rel err = {err2:.3e}  "
          f"(< 1e-9)  →  {'PASS' if pass2 else 'FAIL'}")
    if not pass2:
        print("      ghm_modulus() and assemble_ghm_augmented() encode "
              "different damping models — fix one of them.")

    # ----- Test 3: limit checks -----
    G_zero = ghm_modulus(np.array([1e-8]), p)[0]
    G_inf_lim = ghm_modulus(np.array([1e12]), p)[0]
    plateau = G_inf * (1.0 + p.alpha.sum())
    pass3a = abs(G_zero.real - G_inf) / G_inf < 1e-12
    pass3b = abs(G_inf_lim.real - plateau) / plateau < 1e-6
    print(f"  [3a] G^*(0) = G_inf check: {G_zero.real:.6e} vs {G_inf:.6e}  "
          f"→  {'PASS' if pass3a else 'FAIL'}")
    print(f"  [3b] G^*(inf) = G_inf*(1+sum alpha) check: "
          f"{G_inf_lim.real:.6e} vs {plateau:.6e}  "
          f"→  {'PASS' if pass3b else 'FAIL'}")


if __name__ == "__main__":
    print("=" * 60)
    print("GHM augmentation — self-test")
    print("=" * 60)
    _self_test()
    print("=" * 60)
