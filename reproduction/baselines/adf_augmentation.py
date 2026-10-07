"""Anelastic Displacement Fields (ADF) augmentation method.

Reference
---------
Lesieutre, G.A., Lee, U., 1996. A finite element for beams having
segmented active constrained layers with frequency-dependent
viscoelastics. *Smart Mater. Struct.* 5(5), 615-627.
Cross-checked against the review by Trindade, Benjeddou, Ohayon (2000).

Idea
----
Express the viscoelastic shear modulus as a Prony series:

    G^*(s) = G_e + sum_f G_v^{(f)} * s / (s + Omega_f),    s = j*omega

with:
    G_e        = elastic (low-frequency) modulus        > 0
    G_v^{(f)}  = strength of fth anelastic field        > 0
    Omega_f    = relaxation frequency [rad/s]            > 0

Each field is governed by a *first-order* relaxation ODE in time:

    udot^{(f)} + Omega_f u^{(f)} = Omega_f R x
    sigma_aug = sigma_base + G_v^{(f)} R^T (R x - u^{(f)})

with K_shear = R^T R the rank decomposition. The total system becomes:

    M xddot + [K_base + (G_e + sum_f G_v^{(f)}) K_shear] x
            - sum_f G_v^{(f)} R^T u^{(f)} = F
    udot^{(f)} + Omega_f u^{(f)} - Omega_f R x = 0

Cast as first-order state-space  Ydot = A Y + B_in F,
        Y = [xdot ; x ; u^{(1)} ; ... ; u^{(N)}]^T  in R^{2n + N r}

This module provides:
- ADFParameters dataclass
- adf_modulus(omega, params)
- fit_adf_parameters(omega, G_re, G_im, n_adf, n_starts)
- assemble_adf_state_space(K_base, K_shear, M, F, params)
- adf_frequency_response(...)         per-omega LU on (j*omega I - A)
- adf_modal_response(...)             standard eigvp + modal sum

The module is model-agnostic: any element family that admits the affine
decomposition K(G) = K_base + G K_shear is supported (Amichi 2010 plate,
the sandwich beam example, RKU / Mead-Markus / partial-layerwise).
"""
# @MX:NOTE: ADF state-space augmentation. Applies only to the K(G)-affine element class.
# @MX:ANCHOR: assemble_adf_state_space — entry point that builds the full augmented (A, B_in).
# @MX:REASON: generates the ADF comparison data used as a quantitative baseline in the paper.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as spla
from scipy.optimize import least_squares


# --------------------------------------------------------------------------- #
# 1. Data model                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class ADFParameters:
    """ADF Prony-series parameter set.

    Attributes
    ----------
    G_e : float
        Elastic (low-frequency) modulus [Pa].
    Gv : (N,) ndarray
        Anelastic field strengths [Pa], Gv_f > 0.
    Omega : (N,) ndarray
        Relaxation frequencies [rad/s], Omega_f > 0.
    """
    G_e: float
    Gv: np.ndarray
    Omega: np.ndarray

    def __post_init__(self) -> None:
        self.Gv = np.asarray(self.Gv, dtype=float)
        self.Omega = np.asarray(self.Omega, dtype=float)
        if self.Gv.size != self.Omega.size:
            raise ValueError("Gv and Omega must have identical length")
        if self.G_e <= 0:
            raise ValueError("G_e must be positive")
        if np.any(self.Gv <= 0):
            raise ValueError("Gv must be strictly positive")
        if np.any(self.Omega <= 0):
            raise ValueError("Omega must be strictly positive")

    @property
    def n_adf(self) -> int:
        return int(self.Gv.size)


@dataclass
class ADFStateSpace:
    """ADF first-order state-space (A, B_in)."""
    A: np.ndarray
    B_in: np.ndarray   # column vector applying force at physical DOFs
    n_phys: int
    n_internal: int    # per anelastic field (= rank of K_shear)
    n_adf: int


# --------------------------------------------------------------------------- #
# 2. ADF modulus                                                              #
# --------------------------------------------------------------------------- #
def adf_modulus(omega: np.ndarray | float, params: ADFParameters) -> np.ndarray:
    """Evaluate G^*(omega) for the ADF Prony series."""
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    s = 1j * omega
    accum = np.full_like(s, params.G_e, dtype=complex)
    for Gv, Om in zip(params.Gv, params.Omega):
        accum = accum + Gv * s / (s + Om)
    return accum


# --------------------------------------------------------------------------- #
# 3. Parameter fitting                                                        #
# --------------------------------------------------------------------------- #
def _pack_adf(G_e: float, Gv: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    return np.concatenate([
        [np.log(G_e)],
        np.log(Gv),
        np.log(Omega),
    ])


def _adf_bounds(n_adf: int, omega_lo: float, omega_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Hard bounds for ADF parameters in log space.

    Order: [log G_e | log Gv (N) | log Omega (N)].
    """
    lb = np.concatenate([
        [np.log(1.0e2)],                       # G_e >= 100 Pa
        np.full(n_adf, np.log(1.0e2)),         # Gv >= 100 Pa
        np.full(n_adf, np.log(omega_lo / 3.0)),  # Omega >= omega_min/3
    ])
    ub = np.concatenate([
        [np.log(1.0e11)],                      # G_e <= 1e11 Pa
        np.full(n_adf, np.log(1.0e11)),        # Gv <= 1e11 Pa
        np.full(n_adf, np.log(omega_hi * 3.0)),  # Omega <= omega_max*3
    ])
    return lb, ub


def _unpack_adf(x: np.ndarray, n_adf: int) -> tuple[float, np.ndarray, np.ndarray]:
    G_e = float(np.exp(x[0]))
    Gv = np.exp(x[1: 1 + n_adf])
    Om = np.exp(x[1 + n_adf: 1 + 2 * n_adf])
    return G_e, Gv, Om


def _residual_adf(x: np.ndarray, omega: np.ndarray, G_re: np.ndarray,
                  G_im: np.ndarray, n_adf: int, weight: np.ndarray) -> np.ndarray:
    G_e, Gv, Om = _unpack_adf(x, n_adf)
    p = ADFParameters(G_e=G_e, Gv=Gv, Omega=Om)
    G_pred = adf_modulus(omega, p)
    return np.concatenate([
        weight * (G_pred.real - G_re),
        weight * (G_pred.imag - G_im),
    ])


def fit_adf_parameters(
    omega: np.ndarray,
    G_re: np.ndarray,
    G_im: np.ndarray,
    n_adf: int,
    n_starts: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[ADFParameters, dict]:
    """Multi-start nonlinear LS fit of ADF parameters to G^*(omega).

    Returns
    -------
    params : ADFParameters
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

    weight = 1.0 / np.maximum(np.sqrt(G_re ** 2 + G_im ** 2), 1.0)

    # Initial guess: log-spaced relaxation frequencies in the data band.
    log_om_lo = np.log(omega.min() / 3.0)
    log_om_hi = np.log(omega.max() * 3.0)
    G_e_init = float(max(G_re.min(), 1.0))
    lb, ub = _adf_bounds(n_adf, omega.min(), omega.max())

    best_x: np.ndarray | None = None
    best_norm = np.inf
    norms: list[float] = []

    for k in range(n_starts):
        if k == 0:
            log_om0 = np.linspace(log_om_lo, log_om_hi, n_adf)
            Gv0 = np.full(n_adf, max(G_e_init, 1.0))
            G_e0 = G_e_init
        else:
            log_om0 = rng.uniform(log_om_lo, log_om_hi, size=n_adf)
            log_om0.sort()
            Gv0 = G_e_init * rng.uniform(0.1, 5.0, size=n_adf)
            G_e0 = G_e_init * float(rng.uniform(0.3, 3.0))
        x0 = _pack_adf(G_e0, Gv0, np.exp(log_om0))
        x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)

        try:
            sol = least_squares(
                _residual_adf, x0,
                args=(omega, G_re, G_im, n_adf, weight),
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
        raise RuntimeError("All ADF fitting starts failed")

    G_e, Gv, Om = _unpack_adf(best_x, n_adf)
    order = np.argsort(Om)
    params = ADFParameters(G_e=float(G_e), Gv=Gv[order], Omega=Om[order])

    G_pred = adf_modulus(omega, params)
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
# 4. State-space assembly                                                     #
# --------------------------------------------------------------------------- #
def _kshear_factor(K_shear: np.ndarray, rank_tol: float = 1e-12
                   ) -> tuple[np.ndarray, int]:
    """Factor K_shear = R^T R with R in R^{r x n}, rank r, via eigendecomposition.

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
    return R, r


def assemble_adf_state_space(
    K_base: np.ndarray,
    K_shear: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
    params: ADFParameters,
    rank_tol: float = 1e-10,
) -> ADFStateSpace:
    """Build the ADF first-order state-space (A, B_in).

    State vector:  Y = [xdot ; x ; u^{(1)} ; ... ; u^{(N)}]
    State dim:     n_total = 2*n + N*r,  r = rank(K_shear).
    """
    n = K_base.shape[0]
    if not (K_shear.shape == (n, n) and M.shape == (n, n)):
        raise ValueError("K_base, K_shear, M dimension mismatch")
    if F.shape != (n,):
        raise ValueError("F dimension mismatch")
    R, r = _kshear_factor(K_shear, rank_tol=rank_tol)

    N = params.n_adf
    G_e = params.G_e
    Gv = params.Gv
    Om = params.Omega
    sum_Gv = float(Gv.sum())

    K_inf = K_base + (G_e + sum_Gv) * K_shear
    M_inv = spla.inv(M)
    Ainv_K = M_inv @ K_inf  # n x n

    n_total = 2 * n + N * r
    A = np.zeros((n_total, n_total), dtype=float)
    B_in = np.zeros((n_total, n), dtype=float)

    # Row block layout: rows 0..n-1   = xdot
    #                   rows n..2n-1  = x
    #                   rows 2n + f*r .. 2n + (f+1)*r - 1 = u^{(f)}
    # xdot equation:  xdot' = -M^{-1} K_inf x + sum_f M^{-1} G_v R^T u^{(f)} + M^{-1} F
    A[0:n, n:2*n] = -Ainv_K
    for f in range(N):
        u_lo = 2 * n + f * r
        u_hi = u_lo + r
        A[0:n, u_lo:u_hi] = Gv[f] * (M_inv @ R.T)
    B_in[0:n, :] = M_inv

    # x equation:  xdot = xdot  →  identity coupling
    A[n:2*n, 0:n] = np.eye(n)

    # field equation:  udot = Omega_f R x - Omega_f u^{(f)}
    for f in range(N):
        u_lo = 2 * n + f * r
        u_hi = u_lo + r
        A[u_lo:u_hi, n:2*n] = Om[f] * R
        A[u_lo:u_hi, u_lo:u_hi] = -Om[f] * np.eye(r)

    return ADFStateSpace(A=A, B_in=B_in, n_phys=n, n_internal=r, n_adf=N)


# --------------------------------------------------------------------------- #
# 5. Frequency response                                                       #
# --------------------------------------------------------------------------- #
def adf_frequency_response(
    ss: ADFStateSpace,
    F: np.ndarray,
    omega_array: np.ndarray,
    obs_dof: int,
) -> np.ndarray:
    """Per-frequency LU on (j*omega I - A) F_drive.

    State Y(omega) = (j*omega I - A)^{-1} B_in F.
    Observation = Y[n + obs_dof] (the displacement block).
    """
    omega_array = np.asarray(omega_array, dtype=float)
    n = ss.n_phys
    n_total = ss.A.shape[0]
    drive = (ss.B_in @ F.astype(complex))
    Y = np.zeros(omega_array.size, dtype=complex)
    A_c = ss.A.astype(complex)
    eye_c = np.eye(n_total, dtype=complex)
    for k, om in enumerate(omega_array):
        try:
            x = spla.solve(1j * om * eye_c - A_c, drive)
            Y[k] = x[n + obs_dof]
        except np.linalg.LinAlgError:
            Y[k] = np.nan + 0.0j
    return Y


def adf_modal_response(
    ss: ADFStateSpace,
    F: np.ndarray,
    omega_array: np.ndarray,
    obs_dof: int,
) -> tuple[np.ndarray, dict]:
    """One-shot eigenvalue decomposition + modal sum for state-space ADF.

    A = R diag(lambda) L,  L R = I.
    Y(omega) = sum_k r_k * (l_k^T B_in F) / (j*omega - lambda_k).
    """
    n = ss.n_phys
    eigvals, R = spla.eig(ss.A)
    L = spla.inv(R)
    drive = ss.B_in @ F.astype(complex)
    coeffs = L @ drive  # complex (n_total,)

    omega_array = np.asarray(omega_array, dtype=float)
    Y = np.zeros(omega_array.size, dtype=complex)
    R_disp = R[n:2*n, :]  # displacement block
    for k_om, om in enumerate(omega_array):
        denom = 1j * om - eigvals
        Y[k_om] = np.sum(R_disp[obs_dof, :] * coeffs / denom)

    res = ss.A @ R - R * eigvals[None, :]
    info = dict(
        n_modes=int(eigvals.size),
        eig_residual_max=float(np.linalg.norm(res, axis=0).max()),
    )
    return Y, info


# --------------------------------------------------------------------------- #
# 6. Self-test (Gv -> 0 reduces to constant K_base + G_e*K_shear)             #
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    rng = np.random.default_rng(20260509)
    n = 12
    A = rng.standard_normal((n, n))
    K_base = (A.T @ A) + 5.0 * np.eye(n)
    B = rng.standard_normal((n, n))
    K_shear = (B.T @ B)
    M = np.eye(n)
    F = rng.standard_normal(n)
    omega = np.linspace(1.0, 100.0, 30)

    G_e = 2.5e6
    # Direct reference at G = G_e (constant)
    Y_ref = np.zeros(omega.size, dtype=complex)
    for k, om in enumerate(omega):
        D = K_base + G_e * K_shear - om * om * M
        Y_ref[k] = spla.solve(D.astype(complex), F.astype(complex))[0]

    # ADF with vanishing field strengths → constant G_e
    p = ADFParameters(G_e=G_e, Gv=np.array([1e-12]), Omega=np.array([100.0]))
    ss = assemble_adf_state_space(K_base, K_shear, M, F, p)
    Y_adf = adf_frequency_response(ss, F, omega, obs_dof=0)
    err = np.max(np.abs(Y_adf - Y_ref) / np.maximum(np.abs(Y_ref), 1e-30))
    print(f"  ADF constant-G limit self-test: max rel err = {err:.3e}  "
          f"(must be < 1e-6)  →  {'PASS' if err < 1e-6 else 'FAIL'}")

    # ADF modulus: at low omega → G_e; at high omega → G_e + sum(Gv)
    G_lo = adf_modulus(1e-3, p)[0]
    G_hi = adf_modulus(1e6, p)[0]
    print(f"  ADF modulus low-freq limit  : Re = {G_lo.real:.6e}  (G_e = {G_e:.3e})")
    print(f"  ADF modulus high-freq limit : Re = {G_hi.real:.6e}  "
          f"(G_e + Gv_sum = {G_e + p.Gv.sum():.3e})")


if __name__ == "__main__":
    print("=" * 60)
    print("ADF augmentation — self-test")
    print("=" * 60)
    _self_test()
    print("=" * 60)
