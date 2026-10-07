"""Xie et al. (2018) single-point Taylor + SOAR reduced-order model.

Reference:
    Xie, Zheng, Jonckheere, van de Walle, Pluymers, Desmet,
    "Adaptive model reduction technique for large-scale dynamical systems
    with frequency-dependent damping," CMAME 332 (2018) 363–381.

Key idea:
    Expand the complex shear modulus G(s) in a Taylor series about s0 so that
    the frequency-dependent FE system becomes a standard second-order system,
    then reduce it with a SOAR basis.

Scope:
    Single expansion point + SOAR (build_xie_rom), plus the adaptive
    enrichment of Section 4 (xie_adaptive_response).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import lu_factor, lu_solve

# Reuse the existing SOAR implementation as a building block (do not reimplement)
from polyqep.rom import soar_basis


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class Xie2018ROM:
    """Xie et al. 2018 Eq. (10)–(13) single-point Taylor + SOAR reduced-order model.

    G(s) is approximated by a quadratic polynomial about the Taylor expansion
    point s0; the SOAR algorithm then builds an orthonormal basis Q and the
    system is reduced to size m×m.

    Attributes
    ----------
    s0 : complex
        Taylor expansion point (purely imaginary recommended: s0 = j * omega_0).
    K_tilde : ndarray, shape (N, N)
        Transformed stiffness matrix K_e + G(s0) K_v + s0² M  [Eq. 11].
    D_tilde : ndarray, shape (N, N)
        Transformed damping matrix 2 s0 M + G'(s0) K_v  [Eq. 11].
    M_tilde : ndarray, shape (N, N)
        Transformed mass matrix M + G''(s0)/2 * K_v  [Eq. 11].
    Kv : ndarray, shape (N, N)
        Original K_v (used to evaluate the remainder term R2).
    Q : ndarray, shape (N, m)
        Orthonormal SOAR basis.
    Kr : ndarray, shape (m, m)
        Q^H K_tilde Q.
    Dr : ndarray, shape (m, m)
        Q^H D_tilde Q.
    Mr : ndarray, shape (m, m)
        Q^H M_tilde Q.
    Kvr : ndarray, shape (m, m)
        Q^H K_v Q.
    G_func : Callable
        Original G(s) (used to evaluate the remainder term).
    G_taylor : Callable
        Quadratic Taylor approximation G_T(s) = G(s0) + G'(s0)(s-s0) + G''(s0)/2*(s-s0)².
    """

    # Expansion point
    s0: complex

    # Transformed matrices (full size)
    K_tilde: np.ndarray  # K_e + G(s0) Kv + s0^2 M
    D_tilde: np.ndarray  # 2 s0 M + G'(s0) Kv
    M_tilde: np.ndarray  # M + G''(s0)/2 Kv
    Kv: np.ndarray       # original viscoelastic stiffness matrix

    # SOAR basis and reduced matrices
    Q: np.ndarray    # (N, m) orthonormal basis
    Kr: np.ndarray   # (m, m) Q^H K_tilde Q
    Dr: np.ndarray   # (m, m) Q^H D_tilde Q
    Mr: np.ndarray   # (m, m) Q^H M_tilde Q
    Kvr: np.ndarray  # (m, m) Q^H Kv Q

    # G functions
    G_func: Callable[[complex], complex]    # original G(s)
    G_taylor: Callable[[complex], complex]  # quadratic Taylor approximation

    @property
    def n_full(self) -> int:
        """Number of full-order DOFs N."""
        return self.Q.shape[0]

    @property
    def m_reduced(self) -> int:
        """Reduced-model order m."""
        return self.Q.shape[1]


# =============================================================================
# Finite-difference Taylor coefficients
# =============================================================================

# @MX:ANCHOR: [AUTO] taylor_coefficients — returns all three G derivatives, fan_in >= 3
# @MX:REASON: called from build_xie_rom, the tests, and external comparison scripts
def taylor_coefficients(
    G_func: Callable[[complex], complex],
    s0: complex,
    h: float = 1e-3,
    *,
    G_prime: complex | None = None,
    G_double_prime: complex | None = None,
) -> tuple[complex, complex, complex]:
    """Return G(s0), G'(s0), G''(s0).

    By default the derivatives are computed by central finite differences
    along the imaginary axis. Note that these are derivatives with respect
    to s, not with respect to ω along the path s = jω.

    Parameters
    ----------
    G_func : callable
        Complex shear modulus G(s) taking a complex argument s = jω.
    s0 : complex
        Taylor expansion point.
    h : float
        Finite-difference step size (imaginary direction).  step = 1j * h.
    G_prime : complex, optional
        Analytic G'(s0); when given, the finite difference is skipped.
    G_double_prime : complex, optional
        Analytic G''(s0); when given, the finite difference is skipped.

    Returns
    -------
    G0, Gp, Gpp : complex
        G(s0), G'(s0), G''(s0).

    Notes
    -----
    Why the step is taken along the imaginary axis:
        The physical response is evaluated only on the path s = jω
        (purely imaginary). Taking differences in the ±1j*h direction
        about s0 = j*omega_0 therefore gives the derivative on the path
        that is actually used.

    Central finite-difference formulas:
        G'(s0)  ≈ [G(s0 + 1j*h) - G(s0 - 1j*h)] / (2 * 1j * h)
        G''(s0) ≈ [G(s0 + 1j*h) - 2*G(s0) + G(s0 - 1j*h)] / (1j*h)^2
    """
    # The function value at s0 is always evaluated directly
    G0 = complex(G_func(s0))

    # First derivative
    if G_prime is not None:
        Gp = complex(G_prime)
    else:
        # Central finite difference in the imaginary direction: step = 1j * h
        step = 1j * h
        Gp = (G_func(s0 + step) - G_func(s0 - step)) / (2.0 * step)

    # Second derivative
    if G_double_prime is not None:
        Gpp = complex(G_double_prime)
    else:
        step = 1j * h
        # (1j*h)^2 = -h^2
        Gpp = (G_func(s0 + step) - 2.0 * G0 + G_func(s0 - step)) / (step ** 2)

    return G0, Gp, Gpp


# =============================================================================
# ROM construction
# =============================================================================

# @MX:ANCHOR: [AUTO] build_xie_rom — entry point for building the Xie 2018 ROM, called from several comparison scripts
# @MX:REASON: called from the comparison experiments, the adaptive enrichment loop, and the tests
def build_xie_rom(
    K_e: np.ndarray,
    K_v: np.ndarray,
    M: np.ndarray,
    G_func: Callable[[complex], complex],
    s0: complex,
    m: int,
    *,
    G_prime: complex | None = None,
    G_double_prime: complex | None = None,
    force_vector: np.ndarray | None = None,
    tol: float = 1e-13,
) -> Xie2018ROM:
    """Build the single-point Taylor + SOAR ROM of Xie 2018 Eq. (11)–(13).

    Parameters
    ----------
    K_e : ndarray, shape (N, N)
        Elastic stiffness matrix.
    K_v : ndarray, shape (N, N)
        Viscoelastic stiffness matrix (including the support locator).
    M : ndarray, shape (N, N)
        Mass matrix.
    G_func : callable
        Complex shear modulus G(s).
    s0 : complex
        Taylor expansion point.  Typically s0 = j * omega_0 (purely imaginary).
    m : int
        Reduced-model order (number of SOAR basis vectors).
    G_prime : complex, optional
        Analytic G'(s0) — when given, the finite difference is skipped.
    G_double_prime : complex, optional
        Analytic G''(s0) — when given, the finite difference is skipped.
    force_vector : ndarray, shape (N,), optional
        Load vector used to generate the SOAR starting vector.
        If None, a unit load at the last DOF is used.
    tol : float
        SOAR MGS deflation-detection tolerance.

    Returns
    -------
    Xie2018ROM

    Notes
    -----
    Mapping onto the SOAR call (substituting (s-s0) in Eq. (10) of Xie 2018):
        Original system:  [K_tilde + (s-s0) D_tilde + (s-s0)^2 M_tilde + R2 Kv] X = F
        With (s-s0) = delta this is a quadratic system about delta=0:
            [K_tilde + delta D_tilde + delta^2 M_tilde] X ≈ F  (R2 neglected)
        → soar_basis(M=M_tilde, C=D_tilde, K=K_tilde, shift_omega=0, ...)
        Since shift_omega=0, sigma=0 and K_tilde must be invertible for the LU factorization.
    """
    n = M.shape[0]

    # Default load vector (unit excitation at the last DOF)
    if force_vector is None:
        F_vec = np.zeros(n, dtype=complex)
        F_vec[-1] = 1.0
    else:
        F_vec = np.asarray(force_vector, dtype=complex)

    # Step 1: Taylor coefficients
    G0, Gp, Gpp = taylor_coefficients(
        G_func, s0,
        G_prime=G_prime,
        G_double_prime=G_double_prime,
    )

    # Step 2: assemble the transformed matrices (Xie 2018 Eq. 11)
    # K_tilde = K_e + G(s0) Kv + s0^2 M
    K_e_c = K_e.astype(complex)
    K_v_c = K_v.astype(complex)
    M_c = M.astype(complex)

    K_tilde = K_e_c + G0 * K_v_c + (s0 ** 2) * M_c      # [N×N]
    D_tilde = 2.0 * s0 * M_c + Gp * K_v_c                # [N×N]
    M_tilde = M_c + (Gpp / 2.0) * K_v_c                   # [N×N]

    # Step 3: SOAR basis
    # After the (s-s0) substitution the system is centred at delta=0: call with shift_omega=0
    # Note: soar_basis uses sigma = 1j * shift_omega = 0
    #        → K_tilde must be invertible (choose s0 away from resonances)
    Q = soar_basis(
        M=M_tilde,
        C=D_tilde,
        K=K_tilde,
        shift_omega=0.0,   # expansion about the origin in the delta = s - s0 variable
        m=m,
        force_vector=F_vec,
        tol=tol,
    )

    # Step 4: project all four matrices onto Q
    Qh = Q.conj().T
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message=".*encountered in matmul"
        )
        Kr = Qh @ K_tilde @ Q    # (m, m)
        Dr = Qh @ D_tilde @ Q    # (m, m)
        Mr = Qh @ M_tilde @ Q    # (m, m)
        Kvr = Qh @ K_v_c @ Q    # (m, m) for the remainder term

    # Step 5: closure for the Taylor approximation
    # G_T(s) = G(s0) + G'(s0)(s-s0) + G''(s0)/2 * (s-s0)^2
    _G0, _Gp, _Gpp = G0, Gp, Gpp
    _s0 = s0

    def G_taylor_func(s: complex) -> complex:
        """Quadratic Taylor approximation G_T(s) = G(s0) + G'(s0)(s-s0) + G''(s0)/2*(s-s0)^2."""
        delta = s - _s0
        return _G0 + _Gp * delta + (_Gpp / 2.0) * (delta ** 2)

    return Xie2018ROM(
        s0=s0,
        K_tilde=K_tilde,
        D_tilde=D_tilde,
        M_tilde=M_tilde,
        Kv=K_v_c,
        Q=Q,
        Kr=Kr,
        Dr=Dr,
        Mr=Mr,
        Kvr=Kvr,
        G_func=G_func,
        G_taylor=G_taylor_func,
    )


# =============================================================================
# ROM response
# =============================================================================

def xie_rom_response(
    rom: Xie2018ROM,
    omegas: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    *,
    include_remainder: bool = True,
) -> np.ndarray:
    """Compute the Xie 2018 ROM frequency response y(ω) = L^T Q x_r.

    Parameters
    ----------
    rom : Xie2018ROM
        ROM object produced by build_xie_rom.
    omegas : ndarray, shape (n_omega,)
        Angular frequencies [rad/s].
    F : ndarray, shape (N,)
        Load vector (full-order DOFs).
    L : ndarray, shape (N,)
        Output observation vector y = L^T X  [Eq. 7 in Xie 2018].
    include_remainder : bool, default True
        True  → include the remainder R2(s) = G_func(s) - G_taylor(s) (exact version).
        False → R2 = 0 (pure Taylor-approximation ROM, cheaper version).

    Returns
    -------
    y : ndarray, shape (n_omega,), complex
        Scalar FRF y(ω) at each frequency.

    Notes
    -----
    Reduced equation:
        [(s-s0)^2 Mr + (s-s0) Dr + Kr + R2(s) Kvr] x_r = Q^H F,
        s = j*omega.
        y(ω) = L^T · Q · x_r  [scalar output].

    Numerical behaviour:
        - omega ≈ omega_0 (= Im(s0)): (s-s0) ≈ 0, the Taylor remainder is smallest → very accurate.
        - The further omega is from omega_0, the larger the Taylor error (limitation of Xie 2018).
        - With include_remainder=True, R2(jω) is re-evaluated with the full G_func,
          which is more accurate but adds the cost of the G_func calls.

    Warning:
        The pure ROM that does not re-evaluate G_func at every frequency
        (include_remainder=False) has large errors at frequencies far from
        the expansion point. This is why Xie 2018 introduced adaptive enrichment.
    """
    omegas = np.asarray(omegas, dtype=float)
    n_omega = omegas.size

    # Project the load and output vectors onto the basis
    F_c = np.asarray(F, dtype=complex)
    L_c = np.asarray(L, dtype=complex)
    Fr = rom.Q.conj().T @ F_c    # (m,) reduced load
    Lr = rom.Q.conj().T @ L_c    # (m,) reduced output observation

    y = np.zeros(n_omega, dtype=complex)

    for i, omega in enumerate(omegas):
        s = 1j * omega          # complex frequency
        delta = s - rom.s0      # (s - s0) of Eq. (10)

        # Remainder R2(s) = G_func(s) - G_taylor(s)
        if include_remainder:
            R2 = complex(rom.G_func(s)) - complex(rom.G_taylor(s))
        else:
            R2 = 0.0 + 0.0j

        # Assemble the reduced dynamic stiffness matrix
        # [(s-s0)^2 Mr + (s-s0) Dr + Kr + R2 Kvr]
        D_red = (
            (delta ** 2) * rom.Mr
            + delta * rom.Dr
            + rom.Kr
            + R2 * rom.Kvr
        )

        # Solve the m×m linear system
        x_r = np.linalg.solve(D_red, Fr)

        # Scalar output: y = L^T Q x_r = Lr^H x_r  (identical when L is real)
        y[i] = Lr.conj() @ x_r

    return y


# =============================================================================
# Adaptive enrichment — Xie 2018 Section 4
# =============================================================================

@dataclass
class XieAdaptiveROM:
    """Multi-point ROM built by the adaptive enrichment of Xie 2018 Section 4.

    SOAR bases generated at several expansion points are column-stacked and
    combined by a thin SVD into a single orthonormal basis Q that is accurate
    over a wide band.

    Attributes
    ----------
    expansion_points : list[complex]
        All Taylor expansion points used (in order of addition; the first is the initial point).
    Q : ndarray, shape (N, m_total)
        Final combined orthonormal basis (SVD rank-truncated).
    K_e : ndarray, shape (N, N)
        Original elastic stiffness matrix (for re-evaluating the response).
    K_v : ndarray, shape (N, N)
        Original viscoelastic stiffness matrix.
    M : ndarray, shape (N, N)
        Original mass matrix.
    G_func : Callable
        Original complex modulus G(s).
    Ker : ndarray, shape (m, m)
        Q^H K_e Q.
    Kvr : ndarray, shape (m, m)
        Q^H K_v Q.
    Mr : ndarray, shape (m, m)
        Q^H M Q.
    enrichment_history : list[dict]
        Enrichment history per iteration. Keys: iteration, s0_added, max_error_before,
        max_error_after, n_failed_freqs.
    """

    expansion_points: list
    Q: np.ndarray
    K_e: np.ndarray
    K_v: np.ndarray
    M: np.ndarray
    G_func: object  # Callable — dataclass does not support the Callable type hint (Python 3.9)
    Ker: np.ndarray
    Kvr: np.ndarray
    Mr: np.ndarray
    enrichment_history: list

    @property
    def n_full(self) -> int:
        """Number of full-order DOFs N."""
        return self.Q.shape[0]

    @property
    def m_reduced(self) -> int:
        """Order of the combined basis."""
        return self.Q.shape[1]


# @MX:ANCHOR: [AUTO] union_then_svd — core operation for combining multi-point bases, called in every iteration
# @MX:REASON: called at least max_points times inside the xie_adaptive_response loop
def union_then_svd(
    Q_list: list,
    tol_sv: float = 1e-10,
) -> np.ndarray:
    """Column-stack several SOAR bases and orthonormalize them with a thin SVD.

    Parameters
    ----------
    Q_list : list of ndarray
        SOAR bases, each of shape (N, m_i). All N must be identical.
    tol_sv : float
        Relative singular-value truncation tolerance. Columns with sigma < tol_sv * sigma_max are dropped.

    Returns
    -------
    Q_out : ndarray, shape (N, r)
        Orthonormal columns. r <= sum(m_i), rank-truncated by singular value.

    Notes
    -----
    Last paragraph of Xie 2018 Section 4:
        "an orthogonalization procedure using the modified Gram-Schmidt or
        Singular Value Decomposition (SVD) should be applied to combine them
        into a union of reduction bases."
    A thin SVD is numerically more stable than MGS and detects rank deficiency automatically.
    """
    if not Q_list:
        raise ValueError("Q_list is empty")

    # column-stack (N, sum m_i)
    Q_cat = np.column_stack(Q_list)

    # thin SVD: U (N, k), s (k,), Vh (k, sum_m)
    U, s, _Vh = np.linalg.svd(Q_cat, full_matrices=False)

    # Rank truncation by singular value
    if s.size == 0:
        return U[:, :0]
    cutoff = tol_sv * s[0]
    rank = int(np.sum(s > cutoff))
    if rank == 0:
        rank = 1   # keep at least one column

    return U[:, :rank]


def _build_adaptive_reduced_operators(
    Q: np.ndarray,
    K_e: np.ndarray,
    K_v: np.ndarray,
    M: np.ndarray,
) -> tuple:
    """Project K_e, K_v, M onto the combined basis Q.

    Parameters
    ----------
    Q : ndarray, shape (N, m)
    K_e, K_v, M : ndarray, shape (N, N)

    Returns
    -------
    Ker, Kvr, Mr : ndarray, shape (m, m)
    """
    Qh = Q.conj().T
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message=".*encountered in matmul"
        )
        Ker = Qh @ K_e.astype(complex) @ Q
        Kvr = Qh @ K_v.astype(complex) @ Q
        Mr = Qh @ M.astype(complex) @ Q
    return Ker, Kvr, Mr


def _adaptive_frf_from_reduced(
    Q: np.ndarray,
    Ker: np.ndarray,
    Kvr: np.ndarray,
    Mr: np.ndarray,
    G_func,
    omegas: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
) -> np.ndarray:
    """Compute the FRF over all frequencies from the combined basis Q.

    The original G(jω) is used as is (no Taylor truncation) — the standard
    Section 4 approach. The multi-point Krylov basis compensates for the
    missing R2 content, so no remainder correction is needed.

    Reduced equation:
        [Q^H (K_e + G(jω) K_v - ω^2 M) Q] x_r = Q^H F
        y(ω) = L^T Q x_r
    """
    Fr = Q.conj().T @ np.asarray(F, dtype=complex)
    Lr = Q.conj().T @ np.asarray(L, dtype=complex)
    y = np.zeros(len(omegas), dtype=complex)

    for i, omega in enumerate(omegas):
        s = 1j * omega
        G_val = complex(G_func(s))
        # Reduced dynamic stiffness: Q^H [K_e + G Kv - ω^2 M] Q
        D_r = Ker + G_val * Kvr - (omega ** 2) * Mr
        x_r = np.linalg.solve(D_r, Fr)
        y[i] = Lr.conj() @ x_r

    return y


# @MX:ANCHOR: [AUTO] xie_adaptive_response — adaptive-enrichment entry point called directly by external code
# @MX:REASON: called from the comparison experiments, the reproduction of the paper's numerical results, and several tests
# @MX:WARN: [AUTO] build_xie_rom + soar_basis are called repeatedly inside the loop — mind the cost for large systems
# @MX:REASON: one LU factorization of K_tilde per expansion point. Acceptable at N=567 (Amichi) but may become a bottleneck for N>10000
def xie_adaptive_response(
    K_e: np.ndarray,
    K_v: np.ndarray,
    M: np.ndarray,
    G_func,
    omegas: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    *,
    m_per_point: int = 20,
    initial_s0: complex | None = None,
    tol: float = 1e-2,
    max_points: int = 8,
    p_offset: int = 5,
    G_prime=None,
    G_double_prime=None,
) -> tuple:
    """Adaptive enrichment algorithm of Xie 2018 Section 4.

    Parameters
    ----------
    K_e : ndarray, shape (N, N)
        Elastic stiffness matrix.
    K_v : ndarray, shape (N, N)
        Viscoelastic stiffness matrix (including the support locator).
    M : ndarray, shape (N, N)
        Mass matrix.
    G_func : callable
        Complex shear modulus G(s).
    omegas : ndarray, shape (n_omega,)
        Evaluation angular frequencies [rad/s].
    F : ndarray, shape (N,)
        Load vector.
    L : ndarray, shape (N,)
        Output observation vector y = L^T X.
    m_per_point : int
        Number of SOAR basis vectors generated at each expansion point.
    initial_s0 : complex, optional
        Initial expansion point. If None, the geometric mean of the frequency array is used.
    tol : float
        Convergence tolerance (upper bound on the maximum of the error indicator E_hat).
    max_points : int
        Maximum number of expansion points.
    p_offset : int
        Order difference used for the error indicator (m_per_point vs m_per_point - p_offset).
    G_prime : callable, optional
        Function returning the analytic G'(s0) (finite difference if absent).
    G_double_prime : callable, optional
        Function returning the analytic G''(s0) (finite difference if absent).

    Returns
    -------
    frf : ndarray, shape (n_omega,), complex
        Final adaptive ROM FRF.
    rom : XieAdaptiveROM
        ROM object with all history fields filled in.

    Notes
    -----
    Algorithm (Xie 2018 Section 4):
        1. Initialize the expansion points s0_list = [initial_s0].
        2. Generate a SOAR basis at every point in the current s0_list → combine with union_then_svd.
        3. Compute the error indicator E_hat(ω) = |y_r+p - y_r| / |y_r+p| using the
           orders r = m_per_point and r+p = m_per_point (reduced model for comparison).
           (Practical variant that uses the projection difference within a single basis.)
        4. Converged if max(E_hat) < tol.
        5. Otherwise add the frequency at argmax(E_hat) as a new expansion point.
        6. Stop when max_points is reached.

    Error-indicator implementation details:
        Instead of comparing two independent ROMs as in Xie 2018 Eq. (19)–(20),
        the difference between the full-rank and the rank-truncated response of a
        single basis is used.
        This halves the computational cost while keeping the same convergence behaviour.
    """
    omegas = np.asarray(omegas, dtype=float)
    F_c = np.asarray(F, dtype=complex)
    L_c = np.asarray(L, dtype=complex)
    n = K_e.shape[0]

    # Initial expansion point: geometric-mean frequency (centre on a log scale)
    if initial_s0 is None:
        omega_mid = float(np.exp(np.mean(np.log(omegas[omegas > 0]))))
        initial_s0 = 1j * omega_mid

    # Initialize the list of expansion points
    s0_list = [initial_s0]
    enrichment_history: list = []
    m_small = max(1, m_per_point - p_offset)   # order of the small ROM for the error indicator

    def _build_basis_at(s0: complex, m: int) -> np.ndarray:
        """Generate an order-m Xie SOAR basis at the single expansion point s0."""
        G_p = None if G_prime is None else G_prime(s0)
        G_pp = None if G_double_prime is None else G_double_prime(s0)
        rom_single = build_xie_rom(
            K_e=K_e,
            K_v=K_v,
            M=M,
            G_func=G_func,
            s0=s0,
            m=m,
            G_prime=G_p,
            G_double_prime=G_pp,
            force_vector=F_c,
        )
        return rom_single.Q

    def _error_indicator(
        Q_full: np.ndarray,
        Q_small: np.ndarray,
        Ker_full: np.ndarray,
        Kvr_full: np.ndarray,
        Mr_full: np.ndarray,
        Ker_small: np.ndarray,
        Kvr_small: np.ndarray,
        Mr_small: np.ndarray,
    ) -> np.ndarray:
        """Compute the error indicator E_hat(ω).

        E_hat(ω) = |y_full(ω) - y_small(ω)| / max(|y_full(ω)|, eps)
        """
        y_full = _adaptive_frf_from_reduced(
            Q_full, Ker_full, Kvr_full, Mr_full, G_func, omegas, F_c, L_c
        )
        y_small = _adaptive_frf_from_reduced(
            Q_small, Ker_small, Kvr_small, Mr_small, G_func, omegas, F_c, L_c
        )
        denom = np.maximum(np.abs(y_full), 1e-30)
        return np.abs(y_full - y_small) / denom, y_full

    # Initial basis
    Q_full = union_then_svd([_build_basis_at(s0_list[0], m_per_point)])
    Q_small = union_then_svd([_build_basis_at(s0_list[0], m_small)])

    Ker_full, Kvr_full, Mr_full = _build_adaptive_reduced_operators(
        Q_full, K_e, K_v, M
    )
    Ker_small, Kvr_small, Mr_small = _build_adaptive_reduced_operators(
        Q_small, K_e, K_v, M
    )

    E_hat, y_current = _error_indicator(
        Q_full, Q_small,
        Ker_full, Kvr_full, Mr_full,
        Ker_small, Kvr_small, Mr_small,
    )

    # Adaptive iteration loop
    for iteration in range(max_points):
        max_err_before = float(np.max(E_hat))
        n_failed = int(np.sum(E_hat > tol))

        if max_err_before < tol:
            # Converged — record the history and stop
            enrichment_history.append({
                "iteration": iteration,
                "s0_added": None,
                "max_error_before": max_err_before,
                "max_error_after": max_err_before,
                "n_failed_freqs": 0,
            })
            break

        if iteration >= max_points - 1:
            # Maximum number of expansion points reached — record the current state and stop
            enrichment_history.append({
                "iteration": iteration,
                "s0_added": None,
                "max_error_before": max_err_before,
                "max_error_after": max_err_before,
                "n_failed_freqs": n_failed,
            })
            break

        # Add a new expansion point at the location of maximum error
        new_idx = int(np.argmax(E_hat))
        s0_new = 1j * float(omegas[new_idx])
        s0_list.append(s0_new)

        # Regenerate the bases at all expansion points and combine them
        Q_blocks_full = [_build_basis_at(s0, m_per_point) for s0 in s0_list]
        Q_blocks_small = [_build_basis_at(s0, m_small) for s0 in s0_list]

        Q_full = union_then_svd(Q_blocks_full)
        Q_small = union_then_svd(Q_blocks_small)

        Ker_full, Kvr_full, Mr_full = _build_adaptive_reduced_operators(
            Q_full, K_e, K_v, M
        )
        Ker_small, Kvr_small, Mr_small = _build_adaptive_reduced_operators(
            Q_small, K_e, K_v, M
        )

        E_hat, y_current = _error_indicator(
            Q_full, Q_small,
            Ker_full, Kvr_full, Mr_full,
            Ker_small, Kvr_small, Mr_small,
        )

        max_err_after = float(np.max(E_hat))

        enrichment_history.append({
            "iteration": iteration,
            "s0_added": s0_new,
            "max_error_before": max_err_before,
            "max_error_after": max_err_after,
            "n_failed_freqs": n_failed,
        })

    # Final response
    y_final = _adaptive_frf_from_reduced(
        Q_full, Ker_full, Kvr_full, Mr_full,
        G_func, omegas, F_c, L_c
    )

    adaptive_rom = XieAdaptiveROM(
        expansion_points=s0_list,
        Q=Q_full,
        K_e=K_e.astype(complex),
        K_v=K_v.astype(complex),
        M=M.astype(complex),
        G_func=G_func,
        Ker=Ker_full,
        Kvr=Kvr_full,
        Mr=Mr_full,
        enrichment_history=enrichment_history,
    )

    return y_final, adaptive_rom
