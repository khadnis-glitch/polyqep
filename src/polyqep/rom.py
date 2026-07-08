"""Reduced Order Model (ROM) — dimension reduction for complex QEPs.

Two basis construction strategies:

1) Modal Truncation:
   Solve the full QEP once and select, among the 2n eigenvalues, the m closest
   to the shift
   → theoretically near-exact reproduction with m=n (once n effective
   eigenvalues are available)

2) Shift-Invert Arnoldi:
   Apply the shift-invert operator T = (A − σB)⁻¹B to the linearization (A, B)
   and build the basis from the Krylov subspace K_m(T, v₀)
   → moment-matching effect; accuracy over the frequency band of interest is
   retained even for m ≪ n

Common projection:
    M_r = V_rᴴ M V_r,  C_r = V_rᴴ C V_r,  K_r = V_rᴴ K V_r
    Y(ω) = V_r · (K_r + iωC_r − ω²M_r)⁻¹ · V_rᴴ F
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from polyqep.qep_solver import solve_qep


@dataclass
class ReducedSystem:
    """ROM reduction result (second-order projection form — for modal truncation and the original Arnoldi)."""
    V: np.ndarray       # (n, m) primal basis, orthonormal
    M_r: np.ndarray     # (m, m) complex
    C_r: np.ndarray
    K_r: np.ndarray
    shift_omega: float
    method: str         # "modal" | "arnoldi"

    @property
    def n_full(self) -> int:
        return self.V.shape[0]

    @property
    def m_reduced(self) -> int:
        return self.V.shape[1]


@dataclass
class LinearizedROM:
    """ROM reduction result — Option A (full linearized projection + F-aware).

    Linearization: A z = λ B z where z = [v; λv].
    Reduced:       A_r, B_r ∈ ℂ^{m × m}, r_r = Vᴴ [0; F].
    Response:  z_r(ω) = (A_r − iω B_r)⁻¹ r_r,  Y(ω) = (V z_r)[:n]
    """
    V: np.ndarray        # (2n, m) linearized basis, orthonormal
    A_r: np.ndarray      # (m, m) complex
    B_r: np.ndarray      # (m, m) complex
    r_r: np.ndarray      # (m,)   Vᴴ · [0; F]
    n_full: int          # primal degrees of freedom
    shift_omega: float
    method: str = "arnoldi_improved"

    @property
    def m_reduced(self) -> int:
        return self.V.shape[1]


def modal_truncation_basis(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    shift_omega: float,
    m: int,
) -> np.ndarray:
    """Modal truncation: solve the full QEP, then select the m eigenvectors closest to the shift.

    shift = i·ω₀ (position on the eigenvalue axis corresponding to ω₀ in the transfer-function plane).
    """
    qep = solve_qep(M, C, K)
    shift_lam = 1j * shift_omega
    distances = np.abs(qep.eigenvalues - shift_lam)

    # exclude infinite/NaN entries
    valid = np.isfinite(distances)
    order = np.argsort(distances[valid])
    valid_idx = np.flatnonzero(valid)[order]
    chosen = valid_idx[:m]

    V_raw = qep.eigenvectors[:, chosen]
    # QR orthogonalization (in place of complex Gram-Schmidt)
    V_q, _ = np.linalg.qr(V_raw)
    return V_q


def arnoldi_shift_invert_basis(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    shift_omega: float,
    m: int,
    starting_vector: np.ndarray | None = None,
    tol: float = 1e-12,
    rng_seed: int = 0,
) -> np.ndarray:
    """Shift-invert Arnoldi on linearized QEP → primal basis V_r (n, m).

    Linearization: A = [[0, I], [-K, -C]], B = [[I, 0], [0, M]]
    Shifted op: T = (A − σB)⁻¹ B,  σ = i·ω₀.
    The Krylov subspace K_m(T, v₀) is orthogonalized by Gram-Schmidt.

    The final basis is 2n × m; the primal-space basis is obtained by extracting
    the top n rows followed by QR orthogonalization.
    """
    n = M.shape[0]
    sigma = 1j * shift_omega

    I_n = np.eye(n, dtype=complex)
    Z_n = np.zeros((n, n), dtype=complex)
    A = np.block([[Z_n, I_n], [-K.astype(complex), -C.astype(complex)]])
    B = np.block([[I_n, Z_n], [Z_n, M.astype(complex)]])

    A_sigma = A - sigma * B

    # apply a small perturbation when A − σB is singular
    cond = np.linalg.cond(A_sigma)
    if not np.isfinite(cond) or cond > 1e14:
        A_sigma = A_sigma + 1e-8j * np.eye(2 * n)

    lu, piv = lu_factor(A_sigma)

    # starting vector
    if starting_vector is None:
        rng = np.random.default_rng(rng_seed)
        v0 = rng.standard_normal(2 * n) + 1j * rng.standard_normal(2 * n)
    else:
        if starting_vector.size != 2 * n:
            raise ValueError("starting_vector must have length 2n")
        v0 = starting_vector.astype(complex)

    v0 = v0 / np.linalg.norm(v0)

    V_cols: list[np.ndarray] = [v0]
    for k in range(m):
        # w = (A − σB)⁻¹ B v_k
        w = lu_solve((lu, piv), B @ V_cols[k])
        # Modified Gram-Schmidt with one reorthogonalization pass
        for _ in range(2):
            for i, vi in enumerate(V_cols):
                coef = np.vdot(vi, w)   # vi^H w
                w = w - coef * vi
        nrm = np.linalg.norm(w)
        if nrm < tol:
            break
        V_cols.append(w / nrm)

    V_2n = np.column_stack(V_cols[:m])    # (2n, m) (or fewer)
    # primal space = top n rows
    V_primal = V_2n[:n, :]
    V_q, _ = np.linalg.qr(V_primal)
    return V_q[:, : min(m, V_q.shape[1])]


def project_to_reduced(
    V: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    shift_omega: float,
    method: str,
) -> ReducedSystem:
    """Project (M, C, K) onto the basis V → (M_r, C_r, K_r).

    The transpose congruence V^T(·)V is used to preserve the complex-symmetric
    (T-symmetric) structure. The conjugate transpose V^H would break complex
    symmetry, invalidating bilinear orthogonality and Padé moment matching.
    """
    Vt = V.T
    M_r = Vt @ M.astype(complex) @ V
    C_r = Vt @ C.astype(complex) @ V
    K_r = Vt @ K.astype(complex) @ V
    return ReducedSystem(V=V, M_r=M_r, C_r=C_r, K_r=K_r,
                         shift_omega=shift_omega, method=method)


def reduced_response(
    omega_array: np.ndarray,
    reduced_sys: ReducedSystem,
    force_vector: np.ndarray,
) -> np.ndarray:
    """Compute the ROM response Y(ω) and lift (second-order projection form).

    Y_r(ω) = (K_r + iωC_r − ω²M_r)⁻¹ · V_rᵀ F     (m × m solve)
    Y(ω)   = V_r · Y_r(ω)                           (lift)
    """
    F = np.asarray(force_vector, dtype=complex)
    F_r = reduced_sys.V.T @ F   # transpose congruence (consistent with project_to_reduced)

    omega_array = np.asarray(omega_array, dtype=float)
    n_full = reduced_sys.V.shape[0]
    Y = np.zeros((omega_array.size, n_full), dtype=complex)

    M_r, C_r, K_r = reduced_sys.M_r, reduced_sys.C_r, reduced_sys.K_r
    for i, omega in enumerate(omega_array):
        D_r = K_r + 1j * omega * C_r - (omega ** 2) * M_r
        y_r = np.linalg.solve(D_r, F_r)
        Y[i, :] = reduced_sys.V @ y_r

    return Y


# =============================================================================
# Option A — Full Linearized Projection + F-aware starting vector
# =============================================================================

def arnoldi_improved_rom(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    shift_omega: float,
    m: int,
    force_vector: np.ndarray,
    tol: float = 1e-13,
) -> LinearizedROM:
    """Improved Arnoldi ROM — full linearized projection + force-aware starting.

    Algorithm:
        1. Linearization (A, B) ∈ ℂ^{2n × 2n}
        2. r = [0; F] ∈ ℂ^{2n}  (right-hand side of the transfer function)
        3. Shift-invert operator T = (A − σB)⁻¹ B,  σ = i·shift_omega
        4. Starting vector v₀ = (A − σB)⁻¹ r / ||...||   ← injects the F direction
        5. Arnoldi iteration m times → V ∈ ℂ^{2n × m}, column-orthonormal
        6. Linearized reduction: A_r = Vᴴ A V,  B_r = Vᴴ B V,  r_r = Vᴴ r

    Theory: preserves the order-m Padé approximation property around the shift σ
    (rational Krylov moment matching).
    """
    n = M.shape[0]
    sigma = 1j * shift_omega

    I_n = np.eye(n, dtype=complex)
    Z_n = np.zeros((n, n), dtype=complex)
    A = np.block([[Z_n, I_n], [-K.astype(complex), -C.astype(complex)]])
    B = np.block([[I_n, Z_n], [Z_n, M.astype(complex)]])

    # Linearized transfer-function relation: (iω·B − A) z = [0; F].
    # Shift-invert operator: (σB − A)⁻¹·B
    A_sigma = sigma * B - A
    cond = np.linalg.cond(A_sigma)
    if not np.isfinite(cond) or cond > 1e14:
        A_sigma = A_sigma + 1e-8j * np.eye(2 * n)

    lu, piv = lu_factor(A_sigma)

    # Force-aware starting vector: v₀ = (σB − A)⁻¹ · [0; F]
    F_arr = np.asarray(force_vector, dtype=complex)
    if F_arr.size != n:
        raise ValueError(f"force_vector must have length n={n}, got {F_arr.size}")
    r = np.concatenate([np.zeros(n, dtype=complex), F_arr])

    v0 = lu_solve((lu, piv), r)
    nrm0 = np.linalg.norm(v0)
    if nrm0 < tol:
        raise RuntimeError("starting vector is nearly zero — shift σ is too close to a system pole")
    v0 = v0 / nrm0

    # Arnoldi iteration (vectorized Modified Gram-Schmidt + one reorthogonalization pass)
    # Deflation: stop when the post-orthogonalization norm drops below tol times the pre-orthogonalization norm
    # Preallocate V_mat so that no column_stack is needed at every iteration.
    V_mat = np.empty((2 * n, min(m, 2 * n)), dtype=complex)
    V_mat[:, 0] = v0
    n_built = 1
    for k in range(min(m - 1, 2 * n - 1)):
        w = lu_solve((lu, piv), B @ V_mat[:, k])
        nrm_before = np.linalg.norm(w)
        if nrm_before < tol:
            break   # the input itself is nearly zero
        # vectorized GS: w -= V[:, :n_built] · (V[:, :n_built]^H · w), repeated twice
        V_built = V_mat[:, :n_built]
        for _ in range(2):
            coefs = V_built.conj().T @ w
            w = w - V_built @ coefs
        nrm = np.linalg.norm(w)
        # relative deflation: treat as linearly dependent if >= 99.99% is removed by GS
        if nrm < tol * nrm_before or nrm < 1e-12:
            break   # deflation
        V_mat[:, n_built] = w / nrm
        n_built += 1

    V = V_mat[:, :n_built]    # (2n, n_built)
    Vh = V.conj().T

    # Linearized reduction
    # macOS Accelerate BLAS emits spurious FPE warnings for small m, but the results are correct.
    # Keep the fast BLAS path (chained matmul) and only suppress the warning locally.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message=".*encountered in matmul")
        AV = A @ V
        BV = B @ V
        A_r = Vh @ AV
        B_r = Vh @ BV
        r_r = Vh @ r

    return LinearizedROM(
        V=V, A_r=A_r, B_r=B_r, r_r=r_r,
        n_full=n, shift_omega=shift_omega, method="arnoldi_improved",
    )


def linearized_rom_response(
    omega_array: np.ndarray,
    lin_rom: LinearizedROM,
) -> np.ndarray:
    """Linearized ROM response: Y(ω) = (V · (A_r − iω B_r)⁻¹ r_r) [:n].

    For each ω: solve an m × m linear system, lift to 2n, extract the top n rows.
    """
    omega_array = np.asarray(omega_array, dtype=float)
    n = lin_rom.n_full
    Y = np.zeros((omega_array.size, n), dtype=complex)

    A_r, B_r, r_r, V = lin_rom.A_r, lin_rom.B_r, lin_rom.r_r, lin_rom.V
    for i, omega in enumerate(omega_array):
        # transfer-function relation: (iω·B − A) z = r  →  z = (iω·B_r − A_r)⁻¹ r_r
        z_r = np.linalg.solve(1j * omega * B_r - A_r, r_r)
        z = V @ z_r                     # (2n,)
        Y[i, :] = z[:n]                  # primal components

    return Y


# =============================================================================
# Option B — SOAR (Second-Order Arnoldi, Bai & Su 2005)
# =============================================================================

@dataclass
class SecondOrderROM:
    """ROM reduction result — SOAR (Option B, second-order structure preserving).

    Reference
    ---------
    Bai & Su, "Dimension Reduction of Large-Scale Second-Order Dynamical Systems
    via a Second-Order Arnoldi Method," SIAM J. Matrix Anal. Appl., 2005.

    Structure:
        Q ∈ ℂ^{n × m}  primal basis (orthonormalized in the Hermitian inner product)
        M_r, C_r, K_r ∈ ℂ^{m × m}  second-order reduction via the transpose
                       congruence Q^T(·)Q, preserving complex symmetry
        F_r ∈ ℂ^{m}    Q^T · F (cached)
        Response: Y(ω) = Q · (K_r + iωC_r − ω²M_r)⁻¹ · F_r
    """
    Q: np.ndarray         # (n, m) primal basis, orthonormal
    M_r: np.ndarray       # (m, m) complex
    C_r: np.ndarray
    K_r: np.ndarray
    F_r: np.ndarray       # (m,) cached Q^T F
    shift_omega: float
    method: str = "soar"

    @property
    def n_full(self) -> int:
        return self.Q.shape[0]

    @property
    def m_reduced(self) -> int:
        return self.Q.shape[1]


def soar_basis(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    shift_omega: float,
    m: int,
    force_vector: np.ndarray,
    tol: float = 1e-13,
) -> np.ndarray:
    """SOAR basis Q ∈ ℂ^{n × m_actual} for shifted QEP transfer function.

    Algorithm:
        G(σ) = (K + σC + σ²M)⁻¹            [σ = i·shift_omega]
        P'(σ) = C + 2σM                     [shifted first derivative]
        q₁ = G(σ) · F / ||G(σ) · F||         [force-aware]
        p₁ = 0
        for k = 1..m-1:
            z = -G(σ) [P'(σ) qₖ + M pₖ]      [second-order recursion]
            [MGS twice against q_1..q_k]
            βₖ = ||z_orth||
            if βₖ < tol: break (deflation)
            qₖ₊₁ = z_orth / βₖ
            pₖ₊₁ = (qₖ - Σⱼ hⱼₖ pⱼ) / βₖ      [dual recursion]

    Padé matching: order-m moment matching around the shift σ; no primal-extraction
    trick is needed at the output stage (avoids the weakness of the linearized form).
    """
    n = M.shape[0]
    sigma = 1j * shift_omega

    M_c = M.astype(complex)
    C_c = C.astype(complex)
    K_c = K.astype(complex)

    # G(σ) = (K + σC + σ²M)⁻¹ via LU
    Mat_sigma = K_c + sigma * C_c + (sigma ** 2) * M_c
    cond = np.linalg.cond(Mat_sigma)
    if not np.isfinite(cond) or cond > 1e14:
        Mat_sigma = Mat_sigma + 1e-8j * np.eye(n)
    lu, piv = lu_factor(Mat_sigma)

    F_arr = np.asarray(force_vector, dtype=complex)
    if F_arr.size != n:
        raise ValueError(f"force_vector must have length n={n}, got {F_arr.size}")

    # Force-aware starting vector: q₁ = G(σ) · F (injects the direction)
    q1 = lu_solve((lu, piv), F_arr)
    nrm0 = np.linalg.norm(q1)
    if nrm0 < tol:
        raise RuntimeError("starting vector is nearly zero — shift σ is too close to a system pole")
    q1 = q1 / nrm0

    # P'(σ) = C + 2σM
    P_prime = C_c + 2.0 * sigma * M_c

    # preallocate (n × min(m, n))
    cap = min(m, n)
    Q_mat = np.empty((n, cap), dtype=complex)
    P_mat = np.zeros((n, cap), dtype=complex)
    Q_mat[:, 0] = q1
    # P_mat[:, 0] = 0  (already zeros)
    n_built = 1

    for k in range(min(m - 1, n - 1)):
        q_k = Q_mat[:, k]
        p_k = P_mat[:, k]

        # z = -G(σ) · [P'(σ) qₖ + M pₖ]
        rhs = P_prime @ q_k + M_c @ p_k
        z = -lu_solve((lu, piv), rhs)

        nrm_before = np.linalg.norm(z)
        if nrm_before < tol:
            break  # the input itself is nearly zero

        # MGS twice for numerical stability; coefficients accumulated for the
        # dual recursion (Bai & Su 2005 Alg. SOAR requires the SAME h_ij on p)
        Q_built = Q_mat[:, :n_built]
        h_tot = np.zeros(n_built, dtype=complex)
        for _ in range(2):
            coefs = Q_built.conj().T @ z
            z = z - Q_built @ coefs
            h_tot += coefs

        beta = np.linalg.norm(z)
        # relative deflation (same criterion as Option A)
        if beta < tol * nrm_before or beta < 1e-12:
            break  # deflation

        q_new = z / beta
        # Bai & Su 2005 SOAR dual recursion: p_{k+1} = (q_k - Σ h_ij p_i) / β.
        # (The earlier simplification p=q_k/β stalls the measured moment order at
        #  ~6.4 for m>=5; see the accompanying paper. The original recursion
        #  attains the 2m order in measurements, with no instability observed
        #  under MGS x2.)
        p_new = (q_k - P_mat[:, :n_built] @ h_tot) / beta

        Q_mat[:, n_built] = q_new
        P_mat[:, n_built] = p_new
        n_built += 1

    return Q_mat[:, :n_built]


def build_soar_rom(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    shift_omega: float,
    m: int,
    force_vector: np.ndarray,
    tol: float = 1e-13,
) -> SecondOrderROM:
    """Build the SOAR basis and reduce the second-order system (M, C, K, F) to m×m."""
    Q = soar_basis(M, C, K, shift_omega, m, force_vector, tol)
    # preserve the complex-symmetric (T-symmetric) structure: use the transpose congruence Q^T(·)Q (not Q^H).
    Qt = Q.T

    M_c = M.astype(complex)
    C_c = C.astype(complex)
    K_c = K.astype(complex)
    F_c = np.asarray(force_vector, dtype=complex)

    # locally suppress spurious macOS Accelerate BLAS warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message=".*encountered in matmul"
        )
        M_r = Qt @ M_c @ Q
        C_r = Qt @ C_c @ Q
        K_r = Qt @ K_c @ Q
        F_r = Qt @ F_c

    return SecondOrderROM(
        Q=Q, M_r=M_r, C_r=C_r, K_r=K_r, F_r=F_r,
        shift_omega=shift_omega, method="soar",
    )


def soar_response(
    omega_array: np.ndarray,
    soar_rom: SecondOrderROM,
) -> np.ndarray:
    """SOAR ROM response: Y(ω) = Q · (K_r + iωC_r − ω²M_r)⁻¹ · F_r.

    For each ω: solve an m × m linear system and lift with Q (keeping the
    second-order structure intact).
    """
    omega_array = np.asarray(omega_array, dtype=float)
    n = soar_rom.n_full
    Y = np.zeros((omega_array.size, n), dtype=complex)

    M_r = soar_rom.M_r
    C_r = soar_rom.C_r
    K_r = soar_rom.K_r
    F_r = soar_rom.F_r
    Q = soar_rom.Q

    for i, omega in enumerate(omega_array):
        D_r = K_r + 1j * omega * C_r - (omega ** 2) * M_r
        y_r = np.linalg.solve(D_r, F_r)
        Y[i, :] = Q @ y_r

    return Y


# =============================================================================
# Option C — Multi-Shift SOAR (Rational Krylov, wide-band basis)
# =============================================================================

def multi_shift_soar_basis(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    shift_omegas: list[float] | np.ndarray,
    m_per_shift: int,
    force_vector: np.ndarray,
    rank_tol: float = 1e-10,
) -> np.ndarray:
    """Build SOAR bases at several shifts, then merge and orthogonalize — wide-band coverage.

    Reference
    ---------
    Grimme, "Krylov Projection Methods for Model Reduction," Ph.D. thesis,
    UIUC, 1997 (rational Krylov basics).
    Idea: order-m_per_shift Padé matching at each σ_i → wide-band matching via the union.

    Parameters
    ----------
    shift_omegas : list of float
        ω₀ of each shift [rad/s]. K = len(shift_omegas).
    m_per_shift : int
        SOAR basis order per shift.

    Returns
    -------
    Q : (n, m_actual) ndarray
        Orthogonalized basis. m_actual ≤ K · m_per_shift (after removing linearly dependent columns).
    """
    if len(shift_omegas) == 0:
        raise ValueError("shift_omegas is empty")

    Q_blocks: list[np.ndarray] = []
    for sigma_om in shift_omegas:
        try:
            Q_i = soar_basis(M, C, K, float(sigma_om), m_per_shift, force_vector)
            Q_blocks.append(Q_i)
        except RuntimeError:
            # the starting vector is zero at this shift (on a pole) — skip only this shift
            continue

    if not Q_blocks:
        raise RuntimeError("SOAR failed at every shift — the system is too ill-conditioned")

    Q_concat = np.column_stack(Q_blocks)   # (n, K · m_per_shift)

    # orthogonalize with QR and drop rank-deficient columns
    Q_full, R = np.linalg.qr(Q_concat)
    diag_R = np.abs(np.diag(R))
    if diag_R.size == 0:
        return Q_full[:, :0]
    cutoff = rank_tol * diag_R[0]
    keep_mask = diag_R > cutoff
    Q_orth = Q_full[:, keep_mask]
    return Q_orth


def build_multi_shift_soar_rom(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    shift_omegas: list[float] | np.ndarray,
    m_per_shift: int,
    force_vector: np.ndarray,
    rank_tol: float = 1e-10,
) -> SecondOrderROM:
    """Multi-shift SOAR basis + second-order system reduction.

    Returns a SecondOrderROM as-is (reuses the response-computation routine).
    """
    Q = multi_shift_soar_basis(M, C, K, shift_omegas, m_per_shift,
                               force_vector, rank_tol)
    # preserve the complex-symmetric (T-symmetric) structure: use the transpose congruence Q^T(·)Q (not Q^H).
    Qt = Q.T

    M_c = M.astype(complex)
    C_c = C.astype(complex)
    K_c = K.astype(complex)
    F_c = np.asarray(force_vector, dtype=complex)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message=".*encountered in matmul"
        )
        M_r = Qt @ M_c @ Q
        C_r = Qt @ C_c @ Q
        K_r = Qt @ K_c @ Q
        F_r = Qt @ F_c

    # representative shift = mean (metadata only; does not affect the response computation)
    avg_shift = float(np.mean(np.asarray(shift_omegas, dtype=float)))
    return SecondOrderROM(
        Q=Q, M_r=M_r, C_r=C_r, K_r=K_r, F_r=F_r,
        shift_omega=avg_shift, method="multi_shift_soar",
    )
