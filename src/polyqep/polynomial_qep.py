"""PolyQEP core — per-interval quadratic polynomial fitting + complex QEP + effective-eigenvalue selection.

Substituting the fitted coefficients into the equation of motion yields complex matrices M*, C*, K* independent of ω:

    k(ω)   = k₁·ω² + k₂·ω + k₃
    ω·c(ω) = d₁·ω² + d₂·ω + d₃

    M* = M − (k₁ + i·d₁)·L
    C* = (d₂ − i·k₂)·L
    K* = K_base + (k₃ + i·d₃)·L

The frequency-domain response of this interval is **exactly**:
    D(ω) = K* + iω·C* − ω²·M*   (identical to the original direct method as long as the fit is exact)

Complex QEP: (λ²M* + λC* + K*) v = 0
The same eigenpairs describe the poles of D(ω), and Y(ω) can be reconstructed by modal superposition.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve as linsolve

from polyqep.mdof_6dof import (
    SixDOFParameters,
    build_mass_matrix,
    build_base_stiffness,
    build_support_locator,
)
from polyqep.fitting import QuadraticSegment, QuadraticSpline
from polyqep.qep_solver import solve_qep
from polyqep.transfer_function import modal_superposition
from polyqep.rom import (
    modal_truncation_basis,
    arnoldi_shift_invert_basis,
    project_to_reduced,
    reduced_response,
    ReducedSystem,
    arnoldi_improved_rom,
    linearized_rom_response,
    LinearizedROM,
    build_soar_rom,
    soar_response,
    SecondOrderROM,
    build_multi_shift_soar_rom,
)


@dataclass
class ComplexMatrices:
    """Complex matrices M*, C*, K* for one interval."""
    M_star: np.ndarray
    C_star: np.ndarray
    K_star: np.ndarray
    omega_lower: float
    omega_upper: float


def assemble_complex_matrices(
    k_segment: QuadraticSegment,
    oc_segment: QuadraticSegment,
    M: np.ndarray,
    L: np.ndarray,
    K_base: np.ndarray,
) -> ComplexMatrices:
    """Fitted coefficients of one interval → complex M*, C*, K*."""
    if not np.isclose(k_segment.omega_lower, oc_segment.omega_lower):
        raise ValueError("k and ω·c fits must use the same interval partition")

    k1, k2, k3 = k_segment.a, k_segment.b, k_segment.c
    d1, d2, d3 = oc_segment.a, oc_segment.b, oc_segment.c

    M_star = M.astype(complex) - (k1 + 1j * d1) * L
    C_star = (d2 - 1j * k2) * L.astype(complex)
    K_star = K_base.astype(complex) + (k3 + 1j * d3) * L

    return ComplexMatrices(
        M_star=M_star,
        C_star=C_star,
        K_star=K_star,
        omega_lower=k_segment.omega_lower,
        omega_upper=k_segment.omega_upper,
    )


@dataclass
class IntervalSolution:
    """Complex QEP result for one interval."""
    matrices: ComplexMatrices
    eigenvalues: np.ndarray     # 2n complex λ
    eigenvectors: np.ndarray    # (n, 2n) complex eigenvectors

    @property
    def omega_lower(self) -> float:
        return self.matrices.omega_lower

    @property
    def omega_upper(self) -> float:
        return self.matrices.omega_upper


def solve_per_interval(
    spline_k: QuadraticSpline,
    spline_oc: QuadraticSpline,
    params: SixDOFParameters = SixDOFParameters(),
) -> list[IntervalSolution]:
    """Solve the complex QEP for each interval and return eigenvalues/eigenvectors.

    Parameters
    ----------
    spline_k, spline_oc : QuadraticSpline
        Fitted results for k(ω) and ω·c(ω), respectively. Must use **the same interval partition**.

    Returns
    -------
    list[IntervalSolution]  one per interval
    """
    if not np.allclose(spline_k.breakpoints, spline_oc.breakpoints):
        raise ValueError("interval breakpoints of spline_k and spline_oc do not match")

    M = build_mass_matrix(params)
    K_base = build_base_stiffness(params)
    L = build_support_locator(params)

    results: list[IntervalSolution] = []
    for seg_k, seg_oc in zip(spline_k.segments, spline_oc.segments):
        matrices = assemble_complex_matrices(seg_k, seg_oc, M, L, K_base)
        qep_res = solve_qep(matrices.M_star, matrices.C_star, matrices.K_star)
        results.append(IntervalSolution(
            matrices=matrices,
            eigenvalues=qep_res.eigenvalues,
            eigenvectors=qep_res.eigenvectors,
        ))
    return results


def select_effective_eigenvalues(
    interval_sol: IntervalSolution,
    damping_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the **effective eigenvalues** among the 2n eigenvalues of one interval.

    Selection criteria (in the QEP convention; only the sign differs from the γ convention of the accompanying paper):
        - Im(λ) ∈ [ω_lower, ω_upper)  (frequency within the interval)
        - Re(λ) < −damping_tol        (damped direction → stable)

    In the γ convention this corresponds to Re(γ)>0, Im(γ)<0 (fourth quadrant).
    γ = −λ (because the state-space representation is written as `A·r + B·ṙ = f`).

    An interval may contain multiple effective eigenvalues (modes with close natural frequencies).

    Returns
    -------
    eigvals_eff : (k,) complex
    eigvecs_eff : (n, k) complex
    """
    lams = interval_sol.eigenvalues
    vecs = interval_sol.eigenvectors
    lo = interval_sol.omega_lower
    hi = interval_sol.omega_upper

    mask = (
        np.isfinite(lams)
        & (np.imag(lams) >= lo)
        & (np.imag(lams) < hi)
        & (np.real(lams) < -damping_tol)
    )
    return lams[mask], vecs[:, mask]


def select_effective_with_conjugate(
    interval_sol: IntervalSolution,
    damping_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Effective eigenvalues + their **artificial conjugates** — conjugate-symmetric completion into an equivalent viscously damped system.

    Procedure:
      1. Select the n_eff effective eigenvalues (Re(λ)<0, Im(λ)∈[ω_lo, ω_hi))
      2. For each effective λ_k, generate its conjugate λ̄_k (keep the real part, flip the sign of the imaginary part)
      3. Conjugate the eigenvectors likewise (v̄_k = v_k.conjugate())
      4. Replace the original non-effective eigenvalues with these conjugates, forming 2·n_eff eigenvalues as "equivalent conjugate pairs"

    This effectively approximates the original complex system by a **symmetric
    viscously damped system**. High accuracy is reported at low damping, with
    degraded accuracy at high damping (large ζ, η) in the accompanying paper.

    Returns
    -------
    eigvals_full : (2·n_eff,) complex
        Effective λ's concatenated with their conjugates λ̄
    eigvecs_full : (n, 2·n_eff) complex
        Corresponding eigenvectors and their conjugates
    """
    lams_eff, vecs_eff = select_effective_eigenvalues(interval_sol, damping_tol)
    if lams_eff.size == 0:
        return lams_eff, vecs_eff

    lams_conj = lams_eff.conjugate()
    vecs_conj = vecs_eff.conjugate()

    lams_full = np.concatenate([lams_eff, lams_conj])
    vecs_full = np.concatenate([vecs_eff, vecs_conj], axis=1)
    return lams_full, vecs_full


def direct_response_from_poly(
    omega_array: np.ndarray,
    interval_solutions: list[IntervalSolution],
    force_vector: np.ndarray | None = None,
) -> np.ndarray:
    """Compute D(ω)⁻¹·F directly from the fitted complex matrices (accuracy reference before ROM).

    Uses the (M*, C*, K*) of the interval containing each ω.
    If the fit is exact, this nearly matches `direct_method.direct_frequency_response`.
    """
    n_dof = interval_solutions[0].matrices.M_star.shape[0]

    if force_vector is None:
        F = np.zeros(n_dof, dtype=complex)
        F[-1] = 1.0
    else:
        F = np.asarray(force_vector, dtype=complex)

    # Interval boundary array
    lowers = np.array([s.omega_lower for s in interval_solutions])

    omega_array = np.asarray(omega_array, dtype=float)
    Y = np.zeros((omega_array.size, n_dof), dtype=complex)

    for i, omega in enumerate(omega_array):
        # Index of the interval containing this ω (last interval with lower ≤ ω)
        j = int(np.searchsorted(lowers, omega, side="right") - 1)
        j = max(0, min(j, len(interval_solutions) - 1))
        mat = interval_solutions[j].matrices
        D = mat.K_star + 1j * omega * mat.C_star - (omega ** 2) * mat.M_star
        Y[i, :] = linsolve(D, F)

    return Y


def build_rom_per_interval(
    interval_solutions: list[IntervalSolution],
    reduced_dim: int,
    method: str = "arnoldi",
    force_vector: np.ndarray | None = None,
    n_shifts: int = 2,
) -> list:
    """Build the ROM basis for each interval and return a list of reduced systems.

    Parameters
    ----------
    reduced_dim : int
        Reduced dimension m (beneficial only when less than or equal to the original n).
    method : "modal" | "arnoldi" | "arnoldi_improved" | "soar" | "multi_shift_soar"
        - "modal": full QEP solve + select the m eigenvectors nearest the shift
        - "arnoldi": original shift-invert Arnoldi (primal-only extraction, known deficiency)
        - "arnoldi_improved": Full linearized projection + Force-aware v₀ (option A)
        - "soar": Second-Order Arnoldi (Bai & Su 2005, option B) — preserves second-order structure
        - "multi_shift_soar": ★ union of SOAR bases from K shifts + orthogonalization (option C)
    force_vector : (n,) ndarray | None
        Required for "arnoldi_improved" / "soar" / "multi_shift_soar".
        If None, unit force at the last node.
    n_shifts : int
        Used only for "multi_shift_soar". Number of shifts K distributed within each interval (default 2).
        m_per_shift = ceil(reduced_dim / K).

    Returns
    -------
    list of (ReducedSystem | LinearizedROM | SecondOrderROM)
    """
    n_dof = interval_solutions[0].matrices.M_star.shape[0]
    if force_vector is None:
        F = np.zeros(n_dof, dtype=complex)
        F[-1] = 1.0
    else:
        F = np.asarray(force_vector, dtype=complex)

    rom_systems: list = []
    for sol in interval_solutions:
        mat = sol.matrices
        shift_omega = 0.5 * (mat.omega_lower + mat.omega_upper)

        if method == "modal":
            V = modal_truncation_basis(mat.M_star, mat.C_star, mat.K_star,
                                       shift_omega, reduced_dim)
            red = project_to_reduced(V, mat.M_star, mat.C_star, mat.K_star,
                                     shift_omega, method)
        elif method == "arnoldi":
            V = arnoldi_shift_invert_basis(mat.M_star, mat.C_star, mat.K_star,
                                           shift_omega, reduced_dim)
            red = project_to_reduced(V, mat.M_star, mat.C_star, mat.K_star,
                                     shift_omega, method)
        elif method == "arnoldi_improved":
            red = arnoldi_improved_rom(
                mat.M_star, mat.C_star, mat.K_star,
                shift_omega, reduced_dim, force_vector=F,
            )
        elif method == "soar":
            red = build_soar_rom(
                mat.M_star, mat.C_star, mat.K_star,
                shift_omega, reduced_dim, force_vector=F,
            )
        elif method == "multi_shift_soar":
            # Distribute K shifts uniformly within the interval (25%/75% positions for K=2)
            K = max(1, int(n_shifts))
            interval_lo = mat.omega_lower
            interval_hi = mat.omega_upper
            # Inset slightly from both ends (uniform over the inner 60% of the interval)
            offsets = (np.arange(K) + 0.5) / K   # 1/(2K), 3/(2K), ...
            shifts = interval_lo + 0.2 * (interval_hi - interval_lo) + \
                     0.6 * (interval_hi - interval_lo) * offsets
            m_per = int(np.ceil(reduced_dim / K))
            red = build_multi_shift_soar_rom(
                mat.M_star, mat.C_star, mat.K_star,
                shifts.tolist(), m_per, force_vector=F,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        rom_systems.append(red)
    return rom_systems


def rom_response_piecewise(
    omega_array: np.ndarray,
    interval_solutions: list[IntervalSolution],
    rom_systems: list,
    force_vector: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct Y(ω) over the full frequency band using per-interval ROMs.

    Each ω uses the ROM of its interval. The appropriate response routine is
    selected according to the element type of rom_systems (ReducedSystem vs LinearizedROM).
    """
    n_dof = interval_solutions[0].matrices.M_star.shape[0]
    if force_vector is None:
        F = np.zeros(n_dof, dtype=complex)
        F[-1] = 1.0
    else:
        F = np.asarray(force_vector, dtype=complex)

    omega_array = np.asarray(omega_array, dtype=float)
    Y = np.zeros((omega_array.size, n_dof), dtype=complex)

    for j, (sol, red) in enumerate(zip(interval_solutions, rom_systems)):
        if j < len(interval_solutions) - 1:
            mask = (omega_array >= sol.omega_lower) & (omega_array < sol.omega_upper)
        else:
            mask = omega_array >= sol.omega_lower
        if not mask.any():
            continue

        if isinstance(red, LinearizedROM):
            Y[mask, :] = linearized_rom_response(omega_array[mask], red)
        elif isinstance(red, SecondOrderROM):
            Y[mask, :] = soar_response(omega_array[mask], red)
        else:
            Y[mask, :] = reduced_response(omega_array[mask], red, F)

    return Y


def response_from_modes_piecewise(
    omega_array: np.ndarray,
    interval_solutions: list[IntervalSolution],
    force_vector: np.ndarray | None = None,
    mode: str = "full",
) -> np.ndarray:
    """Reconstruct Y(ω) by piecewise modal superposition.

    For each evaluation frequency ω, modal superposition is applied with the mode set of its interval.

    Parameters
    ----------
    mode : {"full", "effective_only", "effective_mirror"}
        - "full" (default): use **all 2n** QEP modes of each interval
          → complete representation of the original complex system; numerically matches `direct_response_from_poly`
        - "effective_only": use only the n_eff effective eigenvalues of each interval (no conjugate-symmetric completion)
          → not the procedure of the accompanying paper; for reference as a numerical approximation bound
        - "effective_mirror": **the full procedure of the accompanying paper**
          the set of n_eff effective eigenvalues + their n_eff conjugates = 2·n_eff, converted into an equivalent viscously damped system
          → similar to (B)/(C) at low damping, with growing error at high damping (as noted in the accompanying paper)
    """
    n_dof = interval_solutions[0].matrices.M_star.shape[0]
    if force_vector is None:
        F = np.zeros(n_dof, dtype=complex)
        F[-1] = 1.0
    else:
        F = np.asarray(force_vector, dtype=complex)

    lowers = np.array([s.omega_lower for s in interval_solutions])
    omega_array = np.asarray(omega_array, dtype=float)
    Y = np.zeros((omega_array.size, n_dof), dtype=complex)

    # For each interval, apply modal superposition only to the ω points belonging to it
    for j, sol in enumerate(interval_solutions):
        # Indices of ω belonging to this interval
        if j < len(interval_solutions) - 1:
            mask = (omega_array >= sol.omega_lower) & (omega_array < sol.omega_upper)
        else:
            mask = omega_array >= sol.omega_lower   # The last interval includes the right boundary
        if not mask.any():
            continue

        if mode == "full":
            lams, vecs = sol.eigenvalues, sol.eigenvectors
        elif mode == "effective_only":
            lams, vecs = select_effective_eigenvalues(sol)
        elif mode == "effective_mirror":
            lams, vecs = select_effective_with_conjugate(sol)
        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. "
                f"Choose from 'full', 'effective_only', 'effective_mirror'."
            )

        if lams.size == 0:
            continue

        Y_local = modal_superposition(
            omega_array=omega_array[mask],
            eigenvalues=lams,
            eigenvectors=vecs,
            M=sol.matrices.M_star,
            C=sol.matrices.C_star,
            force_vector=F,
        )
        Y[mask, :] = Y_local

    return Y
