"""Frequency-dependent corner mounts — poly-QEP formulation for the Amichi plate.

The mount dynamic stiffness is written as a complex quadratic polynomial in ω:
    k(ω)   = k0 + k1·ω + k2·ω²          (mount stiffness, real)
    ω·C(ω) = C0 + C1·ω + C2·ω²          (loss; C(ω) = C0/ω + C1 + C2·ω)
    K_m*(ω) = k(ω) + i·ω·C(ω)           (complex dynamic stiffness)

Applied through the corner w-locator P (diagonal, 1 at the corner w DOFs), the
equation of motion reduces exactly to an ω-independent complex QEP:
    D(ω) = K_plate − ω²M + K_m*(ω)·P
         = [K_plate+(k0+iC0)P] + ω·(k1+iC1)P + ω²·[(k2+iC2)P − M]
    Mapping onto the poly-QEP form D = K* + iωC* − ω²M*:
        M* = M − (k2 + i·C2)·P
        C* = (C1 − i·k1)·P
        K* = K_plate + (k0 + i·C0)·P
    → (λ²M* + λC* + K*) v = 0,  a single complex QEP.

Choosing the loss ω·C(ω) as a parabola with C2<0 gives a peak at ω* = −C1/(2·C2)
(resonant mount) → strong non-proportional damping of the modes near that
frequency → complex modes.

Unlike a distributed viscoelastic core (hysteretic damping, imaginary part of K),
this module provides a route to strongly complex modes through localized
(low-rank) non-proportional damping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyqep.qep_solver import solve_qep

TWO_PI = 2.0 * np.pi


# =============================================================================
# Mount dynamic-stiffness polynomial
# =============================================================================
@dataclass(frozen=True)
class MountPolynomial:
    """Complex quadratic polynomial coefficients of the corner-mount dynamic stiffness K_m*(ω) = k(ω) + i·ω·C(ω).

    k(ω)   = k0 + k1·ω + k2·ω²       [N/m]
    ω·C(ω) = C0 + C1·ω + C2·ω²       [N/m]  (= loss, imaginary part of the dynamic stiffness)
    """
    k0: float
    k1: float
    k2: float
    C0: float
    C1: float
    C2: float

    def stiffness(self, omega: np.ndarray | float) -> np.ndarray | float:
        """Real stiffness k(ω)."""
        return self.k0 + self.k1 * omega + self.k2 * omega ** 2

    def loss(self, omega: np.ndarray | float) -> np.ndarray | float:
        """Loss ω·C(ω) (imaginary part of the dynamic stiffness)."""
        return self.C0 + self.C1 * omega + self.C2 * omega ** 2

    def dynamic_stiffness(self, omega: np.ndarray | float) -> np.ndarray | complex:
        """Complex dynamic stiffness K_m*(ω) = k(ω) + i·ω·C(ω)."""
        return self.stiffness(omega) + 1j * self.loss(omega)

    @property
    def loss_peak_omega(self) -> float:
        """ω* = −C1/(2·C2) at which the loss ω·C(ω) is extremal (C2≠0)."""
        if self.C2 == 0.0:
            return float("nan")
        return -self.C1 / (2.0 * self.C2)


def resonant_mount(
    omega_star: float,
    width: float,
    peak_loss: float,
    k0: float,
    k1: float = 0.0,
    k2: float = 0.0,
) -> MountPolynomial:
    """Build a parabolic loss ω·C(ω) peaking at ω* with value peak_loss and half-width width.

    ω·C(ω) = peak_loss − (peak_loss/width²)·(ω − ω*)²
        ⇒ C2 = −peak_loss/width²,  C1 = 2·|C2|·ω*,  C0 = peak_loss − |C2|·ω*²
    (the loss vanishes at ω = ω* ± width)

    Parameters
    ----------
    omega_star : float
        Angular frequency of the loss peak [rad/s]; typically ω_n of the target mode.
    width : float
        Half-width of the loss parabola [rad/s].
    peak_loss : float
        Loss value ω·C(ω*) at the peak [N/m].
    k0, k1, k2 : float
        Mount stiffness polynomial coefficients [N/m, N·s/(m·rad), ...].
    """
    abs_c2 = peak_loss / width ** 2
    C2 = -abs_c2
    C1 = 2.0 * abs_c2 * omega_star
    C0 = peak_loss - abs_c2 * omega_star ** 2
    return MountPolynomial(k0=k0, k1=k1, k2=k2, C0=C0, C1=C1, C2=C2)


# =============================================================================
# Corner locator + QEP matrices
# =============================================================================
def corner_w_locator(params) -> tuple[np.ndarray, list[int]]:
    """Diagonal locator P for the transverse-displacement (w) DOFs at the 4 corner nodes of the Amichi plate.

    Parameters
    ----------
    params : SandwichPlateParameters
        Requires the nx, ny, n_dof attributes. 7 DOFs per node, w = offset 4.

    Returns
    -------
    P : (n_dof, n_dof) ndarray
        Diagonal matrix with 1 only at the 4 corner w DOFs.
    w_dofs : list[int]
        The 4 global indices of those corner w DOFs.
    """
    nx, ny = params.nx, params.ny
    ndof = params.n_dof
    npx = nx + 1
    corner_nodes = [0, nx, ny * npx, nx + ny * npx]
    w_dofs = [7 * n + 4 for n in corner_nodes]
    P = np.zeros((ndof, ndof))
    for d in w_dofs:
        P[d, d] = 1.0
    return P, w_dofs


def mount_qep_matrices(
    M: np.ndarray,
    K_plate: np.ndarray,
    P: np.ndarray,
    mount: MountPolynomial,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ω-independent complex QEP matrices (M*, C*, K*) of the plate with mounts.

        M* = M − (k2 + i·C2)·P
        C* = (C1 − i·k1)·P
        K* = K_plate + (k0 + i·C0)·P

    K_plate is preferably the elastic (real) plate stiffness. To combine with a
    distributed viscoelastic core, a complex K* may be passed as K_plate
    (mount + core damping together).
    """
    M_star = M.astype(complex) - (mount.k2 + 1j * mount.C2) * P
    C_star = (mount.C1 - 1j * mount.k1) * P
    K_star = K_plate.astype(complex) + (mount.k0 + 1j * mount.C0) * P
    return M_star, C_star, K_star


# =============================================================================
# Complex modal analysis
# =============================================================================
def modal_phase_collinearity(phi: np.ndarray) -> float:
    """Modal Phase Collinearity (1 = real mode, 0 = fully complex).

    λ1,2 = (Sxx+Syy ± sqrt((Sxx−Syy)²+4Sxy²))/2,   S* = Re/Im inner products
    MPC = ((λ1−λ2)/(λ1+λ2))²
    """
    re, im = phi.real, phi.imag
    Sxx, Syy, Sxy = re @ re, im @ im, re @ im
    if Sxx + Syy < 1e-300:
        return 1.0
    root = np.sqrt((Sxx - Syy) ** 2 + 4.0 * Sxy ** 2)
    l1 = (Sxx + Syy + root) / 2.0
    l2 = (Sxx + Syy - root) / 2.0
    return float(((l1 - l2) / (l1 + l2)) ** 2)


@dataclass
class ComplexModeSet:
    """Complex modal analysis results (ascending frequency)."""
    frequencies: np.ndarray     # f_n [Hz] = |λ|/(2π)
    damping_ratios: np.ndarray  # ζ_n = −Re(λ)/|λ|
    loss_factors: np.ndarray    # η_n = 2·ζ_n
    eigenvalues: np.ndarray     # complex λ
    eigenvectors: np.ndarray    # (n_dof, n_modes) complex

    def mpc(self, w_dofs_all: list[int]) -> np.ndarray:
        """MPC of each mode (based on the shape at the given w DOFs)."""
        return np.array([
            modal_phase_collinearity(self.eigenvectors[w_dofs_all, i])
            for i in range(self.eigenvectors.shape[1])
        ])


def solve_complex_modes(
    M_star: np.ndarray,
    C_star: np.ndarray,
    K_star: np.ndarray,
    n_modes: int = 6,
    rigid_tol_hz: float = 5.0,
) -> ComplexModeSet:
    """Solve the complex QEP (λ²M*+λC*+K*)v=0 → lowest n_modes after removing rigid-body modes.

    Modes below rigid_tol_hz (rigid or near-rigid) are discarded.
    """
    res = solve_qep(M_star, C_star, K_star)
    lam, vec = res.eigenvalues, res.eigenvectors
    f = np.abs(lam) / TWO_PI
    mask = np.isfinite(lam) & (lam.imag > 0) & (f > rigid_tol_hz)
    lam, vec = lam[mask], vec[:, mask]
    order = np.argsort(np.abs(lam))[:n_modes]
    lam, vec = lam[order], vec[:, order]
    f = np.abs(lam) / TWO_PI
    zeta = -lam.real / np.abs(lam)
    return ComplexModeSet(
        frequencies=f,
        damping_ratios=zeta,
        loss_factors=2.0 * zeta,
        eigenvalues=lam,
        eigenvectors=vec,
    )


def solve_mount_complex_modes(
    M: np.ndarray,
    K_plate: np.ndarray,
    P: np.ndarray,
    mount: MountPolynomial,
    n_modes: int = 6,
    rigid_tol_hz: float = 5.0,
) -> ComplexModeSet:
    """Complex modes of the mounted plate (convenience wrapper): assemble the QEP matrices and solve."""
    M_star, C_star, K_star = mount_qep_matrices(M, K_plate, P, mount)
    return solve_complex_modes(M_star, C_star, K_star, n_modes, rigid_tol_hz)
