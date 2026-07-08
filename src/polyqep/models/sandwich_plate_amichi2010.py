"""Amichi, Atalla, Ruokolainen (2010) — Three-layer sandwich plate element.

Reference
---------
Amichi K., Atalla N., Ruokolainen R.,
"A new 3D finite element sandwich plate for predicting the vibroacoustic
response of laminated steel panels,"
Finite Elements in Analysis and Design 46 (2010) 1131-1145.

Element specification (Section 3.1, Fig. 3)
-------------------------------------------
- 4-node rectangular Q4 element with 28 DOFs/element
- 7 DOFs/node: q^n_e = [U_02, V_02, ψ_x, ψ_y, W, W_x, W_y]^T
- Lagrange bilinear interpolation: U_02, V_02, ψ_x, ψ_y
- Hermite cubic interpolation: W (with W, W_x, W_y at each node)
- Master element ξ, η ∈ [-1, 1]

Displacement field (Eq 1, symmetric sandwich z₂ = -z₃ = -h₂/2)
----------------------------------------------------------------
Layer i (i=1,2,3 from bottom face to top face):
    U_i(x,y,z,t) = U_02 - z·∂W/∂x + z_i·ψ_x   (where z_i is layer's z-coordinate)
    V_i(x,y,z,t) = V_02 - z·∂W/∂y + z_i·ψ_y
    W_i(x,y,z,t) = W(x,y,t)

Note: ψ_x = ∂W/∂x + γ_x, ψ_y = ∂W/∂y + γ_y (γ = core shear angle)

Strain decomposition (Eqs 4-7)
------------------------------
    ε_m  = D_um · u  (membrane)         3×1
    ψ    = D_ur · u  (rotational)        3×1
    χ    = D_uf · u  (bending)           3×1
    γ_xz, γ_yz = D_uc · u  (core shear)  2×1

Coefficient matrices (Eqs 10-11)
--------------------------------
    C_m  = h₁·C⁽¹⁾ + h₂·C⁽²⁾ + h₃·C⁽³⁾
    C_f  = (z₂³-z₁³)/3·C⁽¹⁾ + (z₃³-z₂³)/3·C⁽²⁾ + (z₄³-z₃³)/3·C⁽³⁾
    C_r  = z₂²h₁·C⁽¹⁾ + (z₃³-z₂³)/3·C⁽²⁾ + z₃²h₃·C⁽³⁾
    C_mr = z₂h₁·C⁽¹⁾ + (z₃²-z₂²)/2·C⁽²⁾ + z₃h₃·C⁽³⁾
    C_mf = -[(z₂²-z₁²)/2·C⁽¹⁾ + (z₃²-z₂²)/2·C⁽²⁾ + (z₄²-z₃²)/2·C⁽³⁾]
    C_fr = -[z₂(z₂²-z₁²)/2·C⁽¹⁾ + (z₃³-z₂³)/3·C⁽²⁾ + z₃(z₄²-z₃²)/2·C⁽³⁾]
    C_c  = h₂·C⁽c⁾

C⁽i⁾ is the plane-stress behavior matrix for layer i:
    C⁽i⁾ = (E_i/(1-ν_i²)) · [[1, ν_i, 0], [ν_i, 1, 0], [0, 0, (1-ν_i)/2]]
C⁽c⁾ for core shear (assumed isotropic): 2×2 = G_c · I

Element stiffness (Eq 15)
-------------------------
K_e = ∫_A (β_m^T C_m β_m + β_f^T C_f β_f + β_r^T C_r β_r
         + β_m^T C_mf β_f + β_f^T C_mf^T β_m
         + β_m^T C_mr β_r + β_r^T C_mr^T β_m
         + β_f^T C_fr β_r + β_r^T C_fr^T β_f
         + β_c^T C_c β_c) dA

where β_m = D_um·N_interp etc.

Implementation status
---------------------
- Phase 2A: 4-node Q4 + monolithic validation (this module).
- DKT triangle, drilling DOFs, curved plate transformation: future work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# =============================================================================
# Parameters
# =============================================================================

@dataclass(frozen=True)
class SandwichPlateParameters:
    """Three-layer symmetric sandwich plate parameters (Amichi 2010 Section 2)."""

    # Geometry
    Lx: float          # plate dimension in x [m]
    Ly: float          # plate dimension in y [m]
    h1: float          # bottom face thickness [m]
    h2: float          # core thickness [m]
    h3: float          # top face thickness [m] (= h1 for symmetric)

    # Face material (assumed isotropic)
    E1: float          # bottom face Young's modulus [Pa]
    E3: float          # top face Young's modulus [Pa]
    nu1: float         # bottom face Poisson's ratio
    nu3: float         # top face Poisson's ratio
    rho1: float        # bottom face density [kg/m³]
    rho3: float        # top face density [kg/m³]

    # Core (viscoelastic, complex modulus depends on omega — passed separately)
    rho2: float        # core density [kg/m³]
    nu2: float = 0.49  # core Poisson's ratio (rubber-like)
    # Core in-plane stiffness (separate from shear G, since real viscoelastic
    # cores have rubbery in-plane modulus that is independent of shear modulus).
    # Default to a small value (1 MPa) so monolithic-limit tests do not see
    # spurious in-plane core stiffness when G is swept to large values.
    # Set to None to use the legacy E_core = 2·G·(1+ν) coupling.
    E2_in_plane: float | None = 1.0e6  # core in-plane Young's modulus [Pa]

    # Mesh (uniform rectangular)
    nx: int = 8        # number of elements in x
    ny: int = 8        # number of elements in y

    # Asymmetric plate support
    # When False (default), enforces h1==h3 and E1==E3 for paper §5.1 case.
    # Set True to allow asymmetric face thicknesses/moduli (paper §5.3 case
    # or h3 perturbation studies). Element-level kinematics already support
    # asymmetric C / mass coefficients (laminate_*_coefficients use h1, h3
    # independently); this flag just relaxes the post-init guard.
    allow_asymmetric: bool = False

    def __post_init__(self):
        if not self.allow_asymmetric:
            if not np.isclose(self.h1, self.h3):
                raise ValueError(
                    f"Symmetric plate requires h1==h3, got {self.h1}, {self.h3}. "
                    f"Set allow_asymmetric=True to override."
                )
            if not np.isclose(self.E1, self.E3):
                raise ValueError(
                    f"Symmetric plate requires E1==E3, got {self.E1}, {self.E3}. "
                    f"Set allow_asymmetric=True to override."
                )
        if self.nx < 1 or self.ny < 1:
            raise ValueError(f"nx, ny >= 1, got nx={self.nx}, ny={self.ny}")

    # ---- Layer z-coordinates (reference plane = mid-plane of core) ---------
    @property
    def z1(self) -> float:
        """Bottom of bottom face."""
        return -self.h2 / 2.0 - self.h1

    @property
    def z2(self) -> float:
        """Top of bottom face = bottom of core."""
        return -self.h2 / 2.0

    @property
    def z3(self) -> float:
        """Top of core = bottom of top face."""
        return self.h2 / 2.0

    @property
    def z4(self) -> float:
        """Top of top face."""
        return self.h2 / 2.0 + self.h3

    @property
    def element_dx(self) -> float:
        return self.Lx / self.nx

    @property
    def element_dy(self) -> float:
        return self.Ly / self.ny

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def n_dof(self) -> int:
        return 7 * self.n_nodes


# =============================================================================
# Behavior matrices C^(i) and laminate coefficients
# =============================================================================

def plane_stress_C(E: float, nu: float) -> np.ndarray:
    """Plane-stress isotropic 3×3 behavior matrix.

    σ = C·ε where σ = (σ_xx, σ_yy, σ_xy)^T and ε = (ε_xx, ε_yy, γ_xy)^T.
    """
    factor = E / (1.0 - nu * nu)
    return factor * np.array([
        [1.0,    nu,       0.0],
        [nu,     1.0,      0.0],
        [0.0,    0.0,      0.5 * (1.0 - nu)],
    ])


def core_shear_C(G_complex: complex) -> np.ndarray:
    """Core transverse shear 2×2 matrix: τ = C·γ where τ=(τ_xz,τ_yz), γ=(γ_xz,γ_yz)."""
    return G_complex * np.eye(2)


def laminate_coefficient_matrices(
    params: SandwichPlateParameters,
    G_core_complex: complex,
    consistent_zigzag: bool = True,
) -> dict[str, np.ndarray]:
    """Compute laminate stiffness coefficient matrices (Eqs 10, 11 of paper).

    Returns dict with keys: 'C_m', 'C_f', 'C_r', 'C_mr', 'C_mf', 'C_fr', 'C_c'.

    The matrices weight membrane, bending, rotational, and shear contributions
    of each of the three layers into the laminate response.

    consistent_zigzag (default True = physically-consistent corrected element;
    set False for faithful paper Eq 1 / Eq 10/11 reproduction):
        The paper's element under-delivers the composite (core-shear-rigid)
        bending stiffness by ~10% because its zig-zag in-plane mode is tied to
        the rotation ψ (=∂W/∂x at γ=0) and so does NOT switch off when the core
        shear γ→0 — leaving each skin referenced to its core interface (z2/z3)
        rather than the global centroid. Concretely the bending-channel strain
        coefficient is a_f = -z while the rotation channel is a_r = +Φ (zig-zag),
        so at γ=0 (β_r=β_f) the net is -z+Φ ≠ -z.

        With consistent_zigzag=True the zig-zag is re-referenced to the *shear*
        γ via a_f = -(z+Φ): u = U_02 - z·∂W/∂x + Φ(z)·γ, Φ = (skin1:z2, core:z,
        skin3:z3). Then at γ=0 the Φ terms cancel → exact plane-section composite
        (∫E z² dz), while the soft-core decoupled limit C_f - C_fr·C_r⁻¹·C_fr is
        preserved. Only C_f, C_fr, C_mf change; C_r, C_mr, C_m, C_c are identical.
        Verified (patch test): D_eff(G→∞) 16.07→17.91 N·m (analytic 17.94),
        decoupled limit 4.02 preserved.
    """
    h1, h2, h3 = params.h1, params.h2, params.h3
    z1, z2, z3, z4 = params.z1, params.z2, params.z3, params.z4

    C1 = plane_stress_C(params.E1, params.nu1)
    C3 = plane_stress_C(params.E3, params.nu3)
    # Core in-plane stiffness: prefer explicit E2_in_plane parameter to avoid
    # spurious in-plane stiffness when G is swept to large values during
    # monolithic-limit tests. For real viscoelastic cores G and in-plane modulus
    # are independent material properties.
    if params.E2_in_plane is not None:
        E_core = float(params.E2_in_plane)
    else:
        # Legacy fallback: derive from G assuming elastic isotropic core
        G_real = float(G_core_complex.real) if isinstance(G_core_complex, complex) else float(G_core_complex)
        E_core = 2.0 * G_real * (1.0 + params.nu2)
    C2 = plane_stress_C(E_core, params.nu2)

    C_c = h2 * core_shear_C(G_core_complex)  # 2×2

    C_m = h1 * C1 + h2 * C2 + h3 * C3
    # @MX:NOTE: standard ∫z²·C dz form of paper Eq 10 (confirmed by a sympy variational derivation).
    # A previous form was swapped with C_fr of paper Eq 11, causing an indefinite K matrix.
    # Verified against the displacement field of paper Eq 1 (reference solution).
    C_f = (
        (z2**3 - z1**3) / 3.0 * C1
        + (z3**3 - z2**3) / 3.0 * C2
        + (z4**3 - z3**3) / 3.0 * C3
    )
    C_r = (
        z2**2 * h1 * C1
        + (z3**3 - z2**3) / 3.0 * C2
        + z3**2 * h3 * C3
    )
    C_mr = (
        z2 * h1 * C1
        + (z3**2 - z2**2) / 2.0 * C2
        + z3 * h3 * C3
    )
    C_mf = -(
        (z2**2 - z1**2) / 2.0 * C1
        + (z3**2 - z2**2) / 2.0 * C2
        + (z4**2 - z3**2) / 2.0 * C3
    )
    C_fr = -(
        z2 * (z2**2 - z1**2) / 2.0 * C1
        + (z3**3 - z2**3) / 3.0 * C2
        + z3 * (z4**2 - z3**2) / 2.0 * C3
    )

    if consistent_zigzag:
        # Re-reference the zig-zag to the core shear γ (a_f = -(z+Φ)) so it
        # cancels at γ=0 → exact plane-section composite. See docstring.
        # Φ = z2 (skin1), z (core), z3 (skin3). Only C_f, C_fr, C_mf change.
        C_f = (
            ((z2**3 - z1**3) / 3.0 + z2 * (z2**2 - z1**2) + z2**2 * h1) * C1
            + (4.0 * (z3**3 - z2**3) / 3.0) * C2
            + ((z4**3 - z3**3) / 3.0 + z3 * (z4**2 - z3**2) + z3**2 * h3) * C3
        )
        C_fr = -(
            (z2 * (z2**2 - z1**2) / 2.0 + z2**2 * h1) * C1
            + (2.0 * (z3**3 - z2**3) / 3.0) * C2
            + (z3 * (z4**2 - z3**2) / 2.0 + z3**2 * h3) * C3
        )
        C_mf = -(
            ((z2**2 - z1**2) / 2.0 + z2 * h1) * C1
            + (z3**2 - z2**2) * C2
            + ((z4**2 - z3**2) / 2.0 + z3 * h3) * C3
        )

    return {
        "C_m": C_m, "C_f": C_f, "C_r": C_r,
        "C_mr": C_mr, "C_mf": C_mf, "C_fr": C_fr,
        "C_c": C_c,
    }


# =============================================================================
# Lamellar density coefficients (for mass matrix, Eq 13)
# =============================================================================

def laminate_density_coefficients(params: SandwichPlateParameters) -> dict[str, float]:
    """Compute laminate inertia coefficients (Eq 13 of paper).

    Returns dict with keys: 'rho_m' (translation), 'rho_z' (z-translation),
    'rho_r' (rotation about y), 'rho_tz' (rot-trans coupling z),
    'rho_tr' (translation-rotation coupling), 'rho_rz' (rotation-z coupling).
    """
    h1, h2, h3 = params.h1, params.h2, params.h3
    z1, z2, z3, z4 = params.z1, params.z2, params.z3, params.z4
    rho1, rho2, rho3 = params.rho1, params.rho2, params.rho3

    rho_m = rho1 * h1 + rho2 * h2 + rho3 * h3
    rho_z = (z2**3 - z1**3) / 3.0 * rho1 + (z3**3 - z2**3) / 3.0 * rho2 + (z4**3 - z3**3) / 3.0 * rho3
    rho_r = z2**2 * h1 * rho1 + (z3**3 - z2**3) / 3.0 * rho2 + z3**2 * h3 * rho3
    # paper Eq 13: ρ_tz = (z2²−z1²)/2·ρ1 + (z3²−z2²)/2·ρ2 + (z4²−z3²)/2·ρ3
    # (leading sign is POSITIVE; a spurious minus here made the consistent mass
    #  matrix indefinite in asymmetric §5.3 — kinetic energy ½q̇ᵀMq̇≥0 requires M≻0.
    #  Verified: flipping to this sign clears all negative M eigenvalues and leaves
    #  the symmetric §5.1 case unchanged because ρ_tz cancels to 0 there.)
    rho_tz = (
        (z2**2 - z1**2) / 2.0 * rho1
        + (z3**2 - z2**2) / 2.0 * rho2
        + (z4**2 - z3**2) / 2.0 * rho3
    )
    rho_tr = z2 * h1 * rho1 + (z3**2 - z2**2) / 2.0 * rho2 + z3 * h3 * rho3
    rho_rz = (
        z2 * (z2**2 - z1**2) / 2.0 * rho1
        + (z3**3 - z2**3) / 3.0 * rho2
        + z3 * (z4**2 - z3**2) / 2.0 * rho3
    )
    return {
        "rho_m": rho_m, "rho_z": rho_z, "rho_r": rho_r,
        "rho_tz": rho_tz, "rho_tr": rho_tr, "rho_rz": rho_rz,
    }


# =============================================================================
# 4-node Q4 shape functions (Section 3.1 of paper)
# =============================================================================
#
# Element node numbering (paper Fig. 3, counter-clockwise from bottom-left):
#   node i = 1: (ξ, η) = (-1, -1)
#   node j = 2: (ξ, η) = (+1, -1)
#   node k = 3: (ξ, η) = (+1, +1)
#   node l = 4: (ξ, η) = (-1, +1)
#
# Element DOF ordering per node n:
#   q^n_e = (u_02n, v_02n, ψ_xn, ψ_yn, W_n, W_x_n, W_y_n)^T  (7 DOFs/node)
# Total q_e = 28 DOFs/element.
#
# Lagrange bilinear (for u_02, v_02, ψ_x, ψ_y):
#   N_un(ξ, η) = (1/4)(1 + ξ·ξ_n)(1 + η·η_n)
#
# Hermite cubic for W (paper Section 3.1; uses W, W_x, W_y at each node).
# Note: paper Eq 14 contains a sign typo for N_wx; the correct ACM form
# verified via a sympy symbolic derivation is:
#   N_wn       = (1/8)(1+ξ_0)(1+η_0)·[2 + ξ_0(1-ξ_0) + η_0(1-η_0)]
#   N_w_x_n    = -(1/8)·ξ_n·(1+ξ_0)²(1-ξ_0)·(1+η_0)·(dx/2)        ← negative leading sign
#   N_w_y_n    = -(1/8)·η_n·(1+ξ_0)·(1+η_0)²(1-η_0)·(dy/2)
#   where ξ_0 = ξ·ξ_n, η_0 = η·η_n.
#
# DOF convention: W_x = ∂W/∂x, W_y = ∂W/∂y in physical (global) coordinates.
# The (dx/2), (dy/2) scaling factors enforce ∂N_w_x_n/∂x = 1 at node n.
# With the corrected N_wx sign, all 12 polynomial fields up to xy^3 are
# represented exactly (Test 2 of verify_amichi_hermite.py).

NODES_QXI = np.array([-1.0, +1.0, +1.0, -1.0])  # ξ_n for nodes 1..4
NODES_QETA = np.array([-1.0, -1.0, +1.0, +1.0])  # η_n for nodes 1..4


def lagrange_bilinear(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (N_lin, dN/dξ, dN/dη) — each (4,) array, indexed by node 1..4."""
    xi_n = NODES_QXI
    eta_n = NODES_QETA
    N = 0.25 * (1 + xi * xi_n) * (1 + eta * eta_n)
    dN_dxi = 0.25 * xi_n * (1 + eta * eta_n)
    dN_deta = 0.25 * eta_n * (1 + xi * xi_n)
    return N, dN_dxi, dN_deta


def hermite_cubic_w(
    xi: float, eta: float, dx: float, dy: float,
) -> dict[str, np.ndarray]:
    """Return shape functions and their derivatives for W field.

    Parameters
    ----------
    xi, eta : float
        Master coordinates ∈ [-1, 1].
    dx, dy : float
        Element size in x, y (physical).

    Returns
    -------
    dict with keys:
        'N_w'       (3, 4) — rows: [N_w_n, N_wx_n, N_wy_n], cols: nodes 1..4
        'N_w_x'     (3, 4) — derivative ∂/∂x
        'N_w_y'     (3, 4) — derivative ∂/∂y
        'N_w_xx'    (3, 4) — derivative ∂²/∂x²
        'N_w_yy'    (3, 4) — derivative ∂²/∂y²
        'N_w_xy'    (3, 4) — derivative ∂²/∂x∂y
    """
    xi_n = NODES_QXI
    eta_n = NODES_QETA
    a = dx / 2.0   # element half-length in x
    b = dy / 2.0   # element half-length in y

    xi0 = xi * xi_n
    eta0 = eta * eta_n
    dxi0_dxi = xi_n
    deta0_deta = eta_n

    # --- Value -----------------------------------------------------------
    # N_w_n(ξ,η) = (1/8)(1+ξ₀)(1+η₀)[2 + ξ₀(1-ξ₀) + η₀(1-η₀)]
    bracket = 2.0 + xi0 * (1 - xi0) + eta0 * (1 - eta0)
    N_w = (1.0 / 8.0) * (1 + xi0) * (1 + eta0) * bracket
    # N_wx_n(ξ,η) = -(a/8)·ξ_n·(1+ξ₀)²(1-ξ₀)·(1+η₀)  ← negative (ACM form)
    N_wx = -(a / 8.0) * xi_n * (1 + xi0) ** 2 * (1 - xi0) * (1 + eta0)
    # N_wy_n(ξ,η) = -(b/8)·η_n·(1+ξ₀)·(1+η₀)²(1-η₀)
    N_wy = -(b / 8.0) * eta_n * (1 + xi0) * (1 + eta0) ** 2 * (1 - eta0)

    # --- First derivatives w.r.t. xi ------------------------------------
    # d(bracket)/dξ = ξ_n·(1-2ξ₀); d/dη = η_n·(1-2η₀)·(0)... wait, only ξ₀ part.
    # d/dξ [bracket] = (dxi0/dxi)·d/dξ₀[2 + ξ₀(1-ξ₀) + η₀(1-η₀)] = ξ_n·(1 - 2ξ₀)
    dbracket_dxi = xi_n * (1 - 2 * xi0)
    dbracket_deta = eta_n * (1 - 2 * eta0)

    # ∂N_w/∂ξ
    dNw_dxi = (1.0 / 8.0) * (
        xi_n * (1 + eta0) * bracket
        + (1 + xi0) * (1 + eta0) * dbracket_dxi
    )
    dNw_deta = (1.0 / 8.0) * (
        eta_n * (1 + xi0) * bracket
        + (1 + xi0) * (1 + eta0) * dbracket_deta
    )
    # ∂N_wx/∂ξ where N_wx = -(a/8)·ξ_n·(1+ξ₀)²(1-ξ₀)·(1+η₀)  ← negative leading
    # f(ξ₀) = (1+ξ₀)²(1-ξ₀); f'(ξ₀) = 2(1+ξ₀)(1-ξ₀) - (1+ξ₀)² = (1+ξ₀)(1-3ξ₀)
    fp_xi = (1 + xi0) * (1 - 3 * xi0)
    dNwx_dxi = -(a / 8.0) * xi_n * fp_xi * xi_n * (1 + eta0)
    dNwx_deta = -(a / 8.0) * xi_n * (1 + xi0) ** 2 * (1 - xi0) * eta_n
    # ∂N_wy/∂ξ
    fp_eta = (1 + eta0) * (1 - 3 * eta0)
    dNwy_dxi = -(b / 8.0) * eta_n * xi_n * (1 + eta0) ** 2 * (1 - eta0)
    dNwy_deta = -(b / 8.0) * eta_n * (1 + xi0) * fp_eta * eta_n

    # --- Second derivatives ---------------------------------------------
    d2bracket_dxi2 = -2 * xi_n ** 2  # = -2 (since ξ_n²=1)
    d2bracket_deta2 = -2 * eta_n ** 2  # = -2
    # ∂²N_w/∂ξ²
    d2Nw_dxi2 = (1.0 / 8.0) * (
        2 * xi_n * (1 + eta0) * dbracket_dxi
        + (1 + xi0) * (1 + eta0) * d2bracket_dxi2
    )
    d2Nw_deta2 = (1.0 / 8.0) * (
        2 * eta_n * (1 + xi0) * dbracket_deta
        + (1 + xi0) * (1 + eta0) * d2bracket_deta2
    )
    d2Nw_dxideta = (1.0 / 8.0) * (
        xi_n * eta_n * bracket
        + xi_n * (1 + eta0) * dbracket_deta
        + eta_n * (1 + xi0) * dbracket_dxi
    )
    # ∂²N_wx/∂ξ²  (signs flipped to match negative leading sign of N_wx)
    fpp_xi = (1 - 3 * xi0) - 3 * (1 + xi0)  # f''(ξ₀) = (1-3ξ₀)·1 + (1+ξ₀)·(-3) = 1-3ξ₀-3-3ξ₀ = -2-6ξ₀
    fpp_xi = -2 - 6 * xi0
    d2Nwx_dxi2 = -(a / 8.0) * xi_n * (xi_n ** 2) * fpp_xi * (1 + eta0)
    d2Nwx_deta2 = 0.0  # N_wx is linear in η
    d2Nwx_dxideta = -(a / 8.0) * xi_n * fp_xi * xi_n * eta_n
    # ∂²N_wy/∂ξ²
    fpp_eta = -2 - 6 * eta0
    d2Nwy_dxi2 = 0.0
    d2Nwy_deta2 = -(b / 8.0) * eta_n * (eta_n ** 2) * fpp_eta * (1 + xi0)
    d2Nwy_dxideta = -(b / 8.0) * eta_n * fp_eta * eta_n * xi_n

    # Stack: row 0 = N_w (value DOF), row 1 = N_wx (W,x DOF), row 2 = N_wy (W,y DOF)
    N_value = np.vstack([N_w, N_wx, N_wy])
    dN_dxi = np.vstack([dNw_dxi, dNwx_dxi, dNwy_dxi])
    dN_deta = np.vstack([dNw_deta, dNwx_deta, dNwy_deta])
    # Second derivatives — note d2*_dxi2 may be a scalar 0 array
    d2N_dxi2_w = np.array([d2Nw_dxi2, d2Nwx_dxi2 * np.ones_like(xi_n), d2Nwy_dxi2 * np.ones_like(xi_n)])
    d2N_dxi2 = np.vstack([d2Nw_dxi2,
                          d2Nwx_dxi2 * np.ones_like(xi_n) if np.isscalar(d2Nwx_dxi2) else d2Nwx_dxi2,
                          d2Nwy_dxi2 * np.ones_like(xi_n) if np.isscalar(d2Nwy_dxi2) else d2Nwy_dxi2])
    d2N_deta2 = np.vstack([d2Nw_deta2,
                           d2Nwx_deta2 * np.ones_like(xi_n) if np.isscalar(d2Nwx_deta2) else d2Nwx_deta2,
                           d2Nwy_deta2 * np.ones_like(xi_n) if np.isscalar(d2Nwy_deta2) else d2Nwy_deta2])
    d2N_dxideta = np.vstack([d2Nw_dxideta, d2Nwx_dxideta, d2Nwy_dxideta])

    # Convert to physical-coordinate derivatives via Jacobian (uniform rect mesh):
    #   ∂/∂x = (1/a)·∂/∂ξ
    #   ∂/∂y = (1/b)·∂/∂η
    #   ∂²/∂x² = (1/a²)·∂²/∂ξ²
    inv_a = 1.0 / a
    inv_b = 1.0 / b
    return {
        "N_w": N_value,
        "N_w_x": dN_dxi * inv_a,
        "N_w_y": dN_deta * inv_b,
        "N_w_xx": d2N_dxi2 * inv_a ** 2,
        "N_w_yy": d2N_deta2 * inv_b ** 2,
        "N_w_xy": d2N_dxideta * inv_a * inv_b,
    }


def evaluate_q4_interpolation(
    xi: float, eta: float, dx: float, dy: float,
) -> dict[str, np.ndarray]:
    """Build interpolation matrices for 4-node Q4 sandwich plate element at (ξ, η).

    Element DOFs per node n: (u_02n, v_02n, ψ_xn, ψ_yn, w_n, w_x_n, w_y_n).
    Total 28 DOFs.

    Returns
    -------
    dict with keys:
        'N'       (5, 28)  — interpolation: u = N·q_e, where u = [U_02, V_02, ψ_x, ψ_y, W]^T
        'N_x'     (5, 28)  — ∂N/∂x
        'N_y'     (5, 28)  — ∂N/∂y
        'N_xx'    (5, 28)  — ∂²N/∂x² (only W-row is non-zero in practice)
        'N_yy'    (5, 28)  — ∂²N/∂y²
        'N_xy'    (5, 28)  — ∂²N/∂x∂y
    """
    a = dx / 2.0
    b = dy / 2.0

    # Lagrange bilinear (4,)
    N_lin, dN_lin_dxi, dN_lin_deta = lagrange_bilinear(xi, eta)
    # Map to physical derivative
    dN_lin_dx = dN_lin_dxi / a
    dN_lin_dy = dN_lin_deta / b

    # Hermite cubic (3, 4)
    H = hermite_cubic_w(xi, eta, dx, dy)

    # Build 5×28 interpolation matrix
    N = np.zeros((5, 28))
    N_x = np.zeros((5, 28))
    N_y = np.zeros((5, 28))
    N_xx = np.zeros((5, 28))
    N_yy = np.zeros((5, 28))
    N_xy = np.zeros((5, 28))

    for n in range(4):  # node index 0..3 for n=1..4
        col0 = 7 * n  # starting column for node n
        # Lagrange-interpolated rows: 0=U_02, 1=V_02, 2=ψ_x, 3=ψ_y
        N[0, col0 + 0] = N_lin[n]
        N[1, col0 + 1] = N_lin[n]
        N[2, col0 + 2] = N_lin[n]
        N[3, col0 + 3] = N_lin[n]
        N_x[0, col0 + 0] = dN_lin_dx[n]
        N_x[1, col0 + 1] = dN_lin_dx[n]
        N_x[2, col0 + 2] = dN_lin_dx[n]
        N_x[3, col0 + 3] = dN_lin_dx[n]
        N_y[0, col0 + 0] = dN_lin_dy[n]
        N_y[1, col0 + 1] = dN_lin_dy[n]
        N_y[2, col0 + 2] = dN_lin_dy[n]
        N_y[3, col0 + 3] = dN_lin_dy[n]

        # Hermite-interpolated row: 4 = W
        # Per-node DOFs: w (col0+4), w_x (col0+5), w_y (col0+6)
        N[4, col0 + 4] = H["N_w"][0, n]
        N[4, col0 + 5] = H["N_w"][1, n]
        N[4, col0 + 6] = H["N_w"][2, n]
        N_x[4, col0 + 4] = H["N_w_x"][0, n]
        N_x[4, col0 + 5] = H["N_w_x"][1, n]
        N_x[4, col0 + 6] = H["N_w_x"][2, n]
        N_y[4, col0 + 4] = H["N_w_y"][0, n]
        N_y[4, col0 + 5] = H["N_w_y"][1, n]
        N_y[4, col0 + 6] = H["N_w_y"][2, n]
        N_xx[4, col0 + 4] = H["N_w_xx"][0, n]
        N_xx[4, col0 + 5] = H["N_w_xx"][1, n]
        N_xx[4, col0 + 6] = H["N_w_xx"][2, n]
        N_yy[4, col0 + 4] = H["N_w_yy"][0, n]
        N_yy[4, col0 + 5] = H["N_w_yy"][1, n]
        N_yy[4, col0 + 6] = H["N_w_yy"][2, n]
        N_xy[4, col0 + 4] = H["N_w_xy"][0, n]
        N_xy[4, col0 + 5] = H["N_w_xy"][1, n]
        N_xy[4, col0 + 6] = H["N_w_xy"][2, n]

    return {"N": N, "N_x": N_x, "N_y": N_y,
            "N_xx": N_xx, "N_yy": N_yy, "N_xy": N_xy}


# =============================================================================
# Strain operators — β_m, β_f, β_r, β_c (functions of N derivatives)
# =============================================================================

def strain_operators(interp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build strain operator matrices from interpolation dict.

    paper Eqs 4-7:
        ε_m = D_um·u  (membrane, 3-component)
        ψ   = D_ur·u  (rotational, 3-component)
        χ   = D_uf·u  (bending, 3-component)
        γ   = D_uc·u  (core shear, 2-component) — corrected:
              γ_x = ψ_x − ∂W/∂x;  γ_y = ψ_y − ∂W/∂y

    Returns
    -------
    dict with keys 'beta_m' (3,28), 'beta_f' (3,28), 'beta_r' (3,28),
                   'beta_c' (2,28).
    """
    # u = [U_02, V_02, ψ_x, ψ_y, W]^T  (5 components)
    # row index: 0=U_02, 1=V_02, 2=ψ_x, 3=ψ_y, 4=W
    N = interp["N"]
    N_x = interp["N_x"]
    N_y = interp["N_y"]
    N_xx = interp["N_xx"]
    N_yy = interp["N_yy"]
    N_xy = interp["N_xy"]

    # β_m: membrane = (∂U_02/∂x, ∂V_02/∂y, ∂U_02/∂y + ∂V_02/∂x)
    beta_m = np.zeros((3, 28))
    beta_m[0, :] = N_x[0, :]  # ∂U_02/∂x
    beta_m[1, :] = N_y[1, :]  # ∂V_02/∂y
    beta_m[2, :] = N_y[0, :] + N_x[1, :]  # ∂U_02/∂y + ∂V_02/∂x

    # β_r: rotational = (∂ψ_x/∂x, ∂ψ_y/∂y, ∂ψ_x/∂y + ∂ψ_y/∂x)
    beta_r = np.zeros((3, 28))
    beta_r[0, :] = N_x[2, :]  # ∂ψ_x/∂x
    beta_r[1, :] = N_y[3, :]  # ∂ψ_y/∂y
    beta_r[2, :] = N_y[2, :] + N_x[3, :]  # ∂ψ_x/∂y + ∂ψ_y/∂x

    # β_f: bending = (∂²W/∂x², ∂²W/∂y², 2·∂²W/∂x∂y)
    beta_f = np.zeros((3, 28))
    beta_f[0, :] = N_xx[4, :]
    beta_f[1, :] = N_yy[4, :]
    beta_f[2, :] = 2.0 * N_xy[4, :]

    # β_c: core shear = (ψ_x - ∂W/∂x, ψ_y - ∂W/∂y)
    beta_c = np.zeros((2, 28))
    beta_c[0, :] = N[2, :] - N_x[4, :]  # ψ_x - ∂W/∂x
    beta_c[1, :] = N[3, :] - N_y[4, :]  # ψ_y - ∂W/∂y

    return {"beta_m": beta_m, "beta_f": beta_f, "beta_r": beta_r, "beta_c": beta_c}


# =============================================================================
# Element matrices via 2D Gauss-Legendre quadrature
# =============================================================================

def compute_element_matrices(
    params: SandwichPlateParameters,
    G_core_complex: complex,
    n_gauss: int = 3,
    n_gauss_shear: int | None = None,
    include_mass_cross_coupling: bool = True,
    consistent_zigzag: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble element K_e (28×28) and M_e (28×28) for one Q4 element.

    K_e includes membrane, bending, rotational, core shear, and all
    cross-coupling terms (paper Eq 15).

    Selective reduced integration is used for the core shear term β_c^T C_c β_c:
    full 3×3 Gauss for membrane/bending/rotational/coupling (default n_gauss=3),
    and reduced 1×1 (or n_gauss_shear) Gauss for the shear term to avoid
    Mindlin-Reissner shear locking caused by polynomial-space mismatch
    between ψ (Q4 bilinear) and ∂W/∂x (Hermite cubic derivative).

    M_e is the FULL consistent mass matrix (paper Eq 16): translation (ρ_m),
    rotational inertia (ρ_r), z-gradient inertia (ρ_z), and — when
    include_mass_cross_coupling=True (default) — the translation–rotation
    (ρ_tr), translation–∂W (ρ_tz) and rotation–∂W (ρ_rz) coupling terms.
    """
    if n_gauss_shear is None:
        # Default: 2×2 reduced integration for shear (selective integration).
        # 1×1 introduces spurious hourglass modes (verified empirically); 3×3
        # over-constrains the polynomial mismatch between Q4 ψ and cubic Hermite
        # ∂W/∂x → shear locking. 2×2 is the practical compromise.
        n_gauss_shear = 2
    dx = params.element_dx
    dy = params.element_dy
    a = dx / 2.0
    b = dy / 2.0
    jacobian_det = a * b  # uniform rectangular mesh

    coeffs = laminate_coefficient_matrices(
        params, G_core_complex, consistent_zigzag=consistent_zigzag
    )
    densities = laminate_density_coefficients(params)
    # @MX:NOTE: uses all 6 inertia coefficients of paper Eq 13 (paper Eq 16 full mass)
    rho_m = densities["rho_m"]
    rho_r = densities["rho_r"]
    rho_z = densities["rho_z"]
    rho_tr = densities["rho_tr"]
    rho_tz = densities["rho_tz"]
    rho_rz = densities["rho_rz"]

    C_m = coeffs["C_m"]
    C_f = coeffs["C_f"]
    C_r = coeffs["C_r"]
    C_mf = coeffs["C_mf"]
    C_mr = coeffs["C_mr"]
    C_fr = coeffs["C_fr"]
    C_c = coeffs["C_c"]

    K_e = np.zeros((28, 28), dtype=complex)
    M_e = np.zeros((28, 28), dtype=complex)

    # --- Pass 1: full integration for membrane/bending/rotation/coupling + mass ---
    gp, gw = np.polynomial.legendre.leggauss(n_gauss)
    for i, xi in enumerate(gp):
        for j, eta in enumerate(gp):
            w_total = gw[i] * gw[j]

            interp = evaluate_q4_interpolation(xi, eta, dx, dy)
            beta = strain_operators(interp)
            beta_m = beta["beta_m"]
            beta_f = beta["beta_f"]
            beta_r = beta["beta_r"]

            # Non-shear stiffness terms (full Eq 15 minus shear)
            K_local = (
                beta_m.T @ C_m @ beta_m
                + beta_f.T @ C_f @ beta_f
                + beta_r.T @ C_r @ beta_r
                + beta_m.T @ C_mf @ beta_f
                + beta_f.T @ C_mf.T @ beta_m
                + beta_m.T @ C_mr @ beta_r
                + beta_r.T @ C_mr.T @ beta_m
                + beta_f.T @ C_fr @ beta_r
                + beta_r.T @ C_fr.T @ beta_f
            )
            K_e += w_total * jacobian_det * K_local

            # @MX:NOTE: paper Eq 16 full mass matrix (all 6 coefficients of paper
            # Eq 13 applied). Previously: ρ_m + ρ_r only. Added: ρ_z, ρ_tr, ρ_tz, ρ_rz coupling.
            # paper notation mapping:
            #   N_x  = N[0,:] (U_02), N_y  = N[1,:] (V_02)
            #   N_rx = N[2,:] (ψ_x),  N_ry = N[3,:] (ψ_y)
            #   N_zx = N_x[4,:] (∂W/∂x), N_zy = N_y[4,:] (∂W/∂y)
            #   N_zz = N[4,:] (W)
            N_U   = interp["N"][0:1, :]
            N_V   = interp["N"][1:2, :]
            N_psix = interp["N"][2:3, :]
            N_psiy = interp["N"][3:4, :]
            N_W   = interp["N"][4:5, :]
            N_Wx  = interp["N_x"][4:5, :]   # ∂W/∂x interpolation row
            N_Wy  = interp["N_y"][4:5, :]   # ∂W/∂y interpolation row

            # Diagonal mass terms (always included, paper Eq 16 first 3 lines)
            M_local = (
                rho_m  * (N_U.T @ N_U + N_V.T @ N_V + N_W.T @ N_W)
                + rho_r  * (N_psix.T @ N_psix + N_psiy.T @ N_psiy)
                + rho_z  * (N_Wx.T @ N_Wx + N_Wy.T @ N_Wy)
            )
            # Cross-coupling mass terms (paper Eq 16 last 3 lines, ablation flag)
            # In symmetric sandwich (h1==h3, ρ1==ρ3), ρ_tr and ρ_tz are exactly
            # zero by cancellation, so the flag has no effect
            # on paper §5.1 case. The flag matters in asymmetric cases.
            if include_mass_cross_coupling:
                M_local = M_local + (
                    + rho_tr * (N_U.T @ N_psix + N_psix.T @ N_U
                                + N_V.T @ N_psiy + N_psiy.T @ N_V)
                    + rho_tz * (N_U.T @ N_Wx + N_Wx.T @ N_U
                                + N_V.T @ N_Wy + N_Wy.T @ N_V)
                    + rho_rz * (N_Wx.T @ N_psix + N_psix.T @ N_Wx
                                + N_Wy.T @ N_psiy + N_psiy.T @ N_Wy)
                )
            M_e += w_total * jacobian_det * M_local

    # --- Pass 2: reduced integration for core shear term β_c^T C_c β_c ---
    gp_s, gw_s = np.polynomial.legendre.leggauss(n_gauss_shear)
    for i, xi in enumerate(gp_s):
        for j, eta in enumerate(gp_s):
            w_total = gw_s[i] * gw_s[j]
            interp = evaluate_q4_interpolation(xi, eta, dx, dy)
            beta = strain_operators(interp)
            beta_c = beta["beta_c"]
            K_e += w_total * jacobian_det * (beta_c.T @ C_c @ beta_c)

    return M_e, K_e


# =============================================================================
# Global assembly + boundary conditions
# =============================================================================

def assemble_global(
    params: SandwichPlateParameters,
    G_core_complex: complex,
    n_gauss: int = 3,
    include_mass_cross_coupling: bool = True,
    consistent_zigzag: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble global complex M*, K* for uniform rectangular mesh.

    Node numbering: (i, j) → linear index = i + j*(nx+1), i ∈ 0..nx, j ∈ 0..ny.
    DOFs per node: 7. Total = 7·(nx+1)·(ny+1).

    Set include_mass_cross_coupling=False for ablation studies of paper Eq 16
    cross-coupling terms (ρ_tr, ρ_tz, ρ_rz). Default True preserves paper Eq 16
    full formulation.
    """
    nx, ny = params.nx, params.ny
    n_dof = params.n_dof
    M = np.zeros((n_dof, n_dof), dtype=complex)
    K = np.zeros((n_dof, n_dof), dtype=complex)

    # Element matrices are identical for uniform mesh — compute once
    M_e, K_e = compute_element_matrices(
        params, G_core_complex, n_gauss,
        include_mass_cross_coupling=include_mass_cross_coupling,
        consistent_zigzag=consistent_zigzag,
    )

    for ey in range(ny):
        for ex in range(nx):
            # Local node indices (counter-clockwise: i, j, k, l)
            n1 = ex + ey * (nx + 1)            # bottom-left
            n2 = (ex + 1) + ey * (nx + 1)      # bottom-right
            n3 = (ex + 1) + (ey + 1) * (nx + 1)  # top-right
            n4 = ex + (ey + 1) * (nx + 1)      # top-left
            nodes = [n1, n2, n3, n4]
            dof_indices = []
            for n in nodes:
                dof_indices.extend(range(7 * n, 7 * n + 7))
            dof_indices = np.array(dof_indices)
            M[np.ix_(dof_indices, dof_indices)] += M_e
            K[np.ix_(dof_indices, dof_indices)] += K_e

    return M, K


def assemble_K_base_and_shear(
    params: SandwichPlateParameters,
    n_gauss: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear decomposition of K(G) with respect to G: K(G) = K_base + G · K_shear.

    In the element stiffness of paper Eq 15, the only term that depends on G is
    β_c^T·C_c·β_c, with C_c = h2·G·I (core shear matrix). Therefore:
        K_e(G) = K_e_base + G · K_e_shear
    K_base : independent of G (membrane + bending + rotation + cross-coupling)
    K_shear : core shear stiffness operator (element integral with G factored out)

    This affine split is the key ingredient of the ROM construction — passing
    K_shear in the L slot and K_base in the K_base slot of
    `assemble_complex_matrices` directly rebuilds the ω-independent
    complex matrices.

    Returns
    -------
    K_base, K_shear : (n_dof, n_dof) ndarray (real)
        Neither depends on the unit of G (the unit of K_shear is of the form
        [m] · [m] · [/] = N/(Pa·m), so multiplying by G [Pa] yields K [N/m]).

    Note
    ----
    - K_base includes all cross-coupling terms (membrane-flex etc.)
    - K_shear uses element-wise reduced integration (n_gauss_shear=2),
      matching the default of `assemble_global`
    - The result of `assemble_global(G)` must match `K_base + G_real · K_shear`
      (for real G input); the check function `verify_K_decomposition` is recommended
    """
    # K_e(G=0) — assemble the element with only the shear pass excluded
    K_e_base = _compute_element_K_no_shear(params, n_gauss)
    # K_e_shear — shear pass only, normalized with G=1 to factor out the G unit
    K_e_shear = _compute_element_K_shear_only(params)

    nx, ny = params.nx, params.ny
    n_dof = params.n_dof
    K_base = np.zeros((n_dof, n_dof), dtype=float)
    K_shear = np.zeros((n_dof, n_dof), dtype=float)

    for ey in range(ny):
        for ex in range(nx):
            n1 = ex + ey * (nx + 1)
            n2 = (ex + 1) + ey * (nx + 1)
            n3 = (ex + 1) + (ey + 1) * (nx + 1)
            n4 = ex + (ey + 1) * (nx + 1)
            dof_indices = []
            for n in [n1, n2, n3, n4]:
                dof_indices.extend(range(7 * n, 7 * n + 7))
            dof_indices = np.array(dof_indices)
            K_base[np.ix_(dof_indices, dof_indices)] += K_e_base
            K_shear[np.ix_(dof_indices, dof_indices)] += K_e_shear
    return K_base, K_shear


def _compute_element_K_no_shear(
    params: SandwichPlateParameters,
    n_gauss: int = 3,
) -> np.ndarray:
    """Element K_e without the shear term (β_c^T·C_c·β_c) — independent of G."""
    dx = params.element_dx
    dy = params.element_dy
    a = dx / 2.0
    b = dy / 2.0
    jacobian_det = a * b

    # call laminate_coefficient_matrices with G=0 — only C_c is affected; the rest is independent of G
    coeffs = laminate_coefficient_matrices(params, complex(0.0, 0.0))
    C_m = coeffs["C_m"]
    C_f = coeffs["C_f"]
    C_r = coeffs["C_r"]
    C_mf = coeffs["C_mf"]
    C_mr = coeffs["C_mr"]
    C_fr = coeffs["C_fr"]

    K_e = np.zeros((28, 28), dtype=float)
    gp, gw = np.polynomial.legendre.leggauss(n_gauss)
    for i, xi in enumerate(gp):
        for j, eta in enumerate(gp):
            w_total = gw[i] * gw[j]
            interp = evaluate_q4_interpolation(xi, eta, dx, dy)
            beta = strain_operators(interp)
            beta_m = beta["beta_m"]
            beta_f = beta["beta_f"]
            beta_r = beta["beta_r"]

            K_local = (
                beta_m.T @ C_m @ beta_m
                + beta_f.T @ C_f @ beta_f
                + beta_r.T @ C_r @ beta_r
                + beta_m.T @ C_mf @ beta_f
                + beta_f.T @ C_mf.T @ beta_m
                + beta_m.T @ C_mr @ beta_r
                + beta_r.T @ C_mr.T @ beta_m
                + beta_f.T @ C_fr @ beta_r
                + beta_r.T @ C_fr.T @ beta_f
            )
            K_e += w_total * jacobian_det * K_local.real
    return K_e


def _compute_element_K_shear_only(
    params: SandwichPlateParameters,
    n_gauss_shear: int = 2,
) -> np.ndarray:
    """Shear term of element K_e only (normalized with G=1). C_c = h2·I (G factor split off)."""
    dx = params.element_dx
    dy = params.element_dy
    a = dx / 2.0
    b = dy / 2.0
    jacobian_det = a * b
    h2 = params.h2

    # G=1 normalization → C_c = h2·I
    C_c_normalized = h2 * np.eye(2)

    K_e = np.zeros((28, 28), dtype=float)
    gp_s, gw_s = np.polynomial.legendre.leggauss(n_gauss_shear)
    for i, xi in enumerate(gp_s):
        for j, eta in enumerate(gp_s):
            w_total = gw_s[i] * gw_s[j]
            interp = evaluate_q4_interpolation(xi, eta, dx, dy)
            beta = strain_operators(interp)
            beta_c = beta["beta_c"]
            K_e += w_total * jacobian_det * (beta_c.T @ C_c_normalized @ beta_c)
    return K_e


def apply_boundary_conditions(
    M: np.ndarray,
    K: np.ndarray,
    F: np.ndarray,
    params: SandwichPlateParameters,
    bc: str = "simply_supported_w",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply boundary conditions for plate.

    bc options:
        'simply_supported_w' — w=0 on all 4 edges (only W DOF constrained on
                               boundary; other DOFs free)
        'clamped'            — all 7 DOFs = 0 on all 4 edges
        'free'               — no BC

    Returns (M_red, K_red, F_red, free_dofs).
    """
    nx, ny = params.nx, params.ny
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1

    fixed_dofs = set()

    if bc == "simply_supported_w":
        # w (DOF index 4 per node) = 0 on all 4 edges
        for j in range(n_nodes_y):
            for i in range(n_nodes_x):
                if i == 0 or i == nx or j == 0 or j == ny:
                    node_id = i + j * n_nodes_x
                    fixed_dofs.add(7 * node_id + 4)  # W
                    # For SS, also constrain in-plane on all edges
                    fixed_dofs.add(7 * node_id + 0)  # U
                    fixed_dofs.add(7 * node_id + 1)  # V
    elif bc == "clamped":
        for j in range(n_nodes_y):
            for i in range(n_nodes_x):
                if i == 0 or i == nx or j == 0 or j == ny:
                    node_id = i + j * n_nodes_x
                    for off in range(7):
                        fixed_dofs.add(7 * node_id + off)
    elif bc == "free":
        pass
    else:
        raise ValueError(f"Unknown BC: {bc}")

    n_dof = M.shape[0]
    free_dofs = np.array([i for i in range(n_dof) if i not in fixed_dofs])
    M_red = M[np.ix_(free_dofs, free_dofs)]
    K_red = K[np.ix_(free_dofs, free_dofs)]
    F_red = F[free_dofs]
    return M_red, K_red, F_red, free_dofs
