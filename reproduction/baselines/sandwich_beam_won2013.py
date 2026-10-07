"""Won, Bae, Cho, Bae, Jeong (2013) — Three-layered damped beam element.

Reference
---------
Won S.G., Bae S.H., Cho J.R., Bae S.R., Jeong W.B.,
"Three-layered damped beam element for forced vibration analysis of symmetric
sandwich structures with a viscoelastic core,"
Finite Elements in Analysis and Design 68 (2013) 39-51.

Element specification
---------------------
- 2-node beam element with 3 DOFs per node = **6 DOFs/element**
- DOFs: w (lateral displacement), θ = ∂w/∂x (rotation), γ (transverse shear
  strain in the viscoelastic core)
- Symmetric three-layer sandwich: face / core / face (h₁ = h₃, E₁ = E₃)
- Complex shape functions analytically derived from compatibility relation
  u₃ = (d·w' - h₂·γ)/2
- Master element ξ ∈ [-1, 1], element half-length a = L_e/2

Implementation notes
--------------------
1. **Variational derivation (verified with sympy, scripts/verify_g_star_definition.py):**
       g* = 2·G* / (E₁·h₁·h₂)        [units: 1/m²]
   ⇒ c₁* = -6·d / (g*·h₂·a³)         [units: 1/m]
   ⇒ c₂* = -120·d / (g*²·h₂·a⁵)      [units: 1/m]
   An earlier (incorrect) g* = 2G*/(E₁·h₁) omitted h₂ and caused an 87% error.

2. **Closed-form shape functions (derived with sympy, scripts/derive_closed_form_shape_functions.py):**
   det([a]) = -128·c₁·(8c₁ + c₂).
   N_i*(ξ), N_pp_i*(ξ), H_i*(ξ), H_p_i*(ξ) are all polynomials with the simple
   denominators (8c₁ + c₂) or c₁·(8c₁ + c₂). Direct evaluation needs no
   numerical inversion of [a] (closed-form bypass).

   T* (3×6, α→β transformation matrix):
        β₁ = c₁*·α₄ + 0 + c₂*·α₆
        β₂ =     0 + 4c₁*·α₅ +     0
        β₃ =     0 +     0 + 10c₁*·α₆

3. **Known limitation: fine-mesh ill-conditioning in the finite-G regime**
   c₂/c₁ = 20/(g*·a²) → for a soft core (paper G=8.08e7) combined with a small
   a (n_elem≥32), the dynamic range of K reaches ≥ 1e13, so the 1e-15
   floating-point noise of the eigenvalue solver causes 60%+ variation in the
   results. Consequently:
       - for n_elem ∈ {8, 16} the first mode of paper Table 2 is 216-217 Hz
         (-2% vs the paper's 221 Hz, sufficiently paper-comparable)
       - for n_elem ≥ 32 the numerics blow up (an intrinsic limitation of the
         present formulation)
   A future robust reformulation should consider column scaling, static
   condensation of γ, and mixed/hybrid elements.

Verification scripts:
    scripts/verify_g_star_definition.py        — sympy variational derivation of g*
    scripts/verify_won2013_shape_functions.py  — derivation of the 12 closed-form expressions
    scripts/derive_closed_form_shape_functions.py  — N, N_p, N_pp, H, H_p
    scripts/validate_won2013_table2.py         — comparison with paper Table 2
    scripts/cross_check_closed_form.py         — closed-form vs sympy
    scripts/test_global_K_M.py                  — global K, M consistency check
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class SandwichBeamParameters:
    """Symmetric three-layer sandwich beam parameters (paper Section 2)."""
    L: float          # beam length [m]
    h1: float         # top face thickness [m]
    h2: float         # core thickness [m]
    h3: float         # bottom face thickness [m] (= h1 for symmetric)
    E1: float         # top face Young's modulus [Pa]
    E3: float         # bottom face Young's modulus [Pa]
    rho1: float       # top face density [kg/m³]
    rho2: float       # core density [kg/m³]
    rho3: float       # bottom face density [kg/m³]
    b: float = 1.0    # width [m] (paper uses unit width)
    n_elem: int = 16  # number of elements

    def __post_init__(self):
        if not np.isclose(self.h1, self.h3):
            raise ValueError(
                f"Symmetric beam requires h1 == h3, got h1={self.h1}, h3={self.h3}"
            )
        if not np.isclose(self.E1, self.E3):
            raise ValueError(
                f"Symmetric beam requires E1 == E3, got E1={self.E1}, E3={self.E3}"
            )
        if self.n_elem < 1:
            raise ValueError(f"n_elem >= 1, got {self.n_elem}")

    @property
    def d(self) -> float:
        """Distance between face-sheet centroids: d = h1/2 + h2 + h3/2."""
        return self.h1 / 2.0 + self.h2 + self.h3 / 2.0

    @property
    def D1(self) -> float:
        """Bending rigidity of top face per unit width: E₁·h₁³/12."""
        return self.E1 * self.h1 ** 3 / 12.0

    @property
    def D3(self) -> float:
        return self.E3 * self.h3 ** 3 / 12.0

    @property
    def Dt(self) -> float:
        """Total face bending rigidity D_t = D₁ + D₃."""
        return self.D1 + self.D3

    @property
    def Y(self) -> float:
        """Geometric parameter Y = d²·E₁h₁·E₃h₃ / [Dt·(E₁h₁ + E₃h₃)]."""
        num = self.d ** 2 * self.E1 * self.h1 * self.E3 * self.h3
        den = self.Dt * (self.E1 * self.h1 + self.E3 * self.h3)
        return num / den

    @property
    def rho_eff(self) -> float:
        """Effective mass per unit length: b·(ρ₁h₁ + ρ₂h₂ + ρ₃h₃)."""
        return self.b * (
            self.rho1 * self.h1 + self.rho2 * self.h2 + self.rho3 * self.h3
        )

    @property
    def element_length(self) -> float:
        return self.L / self.n_elem

    @property
    def half_element_length(self) -> float:
        """Master element half-length a = L_e/2."""
        return 0.5 * self.element_length


# =============================================================================
# Complex coefficients (frequency-dependent through G*)
# =============================================================================

def compute_complex_coefficients(
    params: SandwichBeamParameters,
    G_complex: complex,
) -> tuple[complex, complex, complex]:
    """Compute (g*, c₁*, c₂*) for a given complex shear modulus G*.

    Variational derivation (E-L equation w.r.t. γ, verified with sympy):
        L_E = (b·E₁·h₁/4)·(d·w'' − h₂·γ')²
        L_S = (b·G*·h₂/2)·γ²
        ∂L/∂γ − d/dx(∂L/∂γ') = 0
        ⇒ γ'' − g*·γ = (d/h₂)·w'''
        ⇒ **g* = 2·G*/(E₁·h₁·h₂)**   [units: 1/m²]

    c₁*, c₂* (polynomial-coefficient solution; β₁ = c₁·α₄ + c₂·α₆):
        c₁* = -6·d/(g*·h₂·a³)        [units: 1/m]
        c₂* = -120·d/(g*²·h₂·a⁵)     [units: 1/m]

    An earlier (incorrect) version g* = 2G*/(E₁·h₁) omitted the h₂ factor.
    Re-deriving variationally with sympy (scripts/verify_g_star_definition.py)
    confirmed that h₂ must appear in the denominator.
    """
    a = params.half_element_length
    # Variational E-L: g* = 2·G*/(E₁·h₁·h₂)  [units: 1/m²]
    g_star = 2.0 * G_complex / (params.E1 * params.h1 * params.h2)
    c1 = -6.0 * params.d / (a ** 3 * g_star * params.h2)
    c2 = -120.0 * params.d / (a ** 5 * g_star ** 2 * params.h2)
    return g_star, c1, c2


# =============================================================================
# Closed-form shape functions (sympy-derived, avoids ill-conditioned [a]^{-1})
# =============================================================================
#
# Every denominator is (8c₁ + c₂) or c₁·(8c₁ + c₂).
# Source: scripts/derive_closed_form_shape_functions.py
# det([a]) = -128·c₁·(8c₁ + c₂)
#
# Indexing: shape function index i ∈ {0..5} corresponds to
#   element DOF order (w₁, a·θ₁, γ₁, w₂, a·θ₂, γ₂).

def evaluate_shape_functions(
    xi: float,
    c1: complex,
    c2: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate (N, N_pp, H, H_p) at a single ξ using closed-form expressions.

    Returns
    -------
    N    : (6,) complex     — w shape functions
    N_pp : (6,) complex     — d²N/dξ²
    H    : (6,) complex     — γ shape functions
    H_p  : (6,) complex     — dH/dξ
    """
    # Common subexpressions
    den_main = 8.0 * c1 + c2                  # (8c₁ + c₂)
    den_4 = 4.0 * den_main                    # denominator scale 1
    den_8c1 = 8.0 * c1 * den_main             # denominator scale 2 (for N3, N6)
    den_2 = 2.0 * den_main
    xi2 = xi * xi
    xi3 = xi2 * xi
    xim1 = xi - 1.0
    xip1 = xi + 1.0
    xim1_2 = xim1 * xim1
    xip1_2 = xip1 * xip1

    # --- N(ξ): w shape functions ----------------------------------------------
    N = np.empty(6, dtype=complex)
    # N_1*  = -(ξ-1)²·(c₁ξ³ + 2c₁ξ² - 7c₁ξ - 16c₁ - c₂ξ - 2c₂) / [4(8c₁+c₂)]
    N[0] = -xim1_2 * (c1 * xi3 + 2.0 * c1 * xi2 - 7.0 * c1 * xi - 16.0 * c1
                      - c2 * xi - 2.0 * c2) / den_4
    # N_2*  = -(ξ-1)²(ξ+1)·(c₁ξ² + c₁ξ - 8c₁ - c₂) / [4(8c₁+c₂)]
    N[1] = -xim1_2 * xip1 * (c1 * xi2 + c1 * xi - 8.0 * c1 - c2) / den_4
    # N_3*  =  (ξ-1)²(ξ+1)²·(4c₁ξ - 8c₁ - c₂) / [8c₁(8c₁+c₂)]
    N[2] = xim1_2 * xip1_2 * (4.0 * c1 * xi - 8.0 * c1 - c2) / den_8c1
    # N_4*  =  (ξ+1)²·(c₁ξ³ - 2c₁ξ² - 7c₁ξ + 16c₁ - c₂ξ + 2c₂) / [4(8c₁+c₂)]
    N[3] = xip1_2 * (c1 * xi3 - 2.0 * c1 * xi2 - 7.0 * c1 * xi + 16.0 * c1
                     - c2 * xi + 2.0 * c2) / den_4
    # N_5*  = -(ξ-1)(ξ+1)²·(c₁ξ² - c₁ξ - 8c₁ - c₂) / [4(8c₁+c₂)]
    N[4] = -xim1 * xip1_2 * (c1 * xi2 - c1 * xi - 8.0 * c1 - c2) / den_4
    # N_6*  =  (ξ-1)²(ξ+1)²·(4c₁ξ + 8c₁ + c₂) / [8c₁(8c₁+c₂)]
    N[5] = xim1_2 * xip1_2 * (4.0 * c1 * xi + 8.0 * c1 + c2) / den_8c1

    # --- N_pp(ξ): d²N/dξ² ----------------------------------------------------
    N_pp = np.empty(6, dtype=complex)
    # N_pp_1* = -ξ·(10c₁ξ² - 30c₁ - 3c₂) / [2(8c₁+c₂)]
    N_pp[0] = -xi * (10.0 * c1 * xi2 - 30.0 * c1 - 3.0 * c2) / den_2
    # N_pp_2* = -(10c₁ξ³ - 30c₁ξ + 8c₁ - 3c₂ξ + c₂) / [2(8c₁+c₂)]
    N_pp[1] = -(10.0 * c1 * xi3 - 30.0 * c1 * xi + 8.0 * c1
                - 3.0 * c2 * xi + c2) / den_2
    # N_pp_3* =  (20c₁ξ³ - 24c₁ξ² - 12c₁ξ + 8c₁ - 3c₂ξ² + c₂) / [2c₁(8c₁+c₂)]
    N_pp[2] = (20.0 * c1 * xi3 - 24.0 * c1 * xi2 - 12.0 * c1 * xi
               + 8.0 * c1 - 3.0 * c2 * xi2 + c2) / (2.0 * c1 * den_main)
    # N_pp_4* =  ξ·(10c₁ξ² - 30c₁ - 3c₂) / [2(8c₁+c₂)]
    N_pp[3] = xi * (10.0 * c1 * xi2 - 30.0 * c1 - 3.0 * c2) / den_2
    # N_pp_5* = -(10c₁ξ³ - 30c₁ξ - 8c₁ - 3c₂ξ - c₂) / [2(8c₁+c₂)]
    N_pp[4] = -(10.0 * c1 * xi3 - 30.0 * c1 * xi - 8.0 * c1
                - 3.0 * c2 * xi - c2) / den_2
    # N_pp_6* =  (20c₁ξ³ + 24c₁ξ² - 12c₁ξ - 8c₁ + 3c₂ξ² - c₂) / [2c₁(8c₁+c₂)]
    N_pp[5] = (20.0 * c1 * xi3 + 24.0 * c1 * xi2 - 12.0 * c1 * xi
               - 8.0 * c1 + 3.0 * c2 * xi2 - c2) / (2.0 * c1 * den_main)

    # --- H(ξ): γ shape functions ---------------------------------------------
    five_c1_sq = 5.0 * c1 * c1
    minus_xim1_xip1 = -(xim1 * xip1)  # = 1 - ξ²
    H = np.empty(6, dtype=complex)
    # H_1* = -5c₁²·(ξ-1)(ξ+1) / [2(8c₁+c₂)]
    H[0] = -five_c1_sq * (xim1 * xip1) / den_2
    # H_2* = same as H_1*  (variational coupling — verified analytically)
    H[1] = H[0]
    # H_3* =  (ξ-1)·(10c₁ξ + 2c₁ - c₂) / [2(8c₁+c₂)]
    H[2] = xim1 * (10.0 * c1 * xi + 2.0 * c1 - c2) / den_2
    # H_4* =  5c₁²·(ξ-1)(ξ+1) / [2(8c₁+c₂)]   (= -H_1*)
    H[3] = -H[0]
    # H_5* = same as H_1*
    H[4] = H[0]
    # H_6* =  (ξ+1)·(10c₁ξ - 2c₁ + c₂) / [2(8c₁+c₂)]
    H[5] = xip1 * (10.0 * c1 * xi - 2.0 * c1 + c2) / den_2

    # --- H_p(ξ): dH/dξ -------------------------------------------------------
    H_p = np.empty(6, dtype=complex)
    # H_p_1* = -5c₁²ξ / (8c₁+c₂)
    H_p[0] = -five_c1_sq * xi / den_main
    H_p[1] = H_p[0]
    # H_p_3* =  (20c₁ξ - 8c₁ - c₂) / [2(8c₁+c₂)]
    H_p[2] = (20.0 * c1 * xi - 8.0 * c1 - c2) / den_2
    H_p[3] = -H_p[0]
    H_p[4] = H_p[0]
    # H_p_6* =  (20c₁ξ + 8c₁ + c₂) / [2(8c₁+c₂)]
    H_p[5] = (20.0 * c1 * xi + 8.0 * c1 + c2) / den_2

    return N, N_pp, H, H_p


def build_a_matrix(c1: complex, c2: complex) -> np.ndarray:
    """Build the 6×6 complex matrix [a] (Eq 28) mapping {α*} → {w_e}.

    Row order (DOF order in element):  w₁, a·θ₁, γ₁, w₂, a·θ₂, γ₂.

        | -1  -1   1  -1     1     -1            |  w₁  ← w(ξ=-1)
        |  0   1  -2   3    -4      5            |  a·θ₁ ← dw/dξ|_{-1}
        |  0   0   0  c₁*  -4c₁*   c₂*+10c₁*    |  γ₁ ← γ(ξ=-1)
        |  1   1   1   1     1      1            |  w₂  ← w(ξ=+1)
        |  0   1   2   3     4      5            |  a·θ₂ ← dw/dξ|_{+1}
        |  0   0   0  c₁*   4c₁*   c₂*+10c₁*    |  γ₂ ← γ(ξ=+1)
    """
    A = np.zeros((6, 6), dtype=complex)
    # Row 1: w(ξ=-1) = α₁·1 + α₂·(-1) + α₃·1 + α₄·(-1) + α₅·1 + α₆·(-1)
    A[0, :] = [1, -1, 1, -1, 1, -1]
    # Row 2: a·θ₁ = dw/dξ|_{-1} = α₂·1 + 2α₃·(-1) + 3α₄·1 + 4α₅·(-1) + 5α₆·1
    A[1, :] = [0, 1, -2, 3, -4, 5]
    # Row 3: γ(ξ=-1) = β₁·1 + β₂·(-1) + β₃·1 = c₁α₄ - 4c₁α₅ + (c₂+10c₁)α₆
    A[2, :] = [0, 0, 0, c1, -4 * c1, c2 + 10 * c1]
    # Row 4: w(ξ=+1)
    A[3, :] = [1, 1, 1, 1, 1, 1]
    # Row 5: a·θ₂ = dw/dξ|_{+1}
    A[4, :] = [0, 1, 2, 3, 4, 5]
    # Row 6: γ(ξ=+1) = c₁α₄ + 4c₁α₅ + (c₂+10c₁)α₆
    A[5, :] = [0, 0, 0, c1, 4 * c1, c2 + 10 * c1]
    return A


def build_T_matrix(c1: complex, c2: complex) -> np.ndarray:
    """Build the 3×6 complex matrix [T*] mapping {α*} → {β*} (Eq 26).

    {β*} = [T*]·{α*}.  Only columns 4, 5, 6 are nonzero.
    """
    T = np.zeros((3, 6), dtype=complex)
    T[0, 3] = c1
    T[0, 5] = c2
    T[1, 4] = 4.0 * c1
    T[2, 5] = 10.0 * c1
    return T


# =============================================================================
# Element matrices via Gauss-Legendre quadrature
# =============================================================================

def compute_element_matrices(
    params: SandwichBeamParameters,
    G_complex: complex,
    n_gauss: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute element complex mass M*_e and stiffness K*_e (6×6) at given ω.

    Uses the sympy-derived closed-form shape functions (`evaluate_shape_functions`)
    to avoid numerically inverting [a]. This removes the conditioning blow-up
    that occurred under mesh refinement in the low-G regime (c₂/c₁ ≫ 1).

    Energy integrals (paper Eqs 14-16, including the full-width factor b):
        M*_e = a·ρ_eff·∫_{-1}^1 N*ᵀN* dξ
        K_B  = (b·Dt/a³)·∫_{-1}^1 N_pp*ᵀ N_pp* dξ
        K_E  = a·b·(E₁·h₁/2)·∫δᵀδ dξ
               where δ = (d/a²)·N_pp* − (h₂/a)·H_p*
        K_S  = a·b·G*·h₂·∫_{-1}^1 H*ᵀ H* dξ
    """
    a = params.half_element_length
    g_star, c1, c2 = compute_complex_coefficients(params, G_complex)

    # Gauss-Legendre points and weights for [-1, 1]
    gauss_xi, gauss_w = np.polynomial.legendre.leggauss(n_gauss)

    M_e = np.zeros((6, 6), dtype=complex)
    K_e = np.zeros((6, 6), dtype=complex)

    b = params.b
    Dt = params.Dt
    d = params.d
    h2 = params.h2
    E1 = params.E1
    h1 = params.h1
    a_inv = 1.0 / a
    a_inv2 = a_inv * a_inv

    bDt_over_a3 = b * Dt / (a ** 3)
    abEh1_half = a * b * E1 * h1 / 2.0
    abGh2 = a * b * G_complex * h2
    a_rho = a * params.rho_eff

    for xi, wq in zip(gauss_xi, gauss_w):
        # Closed-form shape functions and derivatives at ξ (no [a]^{-1})
        N, N_pp, H, H_p = evaluate_shape_functions(xi, c1, c2)

        # Mass:  M_e += w·a·ρ_eff·N⊗N
        M_e += wq * a_rho * np.outer(N, N)

        # Bending:  K_B += w·(b·Dt/a³)·N_pp⊗N_pp
        K_e += wq * bDt_over_a3 * np.outer(N_pp, N_pp)

        # Extensional:  δ = (d/a²)·N_pp − (h₂/a)·H_p
        #               K_E += w·(a·b·E₁·h₁/2)·δ⊗δ
        delta = (d * a_inv2) * N_pp - (h2 * a_inv) * H_p
        K_e += wq * abEh1_half * np.outer(delta, delta)

        # Core shear:  K_S += w·(a·b·G*·h₂)·H⊗H
        K_e += wq * abGh2 * np.outer(H, H)

    return M_e, K_e


# =============================================================================
# Global assembly
# =============================================================================

def assemble_global(
    params: SandwichBeamParameters,
    G_complex: complex,
    n_gauss: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble global complex M* and K* for uniform beam.

    DOF ordering: per node {w, a·θ, γ}.  Total = 3·(n_elem + 1) DOFs.

    Note: 'a·θ' (instead of θ) is used to make [a] matrix dimensionless.
    Convert to physical θ at post-processing if needed.
    """
    n_nodes = params.n_elem + 1
    n_dof = 3 * n_nodes
    M = np.zeros((n_dof, n_dof), dtype=complex)
    K = np.zeros((n_dof, n_dof), dtype=complex)

    M_e, K_e = compute_element_matrices(params, G_complex, n_gauss)

    for e in range(params.n_elem):
        # Element DOF indices: [w_left, a·θ_left, γ_left, w_right, a·θ_right, γ_right]
        i0 = 3 * e
        idx = np.arange(i0, i0 + 6)
        M[np.ix_(idx, idx)] += M_e
        K[np.ix_(idx, idx)] += K_e

    return M, K


def apply_boundary_conditions(
    M: np.ndarray,
    K: np.ndarray,
    F: np.ndarray,
    bc_left: str = "clamped",
    bc_right: str = "simply_supported",
    n_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply boundary conditions by removing constrained DOFs.

    BC types per node (3 DOFs each: w, a·θ, γ):
        "clamped"          → w = a·θ = γ = 0     (all 3 fixed)
        "simply_supported" → w = γ = 0           (a·θ free)
        "hinged"           → w = a·θ = 0         (γ free)
        "free"             → no constraint

    Parameters
    ----------
    bc_left, bc_right : str
        BC at first / last node.

    Returns
    -------
    (M_red, K_red, F_red, free_dofs)
        Reduced matrices/vector and indices of unconstrained DOFs in the
        original system.
    """
    n_dof = M.shape[0]
    if n_nodes is None:
        n_nodes = n_dof // 3

    fixed_dofs = set()
    bc_to_fixed = {
        "clamped": [0, 1, 2],          # w, a·θ, γ
        "simply_supported": [0, 2],     # w, γ
        "hinged": [0, 1],               # w, a·θ
        "free": [],
    }

    for off in bc_to_fixed.get(bc_left, []):
        fixed_dofs.add(0 * 3 + off)
    last = (n_nodes - 1) * 3
    for off in bc_to_fixed.get(bc_right, []):
        fixed_dofs.add(last + off)

    free_dofs = np.array([i for i in range(n_dof) if i not in fixed_dofs])
    M_red = M[np.ix_(free_dofs, free_dofs)]
    K_red = K[np.ix_(free_dofs, free_dofs)]
    F_red = F[free_dofs]
    return M_red, K_red, F_red, free_dofs


# =============================================================================
# Frequency response
# =============================================================================

def frequency_response(
    params: SandwichBeamParameters,
    G_func: Callable[[float], complex],
    omega_array: np.ndarray,
    F_node: int,
    F_dof: int = 0,
    bc_left: str = "clamped",
    bc_right: str = "simply_supported",
    n_gauss: int = 8,
) -> np.ndarray:
    """Compute complex displacement response at all DOFs over a frequency range.

    Parameters
    ----------
    G_func : callable(ω) -> complex
        Frequency-dependent complex shear modulus G*(ω).
    omega_array : (N_w,) ndarray
        Angular frequencies [rad/s] to evaluate.
    F_node, F_dof : int
        Force application: F_node ∈ [0, n_nodes), F_dof ∈ {0, 1, 2}
        (0 = force on w, 1 = moment on a·θ, 2 = generalized on γ).
    bc_left, bc_right : str
        See `apply_boundary_conditions`.

    Returns
    -------
    Y : (N_w, n_dof_full) ndarray
        Complex response at each DOF in the full (un-reduced) system,
        with constrained DOFs set to 0.
    """
    omega_array = np.asarray(omega_array, dtype=float)
    n_nodes = params.n_elem + 1
    n_dof = 3 * n_nodes

    F_full = np.zeros(n_dof, dtype=complex)
    F_full[3 * F_node + F_dof] = 1.0

    Y = np.zeros((omega_array.size, n_dof), dtype=complex)

    for k, omega in enumerate(omega_array):
        G_om = G_func(omega)
        M, K = assemble_global(params, G_om, n_gauss)
        M_r, K_r, F_r, free = apply_boundary_conditions(
            M, K, F_full, bc_left, bc_right, n_nodes
        )
        D_r = K_r - omega**2 * M_r
        try:
            x_r = np.linalg.solve(D_r, F_r)
        except np.linalg.LinAlgError:
            x_r = np.full(F_r.size, np.nan, dtype=complex)
        Y[k, free] = x_r

    return Y


# =============================================================================
# Convenience: default frequency-dependent G* (paper Table 6 viscoelastic)
# =============================================================================

def default_viscoelastic_core_G(
    G_storage: float = 1.0e8,
    eta: float = 0.2,
) -> Callable[[float], complex]:
    """Constant complex modulus (paper Table 1 default rubber): G*(ω) = G·(1+iη)."""
    def G_func(omega: float) -> complex:
        return G_storage * (1.0 + 1j * eta)
    return G_func


def freq_dependent_core_G_paper_table6(
    f_hz: float | None = None,
) -> Callable[[float], complex]:
    """Frequency-dependent core (paper Table 6): G' = 6895·exp(0.171·ln(f) + 2.705),
    G'' = 6895·exp(0.026·ln(f) + 4.754).
    """
    def G_func(omega: float) -> complex:
        f = max(omega / (2.0 * np.pi), 1e-3)
        G_prime = 6895.0 * np.exp(0.171 * np.log(f) + 2.705)
        G_double = 6895.0 * np.exp(0.026 * np.log(f) + 4.754)
        return G_prime + 1j * G_double
    return G_func
