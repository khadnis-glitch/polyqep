"""n-DOF symmetric chain model (general extension of the 3-DOF system in the accompanying paper).

Structure (n=6 example):

    │k1                                              │k1
    m1 ── k2 ── m2 ── k2 ── m3 ── k2 ── m4 ── k2 ── m5 ── k2 ── m6
                │                 │                           │
             k_H,c_H            k_H,c_H                    k_H,c_H
                │                 │                           │
             (ground)          (ground)                   (ground)

- Masses: m1 = mn = 200 ton (both ends), m2..m(n-1) = 500 ton (interior)
- Elastic springs: k1 (2 grounded springs at both ends), k2 (n-1 springs between nodes)
- Frequency-dependent supports: by default at 0-indexed odd nodes (1, 3, 5, ...) = even-numbered nodes
- DOFs: 1 DOF per node (vertical direction), n DOFs in total

Generalization: specify arbitrary N (≥3) via `ChainModelParameters(n_dof=N)`.
The existing 6-DOF remains backward compatible through the `SixDOFParameters()` alias.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class ChainModelParameters:
    """Parameters of the symmetric chain model (units: SI, kg, N/m).

    Attributes
    ----------
    n_dof : int
        Total number of DOFs (nodes). Minimum 3.
    m_end, m_inner : float
        End and interior masses.
    k1, k2 : float
        Grounded end springs / interior coupling springs.
    support_nodes : tuple[int, ...] | None
        Node indices (0-based) of the frequency-dependent supports. If None,
        supports are placed automatically at even-numbered nodes.
    """
    n_dof: int = 6
    m_end: float = 200.0e3
    m_inner: float = 500.0e3
    k1: float = 300.0e6
    k2: float = 400.0e6
    support_nodes: Optional[tuple[int, ...]] = None

    def __post_init__(self):
        if self.n_dof < 3:
            raise ValueError("n_dof must be >= 3 (2 end nodes + at least 1 interior node)")
        if self.support_nodes is None:
            # 0-based odd indices = 1-indexed even-numbered nodes (nodes 2, 4, 6, ...)
            default = tuple(range(1, self.n_dof, 2))
            object.__setattr__(self, "support_nodes", default)
        # Validation
        for idx in self.support_nodes:
            if not (0 <= idx < self.n_dof):
                raise ValueError(f"support_nodes index {idx} is outside range [0, {self.n_dof})")


# Backward compat — alias for existing code (`SixDOFParameters()`)
SixDOFParameters = ChainModelParameters


def build_mass_matrix(params: ChainModelParameters = None) -> np.ndarray:
    """Build the mass matrix M (symmetric chain). m_end at both ends, m_inner inside."""
    if params is None:
        params = ChainModelParameters()
    n = params.n_dof
    diag = np.full(n, params.m_inner, dtype=float)
    diag[0] = params.m_end
    diag[-1] = params.m_end
    return np.diag(diag)


def build_base_stiffness(params: SixDOFParameters = SixDOFParameters()) -> np.ndarray:
    """Build the base stiffness matrix K_base composed of elastic springs only.

    - End nodes: k1 (grounded) + k2 (interior coupling)
    - Interior nodes: 2 × k2
    """
    n = params.n_dof
    K = np.zeros((n, n), dtype=float)
    k1, k2 = params.k1, params.k2

    # Interior coupling springs (node i ↔ i+1): 5 in total (i = 0..4)
    for i in range(n - 1):
        K[i, i] += k2
        K[i + 1, i + 1] += k2
        K[i, i + 1] -= k2
        K[i + 1, i] -= k2

    # Grounded springs at both ends
    K[0, 0] += k1
    K[n - 1, n - 1] += k1

    return K


def build_support_locator(params: SixDOFParameters = SixDOFParameters()) -> np.ndarray:
    """Build the diagonal 0/1 support locator matrix L marking the frequency-dependent supports.

    K_H(ω) = k_H(ω) · L, C_H(ω) = c_H(ω) · L
    """
    L = np.zeros((params.n_dof, params.n_dof), dtype=float)
    for idx in params.support_nodes:
        L[idx, idx] = 1.0
    return L


@dataclass
class AssembledSystem:
    """System matrices assembled at a specific frequency ω.

    Governing equation (frequency domain): [-ω²M + iωC(ω) + K(ω)] X = F
    """
    M: np.ndarray
    K_total: np.ndarray  # K_base + k_H(ω)·L
    C_total: np.ndarray  # c_H(ω)·L
    omega: float

    @property
    def n(self) -> int:
        return self.M.shape[0]


def assemble_system_at_omega(
    omega: float,
    k_h_func,
    c_h_func,
    params: SixDOFParameters = SixDOFParameters(),
    M: np.ndarray | None = None,
    K_base: np.ndarray | None = None,
    L: np.ndarray | None = None,
) -> AssembledSystem:
    """Assemble the full system matrices at a given ω.

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s]
    k_h_func : callable(omega) -> float
        Frequency-dependent stiffness k_H(ω) [N/m]
    c_h_func : callable(omega) -> float
        Frequency-dependent damping c_H(ω) [N·s/m]
    params : SixDOFParameters
    M, K_base, L : cached matrices (optional)
        Can be reused across repeated calls
    """
    if M is None:
        M = build_mass_matrix(params)
    if K_base is None:
        K_base = build_base_stiffness(params)
    if L is None:
        L = build_support_locator(params)

    k_h = float(k_h_func(omega))
    c_h = float(c_h_func(omega))

    K_total = K_base + k_h * L
    C_total = c_h * L

    return AssembledSystem(M=M, K_total=K_total, C_total=C_total, omega=omega)
