"""Quadratic eigenvalue problem (QEP) solver.

QEP:  (λ² M + λ C + K) v = 0

Companion linearization (companion form):
    [  0    I ] [v   ]       [-I   0 ] [v   ]
    [ -K   -C ] [λv  ] = λ · [ 0   M ] [λv  ]

or, equivalently, the generalized eigenvalue problem:
    A w = λ B w,  w = [v; λv]
    A = [ 0   I ], B = [ I   0 ]
        [-K  -C]       [ 0   M ]

Since each system matrix may be complex, scipy.linalg.eig is used.

Reference: Tisseur & Meerbergen, "The Quadratic Eigenvalue Problem", SIAM Review 2001.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eig


@dataclass
class QEPResult:
    """Result of a QEP solve.

    - eigenvalues: length-2n array of complex λ
    - eigenvectors: (n, 2n) complex matrix; each column is the v corresponding to λ
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @property
    def n(self) -> int:
        return self.eigenvectors.shape[0]


def solve_qep(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
) -> QEPResult:
    """Solve (λ²M + λC + K) v = 0 via companion linearization.

    Parameters
    ----------
    M, C, K : (n, n) ndarray
        Each may be complex. In the direct method of the accompanying paper,
        k_H(ω) enters K and c_H(ω) enters C.

    Returns
    -------
    QEPResult
    """
    M = np.asarray(M)
    C = np.asarray(C)
    K = np.asarray(K)
    n = M.shape[0]
    if not (M.shape == C.shape == K.shape == (n, n)):
        raise ValueError(f"M, C, K must all be ({n},{n})")

    # Promote to complex (to handle complex coefficients)
    dtype = np.result_type(M.dtype, C.dtype, K.dtype, np.complex128)
    M = M.astype(dtype)
    C = C.astype(dtype)
    K = K.astype(dtype)

    I = np.eye(n, dtype=dtype)
    Z = np.zeros((n, n), dtype=dtype)

    A = np.block([[Z,   I],
                  [-K, -C]])
    B = np.block([[I,   Z],
                  [Z,   M]])

    eigvals, eigvecs_full = eig(A, B)
    # Eigenvector w = [v; λv] → extract only the v component (upper block)
    v_part = eigvecs_full[:n, :]

    return QEPResult(eigenvalues=eigvals, eigenvectors=v_part)
