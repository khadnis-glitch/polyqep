"""Additional studies — SOAR basis with a sparse shifted factorization.

Same recursion as ``polyqep.rom.soar_basis`` (force-aware start, modified
Gram-Schmidt applied twice, Bai-Su dual recursion, identical deflation rule).
Two implementation choices differ, neither of which changes the algorithm:

* the shifted matrix ``K + sigma C + sigma^2 M`` is factorized by sparse LU
  (SuperLU) instead of dense LU;
* the dense 2-norm condition-number guard (a full SVD of the shifted matrix)
  is omitted.

The released implementation remains the reference; this variant is used in
the revision only to report how much of the offline time is due to the dense
factorization path.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as spsparse
import scipy.sparse.linalg as spspla


def soar_basis_sparse(M_s, C_s, K_s, shift_omega, m, force_vector, tol=1e-13):
    """SOAR basis Q (n x m_actual) from sparse segment matrices (CSC)."""
    n = M_s.shape[0]
    sigma = 1j * shift_omega
    lu = spspla.splu((K_s + sigma * C_s + (sigma**2) * M_s).tocsc())
    P_prime = (C_s + 2.0 * sigma * M_s).tocsr()
    M_csr = M_s.tocsr()

    q1 = lu.solve(np.asarray(force_vector, dtype=complex))
    nrm0 = np.linalg.norm(q1)
    if nrm0 < tol:
        raise RuntimeError("starting vector is numerically zero")
    cap = min(m, n)
    Q = np.empty((n, cap), dtype=complex)
    P = np.zeros((n, cap), dtype=complex)
    Q[:, 0] = q1 / nrm0
    n_built = 1
    for k in range(min(m - 1, n - 1)):
        z = -lu.solve(P_prime @ Q[:, k] + M_csr @ P[:, k])
        nrm_before = np.linalg.norm(z)
        if nrm_before < tol:
            break
        Q_built = Q[:, :n_built]
        h_tot = np.zeros(n_built, dtype=complex)
        for _ in range(2):
            coefs = Q_built.conj().T @ z
            z = z - Q_built @ coefs
            h_tot += coefs
        beta = np.linalg.norm(z)
        if beta < tol * nrm_before or beta < 1e-12:
            break
        Q[:, n_built] = z / beta
        P[:, n_built] = (Q[:, k] - P[:, :n_built] @ h_tot) / beta
        n_built += 1
    return Q[:, :n_built]


def segment_matrices_sparse(spline_re, spline_im, M_s, Kb_s, Ks_s):
    """Sparse counterpart of ``assemble_complex_per_interval_amichi``."""
    out = []
    for seg_re, seg_im in zip(spline_re.segments, spline_im.segments):
        a, b, c = seg_re.a, seg_re.b, seg_re.c
        d, e, f = seg_im.a, seg_im.b, seg_im.c
        out.append(dict(
            M_star=(M_s - (a + 1j * d) * Ks_s).tocsc(),
            C_star=((e - 1j * b) * Ks_s).tocsc(),
            K_star=(Kb_s + (c + 1j * f) * Ks_s).tocsc(),
            omega_lower=seg_re.omega_lower, omega_upper=seg_re.omega_upper))
    return out


def project_pieces_sparse(Q, Kb_s, Ks_s, M_s, F):
    """Q^T K_base Q, Q^T K_shear Q, Q^T M Q, Q^T F with sparse operators."""
    Qt = Q.T
    return (Qt @ (Kb_s @ Q), Qt @ (Ks_s @ Q), Qt @ (M_s @ Q),
            Qt @ np.asarray(F, dtype=complex))


def as_sparse(K_base, K_shear, M):
    return (spsparse.csc_matrix(K_base).astype(complex),
            spsparse.csc_matrix(K_shear).astype(complex),
            spsparse.csc_matrix(M).astype(complex))
