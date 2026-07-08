"""Transfer function reconstruction utilities — based on modal superposition.

Theoretical background
----------------------
Given the solutions {λ_n, v_n} of the QEP `(λ²M + λC + K) v = 0`, the complex
response Y(ω) is expressed as a sum of residues at each pole.

    D(ω) = K − ω²M + iω C
    D(ω) v_n = 0   when  iω = λ_n  (i.e., ω_pole = −i·λ_n)

    D'(ω) = −2ωM + iC
    D'(ω_pole) = i · (2λ_n M + C)  ≡ i·m̃_n

    Y(ω) = Σ_n v_n (v_nᵀF) / [(ω − ω_pole) · i · m_n]
         = Σ_n v_n (v_nᵀF) / [m_n · (iω − λ_n)]              ★

The i in the denominator cancels via the transformation `(ω + iλ_n)·i = iω − λ_n`.
Hence the correct modal superposition formula has **no i in the denominator**
(fixes a bug in an earlier version).

For real symmetric M, C, K systems the eigenvalues come in conjugate pairs, so
the 2n eigenvalues returned by the QEP already include the conjugates →
the default `include_conjugate=False` is automatically correct.
The same holds for complex symmetric M, C, K systems (polynomial fitting
approach) — the 2n eigenvalues are complete by themselves.
"""
from __future__ import annotations
import numpy as np


def _modal_normalization(lam: complex, v: np.ndarray, M: np.ndarray, C: np.ndarray) -> complex:
    """Modal normalization scalar m_n = v^T (2λM + C) v."""
    return complex(v @ (2.0 * lam * (M @ v) + C @ v))


def modal_superposition(
    omega_array: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
    force_vector: np.ndarray,
) -> np.ndarray:
    """Pure modal superposition formula — Y(ω) = Σ v_n (v_nᵀF) / [m_n·(iω − λ_n)].

    Parameters
    ----------
    omega_array : (N_f,) ndarray
        Angular frequencies to evaluate [rad/s]
    eigenvalues : (K,) complex
        Array of λ_n
    eigenvectors : (n, K) complex
        Each column is the corresponding eigenvector
    M, C : (n, n) complex/real
        Mass and damping matrices of the system (used for modal normalization)
    force_vector : (n,) complex
        Force vector

    Returns
    -------
    Y : (N_f, n) complex
    """
    F = np.asarray(force_vector, dtype=complex)
    i_omega = 1j * np.asarray(omega_array, dtype=float)
    n_dof = M.shape[0]
    Y = np.zeros((i_omega.size, n_dof), dtype=complex)

    M_c = np.asarray(M, dtype=complex)
    C_c = np.asarray(C, dtype=complex)

    for k in range(eigenvalues.size):
        lam = complex(eigenvalues[k])
        v = np.asarray(eigenvectors[:, k], dtype=complex)
        m_k = complex(v @ (2.0 * lam * (M_c @ v) + C_c @ v))
        if m_k == 0 or not np.isfinite(lam):
            continue
        numer = (v @ F) / m_k          # scalar
        denom = i_omega - lam          # (N_f,)
        Y += np.outer(1.0 / denom, v) * numer

    return Y


def compute_error_metric(Y_true: np.ndarray, Y_approx: np.ndarray) -> dict:
    """Compare direct vs. extraction responses — absolute error / relative error / RMSE."""
    err_abs = np.abs(Y_true - Y_approx)
    denom = np.abs(Y_true)
    denom[denom == 0] = 1e-30
    err_rel = err_abs / denom

    return {
        "max_abs": float(np.max(err_abs)),
        "mean_abs": float(np.mean(err_abs)),
        "max_rel": float(np.max(err_rel)),
        "mean_rel": float(np.mean(err_rel)),
        "rmse": float(np.sqrt(np.mean(err_abs ** 2))),
    }
