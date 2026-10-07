# === [auto-resized: figsize x1.5] ===
"""Amichi 567-DOF plate — G(ω) polynomial fitting + segmentwise complex QEP + ROM.

Complete implementation of the follow-up to Bae & Cho (IJPEM 2023).

Pipeline
--------
1. (Stage E) G(ω) fine-grid sampling
   piecewise constant G(f) of core_property_at_freq() → fine-grid samples
   Re[G(ω)] and Im[G(ω)] separated

2. (Stage F) piecewise quadratic polynomial fitting
   fit_joint_spline(omega, Re[G], Im[G], boundaries) → C⁰+C¹ continuous spline
   k(ω) ↔ Re[G(ω)], ω·c(ω) ↔ Im[G(ω)]
   on each segment i: Re[G_i(ω)] = a_i·ω² + b_i·ω + c_i,  Im[G_i(ω)] = d_i·ω² + e_i·ω + f_i

3. (Stage G) segmentwise ω-independent complex matrix reconstruction
   K(ω) = K_base + G(ω)·K_shear
        = K_base + (Re[G] + i·Im[G])·K_shear
        = K_base + (a·ω² + b·ω + c)·K_shear + i·(d·ω² + e·ω + f)·K_shear
   D(ω) = K(ω) - ω²·M
        = [K_base + (c + i·f)·K_shear]
          + ω·[(b + i·e)·K_shear]
          - ω²·[M - (a + i·d)·K_shear]
   Mapped onto the form D(ω) = K* + iω·C* - ω²·M*:
     M*_i = M - (a_i + i·d_i)·K_shear
     C*_i = (e_i - i·b_i)·K_shear        ← from iω·C* = ω·(b + i·e)·K_shear
     K*_i = K_base + (c_i + i·f_i)·K_shear

4. (Stage H) one complex QEP per segment + effective-eigenvalue selection
   (λ²·M*_i + λ·C*_i + K*_i)·v = 0
   effective eigenvalues: Im(λ) ∈ [ω_lo, ω_hi), Re(λ) < 0
   conjugate-pair completion (see the paper)

5. (Stage I) segmentwise modal superposition vs Direct method
   ROM applied (modal m=20-50) on each segment

Output
------
stdout: pipeline diagnostics + accuracy/CPU comparison
figures/phase3_G_fitting.png            — G(ω) fit accuracy
figures/phase3_qep_eigenvalues.png      — segmentwise QEP eigenvalue map
figures/phase3_polyqep_frf_overlay.png  — Direct vs Polynomial QEP vs ROM
"""
from __future__ import annotations

import time
from pathlib import Path

# === [auto-inserted: enable _figure_io monkey-patch for .pkl + .source.md] ===
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _figure_io  # noqa: F401
# === [/auto-inserted] ===

import matplotlib

# === [auto-inserted: 2x font scaling for all figures] ===
# === [2/3 font scaling applied: reduce in-figure text size] ===
import matplotlib as _mpl  # noqa: E402
_mpl.rcParams.update({
    "font.size": 40 / 3,        # 20 * 2/3
    "axes.titlesize": 16.0,     # 24 * 2/3
    "axes.labelsize": 40 / 3,   # 20 * 2/3
    "xtick.labelsize": 40 / 3,
    "ytick.labelsize": 40 / 3,
    "legend.fontsize": 32 / 3,  # 16 * 2/3
    "figure.titlesize": 16.0,
})
# === [/auto-inserted] ===


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla

from polyqep.fitting import (
    QuadraticSegment,
    QuadraticSpline,
    detect_breakpoints,
    fit_joint_spline,
)
from polyqep.models.sandwich_plate_amichi2010 import (
    SandwichPlateParameters,
    apply_boundary_conditions,
    assemble_K_base_and_shear,
    assemble_global,
)
from polyqep.rom import (
    arnoldi_shift_invert_basis,
    modal_truncation_basis,
)
from polyqep.qep_solver import solve_qep


# =============================================================================
# Stage E — G(ω) sampling
# =============================================================================
def paper_5_1_params(nx: int = 8, ny: int = 8) -> SandwichPlateParameters:
    return SandwichPlateParameters(
        Lx=0.3048, Ly=0.3048,
        h1=0.475e-3, h2=0.035e-3, h3=0.475e-3,
        E1=2.1e11, E3=2.1e11,
        nu1=0.26, nu3=0.26,
        rho1=7780, rho3=7780,
        rho2=1134, nu2=0.45,
        E2_in_plane=1.0e7,
        nx=nx, ny=ny,
    )


def G_complex_at_freq(freq_hz: float) -> complex:
    """G_core(f) based on the DMA data of Fig. 7 (§5.1) of the reference paper — smooth interpolation for the fine grid.

    The original `core_property_at_freq` is a 4-step piecewise constant. This
    function takes the mid value of each step as an anchor and interpolates
    smoothly in log-frequency.

    The data are smoothed so that the piecewise quadratic polynomial fit is
    meaningful (an estimate of the actual trend of Fig. 7 of the reference paper).
    """
    # paper Fig 7 anchors (mid of each step + extrapolation)
    f_anchors = np.array([10.0, 50.0, 100.0, 200.0, 400.0, 700.0, 1000.0])
    E2_anchors = np.array([4.5e6, 5.0e6, 6.0e6, 8.0e6, 1.4e7, 1.8e7, 2.0e7])
    eta_anchors = np.array([1.40, 1.36, 1.34, 1.30, 1.22, 1.18, 1.15])
    nu2 = 0.45

    log_f = np.log(np.clip(freq_hz, 1.0, 2000.0))
    log_fa = np.log(f_anchors)
    E2 = np.interp(log_f, log_fa, E2_anchors)
    eta = np.interp(log_f, log_fa, eta_anchors)
    G = E2 / (2.0 * (1.0 + nu2))
    return G * (1.0 + 1j * eta)


def sample_G_omega(f_min: float, f_max: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fine-grid sampling of G(ω). Returns (omega [rad/s], Re[G], Im[G])."""
    f_hz = np.linspace(f_min, f_max, n)
    omega = 2.0 * np.pi * f_hz
    G_complex = np.array([G_complex_at_freq(f) for f in f_hz])
    return omega, G_complex.real, G_complex.imag


# =============================================================================
# Stage F — piecewise quadratic fitting
# =============================================================================
def fit_G_polynomial(
    omega: np.ndarray,
    G_re: np.ndarray,
    G_im: np.ndarray,
    n_segments: int = 5,
) -> tuple[QuadraticSpline, QuadraticSpline, np.ndarray]:
    """Piecewise quadratic fit of Re[G(ω)] and Im[G(ω)] with shared breakpoints.

    Returns
    -------
    spline_re, spline_im, boundaries
    """
    if n_segments == 5:
        # near 100, 300, 700 Hz + both ends
        f_brk = np.array([1.0, 50.0, 150.0, 350.0, 700.0, 1500.0])
        boundaries = 2.0 * np.pi * f_brk[(f_brk >= omega[0] / (2 * np.pi))
                                          & (f_brk <= omega[-1] / (2 * np.pi))]
        boundaries = np.concatenate([[omega[0]], boundaries, [omega[-1]]])
        boundaries = np.unique(boundaries)
    else:
        # uniform automatic split
        f_min = omega[0] / (2 * np.pi)
        f_max = omega[-1] / (2 * np.pi)
        f_brk = np.linspace(f_min, f_max, n_segments + 1)
        boundaries = 2.0 * np.pi * f_brk
    spline_re, spline_im = fit_joint_spline(omega, G_re, G_im, boundaries)
    return spline_re, spline_im, boundaries


def evaluate_spline_at(spline: QuadraticSpline, omega: np.ndarray) -> np.ndarray:
    """Evaluate a piecewise quadratic spline on an omega array."""
    y = np.zeros_like(omega, dtype=float)
    for seg in spline.segments:
        mask = (omega >= seg.omega_lower) & (omega < seg.omega_upper)
        if seg is spline.segments[-1]:
            mask = (omega >= seg.omega_lower) & (omega <= seg.omega_upper)
        y[mask] = seg.a * omega[mask] ** 2 + seg.b * omega[mask] + seg.c
    return y


# =============================================================================
# Stage G — segmentwise ω-independent complex matrix reconstruction
# =============================================================================
def assemble_complex_per_interval_amichi(
    spline_re: QuadraticSpline,
    spline_im: QuadraticSpline,
    M: np.ndarray,
    K_base: np.ndarray,
    K_shear: np.ndarray,
) -> list[dict]:
    """Complex matrices (M*, C*, K*) and ω range of each segment.

    Mapping (see the module docstring):
        M*_i = M - (a_i + i·d_i)·K_shear
        C*_i = (e_i - i·b_i)·K_shear
        K*_i = K_base + (c_i + i·f_i)·K_shear
    """
    n = M.shape[0]
    results = []
    M_c = M.astype(complex)
    K_base_c = K_base.astype(complex)
    K_shear_c = K_shear.astype(complex)
    for seg_re, seg_im in zip(spline_re.segments, spline_im.segments):
        a, b, c = seg_re.a, seg_re.b, seg_re.c     # Re[G] coefficients
        d, e, f = seg_im.a, seg_im.b, seg_im.c     # Im[G] coefficients
        M_star = M_c - (a + 1j * d) * K_shear_c
        C_star = (e - 1j * b) * K_shear_c
        K_star = K_base_c + (c + 1j * f) * K_shear_c
        results.append({
            "M_star": M_star,
            "C_star": C_star,
            "K_star": K_star,
            "omega_lower": seg_re.omega_lower,
            "omega_upper": seg_re.omega_upper,
            "coeffs_re": (a, b, c),
            "coeffs_im": (d, e, f),
        })
    return results


# =============================================================================
# Stage H — segmentwise complex QEP + effective-eigenvalue selection
# =============================================================================
def solve_qep_per_interval(
    interval_matrices: list[dict],
    damping_tol: float = 1e-9,
) -> list[dict]:
    """Solve the complex QEP on each segment and select the effective eigenvalues."""
    out = []
    for i, mat in enumerate(interval_matrices):
        result = solve_qep(mat["M_star"], mat["C_star"], mat["K_star"])
        lams = result.eigenvalues
        vecs = result.eigenvectors  # (n, 2n)
        lo = mat["omega_lower"]
        hi = mat["omega_upper"]
        mask = (
            np.isfinite(lams)
            & (np.imag(lams) >= lo)
            & (np.imag(lams) < hi)
            & (np.real(lams) < -damping_tol)
        )
        lams_eff = lams[mask]
        vecs_eff = vecs[:, mask]
        # conjugate-pair completion
        if lams_eff.size > 0:
            lams_full = np.concatenate([lams_eff, lams_eff.conjugate()])
            vecs_full = np.concatenate([vecs_eff, vecs_eff.conjugate()], axis=1)
        else:
            lams_full = lams_eff
            vecs_full = vecs_eff
        out.append({
            "interval_idx": i,
            "omega_lower": lo,
            "omega_upper": hi,
            "lams_all": lams,
            "vecs_all": vecs,            # full QEP eigenvectors (n, 2n)
            "lams_effective": lams_eff,
            "vecs_effective": vecs_eff,
            "lams_full_with_conj": lams_full,
            "vecs_full_with_conj": vecs_full,
            "matrices": mat,
        })
    return out


# =============================================================================
# Modal superposition response (segmentwise)
# =============================================================================
def modal_response_per_interval(
    omega_array: np.ndarray,
    interval_solutions: list[dict],
    F_red: np.ndarray,
    idx_obs: int,
) -> np.ndarray:
    """Modal superposition with the effective + conjugate modes of the segment containing each ω.

    Y(ω) = Σ_k (vᴴ_k · F) / [(λ_k - iω)·(λ̄_k + iω)] · v_k[idx_obs]
    or, more simply, a direct D⁻¹·F solve via decomposition.

    Here the direct (M*, C*, K*) solve is used instead of the modal form
    (accuracy check — equivalent to the modal expansion).
    """
    n_dof = interval_solutions[0]["matrices"]["M_star"].shape[0]
    Y = np.zeros(omega_array.size, dtype=complex)
    lowers = np.array([s["omega_lower"] for s in interval_solutions])
    for i, omega in enumerate(omega_array):
        j = int(np.searchsorted(lowers, omega, side="right") - 1)
        j = max(0, min(j, len(interval_solutions) - 1))
        mat = interval_solutions[j]["matrices"]
        D = mat["K_star"] + 1j * omega * mat["C_star"] - (omega ** 2) * mat["M_star"]
        u = spla.solve(D, F_red)
        Y[i] = u[idx_obs]
    return Y


# =============================================================================
# ROM per interval — reuse the already-solved QEP eigenvectors (build cost ≈ 0)
# =============================================================================
def rom_response_per_interval_reuse(
    omega_array: np.ndarray,
    interval_solutions: list[dict],
    F_red: np.ndarray,
    idx_obs: int,
    m: int,
    use_effective_only: bool = False,
) -> tuple[np.ndarray, float]:
    """Segmentwise ROM. Directly reuses the QEP eigenvectors solved in Stage H.

    Build cost ≈ QR + projection (no QEP re-solve).

    Parameters
    ----------
    use_effective_only : bool
        True  → use only the effective + conjugate modes
        False → pick the m eigenvectors closest to the shift among all eigenvectors

    Returns
    -------
    Y : ndarray (N_omega,)
    t_build : float (s)  measured build cost
    """
    Y = np.zeros(omega_array.size, dtype=complex)
    lowers = np.array([s["omega_lower"] for s in interval_solutions])

    t0 = time.perf_counter()
    M_r_list, C_r_list, K_r_list, F_r_list, V_r_list = [], [], [], [], []
    for sol in interval_solutions:
        mat = sol["matrices"]
        shift_omega = (sol["omega_lower"] + sol["omega_upper"]) / 2.0
        shift_lam = 1j * shift_omega

        if use_effective_only and sol["vecs_full_with_conj"].size > 0:
            V_raw = sol["vecs_full_with_conj"]
            lams_pool = sol["lams_full_with_conj"]
        else:
            # full eigenvector pool — use vecs_all stored in Stage H
            qep_lams = sol["lams_all"]
            qep_vecs_full = sol["vecs_all"]
            valid = np.isfinite(qep_lams)
            V_raw = qep_vecs_full[:, valid]
            lams_pool = qep_lams[valid]

        # pick the m closest eigenvectors
        dists = np.abs(lams_pool - shift_lam)
        idx_pick = np.argsort(dists)[: min(m, V_raw.shape[1])]
        V_pick = V_raw[:, idx_pick]

        # orthogonalize
        V_r, _ = np.linalg.qr(V_pick)
        # Project
        M_r_list.append(V_r.conj().T @ mat["M_star"] @ V_r)
        C_r_list.append(V_r.conj().T @ mat["C_star"] @ V_r)
        K_r_list.append(V_r.conj().T @ mat["K_star"] @ V_r)
        F_r_list.append(V_r.conj().T @ F_red)
        V_r_list.append(V_r)
    t_build = time.perf_counter() - t0

    for i, omega in enumerate(omega_array):
        j = int(np.searchsorted(lowers, omega, side="right") - 1)
        j = max(0, min(j, len(interval_solutions) - 1))
        D_r = K_r_list[j] + 1j * omega * C_r_list[j] - (omega ** 2) * M_r_list[j]
        u_r = spla.solve(D_r, F_r_list[j])
        u = V_r_list[j] @ u_r
        Y[i] = u[idx_obs]
    return Y, t_build


def _retrieve_full_eigvecs(sol: dict) -> np.ndarray:
    """Only the effective vectors are stored by Stage H — recompute if the full set is needed.

    Only the effective vectors are held here, so they are returned as is
    (no reconstruction needed). Even for use_effective_only=False this falls
    back to effective + conjugate.
    """
    if sol.get("vecs_full_with_conj") is not None and sol["vecs_full_with_conj"].size > 0:
        return sol["vecs_full_with_conj"]
    return sol["vecs_effective"]


def modal_weights_blockwise(
    lams: np.ndarray,
    vecs: np.ndarray,
    M_star: np.ndarray,
    C_star: np.ndarray,
    F_red: np.ndarray,
    idx_obs: int,
    cluster_rtol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Residue weights with (near-)degenerate clusters handled blockwise.

    Per-mode form  w_k = v_k[obs]·(v_k^T F)/a_k,  a_k = v_k^T(2λ_k M*+C*)v_k
    assumes DISTINCT eigenvalues: for a symmetry-degenerate cluster (square
    plates: (m,n)/(n,m) pairs) QZ returns an arbitrary eigenspace basis that
    is NOT bilinear-orthogonal, and the diagonal-only a_k errs at O(1)
    (measured 37% on a 3x3 mesh; intra-cluster offdiag/diag up to 1.6).

    Fix: group eigenvalues with |λ_i-λ_j| < cluster_rtol·|λ|.  A cluster whose
    bilinear coupling A_c = V_c^T D'(λ_c) V_c (D'=2λM*+C*) is negligible keeps
    the exact per-mode poles/weights.  A coupled cluster is collapsed to the
    basis-invariant block residue at the mean pole:
        R_c = V_c A_c^{-1} V_c^T,   w_c = V_c[obs,:] A_c^{-1} (V_c^T F).
    The mean pole cancels the QZ splitting noise of a true multiple root
    (individual split values are noise — a Ritz re-split was tried and errs
    at O(split/dist) ~1e-4; the mean errs at ~1e-8).  Measured trade-off:
    3x3 identity error 0.37 -> 3e-8; 8x8 median 3.9e-8 -> 1.2e-6 (max
    improves), far below the truncation floors (>=1e-4) of every caller.
    Singleton clusters reduce exactly to the classical w_k.

    Returns (lams_out, weights_out) so callers keep the usual
    Y(ω) = Σ_c w_c/(iω - λ_c)  evaluation loop.
    """
    lams = np.asarray(lams)
    m = lams.size
    if m == 0:
        return lams, np.zeros(0, dtype=complex)

    # union-find clustering on relative complex distance (m ≤ 2n, O(m²) ok)
    parent = np.arange(m)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    scale = np.maximum(np.abs(lams)[:, None], np.abs(lams)[None, :])
    close = np.abs(lams[:, None] - lams[None, :]) < cluster_rtol * np.maximum(scale, 1e-300)
    for i in range(m):
        for j in np.nonzero(close[i, i + 1 :])[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri

    roots = np.array([find(i) for i in range(m)])
    offdiag_tol = 1e-6  # per-mode error from ignoring coupling ~ this ratio
    lam_out = []
    w_out = []
    for r in np.unique(roots):
        idx = np.nonzero(roots == r)[0]
        Vc = vecs[:, idx]
        lam_c = lams[idx].mean()
        A = Vc.T @ ((2.0 * lam_c * M_star + C_star) @ Vc)
        rhs = Vc.T @ F_red
        if idx.size == 1:
            if abs(A[0, 0]) < 1e-30:
                continue  # near-defective singleton, as before
            lam_out.append(lams[idx[0]])
            w_out.append(Vc[idx_obs, 0] * rhs[0] / A[0, 0])
            continue

        diag = np.abs(np.diag(A))
        off = np.abs(A - np.diag(np.diag(A))).max()
        if off < offdiag_tol * max(diag.min(), 1e-300):
            # QZ basis already bilinear-orthogonal: keep exact per-mode poles
            for t, k in enumerate(idx):
                if abs(A[t, t]) < 1e-30:
                    continue
                lam_out.append(lams[k])
                w_out.append(Vc[idx_obs, t] * rhs[t] / A[t, t])
            continue

        # coupled cluster: collapsed block residue at the mean pole
        try:
            coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        lam_out.append(lam_c)
        w_out.append(Vc[idx_obs, :] @ np.atleast_1d(coef))
    return np.array(lam_out), np.array(w_out)


def modal_superposition_eff(
    omega_array: np.ndarray,
    interval_solutions: list[dict],
    F_red: np.ndarray,
    idx_obs: int,
) -> tuple[np.ndarray, float]:
    """Effective + conjugate modal superposition (analytic form).

    QEP modal expansion (right eigenvector only, normalized form):
      Y(ω) = Σ_k [φ_k^T·F · φ_k[idx]] / [a_k·(iω - λ_k)]
    where a_k = 2λ_k·v_k^T·M·v_k + v_k^T·C·v_k  (QEP modal mass, complex form),
    derived from the dual relation between the left and right eigenvectors of
    the QEP companion form (see the paper).

    Using only the effective + conjugate modes → superposition of m_eff (≈ 2-50)
    modes avoids the N_dof inversion.
    """
    Y = np.zeros(omega_array.size, dtype=complex)
    lowers = np.array([s["omega_lower"] for s in interval_solutions])
    t0 = time.perf_counter()
    cache = []
    for sol in interval_solutions:
        lams = sol["lams_full_with_conj"]
        vecs = sol["vecs_full_with_conj"]
        if lams.size == 0:
            cache.append(None)
            continue
        mat = sol["matrices"]
        # Degenerate clusters use basis-invariant block residues
        # (the per-mode a_k form is valid only under the distinct-eigenvalue assumption)
        lam_c, w_c = modal_weights_blockwise(
            lams, vecs, mat["M_star"], mat["C_star"], F_red, idx_obs
        )
        cache.append((lam_c, w_c))
    t_build = time.perf_counter() - t0

    for i, omega in enumerate(omega_array):
        j = int(np.searchsorted(lowers, omega, side="right") - 1)
        j = max(0, min(j, len(interval_solutions) - 1))
        if cache[j] is None:
            Y[i] = 0.0
            continue
        lam_c, w_c = cache[j]
        # Y = Σ_c w_c / (iω - λ_c)   (cluster-collapsed residue sum)
        Y[i] = np.sum(w_c / (1j * omega - lam_c))
    return Y, t_build


# =============================================================================
# Direct method: per-frequency assemble (true reference)
# =============================================================================
def direct_response_per_freq(
    params: SandwichPlateParameters,
    K_base: np.ndarray,
    K_shear: np.ndarray,
    M: np.ndarray,
    omega_array: np.ndarray,
    F_red: np.ndarray,
    idx_obs: int,
) -> np.ndarray:
    """Evaluate G(ω) at each ω → K(ω) = K_base + G·K_shear → D(ω) inversion."""
    Y = np.zeros(omega_array.size, dtype=complex)
    K_base_c = K_base.astype(complex)
    K_shear_c = K_shear.astype(complex)
    for i, omega in enumerate(omega_array):
        f_hz = omega / (2.0 * np.pi)
        G = G_complex_at_freq(f_hz)
        K = K_base_c + G * K_shear_c
        D = K - (omega ** 2) * M
        u = spla.solve(D, F_red)
        Y[i] = u[idx_obs]
    return Y


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    np.set_printoptions(linewidth=140, precision=4)

    print("=" * 78)
    print("G(ω) polynomial fitting + segmentwise complex QEP + ROM")
    print("=" * 78)

    # -------------------------------------------------------
    # Stage E — G(ω) sampling
    # -------------------------------------------------------
    print("\n[Stage E] G(ω) fine-grid sampling")
    f_min, f_max = 1.0, 1500.0
    n_sample = 400
    omega, G_re, G_im = sample_G_omega(f_min, f_max, n_sample)
    print(f"  f range: {f_min}-{f_max} Hz, {n_sample} samples")
    print(f"  Re[G]: {G_re.min():.3e} – {G_re.max():.3e} Pa")
    print(f"  Im[G]: {G_im.min():.3e} – {G_im.max():.3e} Pa")

    # -------------------------------------------------------
    # Stage F — piecewise quadratic fitting
    # -------------------------------------------------------
    print("\n[Stage F] piecewise quadratic fitting")
    spline_re, spline_im, boundaries = fit_G_polynomial(omega, G_re, G_im, n_segments=5)
    print(f"  segments: {len(spline_re.segments)}")
    print(f"  breakpoints (Hz): {boundaries / (2 * np.pi)}")
    G_re_fit = evaluate_spline_at(spline_re, omega)
    G_im_fit = evaluate_spline_at(spline_im, omega)
    err_re = np.abs(G_re - G_re_fit) / np.maximum(np.abs(G_re), 1e-30) * 100
    err_im = np.abs(G_im - G_im_fit) / np.maximum(np.abs(G_im), 1e-30) * 100
    print(f"  Re[G] fit error: max {err_re.max():.2f}%, mean {err_re.mean():.2f}%")
    print(f"  Im[G] fit error: max {err_im.max():.2f}%, mean {err_im.mean():.2f}%")

    # -------------------------------------------------------
    # Stage G — segmentwise ω-independent complex matrix reconstruction
    # -------------------------------------------------------
    print("\n[Stage G] segmentwise complex matrix reconstruction (M*, C*, K*)")
    params = paper_5_1_params(nx=8, ny=8)
    K_base, K_shear = assemble_K_base_and_shear(params)
    print(f"  K_base, K_shear shape = {K_base.shape}")

    # M is independent of G
    M_full, _ = assemble_global(params, complex(1.0, 0.0))  # M independent of G
    F_full = np.zeros(M_full.shape[0], dtype=complex)
    cx = params.nx // 2
    cy = params.ny // 2
    n_nodes_x = params.nx + 1
    cw_full = 7 * (cx + cy * n_nodes_x) + 4
    F_full[cw_full] = 1.0
    M_red, K_red0, F_red, free_dofs = apply_boundary_conditions(
        M_full, K_base.astype(complex), F_full, params, bc="free"
    )
    cw_red = int(np.where(np.asarray(free_dofs) == cw_full)[0][0])
    print(f"  reduced n_dof = {M_red.shape[0]}, observation DOF = {cw_red}")

    # K_base, K_shear reduced as well
    free_idx = np.asarray(free_dofs)
    K_base_red = K_base[np.ix_(free_idx, free_idx)]
    K_shear_red = K_shear[np.ix_(free_idx, free_idx)]
    M_red_real = (M_red.real + M_red.real.T) / 2.0  # symmetrize (M itself is real)

    interval_mats = assemble_complex_per_interval_amichi(
        spline_re, spline_im, M_red_real, K_base_red, K_shear_red,
    )
    print(f"  interval count: {len(interval_mats)}")
    for k, mat in enumerate(interval_mats):
        f_lo = mat["omega_lower"] / (2 * np.pi)
        f_hi = mat["omega_upper"] / (2 * np.pi)
        print(f"    interval {k}: [{f_lo:>7.1f}, {f_hi:>7.1f}] Hz, "
              f"M*[0,0]={mat['M_star'][0, 0]:.3e}, "
              f"K*[0,0]={mat['K_star'][0, 0]:.3e}")

    # -------------------------------------------------------
    # Stage H — segmentwise complex QEP + effective-eigenvalue selection
    # -------------------------------------------------------
    print("\n[Stage H] segmentwise complex QEP + effective-eigenvalue selection")
    t0 = time.perf_counter()
    interval_sols = solve_qep_per_interval(interval_mats, damping_tol=1e-6)
    t_qep_total = time.perf_counter() - t0
    print(f"  QEP solve total: {t_qep_total*1000:.1f} ms")
    for sol in interval_sols:
        n_eff = sol["lams_effective"].size
        n_total = sol["lams_all"].size
        f_lo = sol["omega_lower"] / (2 * np.pi)
        f_hi = sol["omega_upper"] / (2 * np.pi)
        print(f"    interval {sol['interval_idx']} [{f_lo:>7.1f}-{f_hi:>7.1f} Hz]: "
              f"{n_total} eigvals, {n_eff} effective (after selection)")
        if n_eff > 0:
            for k in range(min(5, n_eff)):
                lam = sol["lams_effective"][k]
                f_k = lam.imag / (2 * np.pi)
                eta_k = -2 * lam.real / lam.imag if abs(lam.imag) > 0 else 0
                print(f"      λ_{k}: {lam:.3e}  →  f={f_k:.2f} Hz, η_eq={eta_k:.4f}")

    # -------------------------------------------------------
    # Stage I — segmentwise response + Direct comparison + ROM
    # -------------------------------------------------------
    print("\n[Stage I] FRF comparison: Direct vs Polynomial QEP vs ROM")
    f_eval = np.linspace(5.0, 500.0, 200)
    omega_eval = 2.0 * np.pi * f_eval

    # Direct method (per-frequency assembly)
    print("\n  [DIRECT] per-frequency K(ω) inversion...")
    t0 = time.perf_counter()
    Y_direct = direct_response_per_freq(
        params, K_base_red, K_shear_red, M_red_real, omega_eval, F_red, cw_red,
    )
    t_direct = time.perf_counter() - t0
    print(f"    CPU: {t_direct*1000:.1f} ms ({t_direct*1000/len(f_eval):.2f} ms/freq)")

    # Polynomial QEP (segmentwise M*, C*, K* full inversion)
    print("\n  [POLY-QEP] segmentwise (M*, C*, K*) full inversion...")
    t0 = time.perf_counter()
    Y_polyqep = modal_response_per_interval(omega_eval, interval_sols, F_red, cw_red)
    t_polyqep = time.perf_counter() - t0
    err_pq_arr = np.abs(Y_direct - Y_polyqep) / np.maximum(np.abs(Y_direct), 1e-30) * 100
    mask = np.abs(Y_direct) > 0.01 * np.abs(Y_direct).max()
    err_pq_max = err_pq_arr[mask].max()
    err_pq_mean = err_pq_arr[mask].mean()
    print(f"    CPU: {t_polyqep*1000:.1f} ms ({t_polyqep*1000/len(f_eval):.2f} ms/freq)")
    print(f"    vs Direct: max err {err_pq_max:.3f}%, mean err {err_pq_mean:.3f}%")

    # effective + conjugate modal superposition (analytic)
    print("\n  [MODAL-SUP] effective + conjugate modal superposition...")
    t0 = time.perf_counter()
    Y_modal, t_build_modal = modal_superposition_eff(
        omega_eval, interval_sols, F_red, cw_red,
    )
    t_modal_total = time.perf_counter() - t0
    err_modal = np.abs(Y_direct - Y_modal) / np.maximum(np.abs(Y_direct), 1e-30) * 100
    print(f"    build: {t_build_modal*1000:.1f} ms │ total: {t_modal_total*1000:.1f} ms")
    print(f"    speedup vs Direct: ×{t_direct/max(t_modal_total,1e-9):.1f}")
    print(f"    vs Direct: max err {err_modal[mask].max():.3f}%, "
          f"mean err {err_modal[mask].mean():.3f}%")

    # NOTE: the rigid-body-corrected modal sum (free-free) is handled fully by
    # cmame_d1_mode_acceleration.py via mode acceleration (F_perp + K_base⁺ static
    # term + corrected modal sum); its figure (cmame_mode_acceleration.png) is the
    # one used in the paper. No separate correction is computed here.

    # ROM per interval — reuse the already-solved QEP eigenvectors (build cost ≈ 0)
    # Note: the ROM reference is Poly-QEP (segmentwise full inversion), not Direct!
    # The Direct vs Poly-QEP difference is set by the G(ω) polynomial fit error (~3%).
    # The true ROM performance must be measured as ROM vs Poly-QEP.
    print("\n  [ROM-REUSE] (reusing Stage H QEP eigvecs)")
    print("    NOTE: ROM error = vs Poly-QEP (true ROM performance)")
    print("          Direct error = vs Direct (includes G(ω) fit error)")
    rom_results = {}
    mask_pq = np.abs(Y_polyqep) > 0.01 * np.abs(Y_polyqep).max()
    for m in [10, 20, 50, 100]:
        for use_eff in [False, True]:
            t0 = time.perf_counter()
            Y_rom, t_build_rom = rom_response_per_interval_reuse(
                omega_eval, interval_sols, F_red, cw_red, m=m,
                use_effective_only=use_eff,
            )
            t_rom = time.perf_counter() - t0
            # ROM vs Poly-QEP (true ROM accuracy)
            err_pq = np.abs(Y_polyqep - Y_rom) / np.maximum(np.abs(Y_polyqep), 1e-30) * 100
            max_err_pq = err_pq[mask_pq].max()
            mean_err_pq = err_pq[mask_pq].mean()
            # ROM vs Direct (for reference)
            err_d = np.abs(Y_direct - Y_rom) / np.maximum(np.abs(Y_direct), 1e-30) * 100
            max_err_d = err_d[mask].max()
            speedup = t_direct / max(t_rom, 1e-9)
            label = "eff+conj only" if use_eff else "all eigvecs"
            rom_results[(use_eff, m)] = (Y_rom, t_rom, max_err_pq, mean_err_pq, t_build_rom, max_err_d)
            print(f"    m={m:>3} [{label:>14}]: "
                  f"build {t_build_rom*1000:>5.1f} ms │ "
                  f"total {t_rom*1000:>6.1f} ms │ "
                  f"speedup ×{speedup:>5.1f} │ "
                  f"err vs PolyQEP {max_err_pq:>6.2f}% │ "
                  f"err vs Direct {max_err_d:>6.2f}%")

    # -------------------------------------------------------
    # Figures
    # -------------------------------------------------------
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    # Fig 1: G(ω) fit
    fig, axes = plt.subplots(1, 2, figsize=(19.5, 6.8))
    f_arr = omega / (2 * np.pi)
    axes[0].plot(f_arr, G_re / 1e6, "k-", lw=2, label="Re[G] data")
    axes[0].plot(f_arr, G_re_fit / 1e6, "r--", lw=1.2, label="piecewise quadratic")
    for b in boundaries / (2 * np.pi):
        axes[0].axvline(b, color="gray", lw=0.5, ls=":")
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("Re[G] [MPa]")
    axes[0].set_title("Re[G(ω)] piecewise quadratic fit")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(f_arr, G_im / 1e6, "k-", lw=2, label="Im[G] data")
    axes[1].plot(f_arr, G_im_fit / 1e6, "r--", lw=1.2, label="piecewise quadratic")
    for b in boundaries / (2 * np.pi):
        axes[1].axvline(b, color="gray", lw=0.5, ls=":")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Im[G] [MPa]")
    axes[1].set_title("Im[G(ω)] piecewise quadratic fit")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase3_G_fitting.png", dpi=140)
    plt.close(fig)
    print(f"\n[saved] figures/phase3_G_fitting.png")

    # Fig 2: QEP eigenvalue map (segmentwise)
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    for sol in interval_sols:
        lams = sol["lams_all"]
        finite = np.isfinite(lams)
        ax.plot(lams[finite].real, lams[finite].imag / (2 * np.pi),
                ".", ms=3, alpha=0.4,
                label=f"interval {sol['interval_idx']} all")
        eff = sol["lams_effective"]
        if eff.size > 0:
            ax.plot(eff.real, eff.imag / (2 * np.pi),
                    "o", ms=8, mec="k", mfc="none",
                    label=f"interval {sol['interval_idx']} effective ({eff.size})")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Re(λ) [1/s]  (damping)")
    ax.set_ylabel("Im(λ) / 2π [Hz]  (frequency)")
    ax.set_title("Interval QEP eigenvalues and\neffective-eigenvalue selection")
    ax.set_ylim(0, 600)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11.7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "phase3_qep_eigenvalues.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] figures/phase3_qep_eigenvalues.png")

    # Fig 3: FRF overlay (2x in-plot font + show both raw and rb-corrected modal sums)
    with plt.rc_context({
        "font.size": 40 / 3,        # 20 * 2/3
        "axes.titlesize": 16.0,     # 24 * 2/3
        "axes.labelsize": 40 / 3,   # 20 * 2/3
        "xtick.labelsize": 40 / 3,
        "ytick.labelsize": 40 / 3,
        "legend.fontsize": 32 / 3,  # 16 * 2/3
    }):
        fig, axes = plt.subplots(2, 1, figsize=(16.5, 11.2), sharex=True)
        axes[0].semilogy(f_eval, np.abs(Y_direct), "k-", lw=1.5,
                         label="Direct (per-freq assemble)")
        axes[0].semilogy(f_eval, np.abs(Y_polyqep), "b--", lw=1.0,
                         label="Polynomial QEP (full inversion)")
        axes[0].semilogy(f_eval, np.abs(Y_modal), "g:", lw=1.2,
                         label=f"Modal sup (raw, eff+conj, "
                               f"max err {err_modal[mask].max():.2f}%)")
        # plot the representative ROM only (m=50, all)
        rep = rom_results.get((False, 50))
        if rep:
            Y_rom, _, max_err_pq, _, _, _ = rep
            axes[0].semilogy(f_eval, np.abs(Y_rom), "r-.", alpha=0.8,
                             label=f"ROM m=50 reuse (vs PolyQEP {max_err_pq:.2f}%)")
        axes[0].set_ylabel("|Y| [m/N]")
        axes[0].set_title("FRF: Direct vs Poly-QEP vs Modal-Sup vs ROM "
                          "(paper §5.1, 567-DOF)")
        axes[0].grid(True, alpha=0.3, which="both")
        axes[0].legend()

        err_pq_full = np.abs(Y_direct - Y_polyqep) / np.maximum(np.abs(Y_direct), 1e-30) * 100
        axes[1].semilogy(f_eval, err_pq_full, "b-", label="Poly-QEP vs Direct (G(ω) fit error)")
        err_modal_full = np.abs(Y_direct - Y_modal) / np.maximum(np.abs(Y_direct), 1e-30) * 100
        axes[1].semilogy(f_eval, err_modal_full, "g:", lw=1.2,
                         label="Modal sup (raw) vs Direct")
        for (use_eff, m), vals in rom_results.items():
            if m not in [20, 50, 100] or use_eff:
                continue
            Y_rom = vals[0]
            err = np.abs(Y_polyqep - Y_rom) / np.maximum(np.abs(Y_polyqep), 1e-30) * 100
            axes[1].semilogy(f_eval, err, "--", alpha=0.6,
                             label=f"ROM m={m} vs Poly-QEP")
        axes[1].set_xlabel("Frequency [Hz]")
        axes[1].set_ylabel("Relative error [%]")
        axes[1].grid(True, alpha=0.3, which="both")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "phase3_polyqep_frf_overlay.png", dpi=140)
        plt.close(fig)
    print(f"[saved] figures/phase3_polyqep_frf_overlay.png")

    print("\n" + "=" * 78)
    print("All 5 stages complete.")
    print("=" * 78)
    print(f"\nKey results:")
    print(f"  G(ω) fit: Re max {err_re.max():.2f}%, Im max {err_im.max():.2f}%")
    print(f"  Direct method:               {t_direct*1000:>6.0f} ms")
    print(f"  Polynomial QEP (full):       {t_polyqep*1000:>6.0f} ms  "
          f"(error vs Direct: {err_pq_max:.3f}%)")
    print(f"  Modal-Sup (eff+conj):        {t_modal_total*1000:>6.0f} ms  "
          f"speedup ×{t_direct/max(t_modal_total,1e-9):.1f}  "
          f"(error: {err_modal[mask].max():.3f}%)")
    for (use_eff, m), vals in rom_results.items():
        Y_rom, t_rom, max_err_pq, _, t_build, max_err_d = vals
        label = "eff+conj" if use_eff else "all     "
        print(f"  ROM m={m:>3} [{label}]:           {t_rom*1000:>6.0f} ms  "
              f"speedup ×{t_direct/max(t_rom,1e-9):>4.1f}  "
              f"(vs PolyQEP {max_err_pq:.2f}%, vs Direct {max_err_d:.2f}%)")


if __name__ == "__main__":
    main()
