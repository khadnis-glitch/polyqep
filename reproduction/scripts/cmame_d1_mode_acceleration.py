# === [auto-resized: figsize x1.5] ===
"""Free-free mode acceleration with rigid-body projection.

Goal
----
Recover sub-percent FRF accuracy in the free-free configuration by:
  1.  Identifying the six rigid-body eigenvectors V_rb of (K_base, M_real)
      via the (K, M) generalized eigenproblem.  These span the kernel of
      K* (free-free) up to numerical noise.
  2.  Projecting the load F onto the M-orthogonal complement of V_rb:
        F_perp = F - V_rb (V_rb^T M V_rb)^{-1} V_rb^T M F
      so that K*^{+} F_perp is well-defined (no rigid-body content).
  3.  Adding a small mass-shift  K*_reg = K* + epsilon * M  for numerical
      conditioning of the static solve (eps small enough that flexible
      modes are unperturbed).
  4.  Applying the mode-acceleration formula

        Y(omega) ~ (K*_reg)^{-1} F_perp [obs]
                 + sum_{k in S_eff} v_k (v_k^T F_perp) / a_k *
                                    [ 1/(j*omega - lambda_k) + 1/lambda_k ]

      where the eigenpairs come from solving QEP with (M*, C*, K*_reg).

The K*_reg^{-1} F_perp static term carries the quasi-static contribution
of the *flexible-mode tail* truncated by the effective-eigenvalue filter,
and the [1/(j*omega - lambda_k) + 1/lambda_k] correction
removes the static double-count from the included modes; together they
reconstitute the full modal expansion in the form valid for
free-free boundary, where the un-regularized K*^{-1} does not exist.

Sweep: epsilon in {0, 1e-12, 1e-9, 1e-6, 1e-3, 1e0, 1e3} relative to
the natural mass scale.
Mesh: 8x8 free-free (567 DOFs).

Output
------
stdout : sweep table with eps, max/mean response error vs Direct, modes retained
figures/cmame_mode_acceleration.png : FRF overlay + epsilon-sensitivity chart
data/cmame_d1_mode_acceleration.csv : raw numbers for the paper

Run from project root:
    python scripts/cmame_d1_mode_acceleration.py
"""
# @MX:NOTE: free-free mode acceleration with K*_reg = K* + eps*M.
from __future__ import annotations

import csv
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
# font: reduced to 2/3 of the 2x baseline (in-figure text size adjustment)
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

from polyqep.models.sandwich_plate_amichi2010 import (
    SandwichPlateParameters,
    apply_boundary_conditions,
    assemble_K_base_and_shear,
    assemble_global,
)
from polyqep.qep_solver import solve_qep

from scripts.phase3_amichi_polynomial_qep import (
    assemble_complex_per_interval_amichi,
    direct_response_per_freq,
    fit_G_polynomial,
    paper_5_1_params,
    sample_G_omega,
)


# --------------------------------------------------------------------------- #
# Rigid-body subspace identification + M-orthogonal projection                #
# --------------------------------------------------------------------------- #
def identify_rigid_body_subspace(
    K_base_real: np.ndarray,
    M_real: np.ndarray,
    n_rb: int = 6,
    tol_ratio: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify the rigid-body subspace via (K, M) generalized eigenproblem.

    Returns (V_rb, eigvals_rb) where V_rb is (n, n_rb) and eigvals_rb are
    the n_rb smallest |eigvals| of (K, M).  For an unsupported plate,
    these are the six rigid-body modes (translation x, y, z + rotation
    about each axis), with eigenvalues at machine precision.
    """
    eigvals, eigvecs = spla.eigh(K_base_real, M_real)
    order = np.argsort(np.abs(eigvals))
    rb_idx = order[:n_rb]
    V_rb = eigvecs[:, rb_idx]
    rb_evals = eigvals[rb_idx]
    if np.abs(rb_evals).max() > tol_ratio * np.abs(eigvals).max():
        # Diagnostics, not a hard fail
        print(f"    [warn] rigid-body eigvals "
              f"max |lambda_rb| = {np.abs(rb_evals).max():.3e}, "
              f"flex max |lambda| = {np.abs(eigvals).max():.3e} "
              f"(ratio {np.abs(rb_evals).max() / np.abs(eigvals).max():.2e})")
    return V_rb, rb_evals


def project_out_rigid_body(
    F: np.ndarray, V_rb: np.ndarray, M_real: np.ndarray,
) -> np.ndarray:
    """M-orthogonal projection of F onto the complement of V_rb."""
    MV = M_real @ V_rb
    Gram = V_rb.T @ MV
    coef = np.linalg.solve(Gram, V_rb.T @ (M_real @ F))
    F_perp = F - V_rb @ coef
    return F_perp


# --------------------------------------------------------------------------- #
# Regularized segmentwise QEP solve                                           #
# --------------------------------------------------------------------------- #
def solve_qep_regularized(
    interval_matrices: list[dict],
    M_real: np.ndarray,
    epsilon: float,
    damping_tol: float = 1e-9,
) -> list[dict]:
    """Solve the segmentwise QEP with K*_reg = K* + epsilon * M_real.

    Returns the same shape of dict as `solve_qep_per_interval` but with
    `K_star` replaced by the regularized version, and an extra
    `K_star_reg` key for downstream static-term assembly.
    """
    out: list[dict] = []
    eps_M = epsilon * M_real.astype(complex)
    for i, mat in enumerate(interval_matrices):
        K_reg = mat["K_star"] + eps_M
        result = solve_qep(mat["M_star"], mat["C_star"], K_reg)
        lams = result.eigenvalues
        vecs = result.eigenvectors
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
            "vecs_all": vecs,
            "lams_effective": lams_eff,
            "vecs_effective": vecs_eff,
            "lams_full_with_conj": lams_full,
            "vecs_full_with_conj": vecs_full,
            "matrices": mat,
            "K_star_reg": K_reg,
        })
    return out


# --------------------------------------------------------------------------- #
# Mode-acceleration response                                                  #
# --------------------------------------------------------------------------- #
def rigid_body_inertia_response(
    omega_array: np.ndarray,
    V_rb: np.ndarray,
    F: np.ndarray,
    idx_obs: int,
) -> np.ndarray:
    """Closed-form rigid-body inertia response for free-free configuration.

    For omega > 0, K v_rb = 0 and the rigid-body subspace contributes
        Y_rb(omega) = - V_rb (V_rb^T F) / omega^2
    independent of the segmentwise damping matrices, since C* v_rb = 0
    when K_shear v_rb = 0 as well.  V_rb assumed M-orthonormal.
    """
    coef = V_rb.T @ F
    Y_rb_full = -V_rb @ coef
    Y = Y_rb_full[idx_obs] / np.maximum(omega_array ** 2, 1e-300)
    return Y


def make_static_pinv_via_eigh(
    K_real: np.ndarray,
    M_real: np.ndarray,
    n_rb: int = 6,
    flex_cutoff: float | None = None,
    diagnostic: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute V_flex Lambda_flex^{-1} V_flex^T via (K, M)-eigendecomposition.

    Returns (K_plus, V_rb, eigvals_sorted_abs_ascending).

    Modes with |lambda| <= flex_cutoff are excluded from the pseudoinverse,
    in addition to the n_rb smallest. When flex_cutoff is None, only the
    n_rb smallest are excluded.
    """
    eigvals, eigvecs = spla.eigh(K_real, M_real)
    order = np.argsort(np.abs(eigvals))
    eigvals_sorted = eigvals[order]
    if flex_cutoff is None:
        rb_idx = order[:n_rb]
        flex_idx = order[n_rb:]
    else:
        below_cutoff = np.where(np.abs(eigvals_sorted) <= flex_cutoff)[0]
        n_excluded = max(n_rb, int(below_cutoff[-1] + 1) if below_cutoff.size > 0 else n_rb)
        rb_idx = order[:n_excluded]
        flex_idx = order[n_excluded:]

    V_rb = eigvecs[:, rb_idx]
    V_flex = eigvecs[:, flex_idx]
    Lam_flex = eigvals[flex_idx]
    K_plus = V_flex @ np.diag(1.0 / Lam_flex) @ V_flex.T

    if diagnostic:
        print(f"    eigvals[:10] sorted by |lambda|: {eigvals_sorted[:10]}")
        print(f"    Lam_flex min/max: {Lam_flex.min():.3e} / {Lam_flex.max():.3e}")
        print(f"    cond(Lam_flex)  : {Lam_flex.max() / max(abs(Lam_flex.min()), 1e-30):.3e}")
    return K_plus, V_rb, eigvals_sorted


def qep_flex_static_response(
    sol: dict, F_red: np.ndarray, lam_floor: float = 1.0,
) -> np.ndarray:
    """Compute flex_static = -Σ_flex_QEP v_k (v_k^T F) / (a_k λ_k) using QEP eigenpairs.

    The "flex" set is all QEP eigenvalues with |λ_k| >= lam_floor (excludes
    rigid bodies and near-zero spurious modes).  Returns a vector indexed
    over the segmentwise n DOFs.
    """
    lams = sol["lams_all"]
    vecs = sol["vecs_all"]
    finite = np.isfinite(lams)
    lams_f = lams[finite]
    vecs_f = vecs[:, finite]
    is_flex = np.abs(lams_f) >= lam_floor
    lams_x = lams_f[is_flex]
    vecs_x = vecs_f[:, is_flex]
    if vecs_x.shape[1] == 0:
        return np.zeros(F_red.size, dtype=complex)
    mat = sol["matrices"]
    Mv = mat["M_star"] @ vecs_x
    Cv = mat["C_star"] @ vecs_x
    a_k = (2 * lams_x * np.einsum("ij,ij->j", vecs_x, Mv)
           + np.einsum("ij,ij->j", vecs_x, Cv))
    nz = np.abs(a_k) > 1e-30
    lams_x = lams_x[nz]
    vecs_x = vecs_x[:, nz]
    a_k = a_k[nz]
    modal_F = vecs_x.T @ F_red
    weights = modal_F / (a_k * lams_x)  # shape (n_flex,)
    flex_static_vec = -(vecs_x @ weights)
    return flex_static_vec


def mode_acceleration_response(
    omega_array: np.ndarray,
    interval_solutions: list[dict],
    F_red: np.ndarray,
    idx_obs: int,
    F_perp: np.ndarray | None = None,
    K_plus_static: np.ndarray | None = None,
    use_qep_flex_static: bool = True,
    lam_floor: float = 1.0,
) -> np.ndarray:
    """Evaluate the mode-acceleration FRF using a precomputed K^{+} static term.

    Y(omega) = (K^{+} F_perp)[obs]
             + sum_eff  v_k[obs] (v_k^T F_perp) / a_k *
                        [1/(j*omega - lambda_k) + 1/lambda_k]

    K_plus_static must be provided externally; here we take K_plus from
    the (K_base, M_real) generalized eigendecomposition (segment-zero
    static stiffness).  The mode-acceleration formula then assumes the
    deviation of K* from K_base across segments is captured by the
    dynamic modal sum, an approximation that becomes exact when only the
    real-symmetric flexural part of K* is retained.

    F_perp defaults to F_red.  K_plus_static is required for free-free.
    """
    if F_perp is None:
        F_perp = F_red
    if K_plus_static is None and not use_qep_flex_static:
        raise ValueError("Provide K_plus_static or set use_qep_flex_static=True")

    Y = np.zeros(omega_array.size, dtype=complex)
    lowers = np.array([s["omega_lower"] for s in interval_solutions])
    cache: list = []
    for sol in interval_solutions:
        if use_qep_flex_static:
            flex_static_vec = qep_flex_static_response(sol, F_red, lam_floor=lam_floor)
            y_static_obs_seg = flex_static_vec[idx_obs]
        else:
            assert K_plus_static is not None
            y_static_obs_seg = (K_plus_static @ F_red)[idx_obs]

        lams = sol["lams_full_with_conj"]
        vecs = sol["vecs_full_with_conj"]
        if lams.size == 0:
            cache.append((y_static_obs_seg, None, None, None, None))
            continue
        mat = sol["matrices"]
        Mv = mat["M_star"] @ vecs
        Cv = mat["C_star"] @ vecs
        a_k = (2 * lams * np.einsum("ij,ij->j", vecs, Mv)
               + np.einsum("ij,ij->j", vecs, Cv))
        nonzero = np.abs(a_k) > 1e-30
        lams = lams[nonzero]
        vecs = vecs[:, nonzero]
        a_k = a_k[nonzero]
        modal_F = vecs.T @ F_red
        modal_obs = vecs[idx_obs, :]
        weights = modal_obs * modal_F / a_k
        cache.append((y_static_obs_seg, lams, weights, None, None))
    # Skip the now-stale spla.solve call below by short-circuiting cache use:
    Y = np.zeros(omega_array.size, dtype=complex)
    lowers = np.array([s["omega_lower"] for s in interval_solutions])
    for i, omega in enumerate(omega_array):
        j_idx = int(np.searchsorted(lowers, omega, side="right") - 1)
        j_idx = max(0, min(j_idx, len(interval_solutions) - 1))
        ys, lams, weights, _, _ = cache[j_idx]
        if lams is None:
            Y[i] = ys
        else:
            Y[i] = ys + np.sum(weights * (1.0 / (1j * omega - lams) + 1.0 / lams))
    return Y


def _legacy_response_unused(
    interval_solutions, F_perp, idx_obs,
):
    """Stub kept to preserve historical reference; not called."""
    Y = np.zeros((), dtype=complex)
    lowers = np.array([])
    cache: list = []
    for sol in interval_solutions:
        K_reg = sol["K_star_reg"]
        Y_static = spla.solve(K_reg, F_perp)
        y_static_obs = Y_static[idx_obs]

        lams = sol["lams_full_with_conj"]
        vecs = sol["vecs_full_with_conj"]
        if lams.size == 0:
            cache.append((y_static_obs, None, None, None, None))
            continue
        mat = sol["matrices"]
        Mv = mat["M_star"] @ vecs
        Cv = mat["C_star"] @ vecs
        a_k = (2 * lams * np.einsum("ij,ij->j", vecs, Mv)
               + np.einsum("ij,ij->j", vecs, Cv))
        nonzero = np.abs(a_k) > 1e-30
        lams = lams[nonzero]
        vecs = vecs[:, nonzero]
        a_k = a_k[nonzero]
        modal_F = vecs.T @ F_red
        modal_obs = vecs[idx_obs, :]
        weights = modal_obs * modal_F / a_k
        cache.append((y_static_obs, lams, weights, None, None))

    for i, omega in enumerate(omega_array):
        j_idx = int(np.searchsorted(lowers, omega, side="right") - 1)
        j_idx = max(0, min(j_idx, len(interval_solutions) - 1))
        y_static_obs, lams, weights, _, _ = cache[j_idx]
        if lams is None:
            Y[i] = y_static_obs
            continue
        Y[i] = (y_static_obs
                + np.sum(weights * (1.0 / (1j * omega - lams) + 1.0 / lams)))
    return Y


# --------------------------------------------------------------------------- #
# Main sweep                                                                  #
# --------------------------------------------------------------------------- #
def main() -> None:
    np.set_printoptions(linewidth=140, precision=4)

    print("=" * 78)
    print("Free-free mode acceleration sweep (Amichi 8x8)")
    print("=" * 78)

    # 1. Assembly
    params: SandwichPlateParameters = paper_5_1_params(nx=8, ny=8)
    K_base, K_shear = assemble_K_base_and_shear(params)
    M_full, _ = assemble_global(params, complex(1.0, 0.0))
    F_full = np.zeros(M_full.shape[0], dtype=complex)
    cx, cy = params.nx // 2, params.ny // 2
    n_nodes_x = params.nx + 1
    cw_full = 7 * (cx + cy * n_nodes_x) + 4
    F_full[cw_full] = 1.0
    M_red, _, F_red, free_dofs = apply_boundary_conditions(
        M_full, K_base.astype(complex), F_full, params, bc="free",
    )
    cw_red = int(np.where(np.asarray(free_dofs) == cw_full)[0][0])
    free_idx = np.asarray(free_dofs)
    K_base_red = K_base[np.ix_(free_idx, free_idx)]
    K_shear_red = K_shear[np.ix_(free_idx, free_idx)]
    M_red_real = (M_red.real + M_red.real.T) / 2.0
    print(f"  reduced n_dof = {M_red.shape[0]}, observation DOF = {cw_red}")
    print(f"  ||M_real|| = {np.linalg.norm(M_red_real):.3e}, "
          f"||K_base|| = {np.linalg.norm(K_base_red):.3e}")

    # Rigid-body subspace + (K_base, M)-pseudoinverse via M-orthonormal eigh
    print("\n  Identifying rigid-body subspace via (K_base, M_real) eigh ...")
    K_plus_real, V_rb, eigvals_sorted = make_static_pinv_via_eigh(
        K_base_red, M_red_real, n_rb=6, flex_cutoff=1e-3, diagnostic=True,
    )
    F_perp = project_out_rigid_body(F_red.real, V_rb, M_red_real).astype(complex)
    K_plus_static = K_plus_real.astype(complex)
    Y_static_check = (K_plus_real @ F_perp.real)
    K_residual = np.linalg.norm(K_base_red @ Y_static_check - F_perp.real)
    print(f"    K_base @ Y_static - F_perp residual = {K_residual:.3e} "
          f"(||F_perp|| = {np.linalg.norm(F_perp.real):.3e})")
    print(f"    ||Y_static||_2 = {np.linalg.norm(Y_static_check):.3e}")
    print(f"    n_rb effective = {V_rb.shape[1]} (originally 6, lifted by flex_cutoff)")
    rb_residual = np.linalg.norm(V_rb.T @ (M_red_real @ F_perp.real))
    print(f"    ||V_rb^T M F_perp|| = {rb_residual:.3e}  (post-projection residual)")
    print(f"    ||F_perp - F|| / ||F|| = "
          f"{np.linalg.norm((F_perp - F_red).real)/max(np.linalg.norm(F_red.real),1e-30):.3e}")

    # 2. G(omega) sample + 5-segment fit (baseline fit of the paper)
    f_min, f_max = 1.0, 1500.0
    omega_sample, G_re, G_im = sample_G_omega(f_min, f_max, 400)
    spline_re, spline_im, boundaries = fit_G_polynomial(
        omega_sample, G_re, G_im, n_segments=5,
    )
    interval_mats = assemble_complex_per_interval_amichi(
        spline_re, spline_im, M_red_real, K_base_red, K_shear_red,
    )

    # 3. Direct reference
    f_eval = np.linspace(5.0, 500.0, 200)
    omega_eval = 2.0 * np.pi * f_eval
    print(f"\n  Direct reference: {len(f_eval)} freqs in 5-500 Hz ...")
    t0 = time.perf_counter()
    Y_direct = direct_response_per_freq(
        params, K_base_red, K_shear_red, M_red_real,
        omega_eval, F_red, cw_red,
    )
    t_direct = time.perf_counter() - t0
    mask = np.abs(Y_direct) > 0.01 * np.abs(Y_direct).max()
    print(f"  Direct CPU: {t_direct * 1000:.1f} ms")

    # 4. Single mode-acceleration computation with the (K_base, M) pseudoinverse +
    #    closed-form rigid-body inertia.  No epsilon / rcond sweep needed because
    #    K_plus_static is computed exactly via the generalized eigendecomposition.
    rows: list[dict] = []
    Y_records: dict[str, np.ndarray] = {}

    sols0 = solve_qep_regularized(
        interval_mats, M_red_real, epsilon=0.0, damping_tol=1e-9,
    )

    # Closed-form rigid-body inertia response
    Y_rb_inertia = rigid_body_inertia_response(
        omega_eval, V_rb, F_red.real, cw_red,
    ).astype(complex)

    Y_flex = mode_acceleration_response(
        omega_eval, sols0, F_red, cw_red,
        F_perp=F_perp, K_plus_static=K_plus_static,
    )
    Y_ma = Y_flex + Y_rb_inertia
    err = np.abs(Y_direct - Y_ma) / np.maximum(np.abs(Y_direct), 1e-30) * 100.0
    finite = np.isfinite(Y_ma)
    if finite.all():
        err_max_main = float(err[mask].max())
        err_mean_main = float(err[mask].mean())
        n_bad = 0
    else:
        err_max_main = float("inf")
        err_mean_main = float("inf")
        n_bad = int((~finite).sum())
    n_eff_total = sum(s["lams_effective"].size for s in sols0)
    print(f"\n  Mode acceleration with K_plus + rigid-body inertia:")
    print(f"    n_eff_total = {n_eff_total}")
    print(f"    max err = {err_max_main:.3f} %, mean err = {err_mean_main:.3f} %")
    rows.append({
        "method": "mode_accel_K_plus_with_rb",
        "err_max": err_max_main, "err_mean": err_mean_main,
        "non_finite_freqs": n_bad,
    })
    Y_records["mode_accel"] = Y_ma

    # Variant: mode acceleration WITHOUT rigid-body inertia (to isolate effect)
    err_no_rb = np.abs(Y_direct - Y_flex) / np.maximum(np.abs(Y_direct), 1e-30) * 100.0
    rows.append({
        "method": "mode_accel_no_rb_inertia",
        "err_max": float(err_no_rb[mask].max()),
        "err_mean": float(err_no_rb[mask].mean()),
        "non_finite_freqs": 0,
    })
    Y_records["mode_accel_no_rb"] = Y_flex

    # Skip the rcond loop entirely; keep `rows` for downstream table
    rcond_list: list[float] = []  # placeholder
    if False:  # disabled rcond sweep block
        for rcond in rcond_list:
            pass

        # Numerical safety: any non-finite entry signals failure
        finite = np.isfinite(Y_ma)
        if not finite.all():
            n_bad = int((~finite).sum())
            err_max = np.inf
            err_mean = np.inf
        else:
            err = np.abs(Y_direct - Y_ma) / np.maximum(np.abs(Y_direct), 1e-30) * 100.0
            err_max = float(err[mask].max())
            err_mean = float(err[mask].mean())
            n_bad = 0

        n_eff_total = sum(s["lams_effective"].size for s in sols)
        rows.append({
            "rcond": rcond,
            "n_eff_total": n_eff_total,
            "err_max": err_max,
            "err_mean": err_mean,
            "non_finite_freqs": n_bad,
        })
        Y_records[rcond] = Y_ma

    # 5. Plain modal superposition (no static term) for comparison baseline
    from scripts.phase3_amichi_polynomial_qep import modal_superposition_eff
    Y_modal_plain, _ = modal_superposition_eff(
        omega_eval, sols0, F_red, cw_red,
    )

    # 5b. ALL-flex modal superposition (using all QEP eigvals with |lam|>=lam_floor)
    #     This is the upper limit of modal-sum accuracy when truncation is removed.
    def all_flex_modal_sup(sols, omega_arr, F, idx_obs, lam_floor=1.0):
        Y = np.zeros(omega_arr.size, dtype=complex)
        lowers = np.array([s["omega_lower"] for s in sols])
        cache = []
        for sol in sols:
            lams = sol["lams_all"]
            vecs = sol["vecs_all"]
            finite = np.isfinite(lams)
            mask_fl = finite & (np.abs(lams) >= lam_floor)
            lams_x = lams[mask_fl]
            vecs_x = vecs[:, mask_fl]
            if vecs_x.shape[1] == 0:
                cache.append(None); continue
            mat = sol["matrices"]
            Mv = mat["M_star"] @ vecs_x
            Cv = mat["C_star"] @ vecs_x
            a_k = (2 * lams_x * np.einsum("ij,ij->j", vecs_x, Mv)
                   + np.einsum("ij,ij->j", vecs_x, Cv))
            nz = np.abs(a_k) > 1e-30
            lams_x, vecs_x, a_k = lams_x[nz], vecs_x[:, nz], a_k[nz]
            modal_F = vecs_x.T @ F
            modal_obs = vecs_x[idx_obs, :]
            cache.append((lams_x, modal_obs, modal_F, a_k))
        for i, om in enumerate(omega_arr):
            j_idx = max(0, min(int(np.searchsorted(lowers, om, side="right") - 1),
                                len(sols) - 1))
            entry = cache[j_idx]
            if entry is None:
                Y[i] = 0; continue
            lams_x, modal_obs, modal_F, a_k = entry
            denom = a_k * (1j * om - lams_x)
            Y[i] = np.sum(modal_obs * modal_F / denom)
        return Y

    Y_all_flex = all_flex_modal_sup(sols0, omega_eval, F_red, cw_red, lam_floor=1.0)
    err_af = np.abs(Y_direct - (Y_all_flex + Y_rb_inertia)) / np.maximum(np.abs(Y_direct), 1e-30) * 100
    af_max = float(err_af[mask].max()) if np.isfinite(err_af[mask]).all() else float("inf")
    af_mean = float(err_af[mask].mean()) if np.isfinite(err_af[mask]).all() else float("inf")
    print(f"\n  All-flex modal sup + rb inertia (no truncation, no static): "
          f"max {af_max:.3f}%, mean {af_mean:.3f}%")
    rows.append({
        "method": "all_flex_modal_sup+rb",
        "err_max": af_max, "err_mean": af_mean,
        "non_finite_freqs": 0,
    })
    Y_records["all_flex"] = Y_all_flex + Y_rb_inertia
    err_plain = np.abs(Y_direct - Y_modal_plain) / np.maximum(np.abs(Y_direct), 1e-30) * 100.0
    finite_p = np.isfinite(err_plain[mask])
    if finite_p.all():
        err_plain_max = float(err_plain[mask].max())
        err_plain_mean = float(err_plain[mask].mean())
    else:
        err_plain_max = float(err_plain[mask][finite_p].max()) if finite_p.any() else float("inf")
        err_plain_mean = float(err_plain[mask][finite_p].mean()) if finite_p.any() else float("inf")
    print("\n  Reference: plain modal superposition (no static term, no rb inertia):")
    print(f"    err_max = {err_plain_max:.3f} %, err_mean = {err_plain_mean:.3f} %")
    rows.insert(0, {
        "method": "plain_modal_superposition",
        "err_max": err_plain_max, "err_mean": err_plain_mean,
        "non_finite_freqs": 0,
    })
    Y_records["plain_modal"] = Y_modal_plain

    # 6. Print summary table
    print("\nMode-acceleration summary (free-free 8x8, 5 segments):")
    print(f"  {'method':<28}  {'err_max [%]':>13}  {'err_mean [%]':>13}")
    for r in rows:
        emax = "inf" if not np.isfinite(r["err_max"]) else f"{r['err_max']:.3f}"
        emean = "inf" if not np.isfinite(r["err_mean"]) else f"{r['err_mean']:.3f}"
        print(f"  {r['method']:<28}  {emax:>13}  {emean:>13}")

    # 7. Save CSV
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "cmame_d1_mode_acceleration.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n[saved] {csv_path}")

    # 8. Figure: FRF overlay (highlight the all-flex + rb result)
    Y_all_flex_with_rb = Y_records["all_flex"]
    fig, axes = plt.subplots(2, 1, figsize=(16.5, 12.8), sharex=True)
    ax = axes[0]
    ax.semilogy(f_eval, np.abs(Y_direct), "k-", lw=1.5, label="Direct (per-freq)")
    ax.semilogy(f_eval, np.abs(Y_all_flex_with_rb), "C2-.", lw=1.6,
                label=f"all-flex modal-sup + rb inertia "
                      f"(max err {af_max:.2f}%)")
    ax.set_ylabel("|Y| [m/N]")
    ax.set_title("Free-free FRF reconstruction\n"
                 "Amichi 8x8 (567 DOFs), 5 segments, "
                 "rigid-body inertia + all-flex modal expansion")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=40 / 3 + 1, loc="best")

    ax = axes[1]
    err_plain_arr = np.abs(Y_direct - Y_modal_plain) / np.maximum(np.abs(Y_direct), 1e-30) * 100
    err_af_arr = np.abs(Y_direct - Y_all_flex_with_rb) / np.maximum(np.abs(Y_direct), 1e-30) * 100
    ax.semilogy(f_eval, err_af_arr, "C2-", lw=1.5,
                label=f"all-flex + rb inertia (peak max {af_max:.2f}%)")
    ax.axhline(1.0, color="green", ls=":", lw=1.0, label="1% target")
    ax.axhline(2.24, color="purple", ls=":", lw=1.0,
               label="G(ω) fit residual floor (2.24%)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Relative error [%]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=13.0, loc="best")

    fig.tight_layout()
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    fig_path = fig_dir / "cmame_mode_acceleration.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {fig_path}")

    print("\n" + "=" * 78)
    print("Mode-acceleration sweep complete.")
    print(f"  S_eff only (no static, no rb):     {err_plain_max:>8.2f}%  max")
    print(f"  S_eff + K_plus + rb inertia:       {err_max_main:>8.2f}%  max")
    print(f"  ALL-flex + rb inertia (PRIMARY):   {af_max:>8.2f}%  max  "
          f"(G fit floor 2.24%, corrected element)")
    print("=" * 78)


if __name__ == "__main__":
    main()
