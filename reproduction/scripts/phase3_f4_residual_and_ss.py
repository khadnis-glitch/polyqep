# === [auto-resized: figsize x1.5] ===
"""Eigenvector residual filter + simply-supported BC trial.

Stage M: measure the residual ‖(λ²M+λC+K)v‖ / max{‖λ²Mv‖, ‖λCv‖, ‖Kv‖} of the
        1134 eigenvectors and filter on it to improve modal-sup accuracy.
Stage N: modal-sup accuracy when switching from free BC to simply-supported BC
        (rigid-body modes removed → catastrophic cancellation avoided).

Goal: modal-sup accuracy within 1% on the 567-DOF Amichi plate.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as spla
# === [auto-inserted: enable _figure_io monkey-patch for .pkl + .source.md] ===
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _figure_io  # noqa: F401
# === [/auto-inserted] ===

import matplotlib

# === [auto-inserted: 2x font scaling for all figures] ===
import matplotlib as _mpl  # noqa: E402
_mpl.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 23,
    "axes.labelsize": 19,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 15,
    "figure.titlesize": 23,
})
# === [/auto-inserted] ===


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from scripts.phase3_amichi_polynomial_qep import (
    sample_G_omega, fit_G_polynomial,
    assemble_complex_per_interval_amichi,
    paper_5_1_params,
)
from polyqep.models.sandwich_plate_amichi2010 import (
    apply_boundary_conditions, assemble_K_base_and_shear, assemble_global,
)


# =============================================================================
# QEP solve + residual computation
# =============================================================================
def solve_qep_with_residual(M, C, K):
    """Solve the QEP and compute the residual of each mode."""
    n = M.shape[0]
    I = np.eye(n, dtype=complex)
    Z = np.zeros((n, n), dtype=complex)
    A = np.block([[Z, I], [-K, -C]])
    B = np.block([[I, Z], [Z, M]])
    eigvals, vr_full = spla.eig(A, B)
    v_part = vr_full[:n, :]

    # Residual: ‖(λ²M + λC + K) v‖_2 / max(...)
    residuals = np.full(eigvals.size, np.nan)
    for k in range(eigvals.size):
        lam = eigvals[k]
        if not np.isfinite(lam):
            continue
        v = v_part[:, k]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-30:
            continue
        Mv = M @ v
        Cv = C @ v
        Kv = K @ v
        lhs = (lam ** 2) * Mv + lam * Cv + Kv
        denom = max(abs(lam) ** 2 * np.linalg.norm(Mv),
                    abs(lam) * np.linalg.norm(Cv),
                    np.linalg.norm(Kv), 1e-30)
        residuals[k] = np.linalg.norm(lhs) / denom
    return eigvals, v_part, residuals


def modal_sum_filtered(
    omega_array, eigvals, v_part, residuals,
    M, C, F, idx_obs,
    abs_lam_min=0.1,
    residual_max=None,
):
    """Modal-sup with filtering applied."""
    valid = (np.isfinite(eigvals)
             & (np.abs(eigvals) > abs_lam_min)
             & (residuals < (residual_max if residual_max is not None else np.inf)))
    lams = eigvals[valid]
    v = v_part[:, valid]
    if lams.size == 0:
        return np.full(omega_array.size, np.nan, dtype=complex), 0
    Mv = M @ v
    Cv = C @ v
    a_k = (2 * lams * np.einsum("ij,ij->j", v, Mv)
           + np.einsum("ij,ij->j", v, Cv))
    a_max = np.abs(a_k).max() if a_k.size else 0.0
    if a_max < 1e-30:
        return np.full(omega_array.size, np.nan, dtype=complex), 0
    nz = np.abs(a_k) > 1e-15 * a_max
    lams = lams[nz]
    v = v[:, nz]
    a_k = a_k[nz]
    if lams.size == 0:
        return np.full(omega_array.size, np.nan, dtype=complex), 0
    F_k = v.T @ F
    obs_k = v[idx_obs, :]
    weights = obs_k * F_k / a_k
    Y = np.zeros(omega_array.size, dtype=complex)
    for i, om in enumerate(omega_array):
        Y[i] = np.sum(weights / (1j * om - lams))
    return Y, weights.size


def direct_response(M, C, K, omega_array, F, idx_obs):
    Y = np.zeros(omega_array.size, dtype=complex)
    for i, om in enumerate(omega_array):
        D = K + 1j * om * C - om**2 * M
        Y[i] = spla.solve(D, F)[idx_obs]
    return Y


# =============================================================================
# Setup helper
# =============================================================================
def setup_amichi(bc: str = "free"):
    """Set up the paper §5.1 case. bc: 'free' or 'simply_supported_w'."""
    params = paper_5_1_params(nx=8, ny=8)
    K_base, K_shear = assemble_K_base_and_shear(params)
    M_full, _ = assemble_global(params, complex(1.0, 0.0))
    F_full = np.zeros(M_full.shape[0], dtype=complex)
    cw_full = 7 * (params.nx // 2 + (params.ny // 2) * (params.nx + 1)) + 4
    F_full[cw_full] = 1.0
    M_red, _, F_red, free_dofs = apply_boundary_conditions(
        M_full, K_base.astype(complex), F_full, params, bc=bc
    )
    cw_red_arr = np.where(np.asarray(free_dofs) == cw_full)[0]
    if len(cw_red_arr) == 0:
        # for SS some W DOFs may be fixed — then use W of the nearest internal node
        n_nodes_x = params.nx + 1
        for r in range(1, params.nx):
            for c in range(1, params.ny):
                node_id = r + c * n_nodes_x
                test_dof = 7 * node_id + 4
                hits = np.where(np.asarray(free_dofs) == test_dof)[0]
                if len(hits) > 0:
                    cw_red = int(hits[0])
                    F_red = np.zeros_like(F_red)
                    F_red[cw_red] = 1.0
                    print(f"  [bc={bc}] center W not free; using internal "
                          f"node ({r},{c}) at reduced DOF {cw_red}")
                    break
            else:
                continue
            break
    else:
        cw_red = int(cw_red_arr[0])
    free_idx = np.asarray(free_dofs)
    K_base_red = K_base[np.ix_(free_idx, free_idx)]
    K_shear_red = K_shear[np.ix_(free_idx, free_idx)]
    M_red_real = (M_red.real + M_red.real.T) / 2.0

    omega_s, G_re, G_im = sample_G_omega(1.0, 1500.0, 400)
    spline_re, spline_im, _ = fit_G_polynomial(omega_s, G_re, G_im, n_segments=5)
    interval_mats = assemble_complex_per_interval_amichi(
        spline_re, spline_im, M_red_real, K_base_red, K_shear_red,
    )
    return interval_mats, F_red, cw_red, M_red_real.shape[0]


def test_bc(bc_label: str, interval_mats, F_red, cw_red, n_dof, target_idx=0):
    """Measure modal-sup accuracy on a single interval (residual filter sweep)."""
    print(f"\n{'='*78}")
    print(f"BC = {bc_label}, interval {target_idx} test")
    print(f"{'='*78}")
    mat = interval_mats[target_idx]
    M_s = mat["M_star"]
    C_s = mat["C_star"]
    K_s = mat["K_star"]

    # conditioning of K_s
    K_real_eigs = np.linalg.eigvalsh((K_s.real + K_s.real.T) / 2)
    print(f"  K_s real-sym eigvals: min {K_real_eigs.min():.3e}, "
          f"max {K_real_eigs.max():.3e}")
    print(f"  K_s n_negative: {int((K_real_eigs < 0).sum())}, "
          f"n_near_zero (<1e-3): {int((np.abs(K_real_eigs) < 1e-3).sum())}")

    f_lo = mat["omega_lower"] / (2 * np.pi)
    f_hi = mat["omega_upper"] / (2 * np.pi)
    f_eval = np.linspace(max(5, f_lo), f_hi, 100)
    omega_eval = 2 * np.pi * f_eval

    # Reference
    Y_ref = direct_response(M_s, C_s, K_s, omega_eval, F_red, cw_red)
    mask = np.abs(Y_ref) > 0.01 * np.abs(Y_ref).max()

    # QEP + residual
    eigvals, v_part, residuals = solve_qep_with_residual(M_s, C_s, K_s)
    finite = np.isfinite(residuals)
    print(f"\n  Residual distribution ({finite.sum()}/{eigvals.size} finite):")
    for thr in [1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e+3]:
        cnt = int((residuals[finite] < thr).sum())
        print(f"    residual < {thr:>1.0e}: {cnt}")

    # Filter sweep
    print(f"\n  Filter sweep (|λ| > 0.1, residual < threshold):")
    print(f"    {'res_max':>10} {'n_used':>7} {'max err [%]':>14} {'mean err [%]':>14}")
    best = None
    for res_max in [None, 1.0, 1e-3, 1e-6, 1e-9, 1e-12]:
        Y_modal, n_used = modal_sum_filtered(
            omega_eval, eigvals, v_part, residuals,
            M_s, C_s, F_red, cw_red,
            abs_lam_min=0.1, residual_max=res_max,
        )
        err = np.abs(Y_ref - Y_modal) / np.maximum(np.abs(Y_ref), 1e-30) * 100
        max_err = err[mask].max()
        mean_err = err[mask].mean()
        label = "None" if res_max is None else f"{res_max:.0e}"
        print(f"    {label:>10} {n_used:>7} {max_err:>14.4f} {mean_err:>14.4f}")
        if best is None or max_err < best[1]:
            best = (label, max_err, mean_err, n_used, Y_modal)
    return best, Y_ref, omega_eval, f_eval, mask


def main():
    np.set_printoptions(linewidth=140, precision=4)
    print("=" * 78)
    print("Eigvec residual filter + SS BC")
    print("=" * 78)

    # Free BC
    print("\n[Phase 1] Free BC")
    int_mats_free, F_free, cw_free, n_free = setup_amichi(bc="free")
    print(f"  reduced n_dof = {n_free}")
    best_free, Yref_f, om_f, fhz_f, mask_f = test_bc(
        "free", int_mats_free, F_free, cw_free, n_free, target_idx=0,
    )

    # SS BC
    print("\n\n[Phase 2] Simply-supported BC")
    int_mats_ss, F_ss, cw_ss, n_ss = setup_amichi(bc="simply_supported_w")
    print(f"  reduced n_dof = {n_ss}  (fewer than free — boundary W/U/V fixed)")
    best_ss, Yref_s, om_s, fhz_s, mask_s = test_bc(
        "SS", int_mats_ss, F_ss, cw_ss, n_ss, target_idx=0,
    )

    # Summary
    print("\n" + "=" * 78)
    print("Conclusion")
    print("=" * 78)
    print(f"  free BC best: residual_max={best_free[0]}, "
          f"max err {best_free[1]:.4f}%, n_modes {best_free[3]}")
    print(f"  SS BC   best: residual_max={best_ss[0]}, "
          f"max err {best_ss[1]:.4f}%, n_modes {best_ss[3]}")

    if best_ss[1] < 1.0:
        print(f"\n  ✅ Accuracy within 1% achieved with SS BC!")
    elif best_free[1] < 1.0 or best_ss[1] < 10.0:
        print(f"\n  🟡 Partial improvement — further work needed")
    else:
        print(f"\n  ⚠️ 1% accuracy not reached — presumably a limitation of scipy eig itself at 567 DOF")

    # Figure
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(19.5, 12))
    # free BC
    Y_modal_f = best_free[4]
    axes[0, 0].semilogy(fhz_f, np.abs(Yref_f), "k-", lw=1.5, label="Direct")
    axes[0, 0].semilogy(fhz_f, np.abs(Y_modal_f), "r--",
                        label=f"Modal-sup ({best_free[1]:.2f}% max)")
    axes[0, 0].set_title(f"Free BC, interval 0 (n={n_free})")
    axes[0, 0].set_ylabel("|Y| [m/N]")
    axes[0, 0].grid(True, alpha=0.3, which="both")
    axes[0, 0].legend()

    err_f = np.abs(Yref_f - Y_modal_f) / np.maximum(np.abs(Yref_f), 1e-30) * 100
    axes[1, 0].semilogy(fhz_f, err_f, "r-")
    axes[1, 0].axhline(1.0, color="g", ls=":", label="1% target")
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].set_ylabel("Relative error [%]")
    axes[1, 0].grid(True, alpha=0.3, which="both")
    axes[1, 0].legend()

    # SS BC
    Y_modal_s = best_ss[4]
    axes[0, 1].semilogy(fhz_s, np.abs(Yref_s), "k-", lw=1.5, label="Direct")
    axes[0, 1].semilogy(fhz_s, np.abs(Y_modal_s), "b--",
                        label=f"Modal-sup ({best_ss[1]:.2f}% max)")
    axes[0, 1].set_title(f"SS BC, interval 0 (n={n_ss})")
    axes[0, 1].grid(True, alpha=0.3, which="both")
    axes[0, 1].legend()

    err_s = np.abs(Yref_s - Y_modal_s) / np.maximum(np.abs(Yref_s), 1e-30) * 100
    axes[1, 1].semilogy(fhz_s, err_s, "b-")
    axes[1, 1].axhline(1.0, color="g", ls=":", label="1% target")
    axes[1, 1].set_xlabel("Frequency [Hz]")
    axes[1, 1].grid(True, alpha=0.3, which="both")
    axes[1, 1].legend()

    fig.suptitle("Modal-sup with eigvec residual filter (free vs SS BC)")
    fig.tight_layout()
    fig.savefig(out_dir / "phase3_f4_modal_residual.png", dpi=140)
    plt.close(fig)
    print(f"\n[saved] figures/phase3_f4_modal_residual.png")


if __name__ == "__main__":
    main()
