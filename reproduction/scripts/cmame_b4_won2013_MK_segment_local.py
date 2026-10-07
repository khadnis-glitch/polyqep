# === [auto-resized: figsize x1.5] ===
"""Won 2013 segment-local Track A+G applied to BOTH K(G) and M(G).

Goal
----
Extend the K(G)-only Track A+G analysis of cmame_b3 to ALSO cover M(G).

Background
----------
Won 2013 sandwich-beam element has complex G-dependent shape functions
N*(ξ;c1*,c2*), H*(ξ;c1*,c2*) where (c1*, c2*) = f(G).  Both the stiffness
matrix and the consistent mass matrix are assembled from these shape
functions, so BOTH K(G) and M(G) are complex and G-dependent.

The existing cmame_b3 script measures only K(G) residual under Track A+G.
This script extends the diagnostic to M(G) to verify whether the
polynomial-QEP framework is truly applicable to Won 2013 when Track A+G
is applied to both matrices.

Per-segment metric:
    eps_K_i(G_t) = ||K(G_t) - K_{0,i} - G_t K_{1,i}||_F / ||K(G_t)||_F
    eps_M_i(G_t) = ||M(G_t) - M_{0,i} - G_t M_{1,i}||_F / ||M(G_t)||_F

After Track G low-rank correction with rank r_i (chosen by sigma_k/sigma_0 > tol):
    eps_K_AG_i = (r_i+1)-th singular value of dK snapshot / mean ||K||_F
    eps_M_AG_i = (r_i+1)-th singular value of dM snapshot / mean ||M||_F

Output
------
data/cmame_b4_won2013_MK_segment_local.csv     per-segment eps_K_A, r_K_i, eps_K_AG,
                                                eps_M_A, r_M_i, eps_M_AG
figures/cmame_b4_won2013_MK_segment_local.png   log-scale residual bar chart (K vs M)
stdout : per-segment summary table
"""
# Standard library
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# Third-party libraries
# === [auto-inserted: enable _figure_io monkey-patch for .pkl + .source.md] ===
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _figure_io  # noqa: F401
# === [/auto-inserted] ===

import matplotlib

# === [auto-inserted: 2x font scaling for all figures] ===
# === [2/3 font scaling applied: smaller in-figure text] ===
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
from scipy.linalg import svd

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.sandwich_beam_won2013 import (  # noqa: E402
    SandwichBeamParameters,
    apply_boundary_conditions,
    assemble_global,
    freq_dependent_core_G_paper_table6,
)


# --------------------------------------------------------------------------- #
# Data classes                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class SegmentResultMK:
    """Track A + Track G analysis results for both K and M on a single segment."""

    index: int
    f_lo: float
    f_hi: float
    G_lo: complex
    G_hi: complex
    dG_abs: float
    # K-side results
    eps_K_A_max: float
    eps_K_A_points: list[float]
    rank_K_i: int
    eps_K_AG_max: float
    sigma_K_first: float
    # M-side results
    eps_M_A_max: float
    eps_M_A_points: list[float]
    rank_M_i: int
    eps_M_AG_max: float
    sigma_M_first: float


# --------------------------------------------------------------------------- #
# Won 2013 paper §5.1 setup                                                   #
# --------------------------------------------------------------------------- #
def won2013_paper_params(n_elem: int = 8) -> SandwichBeamParameters:
    return SandwichBeamParameters(
        L=0.320, h1=0.003, h2=0.003, h3=0.003,
        E1=2.1e11, E3=2.1e11,
        rho1=7850.0, rho2=2000.0, rho3=7850.0,
        b=0.009,
        n_elem=n_elem,
    )


# --------------------------------------------------------------------------- #
# Helpers for extracting K and M together                                     #
# --------------------------------------------------------------------------- #
def compute_MK_won(
    params: SandwichBeamParameters,
    G: complex,
    apply_bc: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (M, K) matrices of the Won 2013 sandwich beam at G."""
    M, K = assemble_global(params, G)
    if not apply_bc:
        return M.astype(complex), K.astype(complex)
    n_nodes = params.n_elem + 1
    F = np.zeros(M.shape[0], dtype=complex)
    M_red, K_red, _, _ = apply_boundary_conditions(
        M, K, F, bc_left="clamped", bc_right="simply_supported", n_nodes=n_nodes,
    )
    return M_red.astype(complex), K_red.astype(complex)


def two_anchor_affine(
    A_lo: np.ndarray,
    A_hi: np.ndarray,
    G_lo: complex,
    G_hi: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-anchor extraction of a general matrix A(G) ≈ A_base + G·A_shear."""
    A_shear = (A_hi - A_lo) / (G_hi - G_lo)
    A_base = A_lo - G_lo * A_shear
    return A_base, A_shear


def track_AG_residual(
    snapshots: list[np.ndarray],
    norms: list[float],
    svd_tol: float,
) -> tuple[int, float, float]:
    """Snapshot matrix -> SVD -> rank r and residual estimate.

    Returns
    -------
    r : int          truncation rank (sigma_k/sigma_0 > tol)
    eps_AG_max : float    estimated residual after rank truncation
    sigma_first : float
    """
    snapshot = np.column_stack([s.ravel() for s in snapshots])
    _, sigma, _ = svd(snapshot, full_matrices=False)
    sigma_first = float(sigma[0]) if sigma.size else 0.0

    if sigma_first > 0:
        ratios = sigma / sigma_first
        r = int(np.sum(ratios > svd_tol))
    else:
        r = 0

    norm_mean = float(np.mean(norms)) if norms else 1.0
    if r < sigma.size:
        eps_AG_max = float(sigma[r]) / max(norm_mean, 1e-30)
    else:
        eps_AG_max = 0.0
    return r, eps_AG_max, sigma_first


def segment_local_MK_analysis(
    params: SandwichBeamParameters,
    G_func,
    f_lo: float,
    f_hi: float,
    seg_idx: int,
    n_test: int = 15,
    svd_tol: float = 1e-6,
    apply_bc: bool = False,
) -> SegmentResultMK:
    """Track A + Track G analysis for both K and M on a single segment."""
    omega_lo = 2.0 * np.pi * f_lo
    omega_hi = 2.0 * np.pi * f_hi
    G_lo = G_func(omega_lo)
    G_hi = G_func(omega_hi)

    # Extract M, K at the two anchor points
    M_lo, K_lo = compute_MK_won(params, G_lo, apply_bc=apply_bc)
    M_hi, K_hi = compute_MK_won(params, G_hi, apply_bc=apply_bc)

    # Track A — segment-local affine factors
    K_base, K_shear = two_anchor_affine(K_lo, K_hi, G_lo, G_hi)
    M_base, M_shear = two_anchor_affine(M_lo, M_hi, G_lo, G_hi)

    # Interior test points
    alphas = np.linspace(0.0, 1.0, n_test + 2)[1:-1]
    eps_K_A_points: list[float] = []
    eps_M_A_points: list[float] = []
    dK_snapshots: list[np.ndarray] = []
    dM_snapshots: list[np.ndarray] = []
    K_norms: list[float] = []
    M_norms: list[float] = []

    for alpha in alphas:
        omega_t = omega_lo + alpha * (omega_hi - omega_lo)
        G_t = G_func(omega_t)
        M_t, K_t = compute_MK_won(params, G_t, apply_bc=apply_bc)

        dK_t = K_t - K_base - G_t * K_shear
        dM_t = M_t - M_base - G_t * M_shear

        K_norm = float(np.linalg.norm(K_t, "fro"))
        M_norm = float(np.linalg.norm(M_t, "fro"))
        dK_norm = float(np.linalg.norm(dK_t, "fro"))
        dM_norm = float(np.linalg.norm(dM_t, "fro"))

        eps_K_A_points.append(dK_norm / max(K_norm, 1e-30))
        eps_M_A_points.append(dM_norm / max(M_norm, 1e-30))
        dK_snapshots.append(dK_t)
        dM_snapshots.append(dM_t)
        K_norms.append(K_norm)
        M_norms.append(M_norm)

    eps_K_A_max = max(eps_K_A_points)
    eps_M_A_max = max(eps_M_A_points)

    # Track G — low-rank residual correction (K and M independently)
    rank_K, eps_K_AG, sigma_K = track_AG_residual(dK_snapshots, K_norms, svd_tol)
    rank_M, eps_M_AG, sigma_M = track_AG_residual(dM_snapshots, M_norms, svd_tol)

    return SegmentResultMK(
        index=seg_idx,
        f_lo=f_lo, f_hi=f_hi,
        G_lo=G_lo, G_hi=G_hi,
        dG_abs=float(abs(G_hi - G_lo)),
        eps_K_A_max=eps_K_A_max,
        eps_K_A_points=eps_K_A_points,
        rank_K_i=rank_K,
        eps_K_AG_max=eps_K_AG,
        sigma_K_first=sigma_K,
        eps_M_A_max=eps_M_A_max,
        eps_M_A_points=eps_M_A_points,
        rank_M_i=rank_M,
        eps_M_AG_max=eps_M_AG,
        sigma_M_first=sigma_M,
    )


# --------------------------------------------------------------------------- #
# Global affine residuals (for reference)                                     #
# --------------------------------------------------------------------------- #
def global_affine_residuals(
    params: SandwichBeamParameters,
    G_func,
    f_anchor_lo: float = 10.0,
    f_anchor_hi: float = 1000.0,
    test_f_list: tuple[float, ...] = (50.0, 200.0, 800.0),
    apply_bc: bool = False,
) -> tuple[list[dict], list[dict]]:
    """K and M residuals of the global two-anchor affine model."""
    G_lo = G_func(2.0 * np.pi * f_anchor_lo)
    G_hi = G_func(2.0 * np.pi * f_anchor_hi)

    M_lo, K_lo = compute_MK_won(params, G_lo, apply_bc=apply_bc)
    M_hi, K_hi = compute_MK_won(params, G_hi, apply_bc=apply_bc)

    K_base, K_shear = two_anchor_affine(K_lo, K_hi, G_lo, G_hi)
    M_base, M_shear = two_anchor_affine(M_lo, M_hi, G_lo, G_hi)

    K_rows: list[dict] = []
    M_rows: list[dict] = []
    for f_t in test_f_list:
        G_t = G_func(2.0 * np.pi * f_t)
        M_t, K_t = compute_MK_won(params, G_t, apply_bc=apply_bc)

        K_pred = K_base + G_t * K_shear
        M_pred = M_base + G_t * M_shear

        K_rows.append({
            "f_Hz": f_t,
            "abs_res": float(np.linalg.norm(K_t - K_pred)),
            "rel_res": float(np.linalg.norm(K_t - K_pred) / max(np.linalg.norm(K_t), 1e-30)),
        })
        M_rows.append({
            "f_Hz": f_t,
            "abs_res": float(np.linalg.norm(M_t - M_pred)),
            "rel_res": float(np.linalg.norm(M_t - M_pred) / max(np.linalg.norm(M_t), 1e-30)),
        })
    return K_rows, M_rows


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    np.set_printoptions(linewidth=140, precision=4)
    print("=" * 88)
    print("Won 2013 segment-local Track A+G applied to BOTH K(G) and M(G)")
    print("=" * 88)

    params = won2013_paper_params(n_elem=8)
    G_func = freq_dependent_core_G_paper_table6()

    f_min, f_max = 5.0, 500.0
    candidates = np.array([1.0, 50.0, 150.0, 350.0, 700.0, 1500.0])
    inner = candidates[(candidates > f_min) & (candidates < f_max)]
    breakpoints_hz = np.unique(np.concatenate([[f_min], inner, [f_max]]))
    n_segments = len(breakpoints_hz) - 1

    print(f"\n[setup] BC = clamped-SS apply_bc=False (27x27 raw), n_elem = {params.n_elem}")
    print(f"[setup] breakpoints [Hz] = {breakpoints_hz}")
    print(f"[setup] segments = {n_segments}  (Won 5–500 Hz)")

    # Global affine residuals (for reference)
    print(f"\n[global affine residuals — anchors at 10 Hz / 1000 Hz]")
    K_glob, M_glob = global_affine_residuals(params, G_func, apply_bc=False)
    print(f"  {'f_test':>8} | {'K rel_res':>11} | {'M rel_res':>11}")
    print("  " + "-" * 38)
    for kr, mr in zip(K_glob, M_glob):
        print(f"  {kr['f_Hz']:>8.1f} | {kr['rel_res']:>11.3e} | {mr['rel_res']:>11.3e}")

    # Segment-local Track A+G (K and M together)
    print("\n[segment-local Track A+G — K and M]")
    print(f"  {'idx':>3} | {'band [Hz]':>14} | {'|dG|':>9} | "
          f"{'eps_K_A':>10} | {'r_K':>3} | {'eps_K_AG':>10} | "
          f"{'eps_M_A':>10} | {'r_M':>3} | {'eps_M_AG':>10}")
    print("  " + "-" * 110)

    results: list[SegmentResultMK] = []
    for i in range(n_segments):
        res = segment_local_MK_analysis(
            params, G_func,
            float(breakpoints_hz[i]), float(breakpoints_hz[i + 1]),
            i, n_test=15, svd_tol=1e-6, apply_bc=False,
        )
        results.append(res)
        print(f"  {res.index:>3} | "
              f"[{res.f_lo:>4.0f},{res.f_hi:>4.0f}]      | "
              f"{res.dG_abs:>9.2e} | "
              f"{res.eps_K_A_max:>10.3e} | {res.rank_K_i:>3d} | {res.eps_K_AG_max:>10.3e} | "
              f"{res.eps_M_A_max:>10.3e} | {res.rank_M_i:>3d} | {res.eps_M_AG_max:>10.3e}")

    # Save CSV
    out_csv = Path(__file__).resolve().parents[1] / "data" / "cmame_b4_won2013_MK_segment_local.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "segment", "f_lo_Hz", "f_hi_Hz",
            "G_lo_re", "G_lo_im", "G_hi_re", "G_hi_im", "abs_dG",
            "eps_K_A_max", "rank_K_i", "eps_K_AG_max",
            "eps_M_A_max", "rank_M_i", "eps_M_AG_max",
        ])
        for res in results:
            w.writerow([
                res.index, res.f_lo, res.f_hi,
                float(np.real(res.G_lo)), float(np.imag(res.G_lo)),
                float(np.real(res.G_hi)), float(np.imag(res.G_hi)),
                res.dG_abs,
                res.eps_K_A_max, res.rank_K_i, res.eps_K_AG_max,
                res.eps_M_A_max, res.rank_M_i, res.eps_M_AG_max,
            ])
    print(f"\n[saved] {out_csv}")

    # Figure — 2-panel bar chart (K and M separately)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.8), sharey=True)
    idx = np.arange(n_segments)
    width = 0.35

    # Panel (a): K(G) residuals
    ax = axes[0]
    eps_K_A_arr = np.array([r.eps_K_A_max for r in results])
    eps_K_AG_arr = np.array([max(r.eps_K_AG_max, 1e-18) for r in results])
    K_global_max = max(r["rel_res"] for r in K_glob)
    ax.bar(idx - width / 2, eps_K_A_arr, width, label="Step 1 (affine)")
    ax.bar(idx + width / 2, eps_K_AG_arr, width, label="Step 2 (+correction)")
    ax.axhline(K_global_max, color="red", linestyle="--", linewidth=1.0,
               label=f"Global affine floor ({K_global_max:.1e})")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([
        f"S{i}\n[{results[i].f_lo:.0f},{results[i].f_hi:.0f}] Hz" for i in idx
    ])
    ax.set_ylabel("Relative Frobenius residual")
    ax.set_title("(a) $\\mathbf{K}(G)$ residual")
    ax.legend(fontsize=10.7, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel (b): M(G) residuals
    ax = axes[1]
    eps_M_A_arr = np.array([r.eps_M_A_max for r in results])
    eps_M_AG_arr = np.array([max(r.eps_M_AG_max, 1e-18) for r in results])
    M_global_max = max(r["rel_res"] for r in M_glob)
    ax.bar(idx - width / 2, eps_M_A_arr, width, label="Step 1 (affine)")
    ax.bar(idx + width / 2, eps_M_AG_arr, width, label="Step 2 (+correction)")
    ax.axhline(M_global_max, color="red", linestyle="--", linewidth=1.0,
               label=f"Global affine floor ({M_global_max:.1e})")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([
        f"S{i}\n[{results[i].f_lo:.0f},{results[i].f_hi:.0f}] Hz" for i in idx
    ])
    ax.set_title("(b) $\\mathbf{M}(G)$ residual")
    ax.legend(fontsize=10.7, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # No figure suptitle: the LaTeX caption carries the title (avoids text overlap)
    fig.tight_layout()
    fig_path = Path(__file__).resolve().parents[1] / "figures" / "cmame_b4_won2013_MK_segment_local.png"
    fig_path.parent.mkdir(exist_ok=True)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {fig_path}")

    # Summary
    print("\n[summary]")
    print(f"  K global affine residual    : max {max(r['rel_res'] for r in K_glob):.3e}")
    print(f"  M global affine residual    : max {max(r['rel_res'] for r in M_glob):.3e}")
    print(f"  K Track A max segment res.  : {max(r.eps_K_A_max for r in results):.3e}")
    print(f"  M Track A max segment res.  : {max(r.eps_M_A_max for r in results):.3e}")
    print(f"  K Track A+G max segment res.: {max(r.eps_K_AG_max for r in results):.3e}")
    print(f"  M Track A+G max segment res.: {max(r.eps_M_AG_max for r in results):.3e}")
    print(f"  K ranks r_K_i               : {[r.rank_K_i for r in results]}")
    print(f"  M ranks r_M_i               : {[r.rank_M_i for r in results]}")

    # Applicability assessment
    K_AG_max = max(r.eps_K_AG_max for r in results)
    M_AG_max = max(r.eps_M_AG_max for r in results)
    print("\n[verdict on framework applicability]")
    if K_AG_max < 1e-14 and M_AG_max < 1e-14:
        print("  Track A+G drives BOTH K(G) and M(G) to LAPACK floor across all segments.")
        print("  → Polynomial-QEP framework is fully applicable to Won 2013 via Track A+G.")
    elif K_AG_max < 1e-14 and M_AG_max >= 1e-14:
        print("  Track A+G drives K(G) to LAPACK floor but M(G) stays above 1e-14.")
        print("  → M-side correction needs higher rank or different ansatz.")
    else:
        print(f"  K Track A+G floor: {K_AG_max:.3e}")
        print(f"  M Track A+G floor: {M_AG_max:.3e}")
        print("  → Re-evaluate svd_tol / rank selection.")


if __name__ == "__main__":
    main()
