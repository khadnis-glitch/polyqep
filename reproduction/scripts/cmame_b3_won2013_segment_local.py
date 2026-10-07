# === [auto-resized: figsize x1.5] ===
"""Won 2013 segment-local affineization + low-rank correction.

Goal
----
Demonstrate that the Won-2013 element's global K(G) residual floor of ~1e-9
(measured in scripts/cmame_b2_won2013_polynomial.py) is a property of the
*global* affine fit and not of the underlying element. Restricting the
affine fit to a single G(omega) segment S_i reduces the residual by the
squared segment-width ratio (Taylor scaling), and the remaining segment-local
residual matrix admits a low-rank approximation U_i D_i(G) V_i^T whose
truncation rank r_i is small (typically r_i <= 5 for the Won-class family).

Per-segment metric (see the paper):
    eps_i(G_t) = ||K(G_t) - K_{0,i} - G_t K_{1,i}||_F / ||K(G_t)||_F

Output
------
data/cmame_b3_won2013_segment_local.csv   per-segment eps_A, r_i, eps_AG
figures/cmame_b3_won2013_segment_local.png  log-scale residual bar chart
stdout : per-segment summary table
"""
# Standard library imports
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

import matplotlib.pyplot as plt

# === [font scaling: reduced to fit figure cleanly] ===
import matplotlib as _mpl  # noqa: E402
_mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
})
# === [/auto-inserted] ===

import numpy as np
from scipy.linalg import svd

# Add the project root to sys.path — same pattern as the other scripts
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
class SegmentResult:
    """Track A + Track G analysis results for a single segment."""

    index: int                # segment index (0-based)
    f_lo: float               # segment lower frequency bound [Hz]
    f_hi: float               # segment upper frequency bound [Hz]
    G_lo: complex             # anchor G at omega_lo
    G_hi: complex             # anchor G at omega_hi
    dG_abs: float             # |G_hi - G_lo|
    eps_A_max: float          # Track A-only residual (max over interior test points)
    eps_A_points: list[float]  # eps_A at each interior test point
    rank_r_i: int             # SVD truncation rank (by tol)
    eps_AG_max: float         # combined Track A+G residual (estimate after rank truncation)
    sigma_first: float        # largest singular value of the residual-snapshot SVD


# --------------------------------------------------------------------------- #
# Won 2013 paper §5.1 — paper-comparable setup                                #
# --------------------------------------------------------------------------- #
def won2013_paper_params(n_elem: int = 8) -> SandwichBeamParameters:
    """Paper Table 1 + §5.1 sandwich beam setup (clamped-SS, n_elem=8)."""
    return SandwichBeamParameters(
        L=0.320, h1=0.003, h2=0.003, h3=0.003,
        E1=2.1e11, E3=2.1e11,
        rho1=7850.0, rho2=2000.0, rho3=7850.0,
        b=0.009,
        n_elem=n_elem,
    )


# --------------------------------------------------------------------------- #
# Core helpers                                                                #
# --------------------------------------------------------------------------- #
def compute_K_won(
    params: SandwichBeamParameters,
    G: complex,
    apply_bc: bool = False,
) -> np.ndarray:
    """Return the K matrix of the Won 2013 sandwich beam at G.

    apply_bc=False (default): full 27x27 K (as used for the cmame_b2 residual measurement).
    apply_bc=True: 22x22 after applying clamped-SS.
    """
    M, K = assemble_global(params, G)
    if not apply_bc:
        return K.astype(complex)
    n_nodes = params.n_elem + 1
    F = np.zeros(M.shape[0], dtype=complex)
    _, K_red, _, _ = apply_boundary_conditions(
        M, K, F, bc_left="clamped", bc_right="simply_supported", n_nodes=n_nodes,
    )
    return K_red.astype(complex)


def two_anchor_affine(
    K_lo: np.ndarray,
    K_hi: np.ndarray,
    G_lo: complex,
    G_hi: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract K_base, K_shear from two anchors (see the paper)."""
    K_shear = (K_hi - K_lo) / (G_hi - G_lo)
    K_base = K_lo - G_lo * K_shear
    return K_base, K_shear


def segment_local_analysis(
    params: SandwichBeamParameters,
    G_func,
    f_lo: float,
    f_hi: float,
    seg_idx: int,
    n_test: int = 3,
    svd_tol: float = 1e-15,
    apply_bc: bool = False,
) -> SegmentResult:
    """Run the Track A + Track G analysis on a single segment."""
    omega_lo = 2.0 * np.pi * f_lo
    omega_hi = 2.0 * np.pi * f_hi
    G_lo = G_func(omega_lo)
    G_hi = G_func(omega_hi)

    # Extract K at the two anchor points → (K_0, K_1)
    K_lo = compute_K_won(params, G_lo, apply_bc=apply_bc)
    K_hi = compute_K_won(params, G_hi, apply_bc=apply_bc)
    K_base, K_shear = two_anchor_affine(K_lo, K_hi, G_lo, G_hi)

    # Interior test points (25/50/75% along the omega axis)
    alphas = np.linspace(0.0, 1.0, n_test + 2)[1:-1]
    eps_A_points: list[float] = []
    dK_snapshots: list[np.ndarray] = []
    K_norms: list[float] = []
    for alpha in alphas:
        omega_t = omega_lo + alpha * (omega_hi - omega_lo)
        G_t = G_func(omega_t)
        K_t = compute_K_won(params, G_t, apply_bc=apply_bc)
        dK_t = K_t - K_base - G_t * K_shear
        K_norm = float(np.linalg.norm(K_t, "fro"))
        dK_norm = float(np.linalg.norm(dK_t, "fro"))
        eps_A_points.append(dK_norm / max(K_norm, 1e-30))
        dK_snapshots.append(dK_t)
        K_norms.append(K_norm)

    eps_A_max = max(eps_A_points)

    # Residual snapshot matrix → SVD → rank r_i
    snapshot = np.column_stack([dK.ravel() for dK in dK_snapshots])
    _, sigma, _ = svd(snapshot, full_matrices=False)
    sigma_first = float(sigma[0]) if sigma.size else 0.0

    if sigma_first > 0:
        ratios = sigma / sigma_first
        r_i = int(np.sum(ratios > svd_tol))
    else:
        r_i = 0

    # Residual estimate after rank truncation: the (r_i+1)-th singular value
    # normalized by the mean ||K||_F → eps_AG_max
    K_norm_mean = float(np.mean(K_norms)) if K_norms else 1.0
    if r_i < sigma.size:
        eps_AG_max = float(sigma[r_i]) / max(K_norm_mean, 1e-30)
    else:
        eps_AG_max = 0.0

    return SegmentResult(
        index=seg_idx,
        f_lo=f_lo,
        f_hi=f_hi,
        G_lo=G_lo,
        G_hi=G_hi,
        dG_abs=float(abs(G_hi - G_lo)),
        eps_A_max=eps_A_max,
        eps_A_points=eps_A_points,
        rank_r_i=r_i,
        eps_AG_max=eps_AG_max,
        sigma_first=sigma_first,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    np.set_printoptions(linewidth=140, precision=4)
    print("=" * 78)
    print("Won 2013 segment-local affineization + low-rank correction")
    print("=" * 78)

    params = won2013_paper_params(n_elem=8)
    G_func = freq_dependent_core_G_paper_table6()

    # Evaluation frequency band of Won 2013 §5.1: 5–500 Hz
    # Reuses the 5-segment candidate breakpoints of phase3_amichi_polynomial_qep.fit_G_polynomial
    f_min, f_max = 5.0, 500.0
    candidates = np.array([1.0, 50.0, 150.0, 350.0, 700.0, 1500.0])
    inner = candidates[(candidates > f_min) & (candidates < f_max)]
    breakpoints_hz = np.unique(np.concatenate([[f_min], inner, [f_max]]))
    n_segments = len(breakpoints_hz) - 1

    print(f"\n[setup] BC = clamped-SS, n_elem = {params.n_elem}, n_dof_free = 22")
    print(f"[setup] breakpoints [Hz] = {breakpoints_hz}")
    print(f"[setup] segments = {n_segments}  (Won 5–500 Hz)")

    results: list[SegmentResult] = []
    print("\n[segment-local analysis]")
    print(f"  {'idx':>3} | {'f_lo':>6}–{'f_hi':>6} Hz | {'|dG|':>10} | "
          f"{'eps_A_max':>11} | {'r_i':>3} | {'eps_AG_max':>11}")
    print("  " + "-" * 72)
    for i in range(n_segments):
        res = segment_local_analysis(
            params, G_func,
            float(breakpoints_hz[i]), float(breakpoints_hz[i + 1]),
            i, n_test=15, svd_tol=1e-6, apply_bc=False,
        )
        results.append(res)
        print(f"  {res.index:>3} | {res.f_lo:>6.1f}–{res.f_hi:>6.1f}    | "
              f"{res.dG_abs:>10.3e} | {res.eps_A_max:>11.3e} | "
              f"{res.rank_r_i:>3d} | {res.eps_AG_max:>11.3e}")

    # Save CSV
    out_csv = Path(__file__).resolve().parents[1] / "data" / "cmame_b3_won2013_segment_local.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "segment", "f_lo_Hz", "f_hi_Hz", "G_lo_re", "G_lo_im",
            "G_hi_re", "G_hi_im", "abs_dG", "eps_A_max", "rank_r_i", "eps_AG_max",
        ])
        for res in results:
            w.writerow([
                res.index, res.f_lo, res.f_hi,
                float(np.real(res.G_lo)), float(np.imag(res.G_lo)),
                float(np.real(res.G_hi)), float(np.imag(res.G_hi)),
                res.dG_abs, res.eps_A_max, res.rank_r_i, res.eps_AG_max,
            ])
    print(f"\n[saved] {out_csv.relative_to(Path.cwd()) if out_csv.is_relative_to(Path.cwd()) else out_csv}")

    # Figure: log-scale bar chart (global affine reference line + Track A + Track A+G)
    fig, ax = plt.subplots(figsize=(11.2, 6.8))
    idx = np.arange(n_segments)
    width = 0.35
    eps_A_arr = np.array([r.eps_A_max for r in results])
    eps_AG_arr = np.array([max(r.eps_AG_max, 1e-18) for r in results])
    ax.bar(idx - width / 2, eps_A_arr, width, label="Track A (segment-affine)")
    ax.bar(idx + width / 2, eps_AG_arr, width, label="Track A+G (low-rank corrected)")
    ax.axhline(1.3e-9, color="red", linestyle="--", linewidth=1.0,
               label="Global affine floor (1.3e-9, cmame_b2 ref.)")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([
        f"S{i}\n[{results[i].f_lo:.0f},{results[i].f_hi:.0f}] Hz" for i in idx
    ], fontsize=10)
    ax.set_ylabel("Relative Frobenius residual")
    ax.set_title("Won 2013 §5.1 — segment-local residual: A vs A+G")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig_path = Path(__file__).resolve().parents[1] / "figures" / "cmame_b3_won2013_segment_local.png"
    fig_path.parent.mkdir(exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {fig_path.relative_to(Path.cwd()) if fig_path.is_relative_to(Path.cwd()) else fig_path}")

    # Summary
    print("\n[summary]")
    print(f"  Global affine residual (cmame_b2 ref.): 1.3e-09")
    print(f"  Max segment-local Track A residual    : {max(r.eps_A_max for r in results):.3e}")
    print(f"  Max segment-local Track A+G residual  : {max(r.eps_AG_max for r in results):.3e}")
    improvement_A = 1.3e-9 / max(max(r.eps_A_max for r in results), 1e-30)
    improvement_AG = 1.3e-9 / max(max(r.eps_AG_max for r in results), 1e-30)
    print(f"  Improvement (global -> A)              : x{improvement_A:.1f}")
    print(f"  Improvement (global -> A+G)            : x{improvement_AG:.1e}")


if __name__ == "__main__":
    main()
