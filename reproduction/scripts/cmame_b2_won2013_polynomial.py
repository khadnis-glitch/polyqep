# === [auto-resized: figsize x1.5] ===
"""Multi-case validation: Won 2013 sandwich beam.

Goal
----
Demonstrate the *applicability boundary* of the K(G) = K_base + G·K_shear
decomposition that underlies the polynomial-QEP framework (see the paper).  Concretely:

  1. Reproduce the per-frequency direct response of the Won-2013 [3]
     sandwich beam at the paper-comparable mesh (n_elem = 8, clamped-SS,
     mid-span transverse drive) under the paper Table 6 frequency-dependent
     viscoelastic core G*(ω).
  2. Diagnose the K(G) decomposition residual for the Won-2013 element
     (shape functions are functions of G via c1*, c2*) versus the
     Amichi-2010 [1] plate element (shape functions G-independent).
  3. Aggregate three validation cases (see the paper):
        (a) Amichi 8x8 free-free   (decomposable, framework applies)
        (b) Amichi 8x8 simply-sup. (decomposable, framework applies)
        (c) Won 2013 sandwich beam (non-decomposable, direct method only)

Output
------
stdout : multi-case summary table + decomposition residual diagnostic
data/cmame_b2_won2013.csv : per-frequency direct response (re/im)
data/cmame_b2_kg_residual.csv : K(G) decomposition residuals (3 verify points)
figures/cmame_b2_won2013_frf.png : Won 2013 FRF (Table 6 viscoelastic core)
figures/cmame_b2_kg_residual.png : K(G) decomposition residual bar chart
                                   (Amichi machine-prec vs Won 2013 nonzero)

Run from project root:
    python scripts/cmame_b2_won2013_polynomial.py
"""
# @MX:NOTE: Multi-case validation. Quantifies the non-decomposability of the
#           Won 2013 K(G) to establish the framework applicability boundary.
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

from baselines.sandwich_beam_won2013 import (
    SandwichBeamParameters,
    apply_boundary_conditions as won_apply_bc,
    assemble_global as won_assemble,
    freq_dependent_core_G_paper_table6,
)
from polyqep.models.sandwich_plate_amichi2010 import (
    apply_boundary_conditions as amichi_apply_bc,
    assemble_K_base_and_shear,
    assemble_global as amichi_assemble,
)

from scripts.phase3_amichi_polynomial_qep import paper_5_1_params


# --------------------------------------------------------------------------- #
# 1.  Won 2013 — paper §5.1 setup (clamped-SS, n_elem = 8)                    #
# --------------------------------------------------------------------------- #
def won2013_paper_params(n_elem: int = 8) -> SandwichBeamParameters:
    """Paper Table 1 / §5.1 sandwich beam configuration."""
    return SandwichBeamParameters(
        L=0.320, h1=0.003, h2=0.003, h3=0.003,
        E1=2.1e11, E3=2.1e11,
        rho1=7850.0, rho2=2000.0, rho3=7850.0,
        b=0.009,
        n_elem=n_elem,
    )


def won2013_direct_response(
    params: SandwichBeamParameters,
    G_func,
    omega_array: np.ndarray,
    F_node: int,
    obs_node: int,
    F_dof: int = 0,
    obs_dof: int = 0,
    bc_left: str = "clamped",
    bc_right: str = "simply_supported",
) -> tuple[np.ndarray, float]:
    """Per-frequency direct response Y(omega) at observation DOF.

    Returns (Y, cpu_seconds).
    """
    n_nodes = params.n_elem + 1
    n_dof = 3 * n_nodes
    F_full = np.zeros(n_dof, dtype=complex)
    F_full[3 * F_node + F_dof] = 1.0
    obs_full = 3 * obs_node + obs_dof

    Y = np.zeros(omega_array.size, dtype=complex)
    t0 = time.perf_counter()
    for k, omega in enumerate(omega_array):
        G_om = G_func(omega)
        M, K = won_assemble(params, G_om)
        M_r, K_r, F_r, free = won_apply_bc(M, K, F_full, bc_left, bc_right, n_nodes)
        D_r = K_r - omega * omega * M_r
        try:
            x_r = np.linalg.solve(D_r, F_r)
        except np.linalg.LinAlgError:
            Y[k] = np.nan + 0.0j
            continue
        full = np.zeros(n_dof, dtype=complex)
        full[free] = x_r
        Y[k] = full[obs_full]
    cpu = time.perf_counter() - t0
    return Y, cpu


# --------------------------------------------------------------------------- #
# 2.  K(G) decomposition residual diagnostic                                  #
# --------------------------------------------------------------------------- #
def kg_residual_won2013(
    params: SandwichBeamParameters,
    G_lo: complex,
    G_hi: complex,
    G_test_list: list[complex],
) -> list[dict]:
    """Estimate the best two-point affine model K(G) ~ K_base + G · K_shear
    from (G_lo, G_hi), then verify at each G_test.

    The Won-2013 element K_e includes K_E (extensional) and K_S (shear) in
    which the integrands depend on shape functions H, N_pp evaluated at
    (c1*, c2*) = f(G).  Hence K(G) is not affine in G; we expect a
    nonzero verification residual scaling with the curvature of K(G).
    """
    _, K_lo = won_assemble(params, G_lo)
    _, K_hi = won_assemble(params, G_hi)
    K_shear_est = (K_hi - K_lo) / (G_hi - G_lo)
    K_base_est = K_lo - G_lo * K_shear_est

    rows: list[dict] = []
    for G_t in G_test_list:
        _, K_act = won_assemble(params, G_t)
        K_pred = K_base_est + G_t * K_shear_est
        num = np.linalg.norm(K_act - K_pred)
        den = np.linalg.norm(K_act)
        rows.append({
            "G_test_re": float(np.real(G_t)),
            "G_test_im": float(np.imag(G_t)),
            "abs_residual": float(num),
            "rel_residual": float(num / max(den, 1e-30)),
        })
    return rows


def kg_residual_amichi() -> list[dict]:
    """Direct check of K_base + G·K_shear vs assemble_global at three G points."""
    params = paper_5_1_params(nx=8, ny=8)
    K_base, K_shear = assemble_K_base_and_shear(params)

    rows: list[dict] = []
    for G_t in [1.0e6 + 0.0j, 1.72e6 + 2.33e6j, 5.0e6 - 1.0e6j]:
        M_full, K_act = amichi_assemble(params, G_t)
        K_pred = K_base.astype(complex) + G_t * K_shear.astype(complex)
        num = np.linalg.norm(K_act - K_pred)
        den = np.linalg.norm(K_act)
        rows.append({
            "G_test_re": float(np.real(G_t)),
            "G_test_im": float(np.imag(G_t)),
            "abs_residual": float(num),
            "rel_residual": float(num / max(den, 1e-30)),
        })
    return rows


# --------------------------------------------------------------------------- #
# 3.  Main                                                                    #
# --------------------------------------------------------------------------- #
def main() -> None:
    np.set_printoptions(linewidth=140, precision=4)

    print("=" * 78)
    print("Multi-case validation (Won 2013 + Amichi)")
    print("=" * 78)

    # 3.1 Won 2013 — direct FRF (paper Table 6 viscoelastic core)
    params_won = won2013_paper_params(n_elem=8)
    n_nodes = params_won.n_elem + 1
    mid_node = n_nodes // 2  # transverse mid-span drive (Won 2013 §5.1 setup)
    print(f"\n[1] Won 2013 setup")
    print(f"    L={params_won.L*1e3:.0f} mm, h={params_won.h1*1e3:.0f}/{params_won.h2*1e3:.0f}"
          f"/{params_won.h3*1e3:.0f} mm, n_elem={params_won.n_elem}, mid_node={mid_node}")
    print(f"    BC: clamped-SS, drive: w-DOF at mid_node, observation: same DOF")

    G_func = freq_dependent_core_G_paper_table6()
    f_eval = np.linspace(5.0, 500.0, 200)
    omega_eval = 2.0 * np.pi * f_eval
    print(f"    Evaluation grid: {len(f_eval)} freqs, {f_eval[0]:.0f}–{f_eval[-1]:.0f} Hz")

    print(f"    Computing per-frequency direct response …")
    Y_won, cpu_won = won2013_direct_response(
        params_won, G_func, omega_eval,
        F_node=mid_node, obs_node=mid_node,
        F_dof=0, obs_dof=0,
        bc_left="clamped", bc_right="simply_supported",
    )
    n_won = 3 * (params_won.n_elem + 1) - 3 - 2  # clamped (3) + SS (2) constraints
    print(f"    Reduced DOF n = {n_won}")
    print(f"    Direct CPU: {cpu_won*1000:.1f} ms ({cpu_won*1000/len(f_eval):.2f} ms/freq)")
    print(f"    |Y| at 1st peak: max |Y| = {np.abs(Y_won).max():.3e}, "
          f"f_peak = {f_eval[int(np.argmax(np.abs(Y_won)))]:.1f} Hz")

    # 3.2 K(G) decomposition residual — Won 2013 (non-decomposable)
    print(f"\n[2] K(G) decomposition residual diagnostic")
    G_lo = G_func(2.0 * np.pi * 10.0)   # f = 10 Hz
    G_hi = G_func(2.0 * np.pi * 1000.0) # f = 1000 Hz
    G_test_list = [
        G_func(2.0 * np.pi * 50.0),
        G_func(2.0 * np.pi * 200.0),
        G_func(2.0 * np.pi * 800.0),
    ]
    won_rows = kg_residual_won2013(params_won, G_lo, G_hi, G_test_list)

    print(f"    Won 2013 element (paper Table 6 G):")
    print(f"      anchors: G(f=10 Hz) = {G_lo.real:.3e} + {G_lo.imag:.3e}j Pa")
    print(f"               G(f=1000 Hz) = {G_hi.real:.3e} + {G_hi.imag:.3e}j Pa")
    for k, r in enumerate(won_rows):
        f_test = [50.0, 200.0, 800.0][k]
        print(f"      verify @ f={f_test:>6.1f} Hz: rel residual = {r['rel_residual']:.3e}  "
              f"(abs = {r['abs_residual']:.3e})")

    # Amichi reference (machine precision expected)
    am_rows = kg_residual_amichi()
    print(f"    Amichi 2010 plate element (paper §5.1 G samples):")
    for k, r in enumerate(am_rows):
        print(f"      verify pt {k+1}: rel residual = {r['rel_residual']:.3e}  "
              f"(abs = {r['abs_residual']:.3e})")

    # 3.3 Multi-case summary table  (cases (a) and (b) are not recomputed here; their numbers are in the paper)
    print(f"\n[3] Multi-case summary (3 cases)")
    summary = [
        {
            "case": "Amichi 8x8 free-free",
            "n_dof": 567,
            "decomposable": "yes (machine prec)",
            "framework": "Polynomial-QEP + ROM (m=20)",
            "accuracy": "see the paper",
            "speedup": "see the paper",
            "ref": "see the paper",
        },
        {
            "case": "Amichi 8x8 simply-supported",
            "n_dof": 471,
            "decomposable": "yes (machine prec)",
            "framework": "Modal superposition",
            "accuracy": "see the paper",
            "speedup": "n/a",
            "ref": "see the paper",
        },
        {
            "case": "Won 2013 (n_elem=8, clamped-SS)",
            "n_dof": n_won,
            "decomposable": f"no (rel res ~{won_rows[1]['rel_residual']:.1e})",
            "framework": "per-frequency direct only",
            "accuracy": "n/a (no ROM)",
            "speedup": "n/a (no ROM)",
            "ref": "this script",
        },
    ]
    for r in summary:
        print(f"    - {r['case']:<32} n={r['n_dof']:>4}  "
              f"decomp={r['decomposable']}  framework={r['framework']}")
        print(f"        accuracy: {r['accuracy']:<32} speedup: {r['speedup']:<24} "
              f"[{r['ref']}]")

    # 3.4 Save CSVs
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    with (data_dir / "cmame_b2_won2013.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f_Hz", "Y_re", "Y_im", "abs_Y"])
        for f, y in zip(f_eval, Y_won):
            w.writerow([f, float(np.real(y)), float(np.imag(y)), float(np.abs(y))])

    with (data_dir / "cmame_b2_kg_residual.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["element", "G_test_re", "G_test_im", "abs_residual", "rel_residual"])
        for r in won_rows:
            w.writerow(["Won2013", r["G_test_re"], r["G_test_im"],
                        r["abs_residual"], r["rel_residual"]])
        for r in am_rows:
            w.writerow(["Amichi2010", r["G_test_re"], r["G_test_im"],
                        r["abs_residual"], r["rel_residual"]])
    print(f"\n[saved] data/cmame_b2_won2013.csv, data/cmame_b2_kg_residual.csv")

    # 3.5 Figure 1 — Won 2013 FRF
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 9), sharex=True,
                             gridspec_kw=dict(hspace=0.08))
    ax_top, ax_bot = axes

    valid = np.isfinite(Y_won)
    ax_top.semilogy(f_eval[valid], np.abs(Y_won[valid]), "k-", lw=1.4,
                    label="Direct (per-ω assembly + factorisation)")
    ax_top.set_ylabel(r"$|Y(\omega)|$ [m/N]")
    ax_top.grid(True, which="both", alpha=0.3)
    ax_top.legend(fontsize=11.0, loc="upper right", framealpha=0.9)
    ax_top.set_title(
        "Won 2013 sandwich beam, paper Table 6 viscoelastic core\n"
        rf"($n_{{\mathrm{{elem}}}}={params_won.n_elem}$, "
        "clamped-SS, mid-span drive)")

    ax_bot.plot(f_eval[valid], np.angle(Y_won[valid], deg=True), "C3-", lw=1.0)
    ax_bot.set_xlabel("Frequency [Hz]")
    ax_bot.set_ylabel(r"$\arg Y(\omega)$  [deg]")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_ylim(-185, 185)

    fig.tight_layout()
    fig_path = fig_dir / "cmame_b2_won2013_frf.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {fig_path}")

    # 3.6 Figure 2 — K(G) residual bar chart (Amichi vs Won 2013)
    fig, ax = plt.subplots(1, 1, figsize=(11.2, 7.2))

    won_res = np.array([r["rel_residual"] for r in won_rows])
    am_res = np.array([r["rel_residual"] for r in am_rows])
    floor = 1e-17  # log-scale floor for machine-precision residuals
    won_plot = np.maximum(won_res, floor)
    am_plot = np.maximum(am_res, floor)

    x_won = np.arange(len(won_res))
    x_am = np.arange(len(am_res)) + len(won_res) + 0.5

    bars_won = ax.bar(x_won, won_plot, width=0.7, color="C3",
                      label="Won 2013 (non-affine in G)")
    bars_am = ax.bar(x_am, am_plot, width=0.7, color="C0",
                     label="Amichi 2010 (affine in G — machine prec.)")
    ax.set_yscale("log")
    ax.set_ylim(floor / 3.0, 1.0)
    ax.axhline(1e-12, color="gray", ls=":", lw=0.8)
    ax.text(x_am[-1] + 0.4, 1e-12, "  machine precision", color="gray",
            fontsize=10.7, va="center")

    ax.set_xticks(np.concatenate([x_won, x_am]))
    ax.set_xticklabels(
        [f"f={f:g} Hz" for f in [50, 200, 800]] +
        [f"pt {k+1}" for k in range(len(am_res))],
        rotation=0, fontsize=12.0,
    )
    ax.set_ylabel(r"$\|\mathbf{K}(G_t)-\mathbf{K}_{\rm base}-G_t\,\mathbf{K}_{\rm shear}\| /"
                  r" \|\mathbf{K}(G_t)\|$")
    ax.set_title("K(G) decomposition residual: Won 2013 vs Amichi 2010 (8×8 plate)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=12.0, loc="upper right")

    # annotate Won residual values
    for x, v in zip(x_won, won_res):
        ax.text(x, max(v, floor) * 1.5, f"{v:.1e}", ha="center", fontsize=10.7)

    fig.tight_layout()
    fig_path2 = fig_dir / "cmame_b2_kg_residual.png"
    fig.savefig(fig_path2, dpi=300)
    plt.close(fig)
    print(f"[saved] {fig_path2}")

    print("\n" + "=" * 78)
    print("Multi-case validation sweep complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
