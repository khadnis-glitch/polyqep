"""Additional studies — corner-mount complex-mode demonstration with a passive mount.

The submitted demonstration used a mount whose loss is a parabola in omega
peaking at the first flexible frequency f1 with zero crossings at -f1 and
3 f1. Above 3 f1 (about 79 Hz) the imaginary part of the mount stiffness is
negative, and the "most complex" modes reported below 120 Hz carry negative
damping ratios: they lie in the band where that mount model is not passive.

This script keeps the plate, the core, and the quadratic-in-omega mount form,
widens the loss parabola so that the mount is passive over the whole search
band, selects the most complex mode among the dissipative ones, and verifies
that every mode in the band is dissipative.

Run::

    PYTHONPATH="$PWD:$PWD/scripts" python scripts/rev_mount_modes.py

Output: ``data/rev_mount_modes.json`` and ``figures/rev_mount_modes.pdf``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

import amichi_three_way_comparison as tw
from baselines.corner_mount import (
    MountPolynomial, corner_w_locator, modal_phase_collinearity,
    mount_qep_matrices, resonant_mount,
)
from polyqep.models.sandwich_plate_amichi2010 import (
    assemble_K_base_and_shear, assemble_global,
)
from polyqep.qep_solver import solve_qep

_ROOT = Path(__file__).resolve().parent.parent
TWO_PI = 2.0 * np.pi
F_SEARCH_HZ = (4.0, 120.0)
BOUNCE_HZ = 18.0


def band_modes(Ms, Cs, Ks, w_dofs):
    res = solve_qep(Ms, Cs, Ks)
    lam, vec = res.eigenvalues, res.eigenvectors
    f = np.abs(lam) / TWO_PI
    keep = (np.isfinite(lam) & (lam.imag > 0)
            & (f > F_SEARCH_HZ[0]) & (f < F_SEARCH_HZ[1]))
    lam, vec, f = lam[keep], vec[:, keep], f[keep]
    mpc = np.array([modal_phase_collinearity(vec[w_dofs, i]) for i in range(lam.size)])
    return f, mpc, -lam.real / np.abs(lam), vec


def main():
    parser = argparse.ArgumentParser(description="passive corner-mount demo")
    parser.add_argument("--width", type=float, default=5.0,
                        help="half-width of the loss parabola in units of omega_1")
    parser.add_argument("--peak", type=float, default=2.0,
                        help="peak loss in units of the mount stiffness k0")
    parser.add_argument("--viscous", type=float, default=None,
                        help="use a viscous dashpot with this loss factor at f1")
    args = parser.parse_args()

    params = tw.paper_params()
    w_dofs = tw.all_w_dofs(params)
    M = np.asarray(assemble_global(params, complex(2.07e6, 0.0))[0]).real.copy()
    K_base, K_shear = assemble_K_base_and_shear(params)
    G50 = tw.core_G_at_freq(50.0)
    K_plate = K_base + G50.real * K_shear
    K_visc = K_base + G50 * K_shear
    P, _ = corner_w_locator(params)

    f1 = np.sqrt(np.clip(sla.eigh(K_plate, M, eigvals_only=True), 0.0, None))[6] / TWO_PI
    w1 = TWO_PI * f1
    m_area = (params.rho1 * params.h1 + params.rho2 * params.h2
              + params.rho3 * params.h3)
    k0 = m_area * params.Lx * params.Ly * (TWO_PI * BOUNCE_HZ) ** 2 / 4.0
    if args.viscous is None:
        mount = resonant_mount(w1, args.width * w1, args.peak * k0, k0)
        passive_up_to_hz = (1.0 + args.width) * f1
        description = (f"loss parabola peaking at f1, zero at {passive_up_to_hz:.1f} Hz, "
                       f"peak loss {args.peak:g} k0")
    else:
        mount = MountPolynomial(k0=k0, k1=0.0, k2=0.0, C0=0.0,
                                C1=args.viscous * k0 / w1, C2=0.0)
        passive_up_to_hz = float("inf")
        description = f"viscous dashpot, loss factor {args.viscous:g} at f1"
    f_band = np.linspace(*F_SEARCH_HZ, 400)
    k_band = np.array([mount.dynamic_stiffness(TWO_PI * f) for f in f_band])
    eta_band = k_band.imag / k_band.real

    zero = np.zeros_like(M, dtype=complex)
    systems = [
        ("Amichi (viscoelastic core, no mount)", "distributed core",
         (M.astype(complex), zero, K_visc)),
        ("Plain plate + corner mount", "local mount only",
         mount_qep_matrices(M, K_plate, P, mount)),
        ("Amichi + corner mount", "core + mount",
         mount_qep_matrices(M, K_visc, P, mount)),
    ]
    rows, picked = [], []
    for name, source, mats in systems:
        f, mpc, zeta, vec = band_modes(*mats, w_dofs)
        dissipative = zeta > 0
        idx = np.flatnonzero(dissipative)[int(np.argmin(mpc[dissipative]))]
        rows.append(dict(system=name, damping_source=source,
                         f_hz=float(f[idx]), mpc=float(mpc[idx]),
                         zeta=float(zeta[idx]), n_modes_in_band=int(f.size),
                         n_non_dissipative=int((~dissipative).sum()),
                         modes=[dict(f_hz=float(a), mpc=float(b), zeta=float(c))
                                for a, b, c in sorted(zip(f, mpc, zeta))]))
        picked.append((name, f[idx], mpc[idx], vec[:, idx]))
        print(f"{name:<40s} f={f[idx]:6.1f} Hz  MPC={mpc[idx]:.3f}  zeta={zeta[idx]:+.3f}"
              f"  ({f.size} modes in band, {(~dissipative).sum()} non-dissipative)")

    result = dict(
        plate="0.3048 x 0.40 m, 8x8 mesh, free-free, core loss eta/3, "
              "core modulus frozen at 50 Hz for the modal analysis",
        mount=description, f1_hz=float(f1), k0_N_per_m=float(k0),
        bounce_hz=BOUNCE_HZ, passive_up_to_hz=passive_up_to_hz,
        search_band_hz=list(F_SEARCH_HZ),
        mount_loss_factor_in_band=[float(eta_band.min()), float(eta_band.max())],
        mount_loss_factor_at_f1=float(
            (mount.dynamic_stiffness(w1).imag / mount.dynamic_stiffness(w1).real)),
        rows=rows)
    tag = (f"viscous{args.viscous:g}" if args.viscous is not None
           else f"w{args.width:g}_p{args.peak:g}")
    with open(_ROOT / "data" / f"rev_mount_modes_{tag}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"mount: {description}; loss factor in band "
          f"{eta_band.min():.2f}-{eta_band.max():.2f}")

    x = np.linspace(0, params.Lx, params.nx + 1)
    y = np.linspace(0, params.Ly, params.ny + 1)
    Xg, Yg = np.meshgrid(x, y)
    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    short = ["Viscoelastic core,\nno mount", "Plain plate\n+ corner mount",
             "Viscoelastic core\n+ corner mount"]
    for i, (lab, (_, fm, mpc_v, v)) in enumerate(zip(short, picked)):
        W = tw.normalize_global_phase(tw.w_field(v, params))
        amp = np.abs(W) / np.abs(W).max()
        ph = np.where(amp < 0.12, np.nan, np.angle(W, deg=True))
        ca = axes[i, 0].contourf(Xg, Yg, amp, levels=20, cmap="viridis")
        cp = axes[i, 1].contourf(Xg, Yg, ph, levels=np.linspace(-180, 180, 25),
                                 cmap="twilight", vmin=-180, vmax=180)
        for ax in axes[i]:
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(f"{lab}\n{fm:.1f} Hz", fontsize=10)
        axes[i, 1].set_title(f"phase [deg], MPC = {mpc_v:.3f}", fontsize=10)
        if i == 0:
            axes[i, 0].set_title("|w| (normalized)", fontsize=10)
        fig.colorbar(ca, ax=axes[i, 0], fraction=0.046, pad=0.04)
        fig.colorbar(cp, ax=axes[i, 1], fraction=0.046, pad=0.04, ticks=[-180, 0, 180])
    fig.tight_layout()
    out_fig = _ROOT / "figures" / f"rev_mount_modes_{tag}.pdf"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
