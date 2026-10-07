"""Three-way comparison by damping mechanism (all free boundaries, frequency-dependent core with η_paper/3).

    sys1  Amichi            : viscoelastic core (η/3), no mount    → distributed damping only
    sys2  plate + mount     : elastic core,             with mount  → mount damping only
    sys3  Amichi + mount    : viscoelastic core (η/3),  with mount  → core + mount

All boundary conditions are free (only the damping mechanisms differ). The loss
factor of the paper's frequency-dependent core G*(ω) is scaled by 1/3 (avoids the
unphysical η>1).

Output (rectangular 0.3048×0.40, degeneracy removed):
    Fig A  overlay of the three near-corner driving-point FRFs (direct method)
    Fig B  most-complex mode of each system, |w| + phase (QEP with the core frozen at a representative 50 Hz)

Saved as png + pkl + source.md + csv/json (save_figure_with_metadata).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.linalg as sla
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

from polyqep.models.sandwich_plate_amichi2010 import (  # noqa: E402
    SandwichPlateParameters,
    assemble_global,
    assemble_K_base_and_shear,
)
from polyqep.qep_solver import solve_qep  # noqa: E402
from baselines.corner_mount import (  # noqa: E402
    corner_w_locator,
    resonant_mount,
    mount_qep_matrices,
    modal_phase_collinearity,
)
from _figure_io import save_figure_with_metadata  # noqa: E402

TWO_PI = 2.0 * np.pi
ETA_SCALE = 1.0 / 3.0
_F_ANCHORS = np.array([10.0, 50.0, 100.0, 200.0, 400.0, 700.0, 1000.0])
_E2_ANCHORS = np.array([4.5e6, 5.0e6, 6.0e6, 8.0e6, 1.4e7, 1.8e7, 2.0e7])
_ETA_ANCHORS = np.array([1.40, 1.36, 1.34, 1.30, 1.22, 1.18, 1.15])
_NU2 = 0.45


def paper_params(nx=8, ny=8):
    return SandwichPlateParameters(
        Lx=0.3048, Ly=0.4000,
        h1=0.475e-3, h2=0.035e-3, h3=0.475e-3,
        E1=2.1e11, E3=2.1e11, nu1=0.26, nu3=0.26,
        rho1=7780, rho3=7780, rho2=1134, nu2=0.45,
        E2_in_plane=1.0e7, nx=nx, ny=ny,
    )


def core_G_at_freq(freq_hz, eta_scale=ETA_SCALE):
    log_f = np.log(np.clip(freq_hz, 1.0, 2000.0))
    E2 = np.interp(log_f, np.log(_F_ANCHORS), _E2_ANCHORS)
    eta = np.interp(log_f, np.log(_F_ANCHORS), _ETA_ANCHORS)
    return E2 / (2.0 * (1.0 + _NU2)) * (1.0 + 1j * eta * eta_scale)


def w_field(phi, params):
    nx, ny = params.nx, params.ny
    npx = nx + 1
    W = np.zeros((ny + 1, nx + 1), dtype=complex)
    for j in range(ny + 1):
        for i in range(nx + 1):
            W[j, i] = phi[7 * (i + j * npx) + 4]
    return W


def normalize_global_phase(W):
    flat = W.ravel()
    return W / flat[int(np.argmax(np.abs(flat)))]


def all_w_dofs(params):
    nx, ny = params.nx, params.ny
    npx = nx + 1
    return [7 * (i + j * npx) + 4 for j in range(ny + 1) for i in range(nx + 1)]


def frf_direct(D_of_omega, omega, F, dof):
    Y = np.empty(omega.size, dtype=complex)
    for k, om in enumerate(omega):
        Y[k] = np.linalg.solve(D_of_omega(om), F)[dof]
    return Y


def most_complex_mode(M_star, C_star, K_star, w_dofs_all, fmax=120.0, rigid_tol=4.0):
    """Return the most complex mode (minimum MPC, below fmax) of the representative QEP as (f, mpc, eigvec)."""
    res = solve_qep(M_star, C_star, K_star)
    lam, vec = res.eigenvalues, res.eigenvectors
    f = np.abs(lam) / TWO_PI
    keep = np.isfinite(lam) & (lam.imag > 0) & (f > rigid_tol) & (f < fmax)
    lam, vec, f = lam[keep], vec[:, keep], f[keep]
    if lam.size == 0:
        return None
    mpcs = np.array([modal_phase_collinearity(vec[w_dofs_all, i]) for i in range(lam.size)])
    idx = int(np.argmin(mpcs))
    return f[idx], float(mpcs[idx]), vec[:, idx]


def main():
    params = paper_params()
    ndof = params.n_dof
    npx = params.nx + 1
    w_dofs_all = all_w_dofs(params)
    drive_dof = 7 * (1 + 1 * npx) + 4
    F = np.zeros(ndof, dtype=complex)
    F[drive_dof] = 1.0

    print("=" * 76)
    print("Three-way comparison — Amichi / plate+mount / Amichi+mount (all free, core η/3)")
    print("=" * 76)

    M_c, _ = assemble_global(params, complex(2.07e6, 0.0))
    M = np.asarray(M_c).real.copy()
    K_base, K_shear = assemble_K_base_and_shear(params)

    G_real = core_G_at_freq(50.0).real           # elastic core / representative stiffness
    K_plate = K_base + G_real * K_shear           # sys2 elastic core
    K_visc_rep = K_base + core_G_at_freq(50.0) * K_shear   # frozen viscoelastic core for the modes

    # Mount
    P, _ = corner_w_locator(params)
    f1 = np.sqrt(np.clip(sla.eigh(K_plate, M, eigvals_only=True), 0.0, None))[6] / TWO_PI
    omega1 = TWO_PI * f1
    A = params.Lx * params.Ly
    m_area = params.rho1 * params.h1 + params.rho2 * params.h2 + params.rho3 * params.h3
    k0 = (m_area * A) * (TWO_PI * 18.0) ** 2 / 4.0
    mount = resonant_mount(omega1, 2.0 * omega1, 0.5 * k0, k0)
    print(f"  core η={_ETA_ANCHORS.min()*ETA_SCALE:.2f}~{_ETA_ANCHORS.max()*ETA_SCALE:.2f}, "
          f"mount bounce≈18Hz, loss peak @ {f1:.1f}Hz")

    # --- Direct-method FRF ---
    freqs = np.linspace(1.0, 250.0, 600)
    omega = TWO_PI * freqs

    def K_core(om):
        return K_base + core_G_at_freq(om / TWO_PI) * K_shear

    print("\n  Direct-method FRFs (3 systems)...")
    Y1 = frf_direct(lambda om: K_core(om) - om ** 2 * M, omega, F, drive_dof)
    Y2 = frf_direct(lambda om: K_plate - om ** 2 * M + mount.dynamic_stiffness(om) * P,
                    omega, F, drive_dof)
    Y3 = frf_direct(lambda om: K_core(om) - om ** 2 * M + mount.dynamic_stiffness(om) * P,
                    omega, F, drive_dof)

    # --- Most complex mode (representative frozen QEP) ---
    zeroP = np.zeros_like(M, dtype=complex)
    m1 = most_complex_mode(M.astype(complex), zeroP, K_visc_rep, w_dofs_all)
    Ms2, Cs2, Ks2 = mount_qep_matrices(M, K_plate, P, mount)
    m2 = most_complex_mode(Ms2, Cs2, Ks2, w_dofs_all)
    # sys3 modes: frozen viscoelastic core (complex K_visc_rep) + mount
    Ms3, Cs3, Ks3 = mount_qep_matrices(M, K_visc_rep, P, mount)
    m3 = most_complex_mode(Ms3, Cs3, Ks3, w_dofs_all)

    labels = ["sys1 Amichi (visco core, no mount)",
              "sys2 plate + mount (elastic core)",
              "sys3 Amichi + mount (visco core)"]
    modes = [m1, m2, m3]
    print("\n  Most complex mode of each system:")
    for lab, m in zip(labels, modes):
        if m:
            print(f"    {lab}: f={m[0]:.1f}Hz, MPC={m[1]:.4f}")

    # =====================================================================
    # Fig A — three FRFs
    # =====================================================================
    figA, ax = plt.subplots(figsize=(11, 6))
    ax.semilogy(freqs, np.abs(Y1), "-", color="tab:blue", lw=1.4,
                label="sys1: Amichi (visco core, no mount)")
    ax.semilogy(freqs, np.abs(Y2), "-", color="tab:green", lw=1.4,
                label="sys2: plate + mount (elastic core)")
    ax.semilogy(freqs, np.abs(Y3), "-", color="tab:red", lw=1.5,
                label="sys3: Amichi + mount (visco core)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$|H(f)|$  (corner driving-point $|w|$) [m/N]")
    ax.set_title("Three-way FRF: damping mechanism comparison (all free BC, core η/3)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0, 250)
    figA.tight_layout()
    save_figure_with_metadata(
        figA, name="amichi_three_way_frf",
        data_dict={"frf": [{"f_Hz": float(ff), "sys1": float(abs(a)),
                            "sys2": float(abs(b)), "sys3": float(abs(c))}
                           for ff, a, b, c in zip(freqs, Y1, Y2, Y3)]},
        source_script=__file__,
        used_in=["3-way damping comparison FRF: Amichi / plate+mount / Amichi+mount"],
        key_numbers={"f1_Hz": f"{f1:.1f}", "eta_core": f"{_ETA_ANCHORS.mean()*ETA_SCALE:.2f}"},
    )

    # =====================================================================
    # Fig B — most complex mode |w| + phase (3 systems)
    # =====================================================================
    x = np.linspace(0, params.Lx, params.nx + 1)
    y = np.linspace(0, params.Ly, params.ny + 1)
    Xg, Yg = np.meshgrid(x, y)
    figB, axes = plt.subplots(3, 2, figsize=(9, 12))
    short = ["sys1 Amichi\n(no mount)", "sys2 plate\n+mount", "sys3 Amichi\n+mount"]
    for i, (lab, m) in enumerate(zip(short, modes)):
        if m is None:
            continue
        fm, mpc_v, v = m
        W = normalize_global_phase(w_field(v, params))
        amp = np.abs(W) / np.abs(W).max()
        ph = np.where(amp < 0.12, np.nan, np.angle(W, deg=True))
        ca = axes[i, 0].contourf(Xg, Yg, amp, levels=20, cmap="viridis")
        axes[i, 0].set_aspect("equal"); axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(f"{lab}\n{fm:.1f}Hz", fontsize=9)
        if i == 0:
            axes[i, 0].set_title("|w| (normalized)", fontsize=10)
        figB.colorbar(ca, ax=axes[i, 0], fraction=0.046, pad=0.04)
        cp = axes[i, 1].contourf(Xg, Yg, ph, levels=np.linspace(-180, 180, 25),
                                 cmap="twilight", vmin=-180, vmax=180)
        axes[i, 1].set_aspect("equal"); axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        axes[i, 1].set_title(f"phase [deg]  MPC={mpc_v:.3f}", fontsize=10)
        figB.colorbar(cp, ax=axes[i, 1], fraction=0.046, pad=0.04, ticks=[-180, 0, 180])

    figB.suptitle("Most-complex mode per system (all free BC)\n"
                  "uniform phase = real-like; spread phase = complex", fontsize=12)
    figB.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure_with_metadata(
        figB, name="amichi_three_way_modes",
        data_dict={"modes": [{"system": lab.replace("\n", " "),
                              "f_Hz": float(m[0]) if m else None,
                              "mpc": float(m[1]) if m else None}
                             for lab, m in zip(short, modes)]},
        source_script=__file__,
        used_in=["3-way damping comparison: most-complex mode shapes (|w|+phase)"],
        key_numbers={f"sys{i+1}_mpc": f"{m[1]:.3f}" for i, m in enumerate(modes) if m},
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
