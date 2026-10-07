"""Won 2013 method comparison: PW (poly-QEP ROM) vs direct.

Goal
----
Won 2013 sandwich-beam element is "almost-affine": both K(G) and M(G) depend
on the complex frequency-dependent shear modulus G.  Segment-local recovery
makes them affine *per segment*:

    K(G) ~ K_{0,i} + G K_{1,i}
    M(G) ~ M_{0,i} + G M_{1,i}

The degree-4 PEP that the M_1 term would create is AVOIDED by dropping M_1
(i.e. M ~ M_{0,i}), reducing the per-frequency problem back to degree-2.

This script (a) quantifies how valid the M_1-drop is via the per-segment
ratio ||G_mid M_{1,i}||_F / ||M_{0,i}||_F, and (b) compares the FRF accuracy
and speedup of several methods against the per-frequency direct reference.

Methods
-------
1. Direct (reference)  : won2013_direct_response — full M(G), non-affine.
2. deg2-M1drop         : per-segment affine K + M_0 only, FULL solve per ω
                          (isolates the M_1-drop + segment-local-affine error).
3. PW-R                : per-segment SOAR ROM from (K_0,i, K_1,i, M_0,i),
                          raw-G online with frequency->segment dispatch.
4. PW-RH               : PW-R basis augmented with each segment's degree-2 QEP
                          eigenvectors, SVD-orthogonalised, raw-G online.
5. PW-RG               : cross-segment thin-SVD of per-segment hybrid bases into
                          a single global basis, raw-G online (no dispatch).

All methods run on the SAME grid / DOF / BC (clamped-SS, mid-span w-drive).

Outputs
-------
data/cmame_b5_won2013_method_comparison.csv   one row per method
figures/cmame_b5_won2013_frf_comparison.png   |Y| overlay + rel-error
stdout                                         clean summary tables

Run from project root:
    python -m scripts.cmame_b5_won2013_method_comparison
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to sys.path (same pattern as the other scripts)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cmame_b2_won2013_polynomial import (  # noqa: E402
    won2013_direct_response,
    won2013_paper_params,
)
from scripts.cmame_b4_won2013_MK_segment_local import (  # noqa: E402
    compute_MK_won,
    two_anchor_affine,
)
from polyqep.rom import build_soar_rom  # noqa: E402
from baselines.sandwich_beam_won2013 import (  # noqa: E402
    apply_boundary_conditions,
    freq_dependent_core_G_paper_table6,
)
from polyqep.qep_solver import solve_qep  # noqa: E402


# --------------------------------------------------------------------------- #
# Data classes                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class SegmentAffine:
    """Segment-local affine factors of a single segment (free-DOF space after BC application)."""

    index: int
    f_lo: float
    f_hi: float
    omega_lo: float
    omega_hi: float
    G_lo: complex
    G_hi: complex
    G_mid: complex
    K0: np.ndarray
    K1: np.ndarray
    M0: np.ndarray
    M1: np.ndarray
    # diagnostics
    ratio_M1: float = 0.0      # ||G_mid·M1||_F / ||M0||_F
    K_affine_resid: float = 0.0  # segment-local K affine residual max (interior)


@dataclass
class SegmentROM:
    """Reduced ROM of a single segment (shared by PW-R / PW-RH)."""

    omega_lo: float
    omega_hi: float
    Q: np.ndarray
    K0_r: np.ndarray
    K1_r: np.ndarray
    M0_r: np.ndarray
    F_r: np.ndarray

    @property
    def m(self) -> int:
        return self.Q.shape[1]


@dataclass
class MethodResult:
    """Accuracy/speed results of a single method."""

    name: str
    Y: np.ndarray
    m: int                       # reduced size (full solve -> n)
    peak_max_err: float = np.nan
    peak_mean_err: float = np.nan
    full_max_err: float = np.nan
    offline_ms: float = 0.0
    online_ms_per_freq: float = np.nan
    speedup: float = np.nan
    notes: str = ""


# --------------------------------------------------------------------------- #
# Common settings                                                             #
# --------------------------------------------------------------------------- #
SEG_BREAKPOINTS_HZ = [5.0, 50.0, 150.0, 350.0, 500.0]  # 4 segments (paper)
M_PER_SEG = 20
M_SOAR_PER_SEG = 15
N_MODAL_AUGMENT = 5
SVD_TOL = 1e-10


def build_force_vector_free(params, mid_node: int) -> tuple[np.ndarray, int, np.ndarray]:
    """Unit load vector F_free in the free-DOF space after BC application, observation reduced index, and free_dofs.

    Drive = observe = w-DOF at mid_node.
    """
    n_nodes = params.n_elem + 1
    n_dof = 3 * n_nodes
    F_full = np.zeros(n_dof, dtype=complex)
    drive_global = 3 * mid_node + 0
    F_full[drive_global] = 1.0
    _, _, F_red, free = apply_boundary_conditions(
        np.zeros((n_dof, n_dof), dtype=complex),
        np.zeros((n_dof, n_dof), dtype=complex),
        F_full, "clamped", "simply_supported", n_nodes,
    )
    obs_global = 3 * mid_node + 0
    obs_reduced = int(np.where(free == obs_global)[0][0])
    return F_red.astype(complex), obs_reduced, free


# --------------------------------------------------------------------------- #
# 1. Segment-local affine factors + M1-drop diagnostics                       #
# --------------------------------------------------------------------------- #
def build_segment_affines(params, G_func) -> list[SegmentAffine]:
    """Compute K0,K1,M0,M1 (apply_bc=True) and diagnostics for each segment."""
    segs: list[SegmentAffine] = []
    bps = SEG_BREAKPOINTS_HZ
    for i in range(len(bps) - 1):
        f_lo, f_hi = bps[i], bps[i + 1]
        omega_lo = 2.0 * np.pi * f_lo
        omega_hi = 2.0 * np.pi * f_hi
        G_lo = G_func(omega_lo)
        G_hi = G_func(omega_hi)

        M_lo, K_lo = compute_MK_won(params, G_lo, apply_bc=True)
        M_hi, K_hi = compute_MK_won(params, G_hi, apply_bc=True)

        K0, K1 = two_anchor_affine(K_lo, K_hi, G_lo, G_hi)
        M0, M1 = two_anchor_affine(M_lo, M_hi, G_lo, G_hi)

        f_mid = 0.5 * (f_lo + f_hi)
        omega_mid = 2.0 * np.pi * f_mid
        G_mid = G_func(omega_mid)

        # M1-drop diagnostic: ||G_mid·M1|| / ||M0||
        ratio_M1 = float(np.linalg.norm(G_mid * M1, "fro")
                         / max(np.linalg.norm(M0, "fro"), 1e-30))

        # segment-local K affine residual (interior test points)
        alphas = np.linspace(0.0, 1.0, 9)[1:-1]
        kres = 0.0
        for a in alphas:
            om_t = omega_lo + a * (omega_hi - omega_lo)
            G_t = G_func(om_t)
            _, K_t = compute_MK_won(params, G_t, apply_bc=True)
            K_pred = K0 + G_t * K1
            r = float(np.linalg.norm(K_t - K_pred, "fro")
                      / max(np.linalg.norm(K_t, "fro"), 1e-30))
            kres = max(kres, r)

        segs.append(SegmentAffine(
            index=i, f_lo=f_lo, f_hi=f_hi,
            omega_lo=omega_lo, omega_hi=omega_hi,
            G_lo=G_lo, G_hi=G_hi, G_mid=G_mid,
            K0=K0, K1=K1, M0=M0, M1=M1,
            ratio_M1=ratio_M1, K_affine_resid=kres,
        ))
    return segs


def segment_dispatch(omega: float, segs: list) -> int:
    """Index of the segment containing frequency ω; outside the range, the nearest end segment."""
    for i, s in enumerate(segs):
        if omega <= s.omega_hi or i == len(segs) - 1:
            return i
    return 0


# --------------------------------------------------------------------------- #
# 2. deg2-M1drop full solve                                                    #
# --------------------------------------------------------------------------- #
def deg2_m1drop_response(
    segs: list[SegmentAffine], G_func, omega_eval: np.ndarray,
    F_free: np.ndarray, obs_reduced: int,
) -> tuple[np.ndarray, float]:
    """per-segment affine K + M0, full solve per ω. Returns (Y, online_seconds)."""
    Y = np.zeros(omega_eval.size, dtype=complex)
    t0 = time.perf_counter()
    for k, omega in enumerate(omega_eval):
        i = segment_dispatch(omega, segs)
        s = segs[i]
        G = G_func(omega)
        D = (s.K0 + G * s.K1) - (omega ** 2) * s.M0
        try:
            x = np.linalg.solve(D, F_free)
        except np.linalg.LinAlgError:
            Y[k] = np.nan
            continue
        Y[k] = x[obs_reduced]
    return Y, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# 3. PW-R / PW-RH per-segment ROM build                                        #
# --------------------------------------------------------------------------- #
def _project(Q, K0, K1, M0, F):
    Qh = Q.conj().T
    return (Qh @ K0 @ Q, Qh @ K1 @ Q, Qh @ M0 @ Q, Qh @ F)


def build_pwr_roms(
    segs: list[SegmentAffine], F_free: np.ndarray,
    m_per_seg: int = M_PER_SEG,
) -> tuple[list[SegmentROM], float]:
    """PW-R: per-segment SOAR ROM (raw-G online). Returns (roms, offline_seconds).

    The SOAR basis is built from the segment-midpoint representative system (M0, 0, K0 + G_center·K1).
    Online: D_r = K0_r + G·K1_r − ω²·M0_r.
    """
    roms: list[SegmentROM] = []
    t0 = time.perf_counter()
    for s in segs:
        omega_center = 0.5 * (s.omega_lo + s.omega_hi)
        K_rep = s.K0 + s.G_mid * s.K1   # segment representative (G at midpoint)
        rom_soar = build_soar_rom(s.M0, np.zeros_like(s.M0), K_rep,
                                  omega_center, m_per_seg, F_free)
        Q = rom_soar.Q
        K0_r, K1_r, M0_r, F_r = _project(Q, s.K0, s.K1, s.M0, F_free)
        roms.append(SegmentROM(
            omega_lo=s.omega_lo, omega_hi=s.omega_hi, Q=Q,
            K0_r=K0_r, K1_r=K1_r, M0_r=M0_r, F_r=F_r,
        ))
    return roms, time.perf_counter() - t0


def build_pwrh_roms(
    segs: list[SegmentAffine], F_free: np.ndarray,
    m_soar_per_seg: int = M_SOAR_PER_SEG, n_modal: int = N_MODAL_AUGMENT,
) -> tuple[list[SegmentROM], float, list[np.ndarray]]:
    """PW-RH: SOAR basis + segment degree-2 QEP eigenvectors, SVD-orthogonalise.

    Returns (roms, offline_seconds, list_of_hybrid_Q) — the hybrid Q are reused by PW-RG.
    """
    roms: list[SegmentROM] = []
    hybrid_Qs: list[np.ndarray] = []
    t0 = time.perf_counter()
    for s in segs:
        omega_center = 0.5 * (s.omega_lo + s.omega_hi)
        K_rep = s.K0 + s.G_mid * s.K1
        rom_soar = build_soar_rom(s.M0, np.zeros_like(s.M0), K_rep,
                                  omega_center, m_soar_per_seg, F_free)
        Q_soar = rom_soar.Q

        # segment representative degree-2 QEP: (λ² M0 + K_rep) v = 0  (C=0)
        qep = solve_qep(s.M0, np.zeros_like(s.M0), K_rep)
        shift_lam = 1j * omega_center
        d = np.abs(qep.eigenvalues - shift_lam)
        valid = np.isfinite(d)
        order = np.argsort(d[valid])
        pick = np.flatnonzero(valid)[order][:n_modal]
        V_modal = qep.eigenvectors[:, pick].astype(complex)

        Q_cat = np.column_stack([Q_soar, V_modal])
        U, sv, _ = np.linalg.svd(Q_cat, full_matrices=False)
        cutoff = SVD_TOL * sv[0] if sv.size else 0.0
        rank = max(1, int(np.sum(sv > cutoff)))
        Q_aug = U[:, :rank]

        hybrid_Qs.append(Q_aug)
        K0_r, K1_r, M0_r, F_r = _project(Q_aug, s.K0, s.K1, s.M0, F_free)
        roms.append(SegmentROM(
            omega_lo=s.omega_lo, omega_hi=s.omega_hi, Q=Q_aug,
            K0_r=K0_r, K1_r=K1_r, M0_r=M0_r, F_r=F_r,
        ))
    return roms, time.perf_counter() - t0, hybrid_Qs


def pw_segment_response(
    roms: list[SegmentROM], segs: list[SegmentAffine], G_func,
    omega_eval: np.ndarray, obs_reduced: int, free: np.ndarray,
) -> tuple[np.ndarray, float]:
    """PW-R/PW-RH online FRF with frequency->segment dispatch.

    obs_reduced is an index in the free-DOF space after BC application (that row is extracted after the Q lift).
    """
    Y = np.zeros(omega_eval.size, dtype=complex)
    t0 = time.perf_counter()
    for k, omega in enumerate(omega_eval):
        i = segment_dispatch(omega, segs)
        rom = roms[i]
        G = G_func(omega)
        D_r = rom.K0_r + G * rom.K1_r - (omega ** 2) * rom.M0_r
        y_r = np.linalg.solve(D_r, rom.F_r)
        u_full = rom.Q @ y_r
        Y[k] = u_full[obs_reduced]
    return Y, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# 4. PW-RG global ROM                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class GlobalROM:
    Q: np.ndarray
    K0_r: np.ndarray
    K1_r: np.ndarray
    M0_r: np.ndarray
    F_r: np.ndarray

    @property
    def m(self) -> int:
        return self.Q.shape[1]


def build_pwrg_rom(
    hybrid_Qs: list[np.ndarray], segs: list[SegmentAffine],
    F_free: np.ndarray, t_hybrid_offline: float,
) -> tuple[GlobalROM, float]:
    """PW-RG: cross-segment thin-SVD of hybrid bases -> single global basis.

    Since K0,K1,M0 differ per segment, no single global affine factor exists.
    PW-RG online therefore uses the (K0,K1,M0) of the segment containing ω,
    projected onto Q_global (a single basis Q_global without dispatch; the
    per-segment reduced operators are cached offline). Here the 4 sets of
    per-segment reduced operators are stored, and online only the reduced
    operator is swapped via segment dispatch.
    NOTE: the basis is a single global one (no dispatch needed), but the
    operators are segment-local, so online only the reduced operator is
    selected (the Q lift uses the single Q_global).
    """
    t0 = time.perf_counter()
    Q_cat = np.column_stack(hybrid_Qs)
    U, sv, _ = np.linalg.svd(Q_cat, full_matrices=False)
    cutoff = SVD_TOL * sv[0] if sv.size else 0.0
    rank = max(1, int(np.sum(sv > cutoff)))
    Q_global = U[:, :rank]

    # per-segment reduced operators (projected onto the single Q_global)
    Qh = Q_global.conj().T
    K0_list, K1_list, M0_list = [], [], []
    for s in segs:
        K0_list.append(Qh @ s.K0 @ Q_global)
        K1_list.append(Qh @ s.K1 @ Q_global)
        M0_list.append(Qh @ s.M0 @ Q_global)
    F_r = Qh @ F_free
    offline = t_hybrid_offline + (time.perf_counter() - t0)
    rom = GlobalROM(Q=Q_global, K0_r=K0_list, K1_r=K1_list,
                    M0_r=M0_list, F_r=F_r)
    return rom, offline


def pwrg_response(
    rom: GlobalROM, segs: list[SegmentAffine], G_func,
    omega_eval: np.ndarray, obs_reduced: int,
) -> tuple[np.ndarray, float]:
    """PW-RG online: single Q_global, per-segment reduced-operator dispatch."""
    Y = np.zeros(omega_eval.size, dtype=complex)
    t0 = time.perf_counter()
    for k, omega in enumerate(omega_eval):
        i = segment_dispatch(omega, segs)
        G = G_func(omega)
        D_r = rom.K0_r[i] + G * rom.K1_r[i] - (omega ** 2) * rom.M0_r[i]
        y_r = np.linalg.solve(D_r, rom.F_r)
        u_full = rom.Q @ y_r
        Y[k] = u_full[obs_reduced]
    return Y, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# 5. Accuracy metrics                                                         #
# --------------------------------------------------------------------------- #
def accuracy_metrics(Y: np.ndarray, Y_ref: np.ndarray) -> tuple[float, float, float]:
    """peak-band max/mean rel error + full-band max rel error vs Y_ref."""
    ref_abs = np.abs(Y_ref)
    valid = np.isfinite(Y) & np.isfinite(Y_ref)
    rel = np.full(Y.size, np.nan)
    denom = np.maximum(ref_abs, 1e-30)
    rel[valid] = np.abs(Y[valid] - Y_ref[valid]) / denom[valid]

    full_max = float(np.nanmax(rel))

    mask = valid & (ref_abs > 0.01 * np.nanmax(ref_abs))
    if np.any(mask):
        peak_max = float(np.nanmax(rel[mask]))
        peak_mean = float(np.nanmean(rel[mask]))
    else:
        peak_max = peak_mean = np.nan
    return peak_max, peak_mean, full_max


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    np.set_printoptions(linewidth=140, precision=4)
    print("=" * 92)
    print("Won 2013 method comparison: PW (poly-QEP ROM) vs direct")
    print("=" * 92)

    params = won2013_paper_params(n_elem=8)
    G_func = freq_dependent_core_G_paper_table6()
    n_nodes = params.n_elem + 1
    mid_node = n_nodes // 2

    f_eval = np.linspace(5.0, 500.0, 200)
    omega_eval = 2.0 * np.pi * f_eval

    F_free, obs_reduced, free = build_force_vector_free(params, mid_node)
    n_red = F_free.size
    print(f"\n[setup] BC clamped-SS, n_elem={params.n_elem}, mid_node={mid_node}")
    print(f"[setup] reduced free-DOF n = {n_red}, obs reduced idx = {obs_reduced}")
    print(f"[setup] grid: {len(f_eval)} freqs, {f_eval[0]:.0f}-{f_eval[-1]:.0f} Hz")
    print(f"[setup] segment breakpoints [Hz] = {SEG_BREAKPOINTS_HZ}")

    # --- Segment affines + M1-drop diagnostic ---
    segs = build_segment_affines(params, G_func)
    print("\n[diagnostic] per-segment M1-drop magnitude and K affine residual")
    print(f"  {'seg':>3} | {'band [Hz]':>12} | {'||G_mid·M1||/||M0||':>20} | {'K affine resid':>15}")
    print("  " + "-" * 60)
    for s in segs:
        print(f"  {s.index:>3} | [{s.f_lo:>4.0f},{s.f_hi:>4.0f}]  | "
              f"{s.ratio_M1:>20.4e} | {s.K_affine_resid:>15.4e}")

    results: list[MethodResult] = []

    # --- Method 1: Direct (reference) ---
    print("\n[run] Direct (per-frequency assembly + factorisation) ...")
    Y_direct, cpu_direct = won2013_direct_response(
        params, G_func, omega_eval,
        F_node=mid_node, obs_node=mid_node, F_dof=0, obs_dof=0,
        bc_left="clamped", bc_right="simply_supported",
    )
    direct_online_per_freq = cpu_direct / len(f_eval)
    res_direct = MethodResult(
        name="Direct", Y=Y_direct, m=n_red,
        peak_max_err=0.0, peak_mean_err=0.0, full_max_err=0.0,
        offline_ms=0.0, online_ms_per_freq=direct_online_per_freq * 1e3,
        speedup=1.0, notes="reference (full M(G))",
    )
    results.append(res_direct)
    print(f"      online {cpu_direct*1e3:.1f} ms ({direct_online_per_freq*1e3:.3f} ms/freq), "
          f"max|Y|={np.nanmax(np.abs(Y_direct)):.3e} at "
          f"{f_eval[int(np.nanargmax(np.abs(Y_direct)))]:.1f} Hz")

    # --- Method 2: deg2-M1drop (full solve) ---
    print("[run] deg2-M1drop (segment-local affine K + M0, full solve) ...")
    Y_deg2, t_deg2 = deg2_m1drop_response(segs, G_func, omega_eval, F_free, obs_reduced)
    pmax, pmean, fmax = accuracy_metrics(Y_deg2, Y_direct)
    res_deg2 = MethodResult(
        name="deg2-M1drop", Y=Y_deg2, m=n_red,
        peak_max_err=pmax, peak_mean_err=pmean, full_max_err=fmax,
        offline_ms=0.0, online_ms_per_freq=(t_deg2 / len(f_eval)) * 1e3,
        speedup=direct_online_per_freq / (t_deg2 / len(f_eval)),
        notes="full solve, no ROM",
    )
    results.append(res_deg2)
    print(f"      online {t_deg2*1e3:.1f} ms, peak_max={pmax:.3e}, peak_mean={pmean:.3e}")

    # --- Method 3: PW-R ---
    print("[run] PW-R (per-segment SOAR ROM, raw-G online) ...")
    roms_pwr, off_pwr = build_pwr_roms(segs, F_free, m_per_seg=M_PER_SEG)
    Y_pwr, t_pwr = pw_segment_response(roms_pwr, segs, G_func, omega_eval, obs_reduced, free)
    pmax, pmean, fmax = accuracy_metrics(Y_pwr, Y_direct)
    m_pwr = [r.m for r in roms_pwr]
    res_pwr = MethodResult(
        name="PW-R", Y=Y_pwr, m=int(round(np.mean(m_pwr))),
        peak_max_err=pmax, peak_mean_err=pmean, full_max_err=fmax,
        offline_ms=off_pwr * 1e3, online_ms_per_freq=(t_pwr / len(f_eval)) * 1e3,
        speedup=direct_online_per_freq / (t_pwr / len(f_eval)),
        notes=f"m_per_seg={m_pwr}",
    )
    results.append(res_pwr)
    print(f"      offline {off_pwr*1e3:.1f} ms, online {t_pwr*1e3:.2f} ms, "
          f"m_per_seg={m_pwr}, peak_max={pmax:.3e}")

    # --- Method 4: PW-RH ---
    print("[run] PW-RH (SOAR + degree-2 QEP eigenvectors, SVD-orth) ...")
    roms_pwrh, off_pwrh, hybrid_Qs = build_pwrh_roms(
        segs, F_free, m_soar_per_seg=20, n_modal=0)  # pure SOAR-20, no modal augmentation
    Y_pwrh, t_pwrh = pw_segment_response(roms_pwrh, segs, G_func, omega_eval, obs_reduced, free)
    pmax, pmean, fmax = accuracy_metrics(Y_pwrh, Y_direct)
    m_pwrh = [r.m for r in roms_pwrh]
    res_pwrh = MethodResult(
        name="PW-RH", Y=Y_pwrh, m=int(round(np.mean(m_pwrh))),
        peak_max_err=pmax, peak_mean_err=pmean, full_max_err=fmax,
        offline_ms=off_pwrh * 1e3, online_ms_per_freq=(t_pwrh / len(f_eval)) * 1e3,
        speedup=direct_online_per_freq / (t_pwrh / len(f_eval)),
        notes=f"m_per_seg={m_pwrh}",
    )
    results.append(res_pwrh)
    print(f"      offline {off_pwrh*1e3:.1f} ms, online {t_pwrh*1e3:.2f} ms, "
          f"m_per_seg={m_pwrh}, peak_max={pmax:.3e}")

    # --- Method 5: PW-RG ---
    print("[run] PW-RG (cross-segment union-SVD global basis, raw-G online) ...")
    rom_g, off_pwrg = build_pwrg_rom(hybrid_Qs, segs, F_free, off_pwrh)
    Y_pwrg, t_pwrg = pwrg_response(rom_g, segs, G_func, omega_eval, obs_reduced)
    pmax, pmean, fmax = accuracy_metrics(Y_pwrg, Y_direct)
    res_pwrg = MethodResult(
        name="PW-RG", Y=Y_pwrg, m=rom_g.m,
        peak_max_err=pmax, peak_mean_err=pmean, full_max_err=fmax,
        offline_ms=off_pwrg * 1e3, online_ms_per_freq=(t_pwrg / len(f_eval)) * 1e3,
        speedup=direct_online_per_freq / (t_pwrg / len(f_eval)),
        notes=f"m_global={rom_g.m}",
    )
    results.append(res_pwrg)
    print(f"      offline {off_pwrg*1e3:.1f} ms, online {t_pwrg*1e3:.2f} ms, "
          f"m_global={rom_g.m}, peak_max={pmax:.3e}")

    # --- Summary table ---
    print("\n" + "=" * 92)
    print("SUMMARY — accuracy + speedup vs direct reference")
    print("=" * 92)
    hdr = (f"{'method':<13} | {'m':>4} | {'peak_max':>10} | {'peak_mean':>10} | "
           f"{'full_max':>10} | {'offl_ms':>9} | {'onl_ms/f':>9} | {'speedup':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.name:<13} | {r.m:>4} | {r.peak_max_err:>10.3e} | "
              f"{r.peak_mean_err:>10.3e} | {r.full_max_err:>10.3e} | "
              f"{r.offline_ms:>9.1f} | {r.online_ms_per_freq:>9.4f} | "
              f"{r.speedup:>9.1f}")

    # --- CSV ---
    out_csv = Path(__file__).resolve().parents[1] / "data" / "cmame_b5_won2013_method_comparison.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "m", "peak_max_err", "peak_mean_err", "full_max_err",
                    "offline_ms", "online_ms_per_freq", "speedup", "notes"])
        for r in results:
            w.writerow([r.name, r.m, r.peak_max_err, r.peak_mean_err, r.full_max_err,
                        r.offline_ms, r.online_ms_per_freq, r.speedup, r.notes])
    print(f"\n[saved] {out_csv}")

    # --- Figure ---
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.0, 9.0), sharex=True,
                                         gridspec_kw=dict(hspace=0.08))
    styles = {
        "Direct": ("k-", 1.6), "deg2-M1drop": ("C0--", 1.1),
        "PW-R": ("C1-", 1.0), "PW-RH": ("C2-.", 1.0), "PW-RG": ("C3:", 1.2),
    }
    for r in results:
        if r.name == "PW-RH":
            continue
        st, lw = styles.get(r.name, ("-", 1.0))
        v = np.isfinite(r.Y)
        ax_top.semilogy(f_eval[v], np.abs(r.Y[v]), st, lw=lw, label=r.name)
    ax_top.set_ylabel(r"$|Y(\omega)|$  [m/N]")
    ax_top.grid(True, which="both", alpha=0.3)
    ax_top.legend(fontsize=11, loc="best", ncol=2)
    ax_top.set_title("Won 2013 sandwich beam — method comparison\n"
                     "(clamped-SS, mid-span w-drive, paper Table 6 core)")

    ref_abs = np.abs(Y_direct)
    for r in results:
        if r.name in ("Direct", "PW-RH"):
            continue
        st, lw = styles.get(r.name, ("-", 1.0))
        rel = np.abs(r.Y - Y_direct) / np.maximum(ref_abs, 1e-30)
        v = np.isfinite(rel)
        ax_bot.semilogy(f_eval[v], rel[v], st, lw=lw, label=r.name)
    ax_bot.set_xlabel("Frequency [Hz]")
    ax_bot.set_ylabel("Relative error vs direct")
    ax_bot.grid(True, which="both", alpha=0.3)
    ax_bot.legend(fontsize=11, loc="best", ncol=2)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.09, hspace=0.08)
    fig_path = Path(__file__).resolve().parents[1] / "figures" / "cmame_b5_won2013_frf_comparison.png"
    fig_path.parent.mkdir(exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {fig_path}")

    # --- Interpretation hint values ---
    max_ratio = max(s.ratio_M1 for s in segs)
    print("\n[interpretation hints]")
    print(f"  max per-segment ||G_mid·M1||/||M0|| = {max_ratio:.3e}")
    print(f"  deg2-M1drop peak_max rel error       = {res_deg2.peak_max_err:.3e}")
    print(f"  PW-R   peak_max / speedup            = {res_pwr.peak_max_err:.3e} / x{res_pwr.speedup:.1f}")
    print(f"  PW-RH  peak_max / speedup            = {res_pwrh.peak_max_err:.3e} / x{res_pwrh.speedup:.1f}")
    print(f"  PW-RG  peak_max / speedup            = {res_pwrg.peak_max_err:.3e} / x{res_pwrg.speedup:.1f}")


if __name__ == "__main__":
    main()
