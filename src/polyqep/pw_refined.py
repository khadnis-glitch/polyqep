"""PW-Refined (PW-R / PW-RH / PW-RG) — piecewise SOAR + raw-G(jω) online.

This module implements a hybrid method that keeps the segment-wise SOAR basis
of the polynomial-QEP framework while using the raw G(jω) directly in the
online FRF evaluation stage.

Key insight
-----------
Existing PW path:
    Offline : G(ω) → piecewise-quadratic fit → per-segment M*, C*, K*
              → SOAR basis Q_i (segment center shift), reduced (M*_r, C*_r, K*_r)
    Online  : D_r(ω) = K*_r + jω C*_r − ω² M*_r        [polynomial G assumption]
              y_r = D_r^{-1} F_r
    → the fit-residual floor sets a lower bound on the response error
      (Amichi §5.1 ≈ 6.16% max)

Xie 2018 XA path:
    Online: D_r(ω) = Q^H (K_e + G(jω) K_v − ω² M) Q   [raw G insertion]
    → no fit residual (XA Amichi §5.1 ≈ 0.0015%)

Three variants
--------------
PW-R (Refined, per-segment, raw-G online):
    Offline : Same per-segment SOAR Q_i as the existing PW path, plus reduced
              operator pieces
              K_base_r = Q_i^H K_base Q_i, K_shear_r = Q_i^H K_shear Q_i, ...
    Online  : Per-frequency segment dispatch + m×m solve with
              D_r(ω) = K_base_r + G_raw·K_shear_r − ω²·M_r.

PW-RH (Refined Hybrid, per-segment, raw-G online):
    Offline : Q_aug_i = orthonormalize([Q_SOAR_i | effective_eigenvecs_QEP_i])
              SOAR + modal augment. SVD rank-truncation.
    Online  : Same as PW-R.

PW-RG (Refined Global, cross-segment union-SVD, raw-G online):
    Offline : [Q_aug_0 | Q_aug_1 | ... | Q_aug_{S-1}] cross-segment column-stack
              → thin SVD orthogonalize → single broadband basis Q_global (rank
              determined automatically). Same idea as the union-SVD of Xie XA,
              except that the source basis is the SOAR + modal hybrid basis.
    Online  : No segment dispatch needed. A single Q_global handles all ω.
              D_r(ω) = K_base_r + G_raw·K_shear_r − ω²·M_r (m_global × m_global).

What the PW-R series does not discard
-------------------------------------
    - per-segment polynomial-QEP eigenvectors → for computing modal
      participation and loss factors
    - the affine identity K(G) = K_base + G·K_shear → online cost independent
      of the segment
    - frequency-independent modal outputs (offline-stage products)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.linalg as spla

from polyqep.rom import build_soar_rom, build_multi_shift_soar_rom


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class PWRefinedSegmentROM:
    """PW-Refined ROM for a single segment.

    Attributes
    ----------
    omega_lower, omega_upper : float
        Segment frequency bounds [rad/s].
    Q : ndarray, shape (n, m)
        Reduced basis — SOAR basis (or SOAR+modal augment for the hybrid
        variant).
    K_base_r : ndarray, shape (m, m), complex
        Q^H K_base Q — frequency-independent.
    K_shear_r : ndarray, shape (m, m), complex
        Q^H K_shear Q — frequency-independent.
    M_r : ndarray, shape (m, m), complex
        Q^H M Q — frequency-independent.
    F_r : ndarray, shape (m,), complex
        Q^H F — projected load.
    m_reduced : int
        Reduced basis dimension (m).
    """
    omega_lower: float
    omega_upper: float
    Q: np.ndarray
    K_base_r: np.ndarray
    K_shear_r: np.ndarray
    M_r: np.ndarray
    F_r: np.ndarray

    @property
    def m_reduced(self) -> int:
        return self.Q.shape[1]


# =============================================================================
# Helpers: project K_base, K_shear, M with Q
# =============================================================================

def _project_operator_pieces(
    Q: np.ndarray,
    K_base: np.ndarray,
    K_shear: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Q^T K_base Q, Q^T K_shear Q, Q^T M Q, Q^T F in one pass.

    Uses the transpose congruence Q^T(·)Q to preserve the complex-symmetric
    (T-symmetric) structure. The conjugate transpose Q^H would break complex
    symmetry and invalidate the bilinear orthogonality / Padé moment matching.
    """
    Qt = Q.T
    Kb_r = Qt @ K_base.astype(complex) @ Q
    Ks_r = Qt @ K_shear.astype(complex) @ Q
    M_r = Qt @ M.astype(complex) @ Q
    F_r = Qt @ F.astype(complex)
    return Kb_r, Ks_r, M_r, F_r


def _orthonormalize_columns_svd(
    Q_cat: np.ndarray,
    tol_sv: float = 1e-10,
) -> np.ndarray:
    """Orthonormalize columns via thin SVD + rank truncation.

    Parameters
    ----------
    Q_cat : (n, k_total) matrix whose columns combine SOAR, modal, etc.
    tol_sv : relative singular-value truncation threshold.

    Returns
    -------
    Q_out : (n, r) orthonormal columns, r <= k_total.
    """
    U, s, _ = np.linalg.svd(Q_cat, full_matrices=False)
    if s.size == 0:
        return U[:, :0]
    cutoff = tol_sv * s[0]
    rank = int(np.sum(s > cutoff))
    if rank == 0:
        rank = 1
    return U[:, :rank]


# =============================================================================
# PW-R basic variant: SOAR basis only
# =============================================================================

def build_pw_refined_segment_roms(
    K_base: np.ndarray,
    K_shear: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
    interval_solutions: list,
    m_per_seg: int = 20,
    use_multi_shift: bool = False,
) -> list[Optional[PWRefinedSegmentROM]]:
    """Build PW-Refined ROMs for all segments at once.

    Parameters
    ----------
    K_base, K_shear, M : ndarray, shape (n, n)
        Original (BC-applied) system matrices.
    F : ndarray, shape (n,)
        Load vector.
    interval_solutions : list
        Output of polyqep.intervals.solve_qep_per_interval.
        Each element is a dict with keys 'omega_lower', 'omega_upper', and
        'matrices' (containing K_star, C_star, M_star).
    m_per_seg : int
        Per-segment SOAR basis dimension (target).
    use_multi_shift : bool
        If True, use 3-shift SOAR within each segment (quartile shifts).

    Returns
    -------
    roms : list[PWRefinedSegmentROM | None]
        Same length as the number of segments. None where the SOAR build
        failed.
    """
    roms: list[Optional[PWRefinedSegmentROM]] = []

    for sol in interval_solutions:
        mat = sol["matrices"]
        K_star = mat["K_star"]
        C_star = mat["C_star"]
        M_star = mat["M_star"]
        seg_center = (sol["omega_lower"] + sol["omega_upper"]) / 2.0

        if use_multi_shift:
            lo = sol["omega_lower"]
            hi = sol["omega_upper"]
            shifts = [
                lo + 0.25 * (hi - lo),
                seg_center,
                lo + 0.75 * (hi - lo),
            ]
            m_per_shift = max(1, m_per_seg // len(shifts))
            try:
                rom_soar = build_multi_shift_soar_rom(
                    M_star, C_star, K_star, shifts, m_per_shift, F,
                )
            except RuntimeError:
                roms.append(None)
                continue
        else:
            try:
                rom_soar = build_soar_rom(
                    M_star, C_star, K_star, seg_center, m_per_seg, F,
                )
            except RuntimeError:
                roms.append(None)
                continue

        Q = rom_soar.Q  # (n, m) complex
        Kb_r, Ks_r, M_r, F_r = _project_operator_pieces(Q, K_base, K_shear, M, F)

        roms.append(PWRefinedSegmentROM(
            omega_lower=sol["omega_lower"],
            omega_upper=sol["omega_upper"],
            Q=Q,
            K_base_r=Kb_r,
            K_shear_r=Ks_r,
            M_r=M_r,
            F_r=F_r,
        ))

    return roms


# =============================================================================
# PW-RH (Hybrid): SOAR basis + polynomial-QEP eigenvectors augmented
# =============================================================================

def build_pw_refined_hybrid_segment_roms(
    K_base: np.ndarray,
    K_shear: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
    interval_solutions: list,
    m_soar_per_seg: int = 15,
    n_modal_augment: int = 5,
    use_multi_shift: bool = False,
    augment_tol_sv: float = 1e-10,
) -> list[Optional[PWRefinedSegmentROM]]:
    """Hybrid PW-Refined: SOAR basis + segment effective eigenvectors augmented.

    Parameters
    ----------
    m_soar_per_seg : int
        Number of SOAR basis columns per segment.
    n_modal_augment : int
        Number of effective eigenvectors within the segment used for the
        augment (closest to the shift first).
    augment_tol_sv : float
        Relative threshold for the SVD rank truncation after concatenation.

    Returns
    -------
    roms : list[PWRefinedSegmentROM | None]
        Actual dim m_reduced = min(m_soar + n_modal, rank_svd) per segment.
        Target total dim ≈ m_soar_per_seg + n_modal_augment (the SVD may
        reduce the rank).
    """
    roms: list[Optional[PWRefinedSegmentROM]] = []

    for sol in interval_solutions:
        mat = sol["matrices"]
        K_star = mat["K_star"]
        C_star = mat["C_star"]
        M_star = mat["M_star"]
        seg_center = (sol["omega_lower"] + sol["omega_upper"]) / 2.0

        if use_multi_shift:
            lo = sol["omega_lower"]
            hi = sol["omega_upper"]
            shifts = [
                lo + 0.25 * (hi - lo),
                seg_center,
                lo + 0.75 * (hi - lo),
            ]
            m_per_shift = max(1, m_soar_per_seg // len(shifts))
            try:
                rom_soar = build_multi_shift_soar_rom(
                    M_star, C_star, K_star, shifts, m_per_shift, F,
                )
            except RuntimeError:
                roms.append(None)
                continue
        else:
            try:
                rom_soar = build_soar_rom(
                    M_star, C_star, K_star, seg_center, m_soar_per_seg, F,
                )
            except RuntimeError:
                roms.append(None)
                continue

        Q_soar = rom_soar.Q

        # Effective eigenvectors at segment center (closest to the shift first)
        lams_pool = sol.get("lams_all")
        vecs_pool = sol.get("vecs_all")
        if lams_pool is None or vecs_pool is None or n_modal_augment <= 0:
            Q_aug = Q_soar
        else:
            valid = np.isfinite(lams_pool)
            lams = lams_pool[valid]
            vecs = vecs_pool[:, valid]
            if lams.size == 0:
                Q_aug = Q_soar
            else:
                shift_lam = 1j * seg_center
                dists = np.abs(lams - shift_lam)
                k = min(n_modal_augment, vecs.shape[1])
                pick = np.argsort(dists)[:k]
                V_modal = vecs[:, pick].astype(complex)
                # W12a: eigenvectors extracted from the linearization carry
                # λ-dependent small norms ‖v‖=1/√(1+|λ|²) — concatenating them
                # without normalization turns the SVD rank cut (tol_sv·σ0) into
                # a magnitude criterion instead of an independence criterion.
                # Unit-normalize each column before combining.
                V_modal = V_modal / np.linalg.norm(V_modal, axis=0, keepdims=True)
                # SVD orthonormalization after column concatenation
                Q_concat = np.column_stack([Q_soar, V_modal])
                Q_aug = _orthonormalize_columns_svd(Q_concat, tol_sv=augment_tol_sv)

        Kb_r, Ks_r, M_r, F_r = _project_operator_pieces(Q_aug, K_base, K_shear, M, F)

        roms.append(PWRefinedSegmentROM(
            omega_lower=sol["omega_lower"],
            omega_upper=sol["omega_upper"],
            Q=Q_aug,
            K_base_r=Kb_r,
            K_shear_r=Ks_r,
            M_r=M_r,
            F_r=F_r,
        ))

    return roms


# =============================================================================
# PW-RG (Global, cross-segment union-SVD)
# =============================================================================

@dataclass
class PWRefinedGlobalROM:
    """Single broadband reduced ROM unified via cross-segment union-SVD.

    Attributes
    ----------
    omega_min, omega_max : float
        Full evaluation frequency range [rad/s].
    Q : ndarray, shape (n, m_global)
        Unified orthonormal basis. Column-stack of the n_seg source bases
        followed by thin SVD.
    K_base_r, K_shear_r, M_r : ndarray, shape (m_global, m_global)
        Q^H · {K_base, K_shear, M} · Q.
    F_r : ndarray, shape (m_global,)
        Q^H F.
    m_reduced : int
        Reduced dimension.
    source_dims : list[int]
        Per-segment source column counts of the SVD input (for diagnostics).
    """
    omega_min: float
    omega_max: float
    Q: np.ndarray
    K_base_r: np.ndarray
    K_shear_r: np.ndarray
    M_r: np.ndarray
    F_r: np.ndarray
    source_dims: list

    @property
    def m_reduced(self) -> int:
        return self.Q.shape[1]


def build_pw_refined_global_rom(
    K_base: np.ndarray,
    K_shear: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
    interval_solutions: list,
    *,
    m_soar_per_seg: int = 15,
    n_modal_augment: int = 5,
    use_multi_shift: bool = False,
    augment_tol_sv: float = 1e-10,
    global_tol_sv: float = 1e-10,
    omega_min: float | None = None,
    omega_max: float | None = None,
) -> PWRefinedGlobalROM:
    """Build the PW-RG ROM. Per-segment hybrid basis → cross-segment union-SVD.

    Parameters
    ----------
    K_base, K_shear, M, F : original system matrices and load.
    interval_solutions : output of polyqep.intervals.solve_qep_per_interval.
    m_soar_per_seg, n_modal_augment : per-segment hybrid basis parameters.
    augment_tol_sv : threshold for the within-segment SVD orthogonalization.
    global_tol_sv : cross-segment union-SVD threshold (smaller → larger dim).
    omega_min, omega_max : if not given, set automatically to the interval
        minimum/maximum.

    Returns
    -------
    PWRefinedGlobalROM
    """
    # 1) Build the per-segment hybrid basis Q_i
    hybrid_roms = build_pw_refined_hybrid_segment_roms(
        K_base, K_shear, M, F, interval_solutions,
        m_soar_per_seg=m_soar_per_seg,
        n_modal_augment=n_modal_augment,
        use_multi_shift=use_multi_shift,
        augment_tol_sv=augment_tol_sv,
    )

    # 2) Collect only the valid segment Q_i
    Q_list = []
    source_dims = []
    for r in hybrid_roms:
        if r is None:
            source_dims.append(0)
            continue
        Q_list.append(r.Q)
        source_dims.append(r.m_reduced)

    if not Q_list:
        raise RuntimeError("PW-RG: SOAR failed for every segment — cannot construct a basis")

    # 3) Column-stack + thin SVD → single broadband basis
    Q_cat = np.column_stack(Q_list)
    Q_global = _orthonormalize_columns_svd(Q_cat, tol_sv=global_tol_sv)

    # 4) Reduced operator pieces
    Kb_r, Ks_r, M_r, F_r = _project_operator_pieces(
        Q_global, K_base, K_shear, M, F,
    )

    # 5) Metadata
    if omega_min is None:
        omega_min = min(s["omega_lower"] for s in interval_solutions)
    if omega_max is None:
        omega_max = max(s["omega_upper"] for s in interval_solutions)

    return PWRefinedGlobalROM(
        omega_min=float(omega_min),
        omega_max=float(omega_max),
        Q=Q_global,
        K_base_r=Kb_r,
        K_shear_r=Ks_r,
        M_r=M_r,
        F_r=F_r,
        source_dims=source_dims,
    )


def pw_refined_global_response(
    omegas: np.ndarray,
    rom: PWRefinedGlobalROM,
    G_func: Callable[[complex], complex],
    idx_obs: int,
) -> np.ndarray:
    """PW-RG online FRF (no segment dispatch needed).

    For each frequency ω:
        D_r(ω) = K_base_r + G_raw(jω) · K_shear_r − ω² · M_r
        y_r    = D_r^{-1} F_r
        y(ω)   = (Q · y_r)[idx_obs]
    """
    y = np.zeros(len(omegas), dtype=complex)
    for i, omega in enumerate(omegas):
        s = 1j * omega
        G_raw = complex(G_func(s))
        D_r = rom.K_base_r + G_raw * rom.K_shear_r - (omega ** 2) * rom.M_r
        y_r = np.linalg.solve(D_r, rom.F_r)
        u_full = rom.Q @ y_r
        y[i] = u_full[idx_obs]
    return y


# =============================================================================
# Online FRF evaluation
# =============================================================================

def pw_refined_response(
    omegas: np.ndarray,
    roms: list[Optional[PWRefinedSegmentROM]],
    G_func: Callable[[complex], complex],
    idx_obs: int,
) -> np.ndarray:
    """Compute the PW-Refined online FRF.

    For each frequency ω:
        1. Identify the containing segment j (binary search on omega_lower)
        2. Evaluate the raw G(jω)
        3. Assemble D_r = K_base_r + G_raw·K_shear_r − ω²·M_r
        4. y_r = D_r^{-1} F_r (m×m solve)
        5. u_full = Q · y_r, output y(ω) = u_full[idx_obs]

    Parameters
    ----------
    omegas : ndarray, shape (n_omega,) — evaluation angular frequencies [rad/s]
    roms : list[PWRefinedSegmentROM | None]
        Output of build_pw_refined_(hybrid_)segment_roms. Segments that are
        None return 0.
    G_func : callable
        Returns the raw G(s) for input s = jω. Same interface as in
        polyqep.intervals.solve_qep_per_interval.
    idx_obs : int
        Observation degree-of-freedom index.

    Returns
    -------
    y : ndarray, shape (n_omega,), complex
    """
    y = np.zeros(len(omegas), dtype=complex)
    lowers = np.array([
        r.omega_lower if r is not None else np.inf for r in roms
    ])

    for i, omega in enumerate(omegas):
        # segment dispatch
        j = int(np.searchsorted(lowers, omega, side="right") - 1)
        j = max(0, min(j, len(roms) - 1))
        rom = roms[j]
        if rom is None:
            y[i] = 0.0 + 0.0j
            continue

        s = 1j * omega
        G_raw = complex(G_func(s))
        D_r = rom.K_base_r + G_raw * rom.K_shear_r - (omega ** 2) * rom.M_r
        y_r = np.linalg.solve(D_r, rom.F_r)
        u_full = rom.Q @ y_r
        y[i] = u_full[idx_obs]

    return y


# =============================================================================
# Diagnostics: timing of the PW-R reduced operator pieces
# =============================================================================

def build_pw_refined_roms_profiled(
    K_base: np.ndarray,
    K_shear: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
    interval_solutions: list,
    *,
    variant: str = "soar",
    m_per_seg: int = 20,
    m_soar_per_seg: int = 15,
    n_modal_augment: int = 5,
    use_multi_shift: bool = False,
    global_tol_sv: float = 1e-10,
):
    """Build the PW-R/PW-RH/PW-RG ROM together with a timing profile.

    variant : 'soar' (PW-R) | 'hybrid' (PW-RH) | 'global' (PW-RG).

    Returns
    -------
    roms_or_rom, profile : tuple
        The per-segment variants return list[PWRefinedSegmentROM | None].
        The global variant returns a single PWRefinedGlobalROM.
    """
    t_soar_0 = time.perf_counter()
    if variant == "soar":
        roms = build_pw_refined_segment_roms(
            K_base, K_shear, M, F, interval_solutions,
            m_per_seg=m_per_seg, use_multi_shift=use_multi_shift,
        )
        t_total = time.perf_counter() - t_soar_0
        m_reduced_per_seg = [r.m_reduced if r is not None else 0 for r in roms]
        profile = {
            "variant": variant,
            "build_total_s": t_total,
            "m_reduced_per_segment": m_reduced_per_seg,
            "m_reduced_sum": int(sum(m_reduced_per_seg)),
            "n_segments": len(roms),
            "n_failed": int(sum(1 for r in roms if r is None)),
        }
        return roms, profile
    elif variant == "hybrid":
        roms = build_pw_refined_hybrid_segment_roms(
            K_base, K_shear, M, F, interval_solutions,
            m_soar_per_seg=m_soar_per_seg,
            n_modal_augment=n_modal_augment,
            use_multi_shift=use_multi_shift,
        )
        t_total = time.perf_counter() - t_soar_0
        m_reduced_per_seg = [r.m_reduced if r is not None else 0 for r in roms]
        profile = {
            "variant": variant,
            "build_total_s": t_total,
            "m_reduced_per_segment": m_reduced_per_seg,
            "m_reduced_sum": int(sum(m_reduced_per_seg)),
            "n_segments": len(roms),
            "n_failed": int(sum(1 for r in roms if r is None)),
        }
        return roms, profile
    elif variant == "global":
        rom_g = build_pw_refined_global_rom(
            K_base, K_shear, M, F, interval_solutions,
            m_soar_per_seg=m_soar_per_seg,
            n_modal_augment=n_modal_augment,
            use_multi_shift=use_multi_shift,
            global_tol_sv=global_tol_sv,
        )
        t_total = time.perf_counter() - t_soar_0
        profile = {
            "variant": variant,
            "build_total_s": t_total,
            "m_reduced_global": int(rom_g.m_reduced),
            "source_dims_per_segment": rom_g.source_dims,
            "source_dims_sum": int(sum(rom_g.source_dims)),
            "n_segments": len(interval_solutions),
        }
        return rom_g, profile
    else:
        raise ValueError(f"unknown variant: {variant!r}")
