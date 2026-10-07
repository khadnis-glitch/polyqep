"""Segmentwise quadratic + per-segment ROM quantification.

Weakness of the earlier ROM study (phase3_amichi_rom_single_G.py):
    the ROM was demonstrated with a constant G_{32Hz} value → the core of the
    framework (segmentwise piecewise-quadratic G(ω)) was not used.

This script:
    1. paper §5.1 8×8 mesh (567 DOF, free-free, center excitation)
    2. G(ω) sampling 1-1500 Hz + 5-segment piecewise-quadratic fit
    3. Per-segment M*_i, C*_i, K*_i assemble (i = 0..4)
    4. Per-segment full QEP solve (offline build cost)
    5. Per-segment modal-truncation basis (m_per_seg ∈ {4, 8, 12, 20, 40, 80})
    6. Frequency dispatch + per-segment reduced QEP solve
    7. Compare to Direct method (per-ω full system) baseline
    8. Report table for the ROM section of the paper

Output
------
stdout: quantitative comparison table (build, solve, speedup, accuracy)
JSON   : data/cache/cmame_e2_segmentwise_rom_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as spla

# reuse the functions of the segmentwise pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_amichi_polynomial_qep import (  # type: ignore
    G_complex_at_freq,
    assemble_complex_per_interval_amichi,
    direct_response_per_freq,
    fit_G_polynomial,
    modal_response_per_interval,
    modal_weights_blockwise,
    paper_5_1_params,
    rom_response_per_interval_reuse,
    sample_G_omega,
    solve_qep_per_interval,
)
from polyqep.models.sandwich_plate_amichi2010 import (
    apply_boundary_conditions,
    assemble_K_base_and_shear,
    assemble_global,
)
from polyqep.rom import (
    build_soar_rom,
    build_multi_shift_soar_rom,
    soar_response,
)


def setup_amichi_8x8(bc: str = "simply_supported_w"):
    """paper §5.1 — 8×8 mesh, BC by argument, center excitation.

    BC selection:
        - 'simply_supported_w': 471 free DOF, K* invertible, matches the SS modal-sum result (0.0000%)
        - 'free': 567 free DOF, rigid-body subspace requires the free-free recovery (skip ROM)
    """
    params = paper_5_1_params(nx=8, ny=8)
    K_base, K_shear = assemble_K_base_and_shear(params)
    M_full, _ = assemble_global(params, complex(1.0, 0.0))
    F_full = np.zeros(M_full.shape[0], dtype=complex)

    cx = params.nx // 2
    cy = params.ny // 2
    n_nodes_x = params.nx + 1
    cw_full = 7 * (cx + cy * n_nodes_x) + 4
    F_full[cw_full] = 1.0

    M_red, _, F_red, free_dofs = apply_boundary_conditions(
        M_full, K_base.astype(complex), F_full, params, bc=bc
    )
    free_idx = np.asarray(free_dofs)
    K_base_red = K_base[np.ix_(free_idx, free_idx)]
    K_shear_red = K_shear[np.ix_(free_idx, free_idx)]
    M_red_real = (M_red.real + M_red.real.T) / 2.0
    cw_red = int(np.where(free_idx == cw_full)[0][0])

    return params, M_red_real, K_base_red, K_shear_red, F_red, cw_red


def build_segments(M_red, K_base_red, K_shear_red, n_seg=5):
    """5-segment piecewise-quadratic fit + per-segment complex matrices."""
    omega_fit, G_re, G_im = sample_G_omega(1.0, 1500.0, 400)
    spline_re, spline_im, _ = fit_G_polynomial(omega_fit, G_re, G_im, n_segments=n_seg)

    interval_mats = assemble_complex_per_interval_amichi(
        spline_re, spline_im, M_red, K_base_red, K_shear_red
    )

    # G(ω) fit accuracy
    g_re_fit = np.zeros_like(omega_fit)
    g_im_fit = np.zeros_like(omega_fit)
    for seg_re, seg_im in zip(spline_re.segments, spline_im.segments):
        mask = (omega_fit >= seg_re.omega_lower) & (omega_fit < seg_re.omega_upper)
        if seg_re is spline_re.segments[-1]:
            mask = (omega_fit >= seg_re.omega_lower) & (omega_fit <= seg_re.omega_upper)
        g_re_fit[mask] = seg_re.a * omega_fit[mask] ** 2 + seg_re.b * omega_fit[mask] + seg_re.c
        g_im_fit[mask] = seg_im.a * omega_fit[mask] ** 2 + seg_im.b * omega_fit[mask] + seg_im.c

    err_re = np.abs(G_re - g_re_fit) / np.maximum(np.abs(G_re), 1e-30) * 100
    err_im = np.abs(G_im - g_im_fit) / np.maximum(np.abs(G_im), 1e-30) * 100

    return interval_mats, {
        "n_segments": n_seg,
        "fit_err_re_max": float(err_re.max()),
        "fit_err_re_mean": float(err_re.mean()),
        "fit_err_im_max": float(err_im.max()),
        "fit_err_im_mean": float(err_im.mean()),
    }


def mode_accelerated_rom_response(
    omega_array, interval_solutions, F_red, idx_obs, m_per_seg,
):
    """Per-segment mode-acceleration ROM (see the paper).

    Each segment i:
        y_static_i = (K*_i)^-1 F                    [full static response]
        Pick m_per_seg modes nearest segment center shift
        w_k = v_k(obs) · v_k^T F / a_k             [modal weight]
        Y_i(ω) = y_static_i[obs]
               + Σ_{k in retained} w_k · [1/(jω - λ_k) + 1/λ_k]

    The static-correction term captures the quasi-static contribution of all
    omitted modes through the analytical identity:
        (K*)^-1 F  =  Σ_all v_k v_k^T F / (a_k (-λ_k))
    so the truncated sum + static correction = exact modal expansion.

    Returns (Y, t_build_ms).
    """
    Y = np.zeros(omega_array.size, dtype=complex)
    lowers = np.array([s["omega_lower"] for s in interval_solutions])
    cache = []

    t0 = time.perf_counter()
    for sol in interval_solutions:
        mat = sol["matrices"]
        K_star = mat["K_star"]
        M_star = mat["M_star"]
        C_star = mat["C_star"]
        shift_omega = (sol["omega_lower"] + sol["omega_upper"]) / 2.0
        shift_lam = 1j * shift_omega

        # 1) Static response (per segment) — small per-segment cost
        y_static_full = spla.solve(K_star, F_red)
        y_static_obs = y_static_full[idx_obs]

        # 2) Pick m_per_seg modes nearest shift from full QEP eigenvectors
        lams_pool = sol["lams_all"]
        vecs_pool = sol["vecs_all"]
        valid = np.isfinite(lams_pool)
        lams_pool = lams_pool[valid]
        vecs_pool = vecs_pool[:, valid]

        dists = np.abs(lams_pool - shift_lam)
        idx_pick = np.argsort(dists)[: min(m_per_seg, vecs_pool.shape[1])]
        lams_k = lams_pool[idx_pick]
        vecs_k = vecs_pool[:, idx_pick]

        # 3) QEP modal weights — (near-)degenerate clusters use
        # basis-invariant block residues (per-mode a_k assumes distinct λ)
        lams_k, weights = modal_weights_blockwise(
            lams_k, vecs_k, M_star, C_star, F_red, idx_obs
        )
        cache.append((y_static_obs, lams_k, weights))
    t_build = time.perf_counter() - t0

    for i, omega in enumerate(omega_array):
        j = int(np.searchsorted(lowers, omega, side="right") - 1)
        j = max(0, min(j, len(interval_solutions) - 1))
        ys, lams, weights = cache[j]
        if lams is None or lams.size == 0:
            Y[i] = ys
        else:
            # Mode-acceleration: retained dynamic + static correction
            # (1/(jω - λ_k) + 1/λ_k) = (1/(jω - λ_k)) - (-1/λ_k)
            Y[i] = ys + np.sum(weights * (1.0 / (1j * omega - lams) + 1.0 / lams))
    return Y, t_build


def per_segment_soar_rom_response(
    omega_array, interval_solutions, F_red, idx_obs, m_per_seg,
    use_multi_shift=False,
):
    """Per-segment SOAR ROM (Bai & Su 2005) with force-aware starting.

    Each segment i:
        - shift = segment center frequency (or 3 shifts at quartiles if multi)
        - SOAR basis Q_i ∈ ℂ^{n × m_per_seg} via force-aware iteration
        - Project M*_i, C*_i, K*_i → reduced m×m matrices
        - Per-frequency: dispatch to segment + small reduced QEP solve

    SOAR matches m-th Padé moments around the shift → smooth FRF coverage
    without truncation cancellation issues of naive modal-truncation.

    Returns (Y, t_build_seconds).
    """
    Y = np.zeros(omega_array.size, dtype=complex)
    lowers = np.array([s["omega_lower"] for s in interval_solutions])
    rom_cache = []

    t0 = time.perf_counter()
    for sol in interval_solutions:
        mat = sol["matrices"]
        M_star = mat["M_star"]
        C_star = mat["C_star"]
        K_star = mat["K_star"]
        seg_center = (sol["omega_lower"] + sol["omega_upper"]) / 2.0

        if use_multi_shift:
            # 3 shifts per segment: lower-quartile, center, upper-quartile
            lo = sol["omega_lower"]
            hi = sol["omega_upper"]
            shifts = [
                lo + 0.25 * (hi - lo),
                seg_center,
                lo + 0.75 * (hi - lo),
            ]
            m_per_shift = max(1, m_per_seg // len(shifts))
            try:
                rom = build_multi_shift_soar_rom(
                    M_star, C_star, K_star, shifts, m_per_shift, F_red,
                )
            except RuntimeError:
                rom = None
        else:
            try:
                rom = build_soar_rom(
                    M_star, C_star, K_star, seg_center, m_per_seg, F_red,
                )
            except RuntimeError:
                rom = None
        rom_cache.append(rom)
    t_build = time.perf_counter() - t0

    for i, omega in enumerate(omega_array):
        j = int(np.searchsorted(lowers, omega, side="right") - 1)
        j = max(0, min(j, len(interval_solutions) - 1))
        rom = rom_cache[j]
        if rom is None:
            Y[i] = 0.0 + 0.0j
            continue
        D_r = rom.K_r + 1j * omega * rom.C_r - (omega ** 2) * rom.M_r
        y_r = np.linalg.solve(D_r, rom.F_r)
        u = rom.Q @ y_r
        Y[i] = u[idx_obs]
    return Y, t_build


def measure_direct(M_red, K_base_red, K_shear_red, F_red, cw_red, params, omega_eval, n_repeat=3):
    """Direct method baseline (averaged over n_repeat)."""
    times = []
    Y_last = None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        Y = direct_response_per_freq(
            params, K_base_red, K_shear_red, M_red, omega_eval, F_red, cw_red
        )
        times.append(time.perf_counter() - t0)
        Y_last = Y
    return Y_last, float(np.mean(times)), float(np.min(times))


def measure_segmentwise_rom_sweep(
    M_red, K_base_red, K_shear_red, F_red, cw_red, params, m_list, n_repeat=3
):
    """Per-segment modal-truncation ROM sweep over m_per_seg list."""
    # G(ω) fit + per-segment matrices
    interval_mats, fit_info = build_segments(M_red, K_base_red, K_shear_red, n_seg=5)
    print(f"  G(ω) fit: Re max {fit_info['fit_err_re_max']:.2f}% / mean "
          f"{fit_info['fit_err_re_mean']:.2f}%; "
          f"Im max {fit_info['fit_err_im_max']:.2f}% / mean "
          f"{fit_info['fit_err_im_mean']:.2f}%")

    # Per-segment QEP build (offline cost)
    t_qep_repeats = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        interval_sols = solve_qep_per_interval(interval_mats, damping_tol=1e-6)
        t_qep_repeats.append(time.perf_counter() - t0)
    t_qep_offline = float(np.min(t_qep_repeats))
    eff_counts = [s["lams_effective"].size for s in interval_sols]
    print(f"  Offline QEP build (5 segments): {t_qep_offline*1000:.0f} ms "
          f"(min of {n_repeat} runs)")
    print(f"  Per-segment effective eigenvalue counts: {eff_counts}")

    # Eval grid
    f_eval = np.linspace(5.0, 500.0, 200)
    omega_eval = 2.0 * np.pi * f_eval

    # Direct baseline
    print(f"\n  Measuring Direct baseline ({n_repeat} runs)...")
    Y_direct, t_direct_mean, t_direct_min = measure_direct(
        M_red, K_base_red, K_shear_red, F_red, cw_red, params, omega_eval, n_repeat
    )
    print(f"  Direct (per-ω full inversion): {t_direct_mean*1000:.1f} ms "
          f"({t_direct_mean*1000/len(f_eval):.3f} ms/freq, min {t_direct_min*1000:.0f} ms)")

    # Polynomial-QEP modal-sup reference (no truncation, for comparison)
    print(f"\n  Measuring PolyQEP full (no ROM) baseline...")
    t_pq_times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        Y_polyqep = modal_response_per_interval(omega_eval, interval_sols, F_red, cw_red)
        t_pq_times.append(time.perf_counter() - t0)
    t_polyqep_min = float(np.min(t_pq_times))
    mask = np.abs(Y_direct) > 0.01 * np.abs(Y_direct).max()
    err_pq = np.abs(Y_direct - Y_polyqep) / np.maximum(np.abs(Y_direct), 1e-30) * 100
    err_pq_max = float(err_pq[mask].max())
    err_pq_mean = float(err_pq[mask].mean())
    print(f"  PolyQEP full: {t_polyqep_min*1000:.1f} ms │ max err {err_pq_max:.3f}% / "
          f"mean {err_pq_mean:.3f}% (vs Direct)")

    # ROM sweep — three variants:
    #   (1) Naive per-segment modal truncation (modal_truncation_basis)
    #   (2) Mode-acceleration corrected (see the paper)
    #   (3) Per-segment SOAR with force-aware starting (Bai & Su 2005) — advanced
    print(f"\n  Segmentwise ROM sweep (3 variants):")
    print(f"  {'m_seg':>6} {'m_tot':>6} {'variant':>14} "
          f"{'build [ms]':>11} {'solve [ms]':>11} {'/freq [ms]':>11} "
          f"{'speedup':>9} {'err max [%]':>11} {'err mean [%]':>12}")
    rom_records = []
    for m in m_list:
        m_total = m * 5

        # Variant 1: Naive truncation
        t_build_repeats = []
        t_solve_repeats = []
        Y_rom_last = None
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            Y_rom, t_build_inner = rom_response_per_interval_reuse(
                omega_eval, interval_sols, F_red, cw_red, m, use_effective_only=False,
            )
            t_total = time.perf_counter() - t0
            t_build_repeats.append(t_build_inner)
            t_solve_repeats.append(t_total - t_build_inner)
            Y_rom_last = Y_rom
        t_build_naive = float(np.min(t_build_repeats))
        t_solve_naive = float(np.min(t_solve_repeats))
        err = np.abs(Y_direct - Y_rom_last) / np.maximum(np.abs(Y_direct), 1e-30) * 100
        err_max_naive = float(err[mask].max())
        err_mean_naive = float(err[mask].mean())
        speedup_naive = t_direct_min / max(t_solve_naive, 1e-12)
        per_freq_naive = t_solve_naive * 1000 / len(f_eval)
        print(f"  {m:>6d} {m_total:>6d} {'naive':>10} "
              f"{t_build_naive*1000:>11.1f} {t_solve_naive*1000:>11.1f} "
              f"{per_freq_naive:>11.3f} ×{speedup_naive:>7.1f} "
              f"{err_max_naive:>11.3f} {err_mean_naive:>12.3f}")

        # Variant 2: Mode-accelerated
        t_build_repeats = []
        t_solve_repeats = []
        Y_ma_last = None
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            Y_ma, t_build_inner = mode_accelerated_rom_response(
                omega_eval, interval_sols, F_red, cw_red, m,
            )
            t_total = time.perf_counter() - t0
            t_build_repeats.append(t_build_inner)
            t_solve_repeats.append(t_total - t_build_inner)
            Y_ma_last = Y_ma
        t_build_ma = float(np.min(t_build_repeats))
        t_solve_ma = float(np.min(t_solve_repeats))
        err = np.abs(Y_direct - Y_ma_last) / np.maximum(np.abs(Y_direct), 1e-30) * 100
        err_max_ma = float(err[mask].max())
        err_mean_ma = float(err[mask].mean())
        speedup_ma = t_direct_min / max(t_solve_ma, 1e-12)
        per_freq_ma = t_solve_ma * 1000 / len(f_eval)
        print(f"  {m:>6d} {m_total:>6d} {'mode-acc':>10} "
              f"{t_build_ma*1000:>11.1f} {t_solve_ma*1000:>11.1f} "
              f"{per_freq_ma:>11.3f} ×{speedup_ma:>7.1f} "
              f"{err_max_ma:>11.3f} {err_mean_ma:>12.3f}")

        # Variant 3: SOAR per segment (force-aware, single shift = segment center)
        t_build_repeats = []
        t_solve_repeats = []
        Y_soar_last = None
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            Y_soar, t_build_inner = per_segment_soar_rom_response(
                omega_eval, interval_sols, F_red, cw_red, m, use_multi_shift=False,
            )
            t_total = time.perf_counter() - t0
            t_build_repeats.append(t_build_inner)
            t_solve_repeats.append(t_total - t_build_inner)
            Y_soar_last = Y_soar
        t_build_soar = float(np.min(t_build_repeats))
        t_solve_soar = float(np.min(t_solve_repeats))
        err = np.abs(Y_direct - Y_soar_last) / np.maximum(np.abs(Y_direct), 1e-30) * 100
        err_max_soar = float(err[mask].max())
        err_mean_soar = float(err[mask].mean())
        speedup_soar = t_direct_min / max(t_solve_soar, 1e-12)
        per_freq_soar = t_solve_soar * 1000 / len(f_eval)
        print(f"  {m:>6d} {m_total:>6d} {'soar(1-shift)':>14} "
              f"{t_build_soar*1000:>11.1f} {t_solve_soar*1000:>11.1f} "
              f"{per_freq_soar:>11.3f} ×{speedup_soar:>7.1f} "
              f"{err_max_soar:>11.3f} {err_mean_soar:>12.3f}")

        # Variant 4: Multi-shift SOAR per segment (3 shifts within segment)
        t_build_repeats = []
        t_solve_repeats = []
        Y_msoar_last = None
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            Y_msoar, t_build_inner = per_segment_soar_rom_response(
                omega_eval, interval_sols, F_red, cw_red, m, use_multi_shift=True,
            )
            t_total = time.perf_counter() - t0
            t_build_repeats.append(t_build_inner)
            t_solve_repeats.append(t_total - t_build_inner)
            Y_msoar_last = Y_msoar
        t_build_msoar = float(np.min(t_build_repeats))
        t_solve_msoar = float(np.min(t_solve_repeats))
        err = np.abs(Y_direct - Y_msoar_last) / np.maximum(np.abs(Y_direct), 1e-30) * 100
        err_max_msoar = float(err[mask].max())
        err_mean_msoar = float(err[mask].mean())
        speedup_msoar = t_direct_min / max(t_solve_msoar, 1e-12)
        per_freq_msoar = t_solve_msoar * 1000 / len(f_eval)
        print(f"  {m:>6d} {m_total:>6d} {'soar(3-shift)':>14} "
              f"{t_build_msoar*1000:>11.1f} {t_solve_msoar*1000:>11.1f} "
              f"{per_freq_msoar:>11.3f} ×{speedup_msoar:>7.1f} "
              f"{err_max_msoar:>11.3f} {err_mean_msoar:>12.3f}")

        rom_records.append({
            "m_per_seg": m,
            "m_total": m_total,
            "naive": {
                "build_ms": t_build_naive * 1000,
                "solve_ms": t_solve_naive * 1000,
                "per_freq_ms": per_freq_naive,
                "speedup_vs_direct": speedup_naive,
                "err_max_pct": err_max_naive,
                "err_mean_pct": err_mean_naive,
            },
            "mode_accelerated": {
                "build_ms": t_build_ma * 1000,
                "solve_ms": t_solve_ma * 1000,
                "per_freq_ms": per_freq_ma,
                "speedup_vs_direct": speedup_ma,
                "err_max_pct": err_max_ma,
                "err_mean_pct": err_mean_ma,
            },
            "soar_1shift": {
                "build_ms": t_build_soar * 1000,
                "solve_ms": t_solve_soar * 1000,
                "per_freq_ms": per_freq_soar,
                "speedup_vs_direct": speedup_soar,
                "err_max_pct": err_max_soar,
                "err_mean_pct": err_mean_soar,
            },
            "soar_3shift": {
                "build_ms": t_build_msoar * 1000,
                "solve_ms": t_solve_msoar * 1000,
                "per_freq_ms": per_freq_msoar,
                "speedup_vs_direct": speedup_msoar,
                "err_max_pct": err_max_msoar,
                "err_mean_pct": err_mean_msoar,
            },
        })

    summary = {
        "setup": {
            "mesh": f"{params.nx}x{params.ny}",
            "n_dof_full": int(params.n_dof),
            "n_dof_free": int(M_red.shape[0]),
            "bc": "simply_supported_w",
            "f_range_hz": [float(f_eval.min()), float(f_eval.max())],
            "n_freq": int(len(f_eval)),
        },
        "g_fit": fit_info,
        "qep_offline": {
            "t_qep_ms": t_qep_offline * 1000,
            "effective_counts_per_segment": eff_counts,
        },
        "direct": {
            "t_total_ms": t_direct_min * 1000,
            "t_per_freq_ms": t_direct_min * 1000 / len(f_eval),
        },
        "polyqep_full": {
            "t_total_ms": t_polyqep_min * 1000,
            "err_max_pct": err_pq_max,
            "err_mean_pct": err_pq_mean,
        },
        "rom_sweep": rom_records,
    }
    return summary


def main():
    np.set_printoptions(linewidth=140, precision=4)
    print("=" * 78)
    print("Segmentwise quadratic + per-segment ROM")
    print("=" * 78)

    # 8×8 mesh setup — simply-supported BC (matches the algorithmically exact SS baseline)
    bc = "simply_supported_w"
    print(f"\n[Setup] paper §5.1 — 8×8 mesh, BC={bc}, center excitation")
    params, M_red, K_base_red, K_shear_red, F_red, cw_red = setup_amichi_8x8(bc=bc)
    print(f"  Free DOF count: {M_red.shape[0]}")

    # m_per_seg sweep
    m_list = [4, 8, 12, 20, 40, 80]
    summary = measure_segmentwise_rom_sweep(
        M_red, K_base_red, K_shear_red, F_red, cw_red, params, m_list, n_repeat=3,
    )

    # Save JSON
    cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "cmame_e2_segmentwise_rom_results.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[saved] {out_path}")

    print("\n" + "=" * 78)
    print("Key results:")
    print("=" * 78)
    direct_pf = summary["direct"]["t_per_freq_ms"]
    print(f"  Direct baseline: {direct_pf:.3f} ms/freq")
    print(f"  PolyQEP full (no ROM): max err {summary['polyqep_full']['err_max_pct']:.2f}%, "
          f"mean {summary['polyqep_full']['err_mean_pct']:.2f}% (vs Direct)")
    print(f"  Segment fit: Re max {summary['g_fit']['fit_err_re_max']:.2f}% / "
          f"mean {summary['g_fit']['fit_err_re_mean']:.2f}%")
    print(f"  Offline QEP build (5 segments): {summary['qep_offline']['t_qep_ms']:.0f} ms")
    print(f"\n  Headline summary — best of 4 variants per m_per_seg:")
    print(f"  {'m_per_seg':>10} {'best variant':>16} {'×speedup':>10} "
          f"{'max err':>10} {'mean err':>10}")
    for rec in summary["rom_sweep"]:
        # Pick variant with lowest max err but speedup > 10
        candidates = []
        for vname in ("naive", "mode_accelerated", "soar_1shift", "soar_3shift"):
            v = rec[vname]
            if v["speedup_vs_direct"] > 10:
                candidates.append((vname, v["err_max_pct"], v["speedup_vs_direct"]))
        if candidates:
            best = min(candidates, key=lambda x: x[1])
            vname, err_max, sp = best
            flag = "  ← good" if err_max < 10 else ""
            print(f"  {rec['m_per_seg']:>10} {vname:>16} ×{sp:>9.1f} "
                  f"{err_max:>9.2f}% {rec[vname]['err_mean_pct']:>9.2f}%{flag}")


if __name__ == "__main__":
    main()
