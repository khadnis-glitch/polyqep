"""Additional studies — LaTeX table bodies of the Supplementary Material.

Every number in the supplementary tables is written from the JSON result
files, never typed by hand. Output: ``tables/*.tex`` (or ``<latex dir>/supp_tables/`` when a directory is given).

Run::

    python scripts/rev_make_supp_tables.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
DATA = _ROOT / "data"


def load(name):
    with open(DATA / name) as fh:
        return json.load(fh)


def sci(x, digits=1):
    """LaTeX scientific notation, e.g. 2.4\\times10^{-10}."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10.0**e
    if round(m, digits) >= 10:
        m, e = m / 10.0, e + 1
    return f"${m:.{digits}f}\\times10^{{{e}}}$"


def num(x):
    """Compact fixed/scientific formatting for a percentage or a time."""
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.2f}"
    if ax >= 0.01:
        return f"{x:.3f}"
    return sci(x)


def write(out_dir, name, lines):
    (out_dir / name).write_text("\n".join(lines) + "\n")
    print(f"wrote supp_tables/{name} ({len(lines)} rows)")


def s1_tables(out):
    rows = load("rev_ghm_adf_diagnosis.json")["order_study"]

    def pick(sampling):
        return [r for r in rows if r["sampling"] == sampling]

    lines = []
    for r in pick("A_uniform_1_1500"):
        f, s = r["fit_grid_errors"], r["starts"]
        cond = s["best_jacobian_condition_free"]
        lines.append(
            f"{r['method']} & {r['order']} & {r['n_params']} & "
            f"{r['objective_rms_rel_pct']:.1f} & "
            f"{f['re_max']:.1f} ({f['f_at_re_max_hz']:.1f}) & "
            f"{f['im_max']:.1f} ({f['f_at_im_max_hz']:.1f}) & "
            f"{f['re_mean']:.1f} / {f['im_mean']:.1f} & "
            f"{r['frf_peak_band_max_pct']:.1f} / {r['frf_peak_band_mean_pct']:.1f} & "
            f"{s['n_within_1pct_of_best']} & {len(r['active_bounds'])} & {sci(cond, 0)} \\\\")
        if r["method"] == "GHM" and r["order"] == 6:
            lines.append("\\midrule")
    write(out, "s1_order_study.tex", lines)

    lines = []
    for r in pick("A_uniform_1_1500"):
        if r["method"] != "GHM":
            continue
        par = r["parameters"]
        lines.append(f"{r['order']} & "
                     + ", ".join(num(w) for w in par["what_hz"]) + " & "
                     + ", ".join(f"{z:.2f}" for z in par["zeta"]) + " \\\\")
    write(out, "s1_ghm_parameters.tex", lines)
    lines = []
    for r in pick("A_uniform_1_1500"):
        if r["method"] != "ADF":
            continue
        par = r["parameters"]
        lines.append(f"{r['order']} & "
                     + ", ".join(num(w) for w in par["Omega_hz"]) + " & "
                     + ", ".join(num(g / 1e6) for g in par["Gv_Pa"]) + " \\\\")
    write(out, "s1_adf_parameters.tex", lines)

    lines = []
    for r in pick("A_uniform_1_1500_wide_box"):
        f = r["fit_grid_errors"]
        base = next(b for b in pick("A_uniform_1_1500")
                    if b["method"] == r["method"] and b["order"] == r["order"])
        lines.append(
            f"{r['method']} & {r['order']} & {base['objective_rms_rel_pct']:.1f} & "
            f"{r['objective_rms_rel_pct']:.1f} & "
            f"{base['frf_peak_band_max_pct']:.1f} / {base['frf_peak_band_mean_pct']:.1f} & "
            f"{r['frf_peak_band_max_pct']:.1f} / {r['frf_peak_band_mean_pct']:.1f} \\\\")
        if r["method"] == "GHM" and r["order"] == 6:
            lines.append("\\midrule")
    write(out, "s1_wide_box.tex", lines)

    lines = []
    samplings = ("A_uniform_1_1500", "B_log_1_1500", "C_log_5_500", "D_uniform_5_500")
    for method in ("GHM", "ADF"):
        for n in (1, 2, 3, 4, 5, 6):
            cells = []
            for sname in samplings:
                r = next(x for x in pick(sname)
                         if x["method"] == method and x["order"] == n)
                cells.append(f"{r['frf_peak_band_max_pct']:.1f} / "
                             f"{r['frf_peak_band_mean_pct']:.1f}")
            lines.append(f"{method} & {n} & " + " & ".join(cells) + " \\\\")
        if method == "GHM":
            lines.append("\\midrule")
    write(out, "s1_samplings.tex", lines)

    lines = []
    labels = {
        "representative, uniform 1-1500 Hz (submitted fit grid)":
            "Representative profile, 1--1500\\,Hz, 200 uniform samples",
        "representative, log 5-500 Hz":
            "Representative profile, 5--500\\,Hz, 200 log-spaced samples",
        "representative, log 10-1000 Hz (anchor span)":
            "Representative profile, 10--1000\\,Hz, 200 log-spaced samples",
        "digitized Fig. 7, 75-639 Hz (142 points)":
            "Digitized DMA curves, 75--639\\,Hz, 142 points",
    }
    for r in load("rev_causality_check.json")["results"]:
        j, s = r["joint_fit"], r["storage_only_fit"]
        lines.append(
            f"{labels[r['data_set']]} & "
            f"{r['data_loss_factor_range'][0]:.2f}--{r['data_loss_factor_range'][1]:.2f} & "
            f"{j['rms_relative_residual_pct']:.1f} & "
            f"{j['re_max']:.1f} / {j['im_max']:.1f} & "
            f"{s['implied_loss_over_data_loss_min']:.2f}--"
            f"{s['implied_loss_over_data_loss_max']:.2f} \\\\")
    write(out, "s1_causality.tex", lines)


def s2_tables(out):
    rows = [r for r in load("rev_partition_sensitivity.json")["rows"]
            if "skipped" not in r]
    lines = []

    def line(group, label, r):
        return (f"{group} & {label} & {r['n_segments']} & "
                f"{r['fit_eval_band']['re_max']:.2f} / {r['fit_eval_band']['im_max']:.2f} & "
                f"{r['frf_fitted_full_order']['max_pct']:.2f} / "
                f"{r['frf_fitted_full_order']['mean_pct']:.2f} & "
                f"{num(r['frf_pw_r']['max_pct'])} & {sci(r['frf_pw_rg']['max_pct'])} & "
                f"{r['m_pw_rg']} & {r['offline_ms']:.0f} & "
                f"{r['pw_rg_per_freq_ms']:.3f} \\\\")

    base_u = next(r for r in rows if r["group"] == "baseline"
                  and r["sampling"] == "uniform400")
    lines.append(line("baseline", "sampling of the paper", base_u))
    last = None
    for r in rows:
        if r["sampling"] != "log2000" or r["group"] == "random":
            continue
        label = (r["label"].replace("N=", "$N=$").replace(" x ", " $\\times$ ")
                 .replace("internal breakpoints", "breakpoints"))
        if last is not None and r["group"] != last:
            lines.append("\\midrule")
        lines.append(line(r["group"] if r["group"] != last else "", label, r))
        last = r["group"]
    rnd = [r for r in rows if r["group"] == "random" and r["sampling"] == "log2000"]

    def rng(key, sub="max_pct"):
        v = [r[key][sub] for r in rnd]
        return min(v), max(v)

    f_lo, f_hi = rng("frf_fitted_full_order")
    fm_lo, fm_hi = rng("frf_fitted_full_order", "mean_pct")
    r_lo, r_hi = rng("frf_pw_r")
    g_lo, g_hi = rng("frf_pw_rg")
    m = [r["m_pw_rg"] for r in rnd]
    lines.append("\\midrule")
    lines.append(
        f"random & {len(rnd)} draws, $\\pm20\\%$ & 5 & --- & "
        f"{f_lo:.2f}--{f_hi:.2f} / {fm_lo:.2f}--{fm_hi:.2f} & "
        f"$\\le$ {num(r_hi)} & $\\le$ {sci(g_hi)} & "
        f"{min(m)}--{max(m)} & --- & --- \\\\")
    write(out, "s2_partition.tex", lines)


def s2_resonance_table(out):
    """Resonances per segment and PW-R error for three per-segment basis sizes."""
    modes = load("rev_partition_modes.json")
    rows = modes["rows"]
    lines = []

    def line(group, label, r):
        p = r["pw_r"]
        return (f"{group} & {label} & {r['n_segments']} & "
                f"{r['max_resonances_reported']} & "
                f"{num(p['10']['max_pct'])} & {num(p['20']['max_pct'])} & "
                f"{num(p['40']['max_pct'])} \\\\")

    last = None
    for r in rows:
        if r["group"] == "random":
            continue
        label = (r["label"].replace("N=", "$N=$").replace(" x ", " $\\times$ ")
                 .replace("internal breakpoints", "breakpoints"))
        if last is not None and r["group"] != last:
            lines.append("\\midrule")
        lines.append(line(r["group"] if r["group"] != last else "", label, r))
        last = r["group"]
    rnd = [r for r in rows if r["group"] == "random"]
    mr = [r["max_resonances_reported"] for r in rnd]
    lines.append("\\midrule")
    lines.append(
        f"random & {len(rnd)} draws, $\\pm20\\%$ & 5 & {min(mr)}--{max(mr)} & "
        + " & ".join(f"$\\le$ {num(max(r['pw_r'][k]['max_pct'] for r in rnd))}"
                     for k in ("10", "20", "40")) + " \\\\")
    write(out, "s2_resonances.tex", lines)


def s2_shapes_table(out):
    """Segment count against modulus shape: fit residual, fitted response, b, raw-modulus ROMs."""
    shapes = load("rev_partition_shapes.json")["shapes"]
    names = {"representative": "DMA-type (paper)", "digitized": "smooth (digitized)",
             "peaked": "loss peak (S5)"}
    lines = []
    first_shape = True
    for shape in ("digitized", "representative", "peaked"):
        if not first_shape:
            lines.append("\\midrule")
        first_shape = False
        last = None
        for r in shapes[shape]["rows"]:
            label = (r["label"].replace("N=", "$N=$").replace("K=", "$K=$"))
            fe, fo = r["fit_eval_band"], r["frf_fitted_full_order"]
            lines.append(
                f"{names[shape] if last is None else ''} & {r['group']} & {label} & "
                f"{r['n_segments']} & {fe['re_max']:.2f} / {fe['im_max']:.2f} & "
                f"{fo['max_pct']:.2f} / {fo['mean_pct']:.2f} & {r['fitted_over_fit_residual']:.2f} & "
                f"{num(r['frf_pw_r']['max_pct'])} & {sci(r['frf_pw_rg']['max_pct'])} & "
                f"{r['m_pw_rg']} & {r['max_resonances_reported']} \\\\")
            last = r["group"]
    write(out, "s2_shapes.tex", lines)


def s3_tables(out):
    study = load("rev_noise_study.json")
    nf = study["noise_free"]
    lines = [f"0 & --- & {nf['e_m_fitted']:.2f} & {nf['e_m_pw_r']:.3f} & "
             f"{sci(nf['e_m_pw_rg'])} & --- \\\\"]
    for lv in study["levels"]:
        def mp(key, fmt="{:.2f}"):
            return f"{fmt.format(lv[key]['mean'])} / {fmt.format(lv[key]['p95'])}"
        lines.append(
            f"{lv['sigma']*100:.0f} & {mp('e_d')} & {mp('e_m_fitted')} & "
            f"{mp('e_m_pw_r', '{:.3f}')} & "
            f"{sci(lv['e_m_pw_rg']['mean'])} / {sci(lv['e_m_pw_rg']['p95'])} & "
            f"{mp('e_t_fitted')} \\\\")
    write(out, "s3_noise.tex", lines)

    rows = load("rev_measured_dma_study.json")["rows"]
    lines, last = [], None
    for r in rows:
        key = (r["method"], r["setting"])
        if r["method"] in ("PW-R", "PW-RG"):
            if last is not None and last[0] not in ("PW-R", "PW-RG"):
                lines.append("\\midrule")
            lines.append(f"{r['method']} & {r['order']} segments, $m={r['m_online']}$ & "
                         f"--- & raw modulus & --- & "
                         f"{num(r['frf']['peak_band_max_pct'])} / "
                         f"{num(r['frf']['peak_band_mean_pct'])} \\\\")
            last = key
            continue
        if last is not None and key != last:
            lines.append("\\midrule")
        name = r["method"] if key != last else ""
        setting = r["setting"] if key != last else ""
        if r["method"] == "Prony floor":
            setting, order = "convex bound, dense pole grid", "---"
        else:
            order = str(r["order"])
        lines.append(
            f"{name} & {setting} & {order} & "
            f"{r['fit']['re_max']:.2f} / {r['fit']['im_max']:.2f} & "
            f"{r['fit']['re_mean']:.2f} / {r['fit']['im_mean']:.2f} & "
            f"{r['frf']['peak_band_max_pct']:.2f} / {r['frf']['peak_band_mean_pct']:.2f} \\\\")
        last = key
    write(out, "s3_measured.tex", lines)


def s4_tables(out):
    rom = load("rev_bench_rom.json")
    sparse8 = load("rev_bench_offline_sparse_nx08.json")
    names = [("g_fit", "Modulus fit (two least-squares problems)"),
             ("segment_assembly", "Segment matrices $\\mathbf{M}^*_i,\\mathbf{C}^*_i,\\mathbf{K}^*_i$"),
             ("soar_bases", "SOAR bases, five segments"),
             ("pw_r_projection", "PW-R projections"),
             ("pw_rg_svd_compression", "PW-RG SVD compression"),
             ("pw_rg_projection", "PW-RG projection")]
    lines = []
    for key, label in names:
        lines.append(f"{label} & {rom['offline_stage_ms'][key]:.1f} & "
                     f"{sparse8['offline_stage_ms'][key]:.1f} \\\\")
    lines.append("\\midrule")
    lines.append(f"Total, PW-R & {rom['summary']['PW-R']['offline_total_ms']:.0f} & "
                 f"{sparse8['offline_total_ms']['PW-R']:.0f} \\\\")
    lines.append(f"Total, PW-RG & {rom['summary']['PW-RG']['offline_total_ms']:.0f} & "
                 f"{sparse8['offline_total_ms']['PW-RG']:.0f} \\\\")
    lines.append("\\midrule")
    lines.append(f"Segment QEP solves (modal output only) & "
                 f"{rom['offline_stage_ms']['segment_qep_modal_output']:.0f} & --- \\\\")
    lines.append(f"XS basis construction & {rom['offline_stage_ms']['xs_build']:.0f} & --- \\\\")
    lines.append(f"XA adaptive procedure & {rom['offline_stage_ms']['xa_adaptive_build']:.0f} & --- \\\\")
    write(out, "s4_offline.tex", lines)

    lines = []
    for name, s in rom["summary"].items():
        lines.append(
            f"{name} & {s['m_online']} & {s['per_freq_ms']:.4f} & "
            f"{s['per_freq_ms_single_output']:.4f} & {s['offline_total_ms']:.0f} & "
            f"{s['break_even_n_freq_vs_dense']:.0f} & "
            f"{s['break_even_n_freq_vs_sparse']:.0f} \\\\")
    write(out, "s4_break_even.tex", lines)

    on = rom["online"]
    lines = [
        f"Dense LU, \\texttt{{numpy.linalg.solve}} (Accelerate) & "
        f"{on['D_dense_numpy']['per_freq_ms']:.2f} \\\\",
        f"Dense LU, \\texttt{{scipy.linalg.solve}} (OpenBLAS; submitted code path) & "
        f"{on['D_dense_scipy']['per_freq_ms']:.2f} \\\\",
        f"Sparse LU, operators kept sparse (SuperLU) & "
        f"{on['D_sparse_native']['per_freq_ms']:.2f} \\\\",
        f"Sparse LU, dense-to-sparse conversion at each frequency (submitted code path) & "
        f"{on['D_sparse']['per_freq_ms']:.2f} \\\\",
    ]
    write(out, "s4_baselines.tex", lines)

    lines_a, lines_b = [], []
    for p in sorted(glob.glob(str(DATA / "rev_bench_scaling_nx*.json"))):
        d = json.load(open(p))
        sp = load(f"rev_bench_offline_sparse_nx{d['nx']:02d}.json")
        pf, r, g = d["per_freq_ms"], d["summary"]["PW-R"], d["summary"]["PW-RG"]
        off_d = g["offline_total_ms"] / 1e3
        off_s = sp["offline_total_ms"]["PW-RG"] / 1e3
        nstar_s = sp["offline_total_ms"]["PW-RG"] / (pf["D_sparse"] - g["per_freq_ms"])
        mesh = f"${d['nx']}\\times{d['nx']}$ & {d['n_free_dof']}"
        lines_a.append(
            f"{mesh} & {num(pf['D_dense_numpy'])} & {num(pf['D_sparse'])} & "
            f"{r['per_freq_ms']:.4f} & {r['speedup_vs_dense']:.0f} / "
            f"{r['speedup_vs_sparse']:.0f} & {r['error_vs_dense']['max_pct']:.3f} & "
            f"{g['m_online']} & {g['per_freq_ms']:.4f} & {g['speedup_vs_dense']:.0f} / "
            f"{g['speedup_vs_sparse']:.0f} & {sci(g['error_vs_dense']['max_pct'])} \\\\")
        lines_b.append(
            f"{mesh} & {num(off_d)} & {g['break_even_n_freq_vs_sparse']:.0f} & "
            f"{num(off_s)} & {nstar_s:.0f} & {sp['m_pw_rg']} & "
            f"{sp['error_vs_sparse_direct']['PW-R']['max_pct']:.3f} & "
            f"{sci(sp['error_vs_sparse_direct']['PW-RG']['max_pct'])} \\\\")
    write(out, "s4_scaling.tex", lines_a)
    write(out, "s4_scaling_offline.tex", lines_b)

    rows = load("rev_basis_size_study.json")["rows"]
    # Table S14 (2026-10-12, user decision): print a selection of the 26 study rows.
    # Kept: the rows cited in the paper and in S3 (the threshold of the round-off
    # floor, the paper settings, the matched-size comparison at m = 30) plus one
    # larger basis per group; the XA tolerance 1e-4 selects the same points and
    # vectors as 1e-2 and is stated in the prose instead.  The full study stays in
    # the JSON and is still used by the check scripts.
    keep = {("all five segments", None, 6), ("all five segments", None, 8),
            ("all five segments", None, 12), ("all five segments", None, 20),
            ("four in-band segments", None, 6), ("four in-band segments", None, 8),
            ("four in-band segments", None, 10), ("four in-band segments", None, 20),
            ("XA", 1e-2, 15), ("XA", 1e-2, 20), ("XA", 1e-2, 30),
            ("XA", 1e-6, 10), ("XA", 1e-6, 15), ("XA", 1e-6, 20)}
    rows = [r for r in rows
            if (r["segments"], r.get("tolerance") if r["segments"] == "XA" else None,
                r["m_per_segment"]) in keep]
    assert len(rows) == len(keep), len(rows)
    lines, last = [], None
    for r in rows:
        if r["segments"] == "XA":
            group = f"XA, tolerance $10^{{{int(np.log10(r['tolerance']))}}}$"
            setting = (f"{r['m_per_segment']} vectors per point, "
                       f"{r['n_points']} points")
        else:
            group = "PW-RG, " + r["segments"]
            setting = f"$m_{{\\rm per\\,seg}}={r['m_per_segment']}$"
        if last is not None and group != last:
            lines.append("\\midrule")
        lines.append(f"{group if group != last else ''} & {setting} & {r['m_global']} & "
                     f"{r['per_freq_ms']:.4f} & {sci(r['max_pct'])} \\\\")
        last = group
    write(out, "s4_basis_size.tex", lines)

    aug = load("rev_bench_augmentation.json")["rows"]
    lines = []
    for name, r in aug.items():
        if name == "Direct":
            continue
        lines.append(f"{name.replace('N=', '$N=$')} & {r['size']} & "
                     f"{num(r['per_freq_ms'])} & {r['offline_ms']/1e3:.2f} \\\\")
    write(out, "s4_augmentation_offline.tex", lines)


def environment_table(out):
    rom = load("rev_bench_rom.json")
    env, prot = rom["environment"], rom["timing_protocol"]
    threads = ", ".join(f"{k}={v}" for k, v in env["thread_env"].items())
    lines = [
        f"Processor & {env['processor']} ({env['machine_model']}), "
        f"{env['cores_physical']} physical cores \\\\",
        f"Memory & {env['memory_gb']:.0f}\\,GB \\\\",
        # records written before the label fix in rev_env.py carry the macOS
        # product version under the name "Darwin"
        f"Operating system & {env['os'].replace('Darwin 26.', 'macOS 26.')} \\\\",
        f"Python & {env['python']} \\\\",
        f"NumPy & {env['numpy']}, BLAS/LAPACK: {env['numpy_blas_lapack']['blas']['name']} \\\\",
        f"SciPy & {env['scipy']}, BLAS/LAPACK: {env['scipy_blas_lapack']['blas']['name']} "
        f"{env['scipy_blas_lapack']['blas']['version']} \\\\",
        f"Threads & one; \\texttt{{{threads}}} \\\\".replace("_", "\\_"),
        f"Clock & \\texttt{{{prot['clock']}}} \\\\".replace("_", "\\_"),
        f"Statistic & {prot['statistic']} of {prot['online_repeats']} online sweeps "
        f"after {prot['warmup_runs']} warm-up sweep; {prot['offline_repeats']} repeats of "
        f"the offline stages; {prot['qep_repeats']} repeats of the segment QEP solves \\\\",
    ]
    write(out, "s4_environment.tex", lines)

    # run-to-run spread of the per-frequency times over the repeated sessions
    names = ["PW", "PW-R", "PW-RG", "XS", "XA"]
    runs = sorted(glob.glob(str(DATA / "rev_bench_rom_run*.json")))
    rows = []
    for p in runs + [str(DATA / "rev_bench_rom.json")]:
        r = json.load(open(p))
        rows.append([r["online"]["D_dense_numpy"]["per_freq_ms"]]
                    + [r["summary"][k]["per_freq_ms"] for k in names])
    a = np.array(rows)
    spread = (a.max(0) - a.min(0)) / np.median(a, 0) * 100.0
    lines = [f"{n} & {a[:, i].min():.4f} & {a[:, i].max():.4f} & {spread[i]:.1f} \\\\"
             for i, n in enumerate(["Direct (dense LU)"] + names)]
    write(out, "s4_repeatability.tex", lines)
    print(f"  sessions: {len(rows)}, largest spread {spread.max():.1f}%")


def mount_table(out):
    """Section S6: most complex mode of the three free-free systems."""
    rows = load("rev_mount_modes_w5_p2.json")["rows"]
    lowest = min(r["mpc"] for r in rows)
    lines = []
    for r in rows:
        mpc = f"{r['mpc']:.3f}"
        if r["mpc"] == lowest:
            mpc = f"\\textbf{{{mpc}}}"
        lines.append(f"{r['system'].replace(' + ', ' $+$ ')} & "
                     f"{r['damping_source'].replace(' + ', ' $+$ ')} & "
                     f"{r['f_hz']:.1f} & {mpc} & {r['zeta']:.3f} \\\\")
    write(out, "s7_mount.tex", lines)


def main():
    out = (Path(sys.argv[1]).resolve() / "supp_tables" if len(sys.argv) > 1
           else _ROOT / "tables")
    out.mkdir(exist_ok=True)
    s1_tables(out)
    s2_tables(out)
    s2_resonance_table(out)
    s2_shapes_table(out)
    s3_tables(out)
    s4_tables(out)
    environment_table(out)
    mount_table(out)


if __name__ == "__main__":
    main()
