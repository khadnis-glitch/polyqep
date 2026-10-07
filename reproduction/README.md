# Reproduction of the paper's tables and figures

This directory regenerates every table and figure of the revised paper and of
its Supplementary Material. The `polyqep` package in `src/` is the method; this
directory adds the comparison baselines, the benchmark scripts, the result
files, and the generated figures.

The result files in `data/` are the ones the paper reports. They were produced
on one computer with one timing protocol:

| Item | Value |
| --- | --- |
| Processor, memory, OS | Apple M5 Max, 128 GB, macOS 26.6.2 |
| Python, NumPy, SciPy | 3.9.6, 2.0.2 (Accelerate BLAS/LAPACK), 1.13.1 (OpenBLAS 0.3.27, SuperLU) |
| Threads | one (`OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` all set to 1) |
| Timing statistic | median of 10 frequency sweeps after one warm-up sweep |

Errors and basis sizes are reproducible on any machine. Timings and speedups
depend on the machine and on the BLAS/LAPACK libraries; the scripts record the
environment they ran in inside every JSON result (`environment` key).

## Setup

```bash
# from the repository root
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r reproduction/requirements.txt

cd reproduction
export PYTHONPATH="$PWD:$PWD/scripts"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Every script is run from this directory, for example
`python scripts/rev_bench_rom.py`. Scripts write their results to `data/`
and their figures to `figures/` (with a `.source.md` provenance stub next to
each figure). `run_all.sh` runs everything in order.

## Layout

| Directory | Contents |
| --- | --- |
| `baselines/` | comparison methods and models that are not part of the package: GHM and ADF augmentations, the single-point and adaptive Krylov reductions of Xie et al. (2018), the frequency-dependent corner mount, the Won et al. (2013) sandwich beam |
| `scripts/` | benchmark generators (`phase3_*`, `cmame_*`, `amichi_three_way_comparison`) and the revision studies (`rev_*`) |
| `data/` | result files read by the tables, the figures, and the text; `amichi_fig7_dma_digitized.csv` holds the digitized DMA curves |
| `figures/` | generated figures |
| `tables/` | LaTeX table bodies of the Supplementary Material, generated from `data/` by `rev_make_supp_tables.py` |

## Scripts, outputs, and where they appear in the paper

Paper = main text, S = Supplementary Material. Runtime classes on the
machine above: short (< 1 min), medium (1--10 min), long (> 10 min).

| Script | Output | Appears in | Runtime |
| --- | --- | --- | --- |
| `phase3_amichi_polynomial_qep.py` | `figures/phase3_G_fitting.png`, fit residuals | Fig. 3, Table 2 | medium |
| `cmame_d1_mode_acceleration.py` | `figures/cmame_mode_acceleration.png` | Fig. 4 | medium |
| `rev_fullorder_checks.py` | `data/rev_fullorder_checks.json` | Sections 3.1, 3.2 (affine residuals, fit residuals, modal-superposition checks) | medium |
| `rev_bench_rom.py` | `data/rev_bench_rom.json`, `rev_bench_rom_frf.npz` | Table 3, S3 (offline breakdown, break-even) | medium |
| `rev_make_fig_rom_frf.py` | `figures/rev_fig_rom_frf.pdf` (from `data/rev_bench_rom_frf.npz`) | Fig. 5 | short |
| `rev_effective_subset_residual.py` | `data/rev_effective_subset_residual.json` | Section 3.2 (residual of the effective-pole sum) | medium |
| `cmame_f1_xie2018_comparison_v3.py` | `figures/cmame_f1_frf_comparison_v3.pdf` | earlier version of the Fig. 5 comparison, not in the paper | medium |
| `rev_bench_scaling.py --nx N` (N = 8, 12, 16, 20, 25) | `data/rev_bench_scaling_nxNN.json` | Section 3.3, S3 (mesh scaling) | medium to long |
| `rev_bench_offline_sparse.py --nx N` | `data/rev_bench_offline_sparse_nxNN.json` | S3 (sparse offline path) | short to medium |
| `rev_bench_augmentation.py` | `data/rev_bench_augmentation.json` | Table 4, S3 | medium |
| `rev_ghm_adf_diagnosis.py` | `data/rev_ghm_adf_diagnosis.json` | S1 (order study, samplings, bounds) | long |
| `rev_causality_check.py` | `data/rev_causality_check.json` | S1 (convex bound) | short |
| `rev_partition_sensitivity.py`, `rev_partition_greedy.py`, `rev_partition_modes.py`, `rev_partition_shapes.py` | `data/rev_partition_sensitivity.json`, `rev_partition_greedy.json`, `rev_partition_modes.json`, `rev_partition_shapes.json` | Section 4 (partition guide); the complete tables are `tables/s2_*.tex` and `figures/rev_figS2_partition.pdf`, which are not part of the Supplementary Material | long |
| `rev_noise_study.py` | `data/rev_noise_study.json`, `rev_noise_example.npz` | S2 | long |
| `rev_measured_dma_study.py` | `data/rev_measured_dma_study.json` | S2 | medium |
| `rev_basis_size_study.py` | `data/rev_basis_size_study.json` | S3 (basis-size study) | medium |
| `cmame_f2_sharp_resonance_synthetic.py` | `data/cmame_f2_sharp_resonance.json`, `figures/cmame_f2_sharp_resonance.pdf` | S4 | medium |
| `rev_mount_modes.py` (`--width 5 --peak 2`) | `data/rev_mount_modes_w5_p2.json`, `figures/rev_mount_modes_w5_p2.pdf` | S6 | short |
| `cmame_b4_won2013_MK_segment_local.py`, `cmame_b5_won2013_method_comparison.py` | `data/cmame_b4_*.csv`, `cmame_b5_*.csv` | Section 3.5 (Won beam) | short |
| `rev_make_figures.py` | `figures/rev_fig_*.pdf`, `figures/rev_figS*.pdf` (from `data/`) | Figs. 6, 7, S1--S4 | short |
| `rev_make_supp_tables.py` | `tables/*.tex` (from `data/`) | all tables of S1--S6, plus the `s2_*` partition tables | short |

Notes.

* Figs. 1 and 2 of the paper are hand-drawn schematics (the effective-eigenvalue
  selection rule and the procedure) and are not generated by a script.
* `data/rev_bench_rom_run1..3.json` are repeated sessions of
  `rev_bench_rom.py`; the repeatability table of S3 is built from them.
* `data/rev_mount_modes_*.json` and the matching figures are the alternative
  mount configurations discussed with the mount example (S6); the paper uses
  `w5_p2`.
* The augmented GHM/ADF systems and the segment eigenproblems take most of the
  time of the medium and long scripts; `rev_ghm_adf_diagnosis.py` runs the
  30-start fits for six orders and four samplings.

## Provenance of the baselines

`baselines/ghm_augmentation.py` and `baselines/adf_augmentation.py` implement
the Golla--Hughes--McTavish and anelastic-displacement-field augmentations in
their standard forms with multi-start bounded least-squares fits (settings in
Supplementary Section S1). `baselines/xie2018_rom.py` implements the
single-point and adaptive second-order Krylov reductions of
Xie, Zheng, Jonckheere, van de Walle, Pluymers and Desmet (2018).
`baselines/corner_mount.py` is the frequency-dependent corner mount of the
complex-mode example and `baselines/sandwich_beam_won2013.py` the sandwich
beam of Won, Sim, Park and Park (2013) used for the almost-affine case.
