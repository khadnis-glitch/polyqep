#!/bin/sh
# Regenerate every result, figure, and table of the paper in order.
# Run from reproduction/ after the setup of README.md. Takes hours in total;
# the long steps are marked.
set -e
export PYTHONPATH="$PWD:$PWD/scripts"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=${PYTHON:-python}

$P scripts/phase3_amichi_polynomial_qep.py            # Fig. 3, Table 2
$P scripts/cmame_d1_mode_acceleration.py              # Fig. 4
$P scripts/rev_fullorder_checks.py                    # Sections 3.1-3.2
$P scripts/rev_bench_rom.py                           # Table 3
$P scripts/rev_make_fig_rom_frf.py                    # Fig. 5 (from the sweep above)
$P scripts/rev_effective_subset_residual.py           # Section 3.2 (effective-pole residual)
$P scripts/cmame_f1_xie2018_comparison_v3.py          # earlier Fig. 5 comparison (not in the paper)
for nx in 8 12 16 20 25; do                           # Section 3.3, S3 (long for 20, 25)
  $P scripts/rev_bench_scaling.py --nx $nx
  $P scripts/rev_bench_offline_sparse.py --nx $nx
done
$P scripts/rev_bench_augmentation.py                  # Table 4
$P scripts/rev_ghm_adf_diagnosis.py                   # S1 (long)
$P scripts/rev_causality_check.py                     # S1
$P scripts/rev_partition_sensitivity.py               # Section 4 partition guide (long)
$P scripts/rev_partition_greedy.py                    # Section 4 partition guide
$P scripts/rev_partition_modes.py                     # Section 4 partition guide (long)
$P scripts/rev_partition_shapes.py                    # Section 4 partition guide (long)
$P scripts/rev_noise_study.py                         # S2 (long)
$P scripts/rev_measured_dma_study.py                  # S2
$P scripts/rev_basis_size_study.py                    # S3
$P scripts/cmame_f2_sharp_resonance_synthetic.py      # S4
$P scripts/rev_mount_modes.py --width 5 --peak 2      # S6
$P scripts/cmame_b4_won2013_MK_segment_local.py       # Section 3.5
$P scripts/cmame_b5_won2013_method_comparison.py      # Section 3.5
$P scripts/rev_make_figures.py                        # Figs. 6, 7, S1-S4
$P scripts/rev_make_supp_tables.py                    # tables/
