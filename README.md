# PolyQEP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21372623.svg)](https://doi.org/10.5281/zenodo.21372623)

Reference implementation of **PolyQEP**, a piecewise-continuous modulus
framework for frequency-response analysis of structures with
frequency-dependent viscoelastic damping, together with the **PW-R** and
**PW-RG** reduced-order models.

This repository accompanies the paper:

> S.-H. Bae, S. Baek, T. Kim, I. Song,
> *PolyQEP: A Piecewise-Continuous Modulus Framework and Reduced-Order Models
> for Efficient Frequency-Response Analysis of Structures with
> Frequency-Dependent Viscoelastic Damping*, submitted to
> the Journal of Sound and Vibration.

## What it does

For structures whose stiffness admits the affine decomposition
`K(w) = K_base + G(w) K_shear` with a frequency-dependent complex shear
modulus `G(w)`, the framework

1. fits `Re[G]` and `Im[G]` with a C0/C1-continuous piecewise-quadratic
   spline on shared segment boundaries (`polyqep.fitting`),
2. recasts the dynamic stiffness segment-wise into an omega-independent
   complex triplet `(M*, C*, K*)` and solves **one complex quadratic
   eigenvalue problem per segment**, selecting the effective eigenvalues
   with a band-and-stability rule (`polyqep.intervals`, `polyqep.qep_solver`),
3. builds reduced-order models on top of the same algebraic structure
   (`polyqep.rom`, `polyqep.pw_refined`):
   - **PW-R** — per-segment SOAR bases with online assembly from the raw
     `G(jw)` through the affine identity (no polynomial-fit residual floor),
   - **PW-RG** — a single broadband basis obtained by a cross-segment
     union-SVD of eigenvector-augmented segment bases, removing the
     per-frequency segment dispatch.

The package also ships the displacement-based sandwich plate element of
Amichi & Atalla (2010) used as the strict-affine benchmark in the paper
(`polyqep.models.sandwich_plate_amichi2010`).

## Installation

Requires Python >= 3.9, NumPy, and SciPy.

```bash
pip install -e .          # library only
pip install -e ".[examples]"  # with matplotlib for the example plot
```

## Quick start

```bash
python examples/amichi_plate_8x8.py
```

The example assembles the 8x8 Amichi sandwich plate (free boundary), fits a
DMA-type complex core modulus over 1-1500 Hz with five segments, solves the
per-segment complex QEPs, builds the PW-R and PW-RG ROMs, and verifies both
FRFs against the dense per-frequency direct solution over 5-500 Hz.

```python
import numpy as np
from polyqep import (
    fit_joint_spline, assemble_complex_per_interval, solve_qep_per_interval,
    build_pw_refined_roms_profiled, pw_refined_global_response,
)

# K(w) = K_base + G(w) K_shear, M: your BC-reduced system matrices
spline_re, spline_im = fit_joint_spline(omega, G_re, G_im, boundaries)
mats = assemble_complex_per_interval(spline_re, spline_im, M, K_base, K_shear)
sols = solve_qep_per_interval(mats)

rom, _ = build_pw_refined_roms_profiled(K_base, K_shear, M, F, sols,
                                        variant="global")
Y = pw_refined_global_response(omegas, rom, G_of_s, idx_obs)
```

## Package layout

| Module | Contents |
| --- | --- |
| `polyqep.fitting` | C0/C1-continuous piecewise-quadratic spline fit of `Re[G]`, `Im[G]` |
| `polyqep.intervals` | per-segment `(M*, C*, K*)` assembly, per-segment QEP + effective-eigenvalue selection |
| `polyqep.qep_solver` | dense quadratic eigenvalue solver (companion linearization) |
| `polyqep.polynomial_qep` | modal expansion of the dynamic-stiffness inverse, per-interval ROMs |
| `polyqep.rom` | SOAR / multi-shift SOAR second-order Krylov reduction |
| `polyqep.pw_refined` | PW-R and PW-RG reduced-order models with raw-`G(jw)` online assembly |
| `polyqep.transfer_function` | modal-superposition FRF utilities |
| `polyqep.mdof_6dof` | 6-DOF lumped demonstration system |
| `polyqep.models.sandwich_plate_amichi2010` | Amichi & Atalla (2010) sandwich plate element (Q4, zig-zag core shear, consistent mass) |

## Data and reproduction

This repository contains the implementation code for the PolyQEP framework
and the PW-R / PW-RG reduced-order models. The remaining materials of the
paper, including the reproduction scripts and validation data, are available
from the corresponding author upon reasonable request.

## Citation

If you use this code, please cite the paper above. A BibTeX entry will be
added upon publication.

## License

MIT — see [LICENSE](LICENSE).
