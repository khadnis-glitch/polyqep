"""PolyQEP — piecewise-continuous modulus framework and reduced-order models
for frequency-response analysis of structures with frequency-dependent
viscoelastic damping.

Reference implementation accompanying:

    S.-H. Bae, S. Baek, T. Kim, I. Song,
    "PolyQEP: A Piecewise-Continuous Modulus Framework and Reduced-Order
    Models for Efficient Frequency-Response Analysis of Structures with
    Frequency-Dependent Viscoelastic Damping" (submitted).
"""

__version__ = "0.1.0"

from polyqep.fitting import (
    QuadraticSegment,
    QuadraticSpline,
    fit_joint_spline,
    detect_breakpoints,
)
from polyqep.qep_solver import solve_qep
from polyqep.intervals import (
    assemble_complex_per_interval,
    solve_qep_per_interval,
)
from polyqep.pw_refined import (
    PWRefinedSegmentROM,
    PWRefinedGlobalROM,
    build_pw_refined_segment_roms,
    build_pw_refined_global_rom,
    build_pw_refined_roms_profiled,
    pw_refined_response,
    pw_refined_global_response,
)
from polyqep.rom import (
    build_soar_rom,
    build_multi_shift_soar_rom,
    soar_response,
)

__all__ = [
    "QuadraticSegment",
    "QuadraticSpline",
    "fit_joint_spline",
    "detect_breakpoints",
    "solve_qep",
    "assemble_complex_per_interval",
    "solve_qep_per_interval",
    "PWRefinedSegmentROM",
    "PWRefinedGlobalROM",
    "build_pw_refined_segment_roms",
    "build_pw_refined_global_rom",
    "build_pw_refined_roms_profiled",
    "pw_refined_response",
    "pw_refined_global_response",
    "build_soar_rom",
    "build_multi_shift_soar_rom",
    "soar_response",
]
