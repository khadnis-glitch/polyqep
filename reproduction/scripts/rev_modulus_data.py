"""Additional studies — the two core-modulus data sets used by the revision studies.

``representative``  the seven-anchor profile of the manuscript benchmark
                    (``G_complex_at_freq``: log-frequency interpolation of
                    estimated anchors that follow the trend of Fig. 7 of
                    Amichi and Atalla, 2010).
``digitized``       the pixel-digitized curves of that Fig. 7 (storage
                    modulus E' and loss factor eta, 75--639 Hz, 4 Hz step,
                    142 points). Shipped as ``data/amichi_fig7_dma_digitized.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from phase3_amichi_polynomial_qep import G_complex_at_freq

NU_CORE = 0.45
_ROOT = Path(__file__).resolve().parent.parent
DIGITIZED_CSV = _ROOT / "data" / "amichi_fig7_dma_digitized.csv"


def representative(f_hz) -> np.ndarray:
    """Complex shear modulus of the manuscript benchmark at ``f_hz`` [Hz]."""
    return np.array([G_complex_at_freq(f) for f in np.atleast_1d(f_hz)])


def digitized_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f_hz, E' [Pa], eta) of the digitized curves, measured band only.

    The first row of the source table (10 Hz) is a linear extrapolation added
    for the element-verification runs, not a digitized point, and is dropped.
    The table is shipped as ``data/amichi_fig7_dma_digitized.csv``.
    """
    if DIGITIZED_CSV.exists():
        f, E, eta = np.loadtxt(DIGITIZED_CSV, delimiter=",", skiprows=1, unpack=True)
        return f, E, eta
    raise FileNotFoundError(f"digitized DMA table not found: {DIGITIZED_CSV}")


def digitized(f_hz) -> np.ndarray:
    """Complex shear modulus interpolated from the digitized table.

    Linear interpolation inside 75--639 Hz; the end values are held outside.
    """
    f, E, eta = digitized_table()
    f_hz = np.atleast_1d(np.asarray(f_hz, dtype=float))
    G = np.interp(f_hz, f, E) / (2.0 * (1.0 + NU_CORE))
    return G * (1.0 + 1j * np.interp(f_hz, f, eta))
