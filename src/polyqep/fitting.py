"""Piecewise quadratic polynomial fitting with C0/C1 continuity (quadratic spline regression).

Core fitting used in the accompanying paper:
    k(ω)   = k₁·ω² + k₂·ω + k₃
    ω·c(ω) = d₁·ω² + d₂·ω + d₃

The two curves share the **same segment partition** — continuity of both fits holds
simultaneously at the same breakpoints, so the complex M*/C*/K* matrices assembled
on each segment depend smoothly on ω.

On each segment j = 0..M, `y_j(ω) = a_j·ω² + b_j·ω + c_j`.

At the interior breakpoints ω_bk (k = 1..M):
    C0 of y_j:  a_{j-1}·ω_bk² + b_{j-1}·ω_bk + c_{j-1} = a_j·ω_bk² + b_j·ω_bk + c_j
    C1 of y_j:  2·a_{j-1}·ω_bk + b_{j-1}            = 2·a_j·ω_bk + b_j

Recursive solution:
    b_j = b_{j-1} + 2·ω_bk·(a_{j-1} - a_j)
    c_j = c_{j-1} - ω_bk²·(a_{j-1} - a_j)

Free parameters: θ = [a_0, b_0, c_0, a_1, a_2, ..., a_M]  (length M+3)

For each data point (ω_i, y_i), assemble the design-matrix row of `ŷ(ω_i) = A_i · θ`
and solve the least-squares problem `A θ ≈ y`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass
class QuadraticSegment:
    """Quadratic polynomial y = a·ω² + b·ω + c on a single segment."""
    a: float
    b: float
    c: float
    omega_lower: float
    omega_upper: float

    def __call__(self, omega: float | np.ndarray) -> float | np.ndarray:
        om = np.asarray(omega)
        return self.a * om ** 2 + self.b * om + self.c


@dataclass
class QuadraticSpline:
    """Quadratic spline with C0/C1 continuity — segment breakpoints and segment list."""
    breakpoints: np.ndarray           # (M+1,) interior+outer breakpoints [rad/s], ascending
    segments: list[QuadraticSegment]  # (M+1,) segment polynomials

    def __call__(self, omega):
        """Automatically select the segment containing ω and return the value.
        Scalar input returns a scalar, array input returns an array."""
        scalar_input = np.isscalar(omega) or (hasattr(omega, "ndim") and omega.ndim == 0)
        om = np.atleast_1d(np.asarray(omega, dtype=float))
        out = np.zeros_like(om)
        # Determine the segment index containing each ω
        # breakpoints[0] ≤ ω < breakpoints[1] → segment 0, ...
        inner = self.breakpoints[1:-1]   # (M-1,) interior breakpoints only
        idx = np.searchsorted(inner, om, side="right")  # 0..M
        idx = np.clip(idx, 0, len(self.segments) - 1)
        for j, seg in enumerate(self.segments):
            mask = idx == j
            if mask.any():
                out[mask] = seg(om[mask])
        return float(out[0]) if scalar_input else out

    @property
    def n_intervals(self) -> int:
        return len(self.segments)

    def coefficients_per_interval(self) -> list[tuple[float, float, float]]:
        """Return the list [(a0,b0,c0), (a1,b1,c1), ...]."""
        return [(s.a, s.b, s.c) for s in self.segments]


def _build_design_matrix(
    omega_data: np.ndarray,
    inner_breaks: np.ndarray,
) -> np.ndarray:
    """Assemble the design matrix A for the piecewise quadratic fit.

    θ = [a_0, b_0, c_0, a_1, a_2, ..., a_M] of length M+3.
    Row i: y_i = A[i,:] · θ.
    """
    n_data = omega_data.size
    M = inner_breaks.size
    n_free = M + 3
    A = np.zeros((n_data, n_free), dtype=float)

    # Determine the segment j containing each data point
    idx_interval = np.searchsorted(inner_breaks, omega_data, side="right")
    # idx_interval ∈ {0, 1, ..., M}

    for i in range(n_data):
        omega_i = omega_data[i]
        j = int(idx_interval[i])

        # Express a_j, b_j, c_j as coefficient vectors over the components of θ
        a_coef = np.zeros(n_free)
        b_coef = np.zeros(n_free)
        c_coef = np.zeros(n_free)

        # a_j: j=0 → θ[0], j≥1 → θ[j+2]
        if j == 0:
            a_coef[0] = 1.0
        else:
            a_coef[j + 2] = 1.0

        # b_j: b_0 + Σ_{k=1}^{j} 2·ω_bk·(a_{k-1} - a_k)
        b_coef[1] = 1.0  # b_0 = θ[1]
        for k in range(1, j + 1):
            omega_bk = inner_breaks[k - 1]
            # a_{k-1} index in θ:
            if k - 1 == 0:
                idx_prev = 0
            else:
                idx_prev = (k - 1) + 2
            idx_curr = k + 2   # a_k
            b_coef[idx_prev] += 2 * omega_bk
            b_coef[idx_curr] -= 2 * omega_bk

        # c_j: c_0 - Σ_{k=1}^{j} ω_bk²·(a_{k-1} - a_k)
        c_coef[2] = 1.0  # c_0 = θ[2]
        for k in range(1, j + 1):
            omega_bk = inner_breaks[k - 1]
            if k - 1 == 0:
                idx_prev = 0
            else:
                idx_prev = (k - 1) + 2
            idx_curr = k + 2
            c_coef[idx_prev] -= omega_bk ** 2
            c_coef[idx_curr] += omega_bk ** 2

        # y(ω_i) = a·ω² + b·ω + c
        A[i, :] = a_coef * omega_i ** 2 + b_coef * omega_i + c_coef

    return A


def _unpack_theta(
    theta: np.ndarray,
    boundaries: np.ndarray,
) -> list[QuadraticSegment]:
    """Compute (a, b, c) for each segment from the optimized θ."""
    M = boundaries.size - 2   # number of interior breakpoints
    inner_breaks = boundaries[1:-1]

    a_list = [theta[0]] + [theta[j + 2] for j in range(1, M + 1)]
    b_list = [theta[1]]
    c_list = [theta[2]]
    for k in range(1, M + 1):
        omega_bk = inner_breaks[k - 1]
        diff = a_list[k - 1] - a_list[k]
        b_list.append(b_list[k - 1] + 2 * omega_bk * diff)
        c_list.append(c_list[k - 1] - omega_bk ** 2 * diff)

    segments = []
    for j in range(M + 1):
        segments.append(QuadraticSegment(
            a=float(a_list[j]),
            b=float(b_list[j]),
            c=float(c_list[j]),
            omega_lower=float(boundaries[j]),
            omega_upper=float(boundaries[j + 1]),
        ))
    return segments


def fit_quadratic_spline(
    omega_data: np.ndarray,
    y_data: np.ndarray,
    boundaries: np.ndarray,
) -> QuadraticSpline:
    """Quadratic spline regression with C0/C1 continuity.

    Parameters
    ----------
    omega_data : (N,) ndarray
        Angular frequencies of the data points [rad/s]
    y_data : (N,) ndarray
        Corresponding material-property values
    boundaries : (M+2,) ndarray
        Segment breakpoints [rad/s]. boundaries[0] = ω_min, boundaries[-1] = ω_max,
        the intermediate entries are interior breakpoints. Produces M+1 segments in total.

    Returns
    -------
    QuadraticSpline
    """
    omega_data = np.asarray(omega_data, dtype=float).ravel()
    y_data = np.asarray(y_data, dtype=float).ravel()
    boundaries = np.asarray(boundaries, dtype=float).ravel()

    if boundaries.size < 2:
        raise ValueError("boundaries must contain at least 2 entries (the 2 outer breakpoints)")
    if not np.all(np.diff(boundaries) > 0):
        raise ValueError("boundaries must be strictly increasing")
    if omega_data.size != y_data.size:
        raise ValueError("omega_data and y_data must have the same length")

    inner_breaks = boundaries[1:-1]
    A = _build_design_matrix(omega_data, inner_breaks)
    theta, *_ = np.linalg.lstsq(A, y_data, rcond=None)

    segments = _unpack_theta(theta, boundaries)
    return QuadraticSpline(breakpoints=boundaries, segments=segments)


def fit_joint_spline(
    omega_data: np.ndarray,
    k_data: np.ndarray,
    omega_c_data: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[QuadraticSpline, QuadraticSpline]:
    """Fit k(ω) and ω·c(ω) separately using the **same segment partition**.

    Returns
    -------
    (spline_k, spline_omega_c) : two QuadraticSpline objects
    """
    spline_k = fit_quadratic_spline(omega_data, k_data, boundaries)
    spline_oc = fit_quadratic_spline(omega_data, omega_c_data, boundaries)
    return spline_k, spline_oc


def _third_derivative(omega: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Approximate y'''(ω) by a fourth-order central finite difference. Boundary entries are padded with 0."""
    n = omega.size
    d3 = np.zeros(n)
    if n < 5:
        return d3
    dx = np.mean(np.diff(omega))
    # 2nd-order central diff for 3rd derivative: (y[i+2] - 2y[i+1] + 2y[i-1] - y[i-2]) / (2 dx^3)
    for i in range(2, n - 2):
        d3[i] = (y[i + 2] - 2 * y[i + 1] + 2 * y[i - 1] - y[i - 2]) / (2 * dx ** 3)
    return d3


def detect_breakpoints(
    omega_data: np.ndarray,
    y_datasets: list[np.ndarray],
    n_breakpoints: int,
    min_separation_rad: float | None = None,
) -> np.ndarray:
    """Automatically select interior breakpoints at inflection points based on the summed absolute third derivatives of multiple curves.

    Parameters
    ----------
    omega_data : (N,) ndarray
    y_datasets : list of (N,) ndarray
        Curves used for the joint fit (e.g. k(ω), ω·c(ω)). Each is normalized before summation.
    n_breakpoints : int
        Number of interior breakpoints to extract (= number of segments − 1)
    min_separation_rad : float, optional
        Minimum spacing between breakpoints [rad/s]. Default is 1/(2·(n_breakpoints+1)) of the full band.

    Returns
    -------
    breakpoints : (n_breakpoints,) ndarray
        Selected interior breakpoint frequencies [rad/s], ascending.
    """
    omega_data = np.asarray(omega_data, dtype=float)
    N = omega_data.size
    if n_breakpoints <= 0:
        return np.array([])

    # Third derivative of each curve → normalized absolute value → sum
    score = np.zeros(N)
    for y in y_datasets:
        y_arr = np.asarray(y, dtype=float)
        d3 = np.abs(_third_derivative(omega_data, y_arr))
        # scale so that each dataset contributes on same footing
        d3_max = d3.max()
        if d3_max > 0:
            d3 = d3 / d3_max
        score += d3

    # Boundary condition: exclude the padded regions at both ends (first/last ~5%)
    pad = max(int(0.05 * N), 3)
    score_interior = score.copy()
    score_interior[:pad] = 0
    score_interior[-pad:] = 0

    if min_separation_rad is None:
        total_span = omega_data[-1] - omega_data[0]
        min_separation_rad = total_span / (2.0 * (n_breakpoints + 1))

    # Greedy selection: descending score, enforcing the minimum spacing from already chosen breakpoints
    order = np.argsort(-score_interior)
    chosen: list[float] = []
    for idx in order:
        candidate_omega = float(omega_data[idx])
        if score_interior[idx] <= 0:
            break
        if all(abs(candidate_omega - c) >= min_separation_rad for c in chosen):
            chosen.append(candidate_omega)
        if len(chosen) >= n_breakpoints:
            break

    # If short, fill in with an even split
    if len(chosen) < n_breakpoints:
        fallback = np.linspace(omega_data[0], omega_data[-1], n_breakpoints + 2)[1:-1]
        for f in fallback:
            if len(chosen) >= n_breakpoints:
                break
            if all(abs(f - c) >= min_separation_rad * 0.5 for c in chosen):
                chosen.append(float(f))

    return np.sort(np.array(chosen[:n_breakpoints], dtype=float))


def auto_boundaries(
    omega_data: np.ndarray,
    y_datasets: list[np.ndarray],
    n_intervals: int,
) -> np.ndarray:
    """Outer breakpoints at both ends + automatic interior breakpoints → full boundaries array.

    Parameters
    ----------
    n_intervals : int
        Desired number of segments (≥ 1). There are n_intervals − 1 interior breakpoints.
    """
    if n_intervals < 1:
        raise ValueError("n_intervals >= 1")
    inner = detect_breakpoints(omega_data, y_datasets, n_intervals - 1)
    return np.concatenate([[omega_data[0]], inner, [omega_data[-1]]])


# =============================================================================
# Resonance-Avoiding Adaptive Splitting
# =============================================================================
#
# Observation from the accompanying paper: inflection-based automatic splitting
# risks placing an inner breakpoint at a resonance. Near a resonance, the large
# values of D(ω)⁻¹ multiply the fitting error, so the response error grows
# explosively. Placing breakpoints in the valleys between resonances lets the
# effect of the fitting error decay naturally in the response.
#
# Reference: the accompanying paper (resonance-avoiding placement performed best)

def estimate_resonance_frequencies(
    M: np.ndarray,
    K_eff: np.ndarray,
) -> np.ndarray:
    """Generalized eigenvalues of (K_eff, M) → undamped natural frequencies ω_n [rad/s], positive only, ascending.

    Parameters
    ----------
    M : (n, n) ndarray
        Mass matrix (symmetric positive definite).
    K_eff : (n, n) ndarray
        Effective stiffness matrix evaluated at the band center (= K_base + k_H(ω_mid)·L).
        Captures the average effect of the frequency-dependent term.

    Returns
    -------
    omega_n : (k,) ndarray
        Positive undamped natural frequencies [rad/s], ascending. k ≤ n.
    """
    from scipy.linalg import eigh
    M_arr = np.asarray(M, dtype=float)
    K_arr = np.asarray(K_eff, dtype=float)
    # If K_eff is not positive definite (negative eigenvalues possible), eigh handles it
    eigvals = eigh(K_arr, M_arr, eigvals_only=True)
    # Positive only (negative eigenvalues are unphysical), sorted
    positive = eigvals[eigvals > 0]
    omega_n = np.sqrt(positive)
    return np.sort(omega_n)


def resonance_avoiding_boundaries(
    omega_data: np.ndarray,
    n_intervals: int,
    resonances_rad: np.ndarray,
    safety_margin: float = 0.05,
) -> np.ndarray:
    """Boundaries with inner breakpoints placed in the valleys that evenly partition the resonances.

    Strategy (quantile-based valley split):
        To split the K in-band resonances evenly into n_intervals groups, use the
        geometric mean of the adjacent resonance pair (k_i, k_{i+1}) at each group
        boundary as the inner breakpoint. Each segment then contains a similar
        number of resonances (balanced fitting domains) while the breakpoints
        naturally land at the valley centers.

    Parameters
    ----------
    omega_data : (N,) ndarray
        Frequency grid of the fitting data [rad/s]. The first/last entries are the band edges.
    n_intervals : int
        Desired number of segments (≥ 1). Preferably at most the number of in-band resonances + 1.
    resonances_rad : (k,) ndarray
        System resonance frequencies [rad/s] (in-band or not). Usually estimated
        with `estimate_resonance_frequencies()`.
    safety_margin : float
        Only the region inset by this fraction from both band edges is used for valley candidates (default 5%).

    Returns
    -------
    boundaries : (n_intervals + 1,) ndarray
        Of the form [omega_min, ..., omega_max], ascending.

    Notes
    -----
    If there are too few in-band resonances (n_intervals > K + 1), fill in with an even split.
    """
    if n_intervals < 1:
        raise ValueError("n_intervals >= 1")

    omega_data = np.asarray(omega_data, dtype=float)
    omega_min = float(omega_data[0])
    omega_max = float(omega_data[-1])

    if n_intervals == 1:
        return np.array([omega_min, omega_max])

    span = omega_max - omega_min
    pad = safety_margin * span
    n_inner = n_intervals - 1

    # In-band resonances only (inside the safety margin)
    res = np.asarray(resonances_rad, dtype=float)
    in_band = np.sort(res[(res > omega_min + pad) & (res < omega_max - pad)])
    K = in_band.size

    boundary_left = omega_min + pad
    boundary_right = omega_max - pad
    if K == 0:
        chosen = list(np.linspace(omega_min, omega_max, n_intervals + 1)[1:-1])
        return np.concatenate([[omega_min], np.asarray(chosen, dtype=float),
                               [omega_max]])

    # 1) Even-split starting points (pattern validated empirically in the accompanying paper)
    even_breaks = np.linspace(omega_min, omega_max, n_intervals + 1)[1:-1]

    # 2) All valley candidates = geometric means between adjacent resonances/edges
    augmented = np.concatenate([[boundary_left], in_band, [boundary_right]])
    valley_centers: list[float] = []
    valley_gaps: list[float] = []
    for i in range(len(augmented) - 1):
        lo, hi = augmented[i], augmented[i + 1]
        if hi <= lo:
            continue
        gap = hi - lo
        if lo > 0 and hi > 0:
            mid = float(np.sqrt(lo * hi))
        else:
            mid = 0.5 * (lo + hi)
        if not (boundary_left <= mid <= boundary_right):
            continue
        valley_centers.append(mid)
        valley_gaps.append(gap)

    if not valley_centers:
        # No valleys (resonances outside the safety margin) — even split
        chosen = list(even_breaks)
        return np.concatenate([[omega_min], np.asarray(chosen, dtype=float),
                               [omega_max]])

    valley_centers_arr = np.array(valley_centers)
    valley_gaps_arr = np.array(valley_gaps)
    # Exclude valleys that are too narrow (less than 5% of the band span)
    min_gap = 0.05 * span
    wide_mask = valley_gaps_arr >= min_gap
    if wide_mask.any():
        valley_centers_arr = valley_centers_arr[wide_mask]
    # else: all valleys are narrow — just use them all

    # 3) Nudge each even_break to the nearest valley (greedy + resonance-risk adaptive)
    #    Default max_nudge = 0.5 * even_step (conservative — prioritize keeping even spacing)
    #    Exception: if an even_break is very close to a resonance (<0.15 * even_step),
    #         nudge aggressively to avoid the resonance (= allow up to even_step)
    chosen: list[float] = []
    available = list(valley_centers_arr)
    even_step = (omega_max - omega_min) / n_intervals
    res_arr = in_band   # in-band resonances (after safety_margin)
    danger_threshold = 0.15 * even_step

    for eb in even_breaks:
        if not available:
            chosen.append(float(eb))
            continue
        # If the even_break is too close to a resonance (dangerous), use the full max_nudge
        if res_arr.size > 0:
            min_dist_to_res = float(np.min(np.abs(res_arr - eb)))
        else:
            min_dist_to_res = float("inf")
        in_danger = min_dist_to_res < danger_threshold
        max_nudge = even_step if in_danger else 0.5 * even_step

        dists = [abs(v - eb) for v in available]
        nearest_idx = int(np.argmin(dists))
        nearest_v = available[nearest_idx]
        if abs(nearest_v - eb) <= max_nudge:
            chosen.append(float(nearest_v))
            available.pop(nearest_idx)
        else:
            chosen.append(float(eb))   # no suitable valley — keep the even position

    chosen = sorted(chosen)
    return np.concatenate([[omega_min], np.asarray(chosen, dtype=float),
                           [omega_max]])
