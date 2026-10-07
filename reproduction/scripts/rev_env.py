"""Additional studies — execution-environment record and timing helpers.

Every revision benchmark script calls :func:`environment_record` and stores
the result next to its numbers, so each reported time stays attributable to a
processor, an operating system, a linear-algebra backend, and a thread count.

Thread control: the BLAS thread caps must be exported *before* numpy/scipy are
imported, so :func:`require_single_thread` only verifies the state and aborts
when the process is not single-threaded.
"""
from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable, TypeVar

T = TypeVar("T")

THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _sysctl(key: str) -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _blas_of(config_module) -> dict:
    deps = config_module.show(mode="dicts")["Build Dependencies"]
    return {
        key: {"name": deps[key]["name"], "version": deps[key].get("version")}
        for key in ("blas", "lapack")
    }


def environment_record() -> dict:
    """Processor, memory, OS, Python stack, BLAS backends, and thread state."""
    import numpy as np
    import scipy
    import numpy.__config__ as np_config
    import scipy.__config__ as sp_config
    import scipy.linalg  # noqa: F401  (loads the scipy BLAS so it is listed)
    from threadpoolctl import threadpool_info

    mem_bytes = _sysctl("hw.memsize")
    return {
        "processor": _sysctl("machdep.cpu.brand_string"),
        "machine_model": _sysctl("hw.model"),
        "cores_physical": _sysctl("hw.physicalcpu"),
        "memory_gb": (int(mem_bytes) / 1024**3) if mem_bytes.isdigit() else None,
        # mac_ver() is the macOS product version; release() is the Darwin kernel.
        "os": (f"macOS {platform.mac_ver()[0]} (Darwin {platform.release()}, "
               f"{platform.machine()})"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numpy_blas_lapack": _blas_of(np_config),
        "scipy_blas_lapack": _blas_of(sp_config),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
        "threadpools": [
            {
                "user_api": t.get("user_api"),
                "internal_api": t.get("internal_api"),
                "version": t.get("version"),
                "num_threads": t.get("num_threads"),
            }
            for t in threadpool_info()
        ],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


def require_single_thread() -> None:
    """Abort unless every BLAS thread cap is set to 1."""
    missing = [k for k in THREAD_ENV_VARS if os.environ.get(k) != "1"]
    if missing:
        raise SystemExit(
            "single-thread run required; export "
            + " ".join(f"{k}=1" for k in missing)
            + " before starting Python"
        )


def timed_repeats(
    fn: Callable[[], T], n_repeat: int, n_warmup: int = 1
) -> tuple[T, dict]:
    """Run ``fn`` ``n_warmup`` + ``n_repeat`` times.

    Returns (last_result, stats) with wall-time statistics in seconds over the
    ``n_repeat`` measured runs (``n_repeat`` >= 1).
    """
    if n_repeat < 1:
        raise ValueError("n_repeat must be >= 1")
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    result = fn()
    samples = [time.perf_counter() - t0]
    for _ in range(n_repeat - 1):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    stats = {
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "n_repeat": n_repeat,
        "n_warmup": n_warmup,
    }
    return result, stats
