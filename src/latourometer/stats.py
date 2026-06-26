"""Bootstrap confidence intervals for Latouromètre accuracy reporting.

Part 1 of PRD ``2_latourometer-statistical-robustness-and-corpus-extension``.

The calibration corpus (currently 39 texts) is treated as a *sample* of a
hypothetical larger population of political texts. Resampling the per-text
correctness vector with replacement and recomputing accuracy yields a
distribution whose 2.5 / 97.5 percentiles bound a 95 % confidence interval.
This lets us report ``0.90 [0.78, 0.98]`` instead of a bare ``19/21`` and,
crucially, say honestly when a calibration fork (γ=1.0 vs λ=3) lives below the
statistical resolution of the corpus.

Pure-numpy, no I/O — the calibration scripts call these and serialize the
returned dicts to ``accuracy_ci.json``.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

# Default percentile bounds for a 95 % CI.
_CI_LO_PCT = 2.5
_CI_HI_PCT = 97.5


def _as_correct_array(correct: Sequence[Any]) -> np.ndarray:
    """Coerce a per-text correctness sequence to a float {0,1} array.

    Accepts bools, ints, the literal strings ``"True"``/``"False"`` (as emitted
    by ``calibrate_latourometre.py`` matrix rows), or anything truthy/falsy.
    """
    out = np.empty(len(correct), dtype=float)
    for i, c in enumerate(correct):
        if isinstance(c, str):
            out[i] = 1.0 if c.strip().lower() == "true" else 0.0
        else:
            out[i] = 1.0 if c else 0.0
    return out


def bootstrap_accuracy_ci(
    correct: Sequence[Any], n_iter: int = 1000, seed: int = 42
) -> Dict[str, Any]:
    """Bootstrap 95 % CI on accuracy from a per-text correctness vector.

    Args:
        correct: per-text correctness (1/True = predicted pole matches label).
        n_iter: number of bootstrap resamples (default 1000).
        seed: RNG seed for reproducibility (default 42).

    Returns:
        ``{point, mean, ci_lo, ci_hi, n, n_iter, seed}`` where ``point`` is the
        observed accuracy and ``mean`` is the bootstrap-resample mean.
    """
    arr = _as_correct_array(correct)
    n = arr.size
    if n == 0:
        return {
            "point": 0.0, "mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
            "n": 0, "n_iter": int(n_iter), "seed": int(seed),
        }
    rng = np.random.default_rng(seed)
    accs = np.array(
        [arr[rng.integers(0, n, size=n)].mean() for _ in range(n_iter)]
    )
    return {
        "point": float(arr.mean()),
        "mean": float(accs.mean()),
        "ci_lo": float(np.percentile(accs, _CI_LO_PCT)),
        "ci_hi": float(np.percentile(accs, _CI_HI_PCT)),
        "n": int(n),
        "n_iter": int(n_iter),
        "seed": int(seed),
    }


def paired_bootstrap_diff_ci(
    correct_a: Sequence[Any],
    correct_b: Sequence[Any],
    n_iter: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap CI on the accuracy *difference* between two methods.

    Resamples the *same* text indices on both methods' per-text results, so the
    difference is measured on matched samples (the texts are the unit of
    resampling, not the methods independently). If the CI on the difference
    includes 0, the choice between A and B is statistically inconclusive at this
    corpus size — used to defend γ=1.0 vs λ=3 honestly.

    Args:
        correct_a, correct_b: per-text correctness for methods A and B. Must be
            aligned text-by-text and of equal length.

    Returns:
        ``{point_a, point_b, mean_diff, ci_lo, ci_hi, inconclusive, n, n_iter,
        seed}``. ``mean_diff`` is A − B; ``inconclusive`` is True when the CI
        straddles 0.
    """
    a = _as_correct_array(correct_a)
    b = _as_correct_array(correct_b)
    if a.size != b.size:
        raise ValueError(
            f"paired bootstrap requires equal-length vectors, "
            f"got {a.size} and {b.size}"
        )
    n = a.size
    if n == 0:
        return {
            "point_a": 0.0, "point_b": 0.0, "mean_diff": 0.0,
            "ci_lo": 0.0, "ci_hi": 0.0, "inconclusive": True,
            "n": 0, "n_iter": int(n_iter), "seed": int(seed),
        }
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_iter, dtype=float)
    for k in range(n_iter):
        idx = rng.integers(0, n, size=n)
        diffs[k] = a[idx].mean() - b[idx].mean()
    ci_lo = float(np.percentile(diffs, _CI_LO_PCT))
    ci_hi = float(np.percentile(diffs, _CI_HI_PCT))
    return {
        "point_a": float(a.mean()),
        "point_b": float(b.mean()),
        "mean_diff": float(diffs.mean()),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "inconclusive": bool(ci_lo <= 0.0 <= ci_hi),
        "n": int(n),
        "n_iter": int(n_iter),
        "seed": int(seed),
    }


def format_ci(ci: Dict[str, Any]) -> str:
    """Render a bootstrap CI dict as ``0.905 [0.762, 1.000]`` for logs/docs."""
    return f"{ci.get('point', ci.get('mean', 0.0)):.3f} [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]"
