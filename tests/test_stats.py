"""Tests for scripts/_stats.py — bootstrap CIs (PRD 2, Part 1)."""
from __future__ import annotations

import numpy as np
import pytest

from latourometer.stats import (
    bootstrap_accuracy_ci,
    format_ci,
    paired_bootstrap_diff_ci,
)


def test_bootstrap_accuracy_ci_perfect():
    ci = bootstrap_accuracy_ci([1, 1, 1, 1, 1])
    assert ci["point"] == 1.0
    assert ci["ci_lo"] == 1.0
    assert ci["ci_hi"] == 1.0
    assert ci["n"] == 5


def test_bootstrap_accuracy_ci_19_of_21():
    # The production headline number: 19/21 correct.
    correct = [1] * 19 + [0] * 2
    ci = bootstrap_accuracy_ci(correct)
    assert ci["point"] == pytest.approx(19 / 21, abs=1e-9)
    # Bracket the point estimate, with a wide CI at this small N.
    assert ci["ci_lo"] < ci["point"] <= ci["ci_hi"]
    assert ci["ci_lo"] < 0.90  # the whole point: lower bound is well below 0.905
    assert ci["n"] == 21


def test_bootstrap_accuracy_ci_accepts_string_passed_flags():
    # calibrate_latourometre.py emits row["passed"] as "True"/"False".
    ci = bootstrap_accuracy_ci(["True", "True", "False", "True"])
    assert ci["point"] == pytest.approx(0.75)
    assert ci["n"] == 4


def test_bootstrap_accuracy_ci_deterministic_under_seed():
    correct = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    a = bootstrap_accuracy_ci(correct, seed=7)
    b = bootstrap_accuracy_ci(correct, seed=7)
    assert a == b


def test_bootstrap_accuracy_ci_empty():
    ci = bootstrap_accuracy_ci([])
    assert ci["n"] == 0
    assert ci["ci_lo"] == 0.0 and ci["ci_hi"] == 0.0


def test_paired_bootstrap_identical_methods_inconclusive():
    same = [1, 0, 1, 1, 0, 1]
    res = paired_bootstrap_diff_ci(same, same)
    assert res["mean_diff"] == pytest.approx(0.0)
    assert res["ci_lo"] == 0.0 and res["ci_hi"] == 0.0
    assert res["inconclusive"] is True


def test_paired_bootstrap_one_text_margin_inconclusive():
    # 19/21 vs 18/21: differ on a single text — should be inconclusive at N=21.
    a = [1] * 19 + [0] * 2
    b = [1] * 18 + [0] * 3
    res = paired_bootstrap_diff_ci(a, b)
    assert res["point_a"] > res["point_b"]
    assert res["inconclusive"] is True  # CI straddles 0


def test_paired_bootstrap_large_gap_conclusive():
    a = [1] * 20
    b = [0] * 20
    res = paired_bootstrap_diff_ci(a, b)
    assert res["mean_diff"] == pytest.approx(1.0)
    assert res["inconclusive"] is False
    assert res["ci_lo"] > 0.0


def test_paired_bootstrap_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_diff_ci([1, 0, 1], [1, 0])


def test_format_ci():
    s = format_ci({"point": 0.905, "ci_lo": 0.762, "ci_hi": 1.0})
    assert s == "0.905 [0.762, 1.000]"
