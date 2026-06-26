"""Unit tests for the text-scoring hardening (per-text CI + Lowe log-odds).

Covers two methodological-hardening features added to the comparison scorer:

  * **Feature 1** — a per-text bootstrap CI on the mean in-lexicon score, with a
    ``sign_robust`` flag (the CI does not cross 0).
  * **Feature 2** — Lowe (2008) log-odds ``theta(w)`` text scoring as an
    alternative to the bounded LBG ``S(w)`` mean.

All math is hand-computable from a 4-word fixture; no spaCy model and no corpus
volume are needed. Run inside the container:

    docker compose -f compose.yml run --rm corpus-builder python -m pytest -q
"""

import math

from latourometer.baselines.compare import (
    ScoredText,
    _LOWE_ALPHA,
    axis_score,
    bootstrap_text_ci,
    compute_lowe_scores,
)


# --- LBG fixture: known S(w), used for axis_score + sign_robust ---------------


def _lbg_lexicon():
    # Hors-Sol words score negative, Terrestre words positive.
    return {
        ("marché", "NOUN"): -0.9,
        ("croissance", "NOUN"): -0.8,
        ("terre", "NOUN"): 0.9,
        ("vivant", "NOUN"): 0.8,
    }


# --- Lowe fixture: known per-pole counts, V=4 --------------------------------
#   marché:     minus=8, plus=0
#   croissance: minus=4, plus=0
#   terre:      minus=0, plus=6
#   vivant:     minus=0, plus=4
# N_minus = 12, N_plus = 10, V = 4, alpha = 0.5.
# denom_minus = 12 + 0.5*4 = 14 ; denom_plus = 10 + 0.5*4 = 12.


def _pole_counts():
    return {
        ("marché", "NOUN"): (8, 0),
        ("croissance", "NOUN"): (4, 0),
        ("terre", "NOUN"): (0, 6),
        ("vivant", "NOUN"): (0, 4),
    }


# === (a) LBG axis_score = expected mean ======================================


def test_lbg_axis_score_is_expected_mean():
    lex = _lbg_lexicon()
    tokens = [("marché", "NOUN"), ("terre", "NOUN"), ("inconnu", "NOUN")]
    score, n_hits = axis_score(tokens, lex)
    assert n_hits == 2
    assert math.isclose(score, (-0.9 + 0.9) / 2, abs_tol=1e-12)  # = 0.0


# === (b) Lowe theta(w) = expected log-ratio ==================================


def test_lowe_theta_matches_hand_computed_log_ratio():
    alpha = _LOWE_ALPHA
    assert alpha == 0.5
    theta = compute_lowe_scores(_pole_counts())

    # Hand formula: theta = log f_plus - log f_minus, f smoothed.
    # marché: f_plus = (0+0.5)/12, f_minus = (8+0.5)/14  -> strongly negative.
    exp_marche = math.log((0 + 0.5) / 12) - math.log((8 + 0.5) / 14)
    # terre: f_plus = (6+0.5)/12, f_minus = (0+0.5)/14   -> strongly positive.
    exp_terre = math.log((6 + 0.5) / 12) - math.log((0 + 0.5) / 14)

    assert math.isclose(theta[("marché", "NOUN")], exp_marche, rel_tol=1e-12)
    assert math.isclose(theta[("terre", "NOUN")], exp_terre, rel_tol=1e-12)
    # Sign convention matches LBG: plus pole positive, minus pole negative.
    assert theta[("marché", "NOUN")] < 0
    assert theta[("terre", "NOUN")] > 0
    # Unbounded: a pole-exclusive word exceeds the LBG [-1,+1] bound.
    assert abs(theta[("terre", "NOUN")]) > 1.0


def test_lowe_text_position_is_mean_of_theta():
    theta = compute_lowe_scores(_pole_counts())
    tokens = [("terre", "NOUN"), ("vivant", "NOUN")]
    score, n_hits = axis_score(tokens, theta)
    expected = (theta[("terre", "NOUN")] + theta[("vivant", "NOUN")]) / 2
    assert n_hits == 2
    assert math.isclose(score, expected, rel_tol=1e-12)
    assert score > 0  # both terrestre words -> positive position


# === (c) sign_robust: same-sign text robust, straddling text not =============


def _scored(score, hits):
    """Build a ScoredText with a bootstrap CI computed from its in-lexicon hits."""
    lo, hi = bootstrap_text_ci(hits)
    return ScoredText(
        slug="t",
        category="seed",
        gold_pole="terrestre",
        on_axis=True,
        axis_score=score,
        n_hits=len(hits),
        n_tokens=len(hits),
        axis_score_ci_low=lo,
        axis_score_ci_high=hi,
    )


def test_sign_robust_true_when_all_tokens_same_sign():
    hits = [0.8, 0.9, 0.7, 0.85, 0.95]  # all positive
    score = sum(hits) / len(hits)
    r = _scored(score, hits)
    assert r.axis_score_ci_low is not None and r.axis_score_ci_low > 0
    assert r.sign_robust is True


def test_sign_robust_false_when_text_straddles_zero():
    hits = [-0.9, 0.9, -0.8, 0.8]  # balanced -> mean ~0, CI crosses 0
    score = sum(hits) / len(hits)
    r = _scored(score, hits)
    assert r.axis_score_ci_low < 0 < r.axis_score_ci_high
    assert r.sign_robust is False


def test_zero_hits_gives_null_ci_and_not_robust():
    lo, hi = bootstrap_text_ci([])
    assert (lo, hi) == (None, None)
    r = ScoredText(
        slug="empty",
        category="holdout",
        gold_pole=None,
        on_axis=False,
        axis_score=0.0,
        n_hits=0,
        n_tokens=10,
        axis_score_ci_low=None,
        axis_score_ci_high=None,
    )
    assert r.sign_robust is False


def test_bootstrap_ci_is_deterministic_for_fixed_seed():
    hits = [-0.9, 0.9, -0.8, 0.8, 0.1]
    assert bootstrap_text_ci(hits) == bootstrap_text_ci(hits)
