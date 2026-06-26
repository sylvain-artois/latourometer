"""Unit tests for the Wordfish baseline core (PRD-9).

These exercise the pure model math only -- no spaCy model and no corpus volume.
Run inside the container:

    docker compose run --rm corpus-builder python -m pytest tests/test_wordfish.py -q
"""

import numpy as np

from latourometer.baselines.wordfish import (
    bootstrap_accuracy,
    fit_wordfish,
    fold_pole_2d,
    orient_axis,
    project_theta,
)


def _two_cluster_corpus():
    """Synthetic 2-cluster corpus: a 'left' word group and a 'right' group.

    Six documents, two latent groups. Group A loads on words 0-2, group B on
    words 3-5; a couple of shared filler words sit in every doc. A correct
    Wordfish fit must separate the two groups along theta.
    """
    # words:           A0  A1  A2  B0  B1  B2  fill fill
    docs = np.array([
        [20, 18, 15,  1,  0,  1,  5, 4],   # A
        [17, 22, 13,  0,  1,  0,  4, 6],   # A
        [19, 16, 20,  1,  1,  0,  6, 5],   # A
        [ 1,  0,  1, 21, 17, 16,  5, 4],   # B
        [ 0,  1,  0, 18, 20, 19,  4, 6],   # B
        [ 1,  1,  0, 16, 22, 18,  6, 5],   # B
    ], dtype=float)
    group = np.array(["A", "A", "A", "B", "B", "B"])
    return docs, group


def test_fit_separates_two_clusters():
    Y, group = _two_cluster_corpus()
    fit = fit_wordfish(Y)
    assert fit.converged
    a = fit.theta[group == "A"]
    b = fit.theta[group == "B"]
    # The two clusters land on opposite sides of the axis (well-separated).
    assert a.mean() * b.mean() < 0
    assert abs(a.mean() - b.mean()) > 1.0


def test_theta_is_standardised():
    Y, _ = _two_cluster_corpus()
    fit = fit_wordfish(Y)
    assert abs(fit.theta.mean()) < 1e-6
    assert abs(fit.theta.std() - 1.0) < 1e-6


def test_fit_is_deterministic():
    Y, _ = _two_cluster_corpus()
    t1 = fit_wordfish(Y).theta
    t2 = fit_wordfish(Y).theta
    assert np.allclose(t1, t2)


def test_orient_axis_puts_plus_pole_positive():
    Y, group = _two_cluster_corpus()
    fit = fit_wordfish(Y)
    # Declare cluster B the "plus" pole; after orientation its mean theta > 0.
    plus = group == "B"
    minus = group == "A"
    theta, beta, _flipped = orient_axis(fit.theta, fit.beta, plus, minus)
    assert theta[plus].mean() > 0
    assert theta[minus].mean() < 0


def test_orient_axis_flips_beta_with_theta():
    theta = np.array([-1.0, 1.0])
    beta = np.array([0.5, -0.3])
    plus = np.array([True, False])   # plus pole is doc 0 (theta negative -> flip)
    minus = np.array([False, True])
    new_theta, new_beta, flipped = orient_axis(theta, beta, plus, minus)
    assert flipped
    assert np.allclose(new_theta, -theta)
    assert np.allclose(new_beta, -beta)


def test_projection_recovers_in_sample_side():
    Y, group = _two_cluster_corpus()
    fit = fit_wordfish(Y)
    plus = group == "B"
    minus = group == "A"
    theta, beta, _ = orient_axis(fit.theta, fit.beta, plus, minus)
    # Project a fresh strongly-A document: should land on the A (negative) side.
    new_doc = np.array([22, 19, 17, 0, 1, 0, 5, 5], dtype=float)
    th = project_theta(new_doc, fit.psi, beta)
    assert th < 0


def test_fold_pole_2d_picks_dominant_axis():
    # Larger |theta| on HS<->T, positive -> terrestre.
    assert fold_pole_2d(1.5, 0.3) == "terrestre"
    assert fold_pole_2d(-1.5, 0.3) == "hors_sol"
    # Larger |theta| on L<->G, positive -> global, negative -> local.
    assert fold_pole_2d(0.2, 1.4) == "global"
    assert fold_pole_2d(0.2, -1.4) == "local"


def test_bootstrap_accuracy_bounds():
    lo, hi = bootstrap_accuracy([True] * 10)
    assert lo == 1.0 and hi == 1.0
    lo, hi = bootstrap_accuracy([True, False, True, False, True, False])
    assert 0.0 <= lo <= hi <= 1.0
    assert bootstrap_accuracy([]) == (None, None)
