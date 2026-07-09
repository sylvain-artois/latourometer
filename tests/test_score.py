"""Functional + golden-regression tests for ``score(text)``.

Skipped automatically when the heavy models / spaCy FR pipeline are not
installed, so the decoupling test still runs in a bare CI. When the models are
present these tests:

* assert the four-pole contract (keys, blended + raw scores, dominant pole);
* freeze a golden output (``golden_score.json``) and check ``score()`` keeps
  reproducing it within float tolerance — the standing calibration regression.
"""

from __future__ import annotations

import json
import pathlib

import pytest

POLES = {"terrestre", "global", "hors_sol", "local"}
GOLDEN = pathlib.Path(__file__).parent / "golden_score.json"

# A short, unambiguously-Terrestre probe (also used to mint the golden).
PROBE = (
    "Nous devons apprendre à habiter la Terre et à composer avec le vivant. "
    "La zone critique dépend de nos sols, de nos forêts et de la biodiversité. "
    "Atterrir, c'est prendre soin des écosystèmes dont nous dépendons."
)


def _score_or_skip(**kwargs):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("spacy")
    from latourometer import score

    try:
        return score(PROBE, **kwargs)
    except OSError as exc:  # spaCy FR model not downloaded
        pytest.skip(str(exc))


def test_four_pole_contract():
    result = _score_or_skip()
    assert set(result["scores"]) == POLES
    assert set(result["cosine_scores"]) == POLES
    assert set(result["stance_scores"]) == POLES
    assert result["dominant_pole"] in POLES
    assert result["blend_gamma"] == 1.0
    # Blended scores form a distribution (renormalised, sums ~1, non-negative).
    assert all(v >= 0 for v in result["scores"].values())
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-6


def test_pure_cosine_path():
    result = _score_or_skip(use_stance=False)
    assert set(result["scores"]) == POLES
    # No stance blend → no stance keys emitted by the metric.
    assert "stance_scores" not in result


def test_golden_regression():
    result = _score_or_skip()
    if not GOLDEN.exists():
        GOLDEN.write_text(
            json.dumps(result["scores"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        pytest.skip(f"minted golden at {GOLDEN}; re-run to assert against it")
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for pole, exp in expected.items():
        assert result["scores"][pole] == pytest.approx(exp, abs=1e-6), pole
