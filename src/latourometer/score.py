"""``score(text)`` — the standalone Latouromètre entrypoint.

Places a French political text on Bruno Latour's two attractor axes
(*Où atterrir?*, 2017)::

        Terrestre (top)
            |
    Local --+-- Global
            |
        Hors-Sol (bottom)

It runs the full scoring in-process, with no server runtime behind it: segment
the text, run the NLI stance metric, then the cosine SemAxis projection blended
with that stance (additive-γ, γ=1.0 — the calibrated default).

The return value is the ``latourometre`` metric block:

    {
      "scores":        {pole: blended_score},   # the four-pole result
      "dominant_pole": "<key>",                 # argmax of scores
      "labels":        {pole: "Terrestre"...},
      "pole_order":    [...],
      "cosine_scores": {pole: ...},             # pre-blend SemAxis projection
      "stance_scores": {pole: ...},             # NLI stance in [-1, +1]
      "blend_gamma":   1.0,
      "stance":        {...},                   # full stance metric payload
    }
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import AnalysisContext, _segment_answer
from .metrics.latour_stance import LatourStanceMetric
from .metrics.latourometre import LatourometreMetric
from .runtime import get_embedder, get_nli_pipeline, get_spacy_nlp, project_root

# Calibrated defaults: seeds v4 + γ=1.0 blend, τ=0.03 per-chunk softmax (the
# "sharper winner-take-all" calibration; the metric's own fallback default is
# 0.05).
_SEEDS_FILE = "config/latourometre-seeds.yml"
_HYPOTHESES_FILE = "config/latourometre-stance-hypotheses.yml"
_DEFAULT_GAMMA = 1.0
_DEFAULT_SOFTMAX_TAU = 0.03


def score(
    text: str,
    *,
    gamma: float = _DEFAULT_GAMMA,
    use_stance: bool = True,
    softmax_tau: Optional[float] = _DEFAULT_SOFTMAX_TAU,
) -> Dict[str, Any]:
    """Score one French text on Latour's four attractors.

    Args:
        text: the raw French text to place on the two axes.
        gamma: additive stance-blend strength (γ=1.0 is the calibrated default;
            0.0 collapses to the pure cosine SemAxis projection).
        use_stance: when False, skip the NLI stance layer entirely and return the
            pure cosine projection (faster — no NLI model load).
        softmax_tau: per-chunk softmax temperature (τ=0.03 is the calibrated
            default; ``None`` falls back to the metric's own 0.05).

    Returns:
        A dict holding the ``latourometre`` metric block, plus a convenience
        ``dominant_pole`` key (the argmax of ``scores``).
    """
    nlp = get_spacy_nlp()
    segments, _strategy = _segment_answer(nlp, text or "")

    ctx = AnalysisContext(
        nlp_doc_per_answer=segments,
        embedder=get_embedder,
        nli=get_nli_pipeline if use_stance else None,
        project_root=project_root(),
        language="fr",
        metrics_so_far={},
    )

    if use_stance:
        stance = LatourStanceMetric({"hypotheses_file": _HYPOTHESES_FILE}).compute(ctx)
        ctx.metrics_so_far["latour_stance"] = stance
    else:
        stance = None

    latour_opts: Dict[str, Any] = {"seeds_file": _SEEDS_FILE}
    if use_stance:
        latour_opts["stance_blend_gamma"] = gamma
    if softmax_tau is not None:
        latour_opts["softmax_tau"] = softmax_tau

    result = LatourometreMetric(latour_opts).compute(ctx)

    scores = result.get("scores") or {}
    if scores:
        result["dominant_pole"] = max(scores, key=scores.get)
    if stance is not None:
        result["stance"] = stance
    return result
