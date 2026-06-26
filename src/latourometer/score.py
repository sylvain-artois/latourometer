"""``score(text)`` — the standalone Latouromètre entrypoint.

Places a French political text on Bruno Latour's two attractor axes
(*Où atterrir?*, 2017)::

        Terrestre (top)
            |
    Local --+-- Global
            |
        Hors-Sol (bottom)

It reproduces the AFK ``words-weight`` production scoring without any of the
pipeline machinery (no Redis, no Postgres, no transcript payload): segment the
text exactly as prod segments one transcript answer, run the NLI stance metric,
then the cosine SemAxis projection blended with that stance (additive-γ, γ=1.0
— the PRD-6 calibrated default).

The return value mirrors the production ``metrics.json`` ``latourometre`` block:

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

# Calibrated production config (PRD-4 embedder, PRD-6 seeds v4 + γ=1.0 blend).
_SEEDS_FILE = "config/latourometre-seeds.yml"
_HYPOTHESES_FILE = "config/latourometre-stance-hypotheses.yml"
_DEFAULT_GAMMA = 1.0


def score(
    text: str,
    *,
    gamma: float = _DEFAULT_GAMMA,
    use_stance: bool = True,
    softmax_tau: Optional[float] = None,
) -> Dict[str, Any]:
    """Score one French text on Latour's four attractors.

    Args:
        text: the raw French text to place on the two axes.
        gamma: additive stance-blend strength (γ=1.0 is the calibrated default;
            0.0 collapses to the pure cosine SemAxis projection).
        use_stance: when False, skip the NLI stance layer entirely and return the
            pure cosine projection (faster — no NLI model load).
        softmax_tau: optional override of the per-chunk softmax temperature.

    Returns:
        A dict mirroring the production ``latourometre`` metric block, plus a
        convenience ``dominant_pole`` key (the argmax of ``scores``).
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
        stance = LatourStanceMetric(
            {"hypotheses_file": _HYPOTHESES_FILE}
        ).compute(ctx)
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
