"""Latouromètre: semantic axis projection against Latour's four attractors.

Operationalises the geometry of Bruno Latour, *Où atterrir?* (2017) — four
attractors arranged on two perpendicular axes:

    Terrestre (top)
        |
Local ----+---- Global
        |
    Hors-Sol (bottom)

Each pole is defined by seed phrases; the metric mean-pools seed embeddings
into a pole vector, then scores each text chunk by summed positive cosine
similarity, yielding a normalised distribution across the four poles.

Methodologically identical to ``ThematicRadarMetric``; the distinction is
semantic (poles vs. themes) and a dedicated chart that imposes the Figure 4
spatial layout.

Optional Phase D blend: when ``stance_blend_alpha`` is set in the metric
options AND a sibling ``LatourStanceMetric`` has run earlier in the same
analyzer pass (its result reachable via ``ctx.metrics_so_far``), the
emitted ``scores`` are a blend of the cosine projection and the NLI
stance score, in the form ``α·cosine + (1−α)·((stance+1)/2)``. This lets
the chart and the synthesizer consume a single ``scores`` field while the
raw ``cosine_scores`` and ``stance_scores`` remain in ``metrics.json``
for forensic audit.

Phase E sharper blend: when ``stance_blend_gamma`` is set (takes
precedence over ``stance_blend_alpha``), the blend uses an additive
correction without a uniform constant, then clips negatives and
renormalises to sum to 1: ``blended[P] = max(0, cos[P] + γ·stance[P]) /
Σ_P max(0, …)``. This preserves the cosine sharpness on neutral-stance
texts (γ·0 = 0 → cosine pass-through) and yields decisive flips on
stance inversions. See ``docs/building_latourometre.md`` § Phase E.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..base import AbstractMetric, AnalysisContext
from .thematic_radar import (
    DEFAULT_SOFTMAX_TAU,
    _chunk_doc,
    _cosine,
    _softmax_aggregate,
)


def _load_seeds(seeds_path: Path) -> Dict[str, Any]:
    with seeds_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _blend(
    cosine_scores: Dict[str, float],
    stance_scores: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    """``α·cosine + (1−α)·((stance+1)/2)`` per pole.

    The stance term is renormalised from ``[-1, +1]`` to ``[0, 1]`` so the
    blended value lives on the same scale as the cosine softmax output
    (``[0, 1]``, summing to 1 across poles before blend). After blend the
    sum may drift slightly from 1; the chart and downstream consumers
    treat blended scores as relative magnitudes, not as a probability —
    the dominant-pole comparison is unaffected.

    Note: this formula adds a uniform ``(1−α)/2`` constant to every pole
    (because neutral stance ≈ 0 maps to ``(0+1)/2 = 0.5``), which floors
    every non-dominant pole and flattens the chart. Phase E ships an
    alternative ``_blend_gamma`` that fixes this — pick that one via
    ``stance_blend_gamma`` in the metric options.
    """
    blended: Dict[str, float] = {}
    for key, cos in cosine_scores.items():
        st = stance_scores.get(key, 0.0)
        st_norm = (st + 1.0) / 2.0
        blended[key] = alpha * cos + (1.0 - alpha) * st_norm
    return blended


def _blend_gamma(
    cosine_scores: Dict[str, float],
    stance_scores: Dict[str, float],
    gamma: float,
) -> Dict[str, float]:
    """``max(0, cos[P] + γ·stance[P])`` per pole, then renormalise to sum to 1.

    Phase E formula. Unlike ``_blend``, this preserves the cosine
    distribution shape when stance is uninformative (``stance[P] ≈ 0``
    for every P → blended ∝ cosine, peakiness intact). When stance is
    strongly negative on pole P, ``cos[P] + γ·stance[P]`` can go below
    zero and gets clipped, effectively suppressing P — which is what we
    want on stance-inversion texts (Bruckner anti-Greta, Butré anti-
    éolien). γ = 1.0 is the calibrated default; see PRD and the offline
    sweep summarised in ``docs/building_latourometre.md`` § Phase E.
    """
    raw: Dict[str, float] = {}
    for key, cos in cosine_scores.items():
        st = stance_scores.get(key, 0.0)
        raw[key] = max(0.0, cos + gamma * st)
    total = sum(raw.values())
    if total <= 0:
        # Degenerate case: every pole clipped to 0 (extremely strong contra
        # stance on every pole). Fall back to uniform so the chart still
        # renders something rather than dividing by zero. With realistic
        # NLI scores this branch should never trigger.
        n = len(raw)
        return {key: 1.0 / n for key in raw}
    return {key: v / total for key, v in raw.items()}


class LatourometreMetric(AbstractMetric):
    id = "latourometre"

    def compute(self, ctx: AnalysisContext) -> Dict[str, Any]:
        import numpy as np  # type: ignore

        rel = self.options.get("seeds_file", "config/latourometre-seeds.yml")
        seeds_path = ctx.project_root / rel
        seeds_cfg = _load_seeds(seeds_path)
        poles = seeds_cfg.get("poles", {})
        labels = {key: meta.get("label", key) for key, meta in poles.items()}

        answer_docs = ctx.nlp_doc_per_answer
        if not poles or not answer_docs:
            empty = {key: 0.0 for key in poles}
            return self._maybe_blend(empty, ctx, labels, list(poles.keys()))

        embedder = ctx.embedder()

        pole_keys = list(poles.keys())
        pole_vectors = []
        for key in pole_keys:
            seeds: List[str] = poles[key].get("seeds", []) or [labels[key]]
            seed_embs = embedder.encode(seeds)
            pole_vectors.append(np.mean(np.asarray(seed_embs), axis=0))

        all_chunks: List[str] = []
        for doc in answer_docs:
            all_chunks.extend(_chunk_doc(doc))

        if not all_chunks:
            empty = {key: 0.0 for key in pole_keys}
            return self._maybe_blend(empty, ctx, labels, pole_keys)

        chunk_embs = embedder.encode(all_chunks)
        if not isinstance(chunk_embs, np.ndarray):
            chunk_embs = np.asarray(chunk_embs)

        tau = float(self.options.get("softmax_tau", DEFAULT_SOFTMAX_TAU))
        cosine_scores = _softmax_aggregate(
            chunk_embs, pole_vectors, pole_keys, tau
        )

        return self._maybe_blend(cosine_scores, ctx, labels, pole_keys)

    def _maybe_blend(
        self,
        cosine_scores: Dict[str, float],
        ctx: AnalysisContext,
        labels: Dict[str, str],
        pole_keys: List[str],
    ) -> Dict[str, Any]:
        """Build the result dict, blending in the stance metric when available.

        When neither ``stance_blend_gamma`` nor ``stance_blend_alpha`` is
        set, or when ``LatourStanceMetric`` did not run / errored, falls
        back to pure cosine: ``scores`` is the cosine projection and no
        ``stance_scores`` key is emitted.

        When both are set, ``stance_blend_gamma`` (Phase E) takes
        precedence. This lets a config opt into the sharper formula
        without removing the legacy α key.
        """
        result: Dict[str, Any] = {
            "scores": cosine_scores,
            "labels": labels,
            "pole_order": pole_keys,
        }

        gamma_opt = self.options.get("stance_blend_gamma")
        alpha_opt = self.options.get("stance_blend_alpha")
        stance_metric_id = self.options.get("stance_metric_id", "latour_stance")
        stance_payload: Optional[Dict[str, Any]] = (
            ctx.metrics_so_far.get(stance_metric_id) if ctx.metrics_so_far else None
        )
        stance_scores = (
            stance_payload.get("scores") if isinstance(stance_payload, dict) else None
        )
        stance_error = (
            stance_payload.get("error") if isinstance(stance_payload, dict) else None
        )

        if (gamma_opt is None and alpha_opt is None) or stance_scores is None or stance_error:
            # Pure-cosine path. Surface the cosine values under the canonical
            # `scores` key so the chart consumes them transparently.
            return result

        if gamma_opt is not None:
            # Phase E sharper blend (recommended). Gamma is unbounded in
            # principle (γ·stance_max ~ γ for stance ∈ [-1, +1]), but
            # values much beyond 1.5 produce degenerate clip-everything
            # behaviour on inversion texts. Calibrated default: 1.0.
            gamma = max(0.0, float(gamma_opt))
            blended = _blend_gamma(cosine_scores, stance_scores, gamma)
            result["scores"] = blended
            result["cosine_scores"] = cosine_scores
            result["stance_scores"] = dict(stance_scores)
            result["blend_gamma"] = gamma
            return result

        # Legacy Phase D blend, kept for backwards-compat.
        alpha = float(alpha_opt)
        alpha = max(0.0, min(1.0, alpha))
        blended = _blend(cosine_scores, stance_scores, alpha)

        result["scores"] = blended
        result["cosine_scores"] = cosine_scores
        result["stance_scores"] = dict(stance_scores)
        result["blend_alpha"] = alpha
        return result
