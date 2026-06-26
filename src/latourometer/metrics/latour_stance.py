"""Latour stance: zero-shot NLI scoring of each chunk against pro / contra
hypotheses per latourian pole.

The cosine-only Latouromètre cannot tell apart "this text endorses pole P"
from "this text uses P's lexicon to attack P" — distributional similarity
is blind to stance. Phase D adds an entailment-based scorer that measures,
for each pole, *how well does the text entail this pro/contra claim?*

Architecture (cf. building_latourometre.md, Plan C1):

  per chunk c:
    nli(c, candidate_labels=[all hypotheses], multi_label=True)
      → {label: P(entail) ∈ [0, 1]}

  per chunk c, per pole P:
    chunk_pro[c, P]   = max over P's pro hypotheses of P(entail at c)
    chunk_contra[c, P] = max over P's contra hypotheses of P(entail at c)

  per pole P:
    stance[P] = mean_c(chunk_pro[c, P]) − mean_c(chunk_contra[c, P])
    clamp to [-1, +1]

The choice of ``max`` (not ``mean``) over hypotheses is deliberate: with
``multi_label=True``, each pro/contra hypothesis is an independent binary
entailment claim that the chunk could plausibly entail. Averaging dilutes
amplitude proportional to the hypothesis count, masking strong stance.
``max`` answers the right question: "did the chunk anywhere express *one*
of these stance claims strongly?" — which is what stance detection wants.
``mean`` is kept across chunks so text length normalises naturally.

The output is consumed downstream by ``LatourometreMetric`` to blend with
the cosine projection (``α·cosine + (1−α)·((stance+1)/2)``).

Graceful by design: any failure path returns a zero-filled payload with
an ``error`` field. The metric never raises, so the pipeline degrades to
pure cosine when NLI is unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..base import AbstractMetric, AnalysisContext
from .thematic_radar import _chunk_doc

logger = logging.getLogger(__name__)


def _load_hypotheses(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _empty_payload(
    poles_cfg: Dict[str, Any],
    error: str,
    *,
    hypotheses_file: str = "",
) -> Dict[str, Any]:
    pole_keys = list(poles_cfg.keys())
    return {
        "scores": {k: 0.0 for k in pole_keys},
        "labels": {k: poles_cfg[k].get("label", k) for k in pole_keys},
        "pole_order": pole_keys,
        "method": "nli_zero_shot",
        "model": None,
        "hypotheses_file": hypotheses_file,
        "details": {},
        "n_chunks": 0,
        "error": error,
    }


class LatourStanceMetric(AbstractMetric):
    id = "latour_stance"

    def compute(self, ctx: AnalysisContext) -> Dict[str, Any]:
        rel = self.options.get(
            "hypotheses_file", "config/latourometre-stance-hypotheses.yml"
        )
        hyp_path = ctx.project_root / rel

        # Load hypotheses first so error payloads can echo pole keys/labels.
        if not hyp_path.exists():
            logger.warning("Stance hypotheses file missing: %s", hyp_path)
            return _empty_payload({}, error="hypotheses_missing", hypotheses_file=rel)

        cfg = _load_hypotheses(hyp_path)
        poles_cfg = cfg.get("poles", {}) or {}
        if not poles_cfg:
            return _empty_payload({}, error="hypotheses_empty", hypotheses_file=rel)

        # FR only in v1 — see building_latourometre.md "Plan C1" decision.
        if ctx.language != "fr":
            return _empty_payload(
                poles_cfg, error="language_not_supported", hypotheses_file=rel
            )

        if ctx.nli is None:
            return _empty_payload(
                poles_cfg, error="nli_model_unavailable", hypotheses_file=rel
            )

        nli = ctx.nli()
        if nli is None:
            return _empty_payload(
                poles_cfg, error="nli_model_unavailable", hypotheses_file=rel
            )

        answer_docs = ctx.nlp_doc_per_answer or []
        all_chunks: List[str] = []
        for doc in answer_docs:
            all_chunks.extend(_chunk_doc(doc))
        all_chunks = [c for c in all_chunks if c.strip()]
        if not all_chunks:
            return _empty_payload(
                poles_cfg, error="empty_text", hypotheses_file=rel
            )

        # Build the global candidate label list once. Tag each label with
        # its pole + polarity so we can route scores back without relying on
        # ordering (HF reorders labels by score in the response).
        pole_keys = list(poles_cfg.keys())
        labels = {k: poles_cfg[k].get("label", k) for k in pole_keys}
        all_candidate_labels: List[str] = []
        # Map from candidate-label string → (pole_key, "pro"|"contra", local_idx)
        label_index: Dict[str, tuple] = {}
        hypotheses_by_pole: Dict[str, Dict[str, List[str]]] = {}
        for pole_key, pole_cfg in poles_cfg.items():
            pro = list(pole_cfg.get("hypotheses_pro", []) or [])
            con = list(pole_cfg.get("hypotheses_contra", []) or [])
            hypotheses_by_pole[pole_key] = {"pro": pro, "contra": con}
            for i, h in enumerate(pro):
                all_candidate_labels.append(h)
                label_index[h] = (pole_key, "pro", i)
            for i, h in enumerate(con):
                all_candidate_labels.append(h)
                label_index[h] = (pole_key, "contra", i)

        if not all_candidate_labels:
            return _empty_payload(
                poles_cfg, error="hypotheses_empty", hypotheses_file=rel
            )

        # Per-chunk × per-pole "best stance signal in either direction".
        # chunk_pro_per_pole[pole][c]   = max P(entail) across pro hypotheses
        # chunk_contra_per_pole[pole][c] = max P(entail) across contra hypotheses
        chunk_pro_per_pole: Dict[str, List[float]] = {k: [] for k in pole_keys}
        chunk_contra_per_pole: Dict[str, List[float]] = {k: [] for k in pole_keys}
        # Also keep the per-hypothesis mean across chunks for the audit trail
        # in metrics.json (so we can see WHICH hypothesis fired, not just the
        # aggregate stance).
        per_hyp_chunk_scores: Dict[str, Dict[str, List[List[float]]]] = {
            k: {
                "pro": [[] for _ in hypotheses_by_pole[k]["pro"]],
                "contra": [[] for _ in hypotheses_by_pole[k]["contra"]],
            }
            for k in pole_keys
        }

        for chunk in all_chunks:
            try:
                result = nli(
                    chunk,
                    candidate_labels=all_candidate_labels,
                    multi_label=True,
                    truncation=True,
                )
            except Exception:  # noqa: BLE001
                logger.exception("NLI inference failed on chunk; skipping")
                continue
            # HF returns {"sequence", "labels": [...], "scores": [...]} with
            # labels sorted by score descending. Build a label→score map.
            returned_labels = result.get("labels", [])
            returned_scores = result.get("scores", [])
            label_to_score: Dict[str, float] = {}
            for lbl, score in zip(returned_labels, returned_scores):
                if lbl in label_index:
                    label_to_score[lbl] = float(score)

            # Per-pole max over each polarity's hypotheses for this chunk.
            for pole_key in pole_keys:
                pro_hyps = hypotheses_by_pole[pole_key]["pro"]
                con_hyps = hypotheses_by_pole[pole_key]["contra"]
                pro_scores_this_chunk = [
                    label_to_score.get(h, 0.0) for h in pro_hyps
                ]
                con_scores_this_chunk = [
                    label_to_score.get(h, 0.0) for h in con_hyps
                ]
                chunk_pro_per_pole[pole_key].append(
                    max(pro_scores_this_chunk) if pro_scores_this_chunk else 0.0
                )
                chunk_contra_per_pole[pole_key].append(
                    max(con_scores_this_chunk) if con_scores_this_chunk else 0.0
                )
                # Audit trail: per-hypothesis per-chunk score (for forensics).
                for i, s in enumerate(pro_scores_this_chunk):
                    per_hyp_chunk_scores[pole_key]["pro"][i].append(s)
                for i, s in enumerate(con_scores_this_chunk):
                    per_hyp_chunk_scores[pole_key]["contra"][i].append(s)

        # Aggregate per pole: stance = mean_chunks(chunk_pro) − mean_chunks(chunk_contra).
        scores: Dict[str, float] = {}
        details: Dict[str, Any] = {}
        for pole_key in pole_keys:
            pro_per_chunk = chunk_pro_per_pole[pole_key]
            con_per_chunk = chunk_contra_per_pole[pole_key]
            pro_mean = (
                sum(pro_per_chunk) / len(pro_per_chunk) if pro_per_chunk else 0.0
            )
            con_mean = (
                sum(con_per_chunk) / len(con_per_chunk) if con_per_chunk else 0.0
            )
            stance = max(-1.0, min(1.0, pro_mean - con_mean))
            scores[pole_key] = stance
            details[pole_key] = {
                "pro_mean": pro_mean,
                "contra_mean": con_mean,
                "pro_per_hypothesis": [
                    (sum(s) / len(s)) if s else 0.0
                    for s in per_hyp_chunk_scores[pole_key]["pro"]
                ],
                "contra_per_hypothesis": [
                    (sum(s) / len(s)) if s else 0.0
                    for s in per_hyp_chunk_scores[pole_key]["contra"]
                ],
            }

        return {
            "scores": scores,
            "labels": labels,
            "pole_order": pole_keys,
            "method": "nli_zero_shot",
            "model": cfg.get("model") or "cmarkea/distilcamembert-base-nli",
            "hypotheses_file": rel,
            "hypotheses": hypotheses_by_pole,
            "details": details,
            "n_chunks": len(all_chunks),
        }
