"""Thematic radar: cosine similarity of each answer against seed-defined
themes, aggregated and normalized to a probability distribution.

Long answers (e.g. a full speech as a single block) are chunked at sentence
boundaries before embedding so we stay under the embedder's 512-token
position limit and so that thematic mass reflects the whole answer rather
than just the truncated first paragraph.

The French embedder (CamemBERT-large) is STS-oriented, producing a wide cosine
spread across thematically-distinct chunks — what we need for semantic axis
projection. Seeds must be written in the target language.

Aggregation strategy: per-chunk softmax with temperature, then mean over
chunks. A plain sum of cosines biases toward the theme with the highest
baseline similarity rather than the winner per chunk. Softmax-per-chunk
produces a winner-take-all signal that averages cleanly across the text.

This module is the shared geometry backbone: ``LatouromètreMetric`` and
``LatourStanceMetric`` import ``_cosine`` / ``_softmax_aggregate`` / ``_chunk_doc``
from here so the two metrics chunk and project text identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..base import AbstractMetric, AnalysisContext

# Approximate token budget per chunk; CamemBERT max position is 512, we
# leave headroom for special tokens and underestimate-vs-tokenizer slack.
CHUNK_TOKEN_TARGET = 256

# Softmax temperature on per-chunk cosines. Lower tau = more winner-take-
# all. Default calibrated against CamemBERT-large (spread [0.3, 0.9]) on
# French political speech. Overridable per-pipeline in the YAML config.
DEFAULT_SOFTMAX_TAU = 0.05


def _cosine(a, b) -> float:
    import numpy as np  # type: ignore

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _softmax_aggregate(
    chunk_embs,
    theme_vectors,
    theme_keys,
    tau: float,
) -> Dict[str, float]:
    """Per-chunk softmax over themes at temperature ``tau``, averaged across
    chunks. Returns a dict ``{theme_key: probability}`` that sums to 1.

    Handles the "narrow high cosine band" pathology: cosines in [0.75, 0.90]
    sum to a near-uniform distribution under plain averaging, so we softmax
    within each chunk first, which amplifies the leader before we average
    across chunks.
    """
    import numpy as np  # type: ignore

    if len(chunk_embs) == 0 or len(theme_vectors) == 0:
        return {key: 0.0 for key in theme_keys}

    tv = np.asarray(theme_vectors)  # (T, D)
    ce = np.asarray(chunk_embs)  # (C, D)
    tv_norms = np.linalg.norm(tv, axis=1)
    ce_norms = np.linalg.norm(ce, axis=1)
    safe_tv = np.where(tv_norms == 0, 1.0, tv_norms)
    safe_ce = np.where(ce_norms == 0, 1.0, ce_norms)
    # sims: (C, T)
    sims = (ce @ tv.T) / (safe_ce[:, None] * safe_tv[None, :])

    # Per-chunk softmax (row-wise) with numerical-stability offset.
    z = sims / max(tau, 1e-6)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)  # (C, T)

    mean_probs = probs.mean(axis=0)  # (T,)
    total = float(mean_probs.sum())
    if total == 0:
        return {key: 0.0 for key in theme_keys}
    return {key: float(mean_probs[i] / total) for i, key in enumerate(theme_keys)}


def _load_seeds(seeds_path: Path) -> Dict[str, Any]:
    with seeds_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _chunk_doc(doc, target_tokens: int = CHUNK_TOKEN_TARGET) -> List[str]:
    """Pack sentences from a spaCy doc into chunks under ``target_tokens`` words.

    Falls back to a single whitespace split when ``doc.sents`` is empty (e.g.
    truly short text). Each emitted chunk is at most ``target_tokens`` words —
    well under the camembert 512-position limit.
    """
    sentences = list(getattr(doc, "sents", []))
    if not sentences:
        words = (getattr(doc, "text", "") or "").split()
        if not words:
            return []
        return [
            " ".join(words[i : i + target_tokens])
            for i in range(0, len(words), target_tokens)
        ]

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for sent in sentences:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        sent_len = len(sent_text.split())
        if buf and buf_len + sent_len > target_tokens:
            chunks.append(" ".join(buf))
            buf = [sent_text]
            buf_len = sent_len
        else:
            buf.append(sent_text)
            buf_len += sent_len
    if buf:
        chunks.append(" ".join(buf))
    return chunks


class ThematicRadarMetric(AbstractMetric):
    id = "thematic_radar"

    def compute(self, ctx: AnalysisContext) -> Dict[str, Any]:
        import numpy as np  # type: ignore

        rel = self.options.get("seeds_file", "config/seeds.yml")
        seeds_path = ctx.project_root / rel
        seeds_cfg = _load_seeds(seeds_path)
        themes = seeds_cfg.get("themes", {})
        labels = {key: meta.get("label", key) for key, meta in themes.items()}

        answer_docs = ctx.nlp_doc_per_answer
        if not themes or not answer_docs:
            return {
                "scores": {key: 0.0 for key in themes},
                "labels": labels,
            }

        embedder = ctx.embedder()

        # Mean-pool each theme's seed phrases.
        theme_keys = list(themes.keys())
        theme_vectors = []
        for key in theme_keys:
            seeds: List[str] = themes[key].get("seeds", []) or [labels[key]]
            seed_embs = embedder.encode(seeds)
            theme_vectors.append(np.mean(np.asarray(seed_embs), axis=0))

        # Flatten all answers into chunks, embed in one batch.
        all_chunks: List[str] = []
        for doc in answer_docs:
            all_chunks.extend(_chunk_doc(doc))

        if not all_chunks:
            return {
                "scores": {key: 0.0 for key in theme_keys},
                "labels": labels,
            }

        chunk_embs = embedder.encode(all_chunks)
        if not isinstance(chunk_embs, np.ndarray):
            chunk_embs = np.asarray(chunk_embs)

        tau = float(self.options.get("softmax_tau", DEFAULT_SOFTMAX_TAU))
        scores = _softmax_aggregate(chunk_embs, theme_vectors, theme_keys, tau)

        return {"scores": scores, "labels": labels}
