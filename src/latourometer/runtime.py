"""Lazy singletons for the heavy NLP dependencies (spaCy + embedder + NLI).

Standalone, French-only port of the AFK ``words-weight`` runtime. It loads the
exact same two models the production Latouromètre uses, so scores reproduce:

- embedder : ``dangvantuan/sentence-camembert-large`` (frozen by PRD-4 — no swap)
- NLI head : ``cmarkea/distilcamembert-base-nli`` (zero-shot stance)
- spaCy    : ``fr_core_news_lg`` (segmentation + sentence boundaries)

An STS-oriented French embedder keeps the cosine spread wide enough for the
SemAxis projection; multilingual retrieval embedders (E5 / BGE) collapse cosines
into a narrow band and kill the discriminance the metric relies on.

This module is Redis/Postgres-free and imports nothing from the AFK monorepo —
that is the whole point of the standalone extraction. The two ``AFK_*_DEVICE``
env gates are carried over unchanged: they default to CPU (safe everywhere) and
opt a model onto CUDA when a GPU is free, which is the wall-time bottleneck for
batch calibration.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SPACY_MODEL = "fr_core_news_lg"
EMBEDDER_MODEL = "dangvantuan/sentence-camembert-large"
NLI_MODEL = "cmarkea/distilcamembert-base-nli"


def project_root() -> Path:
    """Package root — resolves the packaged ``config/*.yml`` seed/hypothesis files."""
    # src/latourometer/runtime.py -> src/latourometer/
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def get_spacy_nlp() -> Any:
    import spacy  # type: ignore

    logger.info("Loading spaCy model: %s", SPACY_MODEL)
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:  # pragma: no cover - first-run guidance
        raise OSError(
            f"spaCy model {SPACY_MODEL!r} is not installed. Install it with:\n"
            f"    python -m spacy download {SPACY_MODEL}"
        ) from exc


def _resolve_embedder_device() -> str:
    """Device string for the sentence embedder. Default ``"cpu"``.

    ``AFK_EMBEDDER_DEVICE`` opts the embedder onto a CUDA device (e.g. ``"0"`` or
    ``"cuda:0"``) for offline batch calibration, where the GPU is free and the
    CPU embedder is the wall-time bottleneck. Falls back to CPU with a warning if
    CUDA is requested but absent, so a stale env var never crashes a run.
    """
    raw = os.environ.get("AFK_EMBEDDER_DEVICE")
    if raw is None or raw.strip() in ("", "-1", "cpu"):
        return "cpu"
    dev = raw.strip()
    if dev.isdigit():
        dev = f"cuda:{dev}"
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            logger.warning(
                "AFK_EMBEDDER_DEVICE=%r requested but CUDA unavailable — using CPU",
                raw,
            )
            return "cpu"
    except Exception:  # noqa: BLE001 — torch missing/broken → safe CPU fallback
        return "cpu"
    logger.info("AFK_EMBEDDER_DEVICE override active: embedder on %s", dev)
    return dev


@lru_cache(maxsize=1)
def get_embedder() -> Any:
    from sentence_transformers import SentenceTransformer  # type: ignore

    device = _resolve_embedder_device()
    logger.info("Loading embedder: %s on %s", EMBEDDER_MODEL, device)
    model = SentenceTransformer(EMBEDDER_MODEL, device=device)
    # The FR CamemBERT models ship no ST wrapper, so max_seq_length defaults to
    # the raw 514 of max_position_embeddings while the tokenizer reports an
    # unbounded model_max_length — long inputs are NOT truncated and overflow
    # CamemBERT's RoBERTa-style position table (valid token budget is 512, not
    # 514) with "IndexError: index out of range in self". Cap at 512 so
    # sentence-transformers truncates instead of crashing.
    if getattr(model, "max_seq_length", None):
        model.max_seq_length = min(model.max_seq_length, 512)
    return model


def _resolve_nli_device() -> int:
    """Device for the NLI pipeline. Default ``-1`` (CPU).

    ``AFK_NLI_DEVICE`` opts the zero-shot stance head onto a CUDA device (e.g.
    ``"0"``) where NLI inference dominates CPU wall time. Falls back to CPU with
    a warning if CUDA is requested but absent.
    """
    raw = os.environ.get("AFK_NLI_DEVICE")
    if raw is None or raw.strip() in ("", "-1"):
        return -1
    try:
        dev = int(raw)
    except ValueError:
        logger.warning("AFK_NLI_DEVICE=%r is not an int — falling back to CPU", raw)
        return -1
    if dev < 0:
        return -1
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            logger.warning(
                "AFK_NLI_DEVICE=%d requested but CUDA unavailable — using CPU", dev
            )
            return -1
    except Exception:  # noqa: BLE001 — torch missing/broken → safe CPU fallback
        return -1
    logger.info("AFK_NLI_DEVICE override active: NLI on cuda:%d", dev)
    return dev


@lru_cache(maxsize=1)
def get_nli_pipeline() -> Any:
    from transformers import pipeline  # type: ignore

    logger.info("Loading NLI model: %s", NLI_MODEL)
    return pipeline(
        task="zero-shot-classification",
        model=NLI_MODEL,
        tokenizer=NLI_MODEL,
        device=_resolve_nli_device(),
    )
