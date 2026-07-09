"""Minimal, dependency-free analysis primitives for the Latouromètre.

Scoring one text needs only two small primitives, and this module defines both
with no server runtime, database client, or orchestrator behind them:

* :class:`AbstractMetric` — a metric is an object with ``options`` and a
  ``compute(ctx)`` method returning a JSON-serialisable dict.
* :class:`AnalysisContext` — the per-run bag the metrics read: the segmented
  spaCy docs, lazy ``embedder`` / ``nli`` loaders, the source language, a
  ``project_root`` to resolve the packaged YAML config, and ``metrics_so_far``
  so the Latouromètre can blend the stance metric that ran before it.

Keeping this layer free of Redis, Postgres, and any analyzer registry is what
makes the package a self-contained library; ``tests/test_decoupling.py`` guards
that property.

``_segment_answer`` splits an input text into the same analysis segments the
scorer was calibrated on, so ``score(text)`` and the calibration figures agree
within float tolerance.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple


@dataclass
class AnalysisContext:
    """Shared per-run context passed to every metric.

    ``nlp_doc_per_answer`` is the list of spaCy docs the metrics chunk and
    embed. ``embedder`` / ``nli`` are zero-arg callables so a metric that does
    not need a model pays no load cost. ``project_root`` resolves the packaged
    config (``config/latourometre-seeds.yml`` …). ``metrics_so_far`` lets a
    later metric read an earlier one's output (the Latouromètre blends the
    stance scores produced by ``LatourStanceMetric``).
    """

    nlp_doc_per_answer: List[Any]
    embedder: Callable[[], Any]
    project_root: Path
    language: str = "fr"
    nli: Optional[Callable[[], Any]] = None
    metrics_so_far: Dict[str, Any] = field(default_factory=dict)


class AbstractMetric(ABC):
    """One NLP computation. ``compute`` returns a JSON-serializable dict."""

    id: ClassVar[str]

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        self.options = options or {}

    @abstractmethod
    def compute(self, ctx: AnalysisContext) -> Dict[str, Any]:
        raise NotImplementedError


_PARAGRAPH_SEP = re.compile(r"\n\s*\n")
_SINGLE_NEWLINE_SEP = re.compile(r"\n")
_MIN_PARAGRAPHS = 3
_WINDOW_SIZE = 3
_DEGRADED_OCR_MIN_CHARS = 1500


def _segment_answer(nlp, text: str) -> Tuple[List[Any], str]:
    """Break a single answer into analysis segments.

    Paragraphs (≥3 blank-line-separated) → one doc per paragraph.
    Degraded OCR (≥3 single-newline paragraphs on long text without blank lines)
    → one doc per single-newline paragraph.
    Otherwise, >3 sentences → sliding window of 3 sentences, step 1.
    Short answers → returned as-is.

    Returns ``(segments, strategy_label)`` where strategy is one of
    ``"paragraphs" | "single_newline" | "sentence_window" | "short"``.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_SEP.split(text) if p.strip()]
    if len(paragraphs) >= _MIN_PARAGRAPHS:
        return [nlp(p) for p in paragraphs], "paragraphs"

    if len(paragraphs) < _MIN_PARAGRAPHS and len(text) > _DEGRADED_OCR_MIN_CHARS:
        single_nl = [p.strip() for p in _SINGLE_NEWLINE_SEP.split(text) if p.strip()]
        if len(single_nl) >= _MIN_PARAGRAPHS:
            return [nlp(p) for p in single_nl], "single_newline"

    doc = nlp(text)
    sents = list(doc.sents)
    if len(sents) <= _WINDOW_SIZE:
        return [doc], "short"

    segments = []
    for i in range(len(sents) - _WINDOW_SIZE + 1):
        window = doc[sents[i].start : sents[i + _WINDOW_SIZE - 1].end]
        segments.append(window.as_doc())
    return segments, "sentence_window"
