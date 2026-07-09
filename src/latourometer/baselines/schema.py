"""Minimal schema for the Wordscores lexicon rows.

Only :class:`LexiconRow` is needed by the calibration baseline — a single row of
the emitted ``lexicon.csv`` / ``lexicon.parquet``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class LexiconRow(BaseModel):
    """A single Wordscores lexicon entry (one row of lexicon.csv / lexicon.parquet).

    Produced by ``latourometer.baselines.calibrate`` on one of Latour's polar
    axes (Hors-Sol <-> Terrestre, Local <-> Global). Contains only lemmas + POS +
    statistics -- no source prose -- so the published artifact is a clean
    non-consumptive derivative.

    ``freq_minus`` / ``freq_plus`` are the raw counts in the -1 / +1 reference
    pole; the *names* of those poles are axis-specific and are written into the
    CSV / Parquet column headers at output time (``freq_hors_sol`` / ``freq_local``
    etc.), so each published lexicon self-documents its own poles.
    """

    word: str  # surface lemma (lowercased)
    pos: str  # NOUN | VERB | ADJ | PROPN
    score: float  # Wordscores, in [-1, +1]; -1 = minus pole, +1 = plus pole
    freq_minus: int  # raw count in the -1 pole (hors_sol / local)
    freq_plus: int  # raw count in the +1 pole (terrestre / global)
    n_docs: int  # docs the word appears in (both poles combined)
    ci_low: Optional[float] = None  # bootstrap CI (Lowe 2008); None if n_docs < 3
    ci_high: Optional[float] = None
