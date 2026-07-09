"""Wordscores calibration of Latour's Hors-Sol <-> Terrestre lexicon.

Runs Wordscores (Laver, Benoit & Garry 2003) over the finalized 39-text
Latouromètre calibration corpus (``corpus_latourometre/``) and writes a
``word -> score`` lexicon on Latour's politically-active axis:

    S(w) = Σ_pole P(pole | w) · ref_score(pole)
         with ref_score(hors_sol) = -1, ref_score(terrestre) = +1

This is a CPU-only lexical baseline -- counting + spaCy lemmatization, no
embeddings, no NLI. The corpus on-disk format is owned by ``_corpus_loader``;
this module only counts and scores.

Usage:

    python -m latourometer.baselines.calibrate --axis hors-sol-terrestre \\
        [--dry-run] [--min-docs-per-pole 5] [--bootstrap-n 500]

Point ``CORPUS_BASE_PATH`` at a corpus checkout (it expects a
``corpus_latourometre/`` subdir). Outputs (under CORPUS_BASE_PATH):

    axes/hors-sol-terrestre/lexicon.csv      -- canonical, human-readable
    axes/hors-sol-terrestre/lexicon.parquet  -- for the Hugging Face viewer

Each score ships with a bootstrap confidence interval (Lowe 2008): documents
are resampled with replacement *within each pole* ``--bootstrap-n`` times
(default 500), ``S(w)`` is recomputed per draw, and ``ci_low`` / ``ci_high`` are
the 2.5 / 97.5 percentiles of the per-word distribution. Words seen in fewer
than 3 documents get null CIs (below the corpus's statistical resolution).
``--bootstrap-n 0`` skips the bootstrap entirely (the fast point-estimate path).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# The single source of truth for the on-disk corpus format AND for the
# audit-flag selection (anti-circularity wiring): seed_pool returns only
# ``audit.use_for_seeds: true`` texts, role_of/ROLE_INVERSION_HOLD identify the
# held-out inversions. No selection logic is duplicated here.
from ._corpus_loader import (
    ROLE_INVERSION_HOLD,
    load_corpus,
    role_of,
    seed_pool,
)
from .schema import LexiconRow

logger = logging.getLogger("latourometer.baselines.calibrate")

# Supported axes -> the ordered (minus_pole, plus_pole) keys that feed the
# calibration. Two of Latour's perpendicular vectors are calibrated independently:
#   - hors-sol-terrestre: the new, politically-active axis (-1 HS / +1 Terrestre)
#   - local-global:       the old modernization front (-1 Local / +1 Global)
SUPPORTED_AXES: Dict[str, Tuple[str, ...]] = {
    "hors-sol-terrestre": ("hors_sol", "terrestre"),
    "local-global": ("local", "global"),
}

# Reference pole scores per axis: the minus pole is -1, the plus pole +1.
AXIS_REF_SCORE: Dict[str, Dict[str, float]] = {
    axis: {poles[0]: -1.0, poles[1]: 1.0} for axis, poles in SUPPORTED_AXES.items()
}

# Default (back-compat) reference map: the politically-active axis.
REF_SCORE: Dict[str, float] = AXIS_REF_SCORE["hors-sol-terrestre"]

# spaCy POS tags kept for the lexicon (content words only).
_KEEP_POS = {"NOUN", "VERB", "ADJ", "PROPN"}

# A token key is the (lemma, POS) pair: the same lemma under two POS tags is two
# lexicon rows, matching LexiconRow carrying a single ``pos`` per ``word``.
TokenKey = Tuple[str, str]

_SPACY_MODEL = "fr_core_news_lg"

# Bootstrap CI parameters (Lowe 2008 document-resampling correction).
_BOOTSTRAP_PCT = (2.5, 97.5)  # 95% percentile CI
_MIN_DOCS_FOR_CI = 3  # below this, the CI is undefined (null)
_BOOTSTRAP_SEED = 12345  # fixed so re-runs on a fixed corpus reproduce the CIs


def _parquet_schema(poles: Tuple[str, ...]) -> pa.Schema:
    """Explicit Parquet schema with axis-specific frequency column names.

    ci_low / ci_high stay float64 even when a row's CI is null (n_docs < 3) --
    pyarrow would otherwise infer a null-typed column from an all-null slice,
    which the HF viewer dislikes. The two freq columns carry the pole names
    (``freq_hors_sol`` / ``freq_local`` …) so each published lexicon self-documents.
    """
    return pa.schema(
        [
            ("word", pa.string()),
            ("pos", pa.string()),
            ("score", pa.float64()),
            (f"freq_{poles[0]}", pa.int64()),
            (f"freq_{poles[1]}", pa.int64()),
            ("n_docs", pa.int64()),
            ("ci_low", pa.float64()),
            ("ci_high", pa.float64()),
        ]
    )


def corpus_base() -> Path:
    return Path(os.environ.get("CORPUS_BASE_PATH", "/data/corpus"))


def corpus_dir() -> Path:
    return corpus_base() / "corpus_latourometre"


def output_dir(axis: str) -> Path:
    return corpus_base() / "axes" / axis


def _load_spacy():
    """Load the French spaCy pipeline used to lemmatize the corpus."""
    import spacy

    return spacy.load(_SPACY_MODEL)


def lemmatize(nlp, text: str) -> List[TokenKey]:
    """Lemmatize, lowercase, keep content POS, drop stopwords/punct/len<2.

    Content-POS lemmatization (no embeddings, no NLI) — the lexical baseline's
    only preprocessing step.
    """
    doc = nlp(text)
    tokens: List[TokenKey] = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        if tok.pos_ not in _KEEP_POS:
            continue
        lemma = tok.lemma_.lower()
        if len(lemma) < 2:
            continue
        tokens.append((lemma, tok.pos_))
    return tokens


def compute_wordscores(
    pole_counts: Dict[str, "Counter[TokenKey]"],
    ref_score: Dict[str, float] | None = None,
) -> Dict[TokenKey, float]:
    """Wordscores point estimate per token key, in [-1, +1].

    ``S(w) = Σ_pole P(pole | w) · ref_score(pole)`` where ``P(pole | w)`` is
    derived from per-pole *relative* frequencies (so unequal pole sizes do not
    bias the score). Pure function of count dictionaries -- no spaCy, no corpus
    -- so the bootstrap can call it once per resample. ``ref_score`` defaults to
    the Hors-Sol <-> Terrestre map (back-compat); pass the axis map for others.
    """
    ref_score = ref_score or REF_SCORE
    totals = {pole: sum(counts.values()) for pole, counts in pole_counts.items()}

    vocab: set[TokenKey] = set()
    for counts in pole_counts.values():
        vocab |= set(counts)

    scores: Dict[TokenKey, float] = {}
    for key in vocab:
        rel = {
            pole: (counts.get(key, 0) / totals[pole] if totals[pole] else 0.0)
            for pole, counts in pole_counts.items()
        }
        denom = sum(rel.values())
        if denom == 0:
            continue
        scores[key] = sum((rel[pole] / denom) * ref_score[pole] for pole in pole_counts)
    return scores


def bootstrap_cis(
    doc_counts: List[dict],
    poles: Tuple[str, ...],
    doc_freq: "Counter[TokenKey]",
    n_resamples: int,
    seed: int = _BOOTSTRAP_SEED,
    ref_score: Dict[str, float] | None = None,
) -> Dict[TokenKey, Tuple[float, float]]:
    """Lowe (2008) document-resampling bootstrap CIs per token key.

    Resamples documents *within each pole* with replacement ``n_resamples``
    times, rebuilds the per-pole counts from the cached per-document counts, and
    recomputes ``S(w)`` each draw. Returns ``{key: (ci_low, ci_high)}`` at the
    95% percentile. Keys with ``n_docs < 3`` are omitted (the caller emits null);
    documents are the resampling unit, per Lowe's correction, not tokens.

    A word absent from a given resample has no score that draw (its relative
    frequency is 0 in both poles) and simply contributes no sample -- the CI is
    taken over the draws in which the word appears.
    """
    docs_by_pole: Dict[str, List["Counter[TokenKey]"]] = {
        pole: [d["counts"] for d in doc_counts if d["pole"] == pole] for pole in poles
    }
    rng = np.random.default_rng(seed)

    samples: Dict[TokenKey, List[float]] = defaultdict(list)
    for _ in range(n_resamples):
        pole_counts: Dict[str, "Counter[TokenKey]"] = {}
        for pole in poles:
            docs = docs_by_pole[pole]
            agg: "Counter[TokenKey]" = Counter()
            if docs:
                for i in rng.integers(0, len(docs), size=len(docs)):
                    agg.update(docs[i])
            pole_counts[pole] = agg
        for key, score in compute_wordscores(pole_counts, ref_score).items():
            samples[key].append(score)

    cis: Dict[TokenKey, Tuple[float, float]] = {}
    for key, values in samples.items():
        if doc_freq[key] < _MIN_DOCS_FOR_CI or not values:
            continue
        lo, hi = np.percentile(values, _BOOTSTRAP_PCT)
        cis[key] = (float(lo), float(hi))
    return cis


def load_axis_corpus(axis: str) -> Tuple[List[dict], List[str]]:
    """Load the seed basin for ``axis``, plus the held-out inversion slugs.

    Selection is driven entirely by the audit frontmatter via the shared
    ``_corpus_loader`` helpers (no ``sub_pole``/``attracteur`` logic here):

    * ``kept``     = ``seed_pool(corpus_dir())`` (only ``audit.use_for_seeds:
      true``) filtered to this axis's two poles. ``seed_pool`` already drops
      every INVERSION-HOLD text and the whole ``tests/`` dir, so the seed basin
      cannot be poisoned.
    * ``excluded`` = the on-axis ``INVERSION-HOLD`` slugs (``role_of(e) ==
      ROLE_INVERSION_HOLD``), reported for transparency and kept aside for the
      comparison stress test. This is what fixes the Bruckner leak: he carries
      ``sub_pole: null`` but ``audit.role: INVERSION-HOLD`` /
      ``use_for_seeds: false``, so the audit-flag filter excludes him where the
      old ``sub_pole`` filter missed him.
    """
    poles = set(SUPPORTED_AXES[axis])

    kept = [e for e in seed_pool(corpus_dir()) if e["expected_pole"] in poles]
    excluded = [
        e["slug"]
        for e in load_corpus(corpus_dir())
        if role_of(e) == ROLE_INVERSION_HOLD and e["expected_pole"] in poles
    ]
    return kept, excluded


def lemmatize_corpus(nlp, entries: List[dict]) -> List[dict]:
    """Lemmatize each entry once, caching per-document token counts.

    Returns a list of ``{slug, pole, counts}`` -- the cache the bootstrap
    resamples over without re-running spaCy.
    """
    doc_counts: List[dict] = []
    for entry in entries:
        counts: "Counter[TokenKey]" = Counter(lemmatize(nlp, entry["body"]))
        doc_counts.append(
            {"slug": entry["slug"], "pole": entry["expected_pole"], "counts": counts}
        )
        logger.info("  %-55s  %d tokens", entry["slug"][:55], sum(counts.values()))
    return doc_counts


def aggregate(
    doc_counts: List[dict], poles: Tuple[str, ...]
) -> Tuple[Dict[str, "Counter[TokenKey]"], "Counter[TokenKey]", "Counter[str]"]:
    """Roll per-document counts up to per-pole counts, doc-frequency, pole sizes."""
    pole_counts: Dict[str, "Counter[TokenKey]"] = {pole: Counter() for pole in poles}
    doc_freq: "Counter[TokenKey]" = Counter()
    texts_per_pole: "Counter[str]" = Counter()

    for doc in doc_counts:
        pole_counts[doc["pole"]].update(doc["counts"])
        texts_per_pole[doc["pole"]] += 1
        for key in doc["counts"]:
            doc_freq[key] += 1

    return pole_counts, doc_freq, texts_per_pole


def build_rows(
    scores: Dict[TokenKey, float],
    pole_counts: Dict[str, "Counter[TokenKey]"],
    doc_freq: "Counter[TokenKey]",
    cis: Dict[TokenKey, Tuple[float, float]] | None = None,
    poles: Tuple[str, ...] = ("hors_sol", "terrestre"),
) -> List[LexiconRow]:
    cis = cis or {}
    minus_pole, plus_pole = poles
    rows: List[LexiconRow] = []
    for lemma, pos in sorted(scores):
        key = (lemma, pos)
        ci = cis.get(key)
        rows.append(
            LexiconRow(
                word=lemma,
                pos=pos,
                score=round(scores[key], 6),
                freq_minus=pole_counts[minus_pole].get(key, 0),
                freq_plus=pole_counts[plus_pole].get(key, 0),
                n_docs=doc_freq[key],
                ci_low=round(ci[0], 6) if ci else None,
                ci_high=round(ci[1], 6) if ci else None,
            )
        )
    return rows


def _output_columns(poles: Tuple[str, ...]) -> Tuple[str, str]:
    """The axis-specific CSV/Parquet header names for the two freq columns."""
    return f"freq_{poles[0]}", f"freq_{poles[1]}"


def write_lexicon(
    rows: List[LexiconRow],
    axis: str,
    poles: Tuple[str, ...] = ("hors_sol", "terrestre"),
) -> Tuple[Path, Path]:
    out_dir = output_dir(axis)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lexicon.csv"
    parquet_path = out_dir / "lexicon.parquet"

    minus_col, plus_col = _output_columns(poles)
    # Map the model's generic freq_minus / freq_plus onto axis-specific headers
    # (freq_hors_sol / freq_local …) so each lexicon names its own poles.
    fieldnames = [
        "word",
        "pos",
        "score",
        minus_col,
        plus_col,
        "n_docs",
        "ci_low",
        "ci_high",
    ]

    def _record(row: LexiconRow) -> dict:
        d = row.model_dump()
        return {
            "word": d["word"],
            "pos": d["pos"],
            "score": d["score"],
            minus_col: d["freq_minus"],
            plus_col: d["freq_plus"],
            "n_docs": d["n_docs"],
            "ci_low": d["ci_low"],
            "ci_high": d["ci_high"],
        }

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = _record(row)
            # Empty string for null CIs keeps the CSV human-readable.
            record["ci_low"] = "" if record["ci_low"] is None else record["ci_low"]
            record["ci_high"] = "" if record["ci_high"] is None else record["ci_high"]
            writer.writerow(record)

    table = pa.Table.from_pylist(
        [_record(row) for row in rows], schema=_parquet_schema(poles)
    )
    pq.write_table(table, parquet_path, compression="snappy")
    return csv_path, parquet_path


def calibrate(
    axis: str,
    dry_run: bool = False,
    min_docs_per_pole: int = 5,
    bootstrap_n: int = 0,
) -> int:
    if axis not in SUPPORTED_AXES:
        print(
            f"ERROR: unknown axis {axis!r}. Available: {list(SUPPORTED_AXES)}",
            file=sys.stderr,
        )
        return 1

    poles = SUPPORTED_AXES[axis]
    ref_score = AXIS_REF_SCORE[axis]

    kept, excluded = load_axis_corpus(axis)
    if excluded:
        logger.info(
            "Excluded %d INVERSION-HOLD text(s): %s", len(excluded), sorted(excluded)
        )
    if not kept:
        print(
            f"ERROR: no on-axis texts found for {axis!r} in {corpus_dir()}",
            file=sys.stderr,
        )
        return 1

    logger.info("Lemmatizing %d on-axis text(s)…", len(kept))
    nlp = _load_spacy()
    doc_counts = lemmatize_corpus(nlp, kept)

    pole_counts, doc_freq, texts_per_pole = aggregate(doc_counts, poles)

    for pole in poles:
        n = texts_per_pole.get(pole, 0)
        if n < min_docs_per_pole:
            logger.warning(
                "pole %r has %d doc(s) < --min-docs-per-pole %d — scores/CIs will be noisy",
                pole,
                n,
                min_docs_per_pole,
            )

    scores = compute_wordscores(pole_counts, ref_score)

    # The bootstrap is the expensive step; skip it on dry-run (which writes
    # nothing anyway) and when --bootstrap-n 0 (the point-estimate fast path).
    cis: Dict[TokenKey, Tuple[float, float]] = {}
    if bootstrap_n and not dry_run:
        logger.info("Bootstrapping %d resamples (Lowe 2008)…", bootstrap_n)
        cis = bootstrap_cis(
            doc_counts, poles, doc_freq, bootstrap_n, ref_score=ref_score
        )

    rows = build_rows(scores, pole_counts, doc_freq, cis, poles)

    print("\nCalibration stats")
    print(f"  axis:            {axis}")
    for pole in poles:
        print(f"  texts ({pole}): {texts_per_pole.get(pole, 0)}")
    print(f"  excluded (INVERSION-HOLD): {len(excluded)} {sorted(excluded)}")
    print(f"  scored lemmas:   {len(rows)}")
    if dry_run:
        print(f"  bootstrap:       skipped (dry-run); --bootstrap-n {bootstrap_n}")
    elif bootstrap_n:
        print(
            f"  lemmas with CI:  {sum(1 for r in rows if r.ci_low is not None)} (bootstrap-n {bootstrap_n})"
        )
    else:
        print("  bootstrap:       skipped (--bootstrap-n 0); CIs null")

    if dry_run:
        print("\n(dry-run: nothing written)")
        return 0

    csv_path, parquet_path = write_lexicon(rows, axis, poles)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {parquet_path} ({parquet_path.stat().st_size:,} bytes)")
    return 0


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis",
        default="hors-sol-terrestre",
        choices=list(SUPPORTED_AXES),
        help="Polar axis to calibrate: hors-sol-terrestre (new axis) or local-global (old axis)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + print stats without writing the lexicon",
    )
    parser.add_argument(
        "--min-docs-per-pole",
        type=int,
        default=5,
        help="warn (not abort) when a pole has fewer docs than this",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=500,
        help="Lowe 2008 document-resampling draws for the per-word CIs "
        "(default 500); 0 skips the bootstrap and emits null CIs",
    )
    args = parser.parse_args(argv)

    return calibrate(
        axis=args.axis,
        dry_run=args.dry_run,
        min_docs_per_pole=args.min_docs_per_pole,
        bootstrap_n=args.bootstrap_n,
    )


if __name__ == "__main__":
    sys.exit(main())
