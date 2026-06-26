"""Score texts with the Wordscores lexicon and compare against the Latouromètre.

STORY-185 / PRD-3 Goal 1. Consumes the lexicon written by ``calibrate`` and
scores whole texts on the 1-D Hors-Sol <-> Terrestre axis:

    axis_score(text) = mean over in-lexicon lemmas of S(w)
    predicted pole   = sign(axis_score)   (>0 terrestre, <0 hors_sol)

It scores four buckets and writes a CSV + a short markdown report:

  * **seeds**      -- the on-axis calibration texts (the easy HS-vs-T problem);
                      these give the headline non-inverted accuracy.
  * **inversions** -- the four ``sub_pole: inversion-stance`` texts (Fink,
                      Pouyanné, Damasio, Brunel) that wield a pole's lexicon
                      *against* it. Wordscores is expected to misclassify them:
                      this is the scientific payload (lexical scaling is
                      necessary but not sufficient; NLI stance is the missing
                      layer). The production cosine+NLI Latouromètre recovers
                      them -- contrasted statically from ``building_latourometre.md``.
  * **hold-out**   -- the unseen ``corpus_latourometre/tests/`` set (diagnostic,
                      mirroring ``score_tests_corpus.py``; gold often absent).
  * **oracle**     -- Latour's own *Où atterrir?* (``BrunoLatour_OuAttérir`` on
                      the samba mount, OCR'd, never redistributed). A positive
                      control: the lexicon should recognize its own source text.
                      Only the *score* is emitted -- never the prose.

This module never calls the words-weight runtime: the production figures are
read statically from the docs. A live re-score would be a follow-up.

Usage (inside the corpus-builder container):

    python -m corpus_builder.compare --axis hors-sol-terrestre [--dry-run] \\
        [--lexicon <path>] [--tests-dir <path>] [--oracle-path <path>]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reused from the read-only words-weight scripts mount (single source of truth
# for the on-disk corpus format AND the audit-flag selection) -- see compose.yml
# / PYTHONPATH. seed_pool / role_of / ROLE_INVERSION_HOLD bucket the corpus by
# audit role so no sub_pole/attracteur selection is duplicated here.
from ._corpus_loader import (
    ROLE_INVERSION_HOLD,
    load_corpus,
    parse_corpus_file,
    role_of,
    seed_pool,
)

# Reused from the sibling calibration module: same spaCy model + lemmatizer, so
# texts are tokenized exactly as the lexicon was built. No duplication.
from .calibrate import (
    SUPPORTED_AXES,
    TokenKey,
    _load_spacy,
    corpus_base,
    corpus_dir,
    lemmatize,
    output_dir,
)

logger = logging.getLogger("corpus_builder.compare")

# The four inversion-stance texts and the pole each author actually argues for,
# projected onto the Hors-Sol <-> Terrestre axis. Read from the frontmatter
# ``attracteur`` where on-axis; Brunel's gold is ``local`` (off this 1-D axis),
# so its prediction is reported but excluded from HS-vs-T accuracy.
# (Documented in latourometre-corpus-note.md §"L'inversion de stance".)

# Default oracle location on the samba mount, relative to CORPUS_BASE_PATH.
_ORACLE_DIRNAME = "BrunoLatour_OuAttérir"

# Text-scoring modes (STORY hardening): LBG = Laver-Benoit-Garry 2003 mean of
# S(w) (bounded [-1,+1], sign-calibrated but magnitude-compressed); LOWE = Lowe
# 2008 log-odds (unbounded, uncompressed calibrated positions).
SCORING_MODES = ("lbg", "lowe")

# Per-text bootstrap-CI parameters: resample the text's in-lexicon token scores
# with replacement (the unit is the in-lexicon TOKEN, not the document) to test
# whether the text's side is robust to its own word composition. Seed matches
# the project convention (calibrate._BOOTSTRAP_SEED is 12345; the per-text CI
# uses the prompt-specified 42 so the two resamplers are independently labelled).
_TEXT_BOOTSTRAP_N = 1000
_TEXT_BOOTSTRAP_SEED = 42
_TEXT_BOOTSTRAP_PCT = (2.5, 97.5)

# Lowe (2008) Dirichlet/Laplace smoothing constant. Calibrate applies NO
# smoothing to S(w) (compute_wordscores uses raw relative frequencies and drops
# words absent from both poles), so there is no project alpha to reuse; we adopt
# alpha=0.5 (Jeffreys prior), required because a word present in only one pole
# would otherwise give log(0) = -inf.
_LOWE_ALPHA = 0.5

# Reference doc for the production figures cited in the report.
_PROD_DOC = "editorial-pipelines/words-weight/docs/building_latourometre.md"


@dataclass
class ScoredText:
    slug: str
    category: str  # seed | inversion | holdout | oracle
    gold_pole: Optional[str]  # folded pole, or None when unknown / off-axis
    on_axis: bool  # gold_pole in the axis's two poles
    axis_score: float
    n_hits: int  # in-lexicon lemmas
    n_tokens: int  # content lemmas after POS filter
    pole_minus: str = "hors_sol"  # the -1 pole label for this axis
    pole_plus: str = "terrestre"  # the +1 pole label for this axis
    # Per-text bootstrap CI on axis_score (None when 0 in-lexicon hits).
    axis_score_ci_low: Optional[float] = None
    axis_score_ci_high: Optional[float] = None

    @property
    def sign_robust(self) -> bool:
        """True iff the CI does not cross 0 (both bounds share axis_score's sign).

        0 in-lexicon hits, a null CI, or an exact-0 axis_score are not robust.
        """
        lo, hi = self.axis_score_ci_low, self.axis_score_ci_high
        if lo is None or hi is None or self.axis_score == 0.0:
            return False
        if self.axis_score > 0:
            return lo > 0 and hi > 0
        return lo < 0 and hi < 0

    @property
    def predicted_pole(self) -> Optional[str]:
        return predict_pole(self.axis_score, self.n_hits, self.pole_minus, self.pole_plus)

    @property
    def correct(self) -> Optional[bool]:
        """True/False on the HS-vs-T axis; None when gold is unknown/off-axis."""
        if not self.on_axis or self.gold_pole is None:
            return None
        return self.predicted_pole == self.gold_pole


# --- pure scoring core (no spaCy, no corpus -- unit-testable) ---------------


def load_lexicon(path: Path) -> Dict[TokenKey, float]:
    """Read ``lexicon.csv`` into ``{(word, pos): score}``."""
    lexicon: Dict[TokenKey, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lexicon[(row["word"], row["pos"])] = float(row["score"])
    return lexicon


def load_pole_counts(path: Path) -> Dict[TokenKey, Tuple[int, int]]:
    """Read ``lexicon.csv`` into ``{(word, pos): (freq_minus, freq_plus)}``.

    The two pole-frequency columns are axis-specific (``freq_hors_sol`` /
    ``freq_terrestre`` or ``freq_local`` / ``freq_global``); they are the 4th and
    5th columns by position (``word, pos, score, freq_minus, freq_plus, …``), so
    we read them positionally and stay axis-agnostic. Needed only for the Lowe
    (2008) log-odds scoring, which recomputes per-word positions from raw counts.
    """
    counts: Dict[TokenKey, Tuple[int, int]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        minus_col, plus_col = header[3], header[4]  # freq_<minus>, freq_<plus>
        for row in reader:
            counts[(row[0], row[1])] = (int(row[3]), int(row[4]))
    logger.debug("pole-count columns: %s / %s", minus_col, plus_col)
    return counts


def compute_lowe_scores(
    pole_counts: Dict[TokenKey, Tuple[int, int]], alpha: float = _LOWE_ALPHA
) -> Dict[TokenKey, float]:
    """Lowe (2008) per-word log-odds positions ``theta(w)`` over the lexicon.

    ``theta(w) = log f_plus(w) − log f_minus(w)`` where
    ``f_pole(w) = (count_pole(w) + alpha) / (N_pole + alpha·V)`` is the
    Dirichlet/Laplace-smoothed relative frequency of ``w`` in that pole, ``V`` =
    lexicon vocab size, ``N_pole`` = total tokens in that pole over the lexicon
    vocabulary. Smoothing is required (a word absent from one pole → log 0).
    Sign convention matches LBG: the +1 (plus) pole drives ``theta`` positive.
    """
    vocab_size = len(pole_counts)
    n_minus = sum(c[0] for c in pole_counts.values())
    n_plus = sum(c[1] for c in pole_counts.values())
    denom_minus = n_minus + alpha * vocab_size
    denom_plus = n_plus + alpha * vocab_size

    scores: Dict[TokenKey, float] = {}
    for key, (c_minus, c_plus) in pole_counts.items():
        f_minus = (c_minus + alpha) / denom_minus
        f_plus = (c_plus + alpha) / denom_plus
        scores[key] = math.log(f_plus) - math.log(f_minus)
    return scores


def axis_score(tokens: List[TokenKey], lexicon: Dict[TokenKey, float]) -> Tuple[float, int]:
    """Mean lexicon score over the tokens that appear in the lexicon.

    Returns ``(score, n_hits)``. Out-of-lexicon tokens are skipped (they carry no
    axis signal). With zero hits the score is 0.0 and the caller treats the pole
    as undecided. Works identically for LBG ``S(w)`` and Lowe ``theta(w)``
    lexicons (mean of the per-token scores either way).
    """
    hits = [lexicon[t] for t in tokens if t in lexicon]
    if not hits:
        return 0.0, 0
    return sum(hits) / len(hits), len(hits)


def bootstrap_text_ci(
    hits: List[float],
    n_resamples: int = _TEXT_BOOTSTRAP_N,
    seed: int = _TEXT_BOOTSTRAP_SEED,
) -> Tuple[Optional[float], Optional[float]]:
    """Bootstrap CI on a text's mean in-lexicon score.

    Resamples the text's in-lexicon token scores (``hits``) with replacement
    ``n_resamples`` times, recomputes the mean each draw, and returns the
    ``(2.5, 97.5)`` percentiles. The resampling unit is the in-lexicon TOKEN
    occurrence, so the CI captures whether the text's side is robust to its own
    word composition. With 0 hits the CI is ``(None, None)``.
    """
    if not hits:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.asarray(hits, dtype=float)
    draws = arr[rng.integers(0, len(arr), size=(n_resamples, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(draws, _TEXT_BOOTSTRAP_PCT)
    return float(lo), float(hi)


def predict_pole(
    score: float,
    n_hits: int,
    pole_minus: str = "hors_sol",
    pole_plus: str = "terrestre",
) -> Optional[str]:
    """Sign of the axis score -> predicted pole; None when undecided.

    >0 -> the +1 (plus) pole, <0 -> the -1 (minus) pole. Zero hits or an exact
    0.0 tie is undecided (None) -- counted as a miss in accuracy, with a note.
    Defaults are the Hors-Sol <-> Terrestre poles (back-compat).
    """
    if n_hits == 0 or score == 0.0:
        return None
    return pole_plus if score > 0 else pole_minus


def accuracy(rows: List[ScoredText], category: str = "seed") -> Tuple[int, int]:
    """(correct, total) over the on-axis, known-gold rows of ``category``.

    ``category="seed"`` is the *in-sample* accuracy (reference == scored texts,
    circular). ``category="holdout"`` is the *out-of-sample* accuracy on the
    hold-out — the real generalization signal — graded only on the HS/T golds
    (Global/Local golds are off this 1-D axis and ungradable here).
    """
    graded = [r for r in rows if r.category == category and r.correct is not None]
    correct = sum(1 for r in graded if r.correct)
    return correct, len(graded)


def load_latourometre_preds(latour_dir: Path) -> Dict[str, str]:
    """Read the production Latouromètre per-text predictions from a run dir.

    Reads ``<slug>.latourometre.json`` (written by words-weight's
    ``score_tests_corpus.py``) and returns ``{slug: predicted_pole}`` over the
    four poles. Only the *prediction* is consumed; the gold comes from the live
    corpus frontmatter, never from these (possibly stale) JSON headers.
    """
    preds: Dict[str, str] = {}
    if not latour_dir.is_dir():
        logger.warning("latourometre dir not found, skipping head-to-head: %s", latour_dir)
        return preds
    for p in sorted(latour_dir.glob("*.latourometre.json")):
        if p.name.startswith("._"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        slug = d.get("slug") or p.name.replace(".latourometre.json", "")
        if d.get("predicted_pole"):
            preds[slug] = d["predicted_pole"]
    return preds


def headtohead(
    holdout: List[ScoredText],
    latour: Dict[str, str],
    poles: Tuple[str, str] = ("hors_sol", "terrestre"),
) -> dict:
    """Wordscores vs Latouromètre on the hold-out (pure, testable).

    - Wordscores is graded only on this axis's two poles.
    - The Latouromètre is graded on every non-null gold it has a prediction for
      (all four poles).
    - ``div`` lists the on-axis texts where exactly one method is right (the
      complementary-failure cases — the scientific payload).
    """
    ws = [r for r in holdout if r.gold_pole in poles]
    ws_ok = sum(1 for r in ws if r.predicted_pole == r.gold_pole)
    lat = [r for r in holdout if r.gold_pole and latour.get(r.slug)]
    lat_ok = sum(1 for r in lat if latour[r.slug] == r.gold_pole)
    div = []
    for r in ws:
        b = latour.get(r.slug)
        if b is not None and (r.predicted_pole == r.gold_pole) != (b == r.gold_pole):
            div.append((r.slug, r.gold_pole, r.predicted_pole, b))
    return {"ws_ok": ws_ok, "ws_tot": len(ws), "lat_ok": lat_ok, "lat_tot": len(lat), "div": div}


# --- corpus loading + scoring (spaCy) --------------------------------------


def _score_tokens(
    tokens: List[TokenKey], lexicon: Dict[TokenKey, float]
) -> Tuple[float, int, int, Tuple[Optional[float], Optional[float]]]:
    """Score + per-text bootstrap CI on the mean in-lexicon score.

    Returns ``(score, n_hits, n_tokens, (ci_low, ci_high))``. The CI resamples
    the in-lexicon token scores; it is null when there are no hits.
    """
    in_lex = [lexicon[t] for t in tokens if t in lexicon]
    score, n_hits = axis_score(tokens, lexicon)
    ci = bootstrap_text_ci(in_lex)
    return score, n_hits, len(tokens), ci


def _load_holdout(tests_dir: Path) -> List[dict]:
    """Tolerant load of the hold-out: ``tests/`` files often have no attracteur.

    Reuses ``parse_corpus_file`` (so the on-disk format knowledge stays in
    ``_corpus_loader``); on a missing/unknown ``attracteur`` it falls back to a
    header-less parse and records ``expected_pole = None``.
    """
    import frontmatter

    entries: List[dict] = []
    if not tests_dir.is_dir():
        logger.warning("hold-out dir not found, skipping: %s", tests_dir)
        return entries
    for path in sorted(tests_dir.glob("*.md")):
        if path.name.startswith("._") or path.name.endswith(".en.md"):
            continue
        try:
            entries.append(parse_corpus_file(path))
        except ValueError:
            # Empty / unknown attracteur -> keep the text, gold unknown.
            post = frontmatter.loads(path.read_text(encoding="utf-8"))
            entries.append(
                {
                    "slug": path.stem,
                    "header": dict(post.metadata),
                    "body": post.content,
                    "expected_pole": None,
                }
            )
    return entries


def _read_oracle_text(oracle_path: Path) -> Optional[str]:
    """Read the OCR'd Latour book (a single file or a directory of text files).

    Returns the concatenated text, or None if nothing readable is found. The
    text is used only to derive a score; it is never written to any artifact.
    """
    if not oracle_path.exists():
        logger.warning("oracle path not found, skipping: %s", oracle_path)
        return None

    files: List[Path]
    if oracle_path.is_dir():
        files = sorted(
            p
            for p in oracle_path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".txt", ".md", ""}
            and not p.name.startswith("._")
        )
    else:
        files = [oracle_path]

    chunks: List[str] = []
    for p in files:
        for encoding in ("utf-8", "latin-1"):
            try:
                chunks.append(p.read_text(encoding=encoding))
                break
            except (UnicodeDecodeError, OSError):
                continue
    text = "\n".join(chunks).strip()
    return text or None


def score_corpus(axis: str, lexicon: Dict[TokenKey, float], tests_dir: Path) -> List[ScoredText]:
    """Score seeds + inversions + hold-out against the lexicon."""
    poles = SUPPORTED_AXES[axis]
    minus_pole, plus_pole = poles
    on_axis_poles = set(poles)
    nlp = _load_spacy()
    rows: List[ScoredText] = []

    def _mk(
        slug: str,
        category: str,
        gold: Optional[str],
        score: float,
        n_hits: int,
        n_tokens: int,
        ci: Tuple[Optional[float], Optional[float]] = (None, None),
    ) -> ScoredText:
        return ScoredText(
            slug=slug,
            category=category,
            gold_pole=gold,
            on_axis=gold in on_axis_poles,
            axis_score=score,
            n_hits=n_hits,
            n_tokens=n_tokens,
            pole_minus=minus_pole,
            pole_plus=plus_pole,
            axis_score_ci_low=ci[0],
            axis_score_ci_high=ci[1],
        )

    # Seeds + inversions live at the top level of corpus_latourometre/. The two
    # buckets are split by audit role (not sub_pole): the seed bucket is the
    # ``audit.use_for_seeds: true`` basin, the inversion bucket every
    # ``INVERSION-HOLD`` text. This is what lands Bruckner (sub_pole: null but
    # role INVERSION-HOLD) in the inversion stress group, not the seeds.

    # Seeds: only this axis's two poles feed the headline accuracy.
    for entry in seed_pool(corpus_dir()):
        gold = entry["expected_pole"]
        if gold not in on_axis_poles:
            continue
        score, n_hits, n_tokens, ci = _score_tokens(lemmatize(nlp, entry["body"]), lexicon)
        rows.append(_mk(entry["slug"], "seed", gold, score, n_hits, n_tokens, ci))

    # Inversions (INVERSION-HOLD): all 8 are scored. Off-axis ones (gold local
    # on this axis) are reported, not graded — a transparency check.
    for entry in load_corpus(corpus_dir()):
        if role_of(entry) != ROLE_INVERSION_HOLD:
            continue
        gold = entry["expected_pole"]
        score, n_hits, n_tokens, ci = _score_tokens(lemmatize(nlp, entry["body"]), lexicon)
        rows.append(_mk(entry["slug"], "inversion", gold, score, n_hits, n_tokens, ci))

    # Hold-out (diagnostic): gold often absent.
    for entry in _load_holdout(tests_dir):
        gold = entry["expected_pole"]
        score, n_hits, n_tokens, ci = _score_tokens(lemmatize(nlp, entry["body"]), lexicon)
        rows.append(_mk(entry["slug"], "holdout", gold, score, n_hits, n_tokens, ci))

    return rows


def score_oracle(lexicon: Dict[TokenKey, float], oracle_path: Path) -> Optional[ScoredText]:
    text = _read_oracle_text(oracle_path)
    if text is None:
        return None
    nlp = _load_spacy()
    score, n_hits, n_tokens, ci = _score_tokens(lemmatize(nlp, text), lexicon)
    return ScoredText(
        slug=oracle_path.name,
        category="oracle",
        gold_pole="terrestre",  # the expected positive-control pole
        on_axis=True,
        axis_score=score,
        n_hits=n_hits,
        n_tokens=n_tokens,
        axis_score_ci_low=ci[0],
        axis_score_ci_high=ci[1],
    )


# --- writers ---------------------------------------------------------------

_CSV_FIELDS = [
    "slug",
    "category",
    "gold_pole",
    "predicted_pole",
    "axis_score",
    "axis_score_ci_low",
    "axis_score_ci_high",
    "sign_robust",
    "n_lexicon_hits",
    "n_tokens",
    "on_axis_correct",
    "scoring",
]


def _fmt_pole(pole: Optional[str]) -> str:
    return pole if pole else "—"


def write_comparison_csv(rows: List[ScoredText], path: Path, scoring: str = "lbg") -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            correct = "" if r.correct is None else ("1" if r.correct else "0")
            ci_lo = "" if r.axis_score_ci_low is None else round(r.axis_score_ci_low, 6)
            ci_hi = "" if r.axis_score_ci_high is None else round(r.axis_score_ci_high, 6)
            writer.writerow(
                {
                    "slug": r.slug,
                    "category": r.category,
                    "gold_pole": r.gold_pole or "",
                    "predicted_pole": r.predicted_pole or "",
                    "axis_score": round(r.axis_score, 6),
                    "axis_score_ci_low": ci_lo,
                    "axis_score_ci_high": ci_hi,
                    "sign_robust": str(r.sign_robust),
                    "n_lexicon_hits": r.n_hits,
                    "n_tokens": r.n_tokens,
                    "on_axis_correct": correct,
                    "scoring": scoring,
                }
            )


def _inversion_table(rows: List[ScoredText]) -> Tuple[str, int, int]:
    """Markdown table for the inversion stress test + (misclassified, gradable)."""
    inv = [r for r in rows if r.category == "inversion"]
    lines = [
        "| Text | Author argues for | Wordscores predicts | Axis score | Hits | Outcome |",
        "|---|---|---|---|---|---|",
    ]
    misclassified = gradable = 0
    for r in sorted(inv, key=lambda x: x.slug):
        if r.correct is None:
            outcome = f"off-axis (gold *{_fmt_pole(r.gold_pole)}*) — reported, not graded"
        else:
            gradable += 1
            if r.correct:
                outcome = "✅ correct (rare for a lexical scaler)"
            else:
                misclassified += 1
                outcome = "❌ **misclassified** — speaks the opposing lexicon"
        lines.append(
            f"| `{r.slug}` | {_fmt_pole(r.gold_pole)} | {_fmt_pole(r.predicted_pole)} "
            f"| {r.axis_score:+.3f} | {r.n_hits} | {outcome} |"
        )
    return "\n".join(lines), misclassified, gradable


def _holdout_table(rows: List[ScoredText], latour: Optional[Dict[str, str]] = None) -> str:
    ho = sorted((r for r in rows if r.category == "holdout"), key=lambda x: x.slug)
    if not ho:
        return "_No hold-out texts found._"
    has_lat = bool(latour)
    head = "| Text | Gold | Wordscores | Axis score | Hits |"
    sep = "|---|---|---|---|---|"
    if has_lat:
        head = "| Text | Gold | Wordscores | Latouromètre | Axis score | Hits |"
        sep = "|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in ho:
        cols = [f"`{r.slug}`", _fmt_pole(r.gold_pole), _fmt_pole(r.predicted_pole)]
        if has_lat:
            cols.append(_fmt_pole(latour.get(r.slug)))
        cols += [f"{r.axis_score:+.3f}", str(r.n_hits)]
        lines.append("| " + " | ".join(cols) + " |")
    return "\n".join(lines)


def _headtohead_section(holdout: List[ScoredText], latour: Optional[Dict[str, str]]) -> str:
    """Out-of-sample accuracy: Wordscores alone, or head-to-head vs Latouromètre."""
    ws_ok, ws_tot = accuracy(holdout if isinstance(holdout, list) else list(holdout), "holdout")
    ws_pct = f"{100 * ws_ok / ws_tot:.1f}%" if ws_tot else "n/a"

    if not latour:
        return (
            f"**Wordscores out-of-sample accuracy (hold-out, HS↔T golds): "
            f"{ws_ok}/{ws_tot} ({ws_pct}).** Unlike the in-sample seeds, the "
            f"hold-out texts never entered the lexicon — this is the real "
            f"generalization signal. Only HS/T golds are graded (Global/Local "
            f"golds are off this 1-D axis). Pass `--latourometre-dir` to add the "
            f"head-to-head against the production metric."
        )

    h = headtohead(list(holdout), latour)
    lat_pct = f"{100 * h['lat_ok'] / h['lat_tot']:.1f}%" if h["lat_tot"] else "n/a"
    div_lines = "\n".join(
        f"- `{slug}` (gold **{g}**): Wordscores **{_fmt_pole(a)}** "
        f"{'✓' if a == g else '✗'}, Latouromètre **{_fmt_pole(b)}** {'✓' if b == g else '✗'}"
        for slug, g, a, b in sorted(h["div"], key=lambda x: x[0])
    ) or "- _(none — the two methods agree on every gradable HS/T text)_"

    return f"""Out-of-sample, on the recalibrated hold-out (golds assigned independently of
either method's output, so this is non-circular):

| Method | Scope | Out-of-sample accuracy |
|---|---|---|
| **Wordscores** (lexical, 1-D HS↔T) | {h['ws_tot']} HS/T golds | **{h['ws_ok']}/{h['ws_tot']} ({ws_pct})** |
| **Latouromètre** (cosine + NLI, 4 poles) | {h['lat_tot']} golds (all poles) | **{h['lat_ok']}/{h['lat_tot']} ({lat_pct})** |

The two are **not** ranked — they cover different problems (Wordscores cannot
represent Global/Local; the Latouromètre does all four poles) and, crucially,
**fail on different texts**. Where exactly one is right on the shared HS↔T
sub-problem:

{div_lines}

These divergences are the payload: a purely-lexical scaler is *more robust* than
the NLI metric on texts whose vocabulary is unambiguously Hors-Sol (techno-
imperial, transhumanist), where the NLI stance layer over-reasons; the metric in
turn wins wherever Global/Local or stance-inversion matters. (Latouromètre
predictions read from the run artifacts; no live words-weight call.)"""


def _oracle_section(oracle: Optional[ScoredText]) -> str:
    if oracle is None:
        return (
            "The Latour oracle was not scored — `BrunoLatour_OuAttérir` was not "
            "found on the corpus mount. Re-run with `--oracle-path` pointing at "
            "the OCR'd book."
        )
    pole = oracle.predicted_pole
    half = "**Terrestre half**" if pole == "terrestre" else "**Hors-Sol half**"
    verdict = (
        "The positive control passes: the lexicon recognizes its own founding "
        "text as Terrestre."
        if pole == "terrestre"
        else (
            "The positive control is **ambiguous**: Latour's book devotes long "
            "passages to *describing* the Hors-Sol attractor in order to "
            "critique it, so a purely-lexical scaler reads that exposition as "
            "Hors-Sol vocabulary — the very stance-vs-topic confusion this report "
            "documents. Even the axis's founding text trips Wordscores; the "
            "score is the result, not a failure to force."
        )
    )
    return (
        f"Scoring *Où atterrir?* (`{oracle.slug}`, OCR'd, never redistributed — "
        f"only the score is emitted) lands at **axis_score = {oracle.axis_score:+.3f}** "
        f"over {oracle.n_hits} in-lexicon lemmas → the {half}. {verdict}"
    )


_POLE_DISPLAY = {
    "hors_sol": "Hors-Sol",
    "terrestre": "Terrestre",
    "local": "Local",
    "global": "Global",
}


def _disp(pole: str) -> str:
    return _POLE_DISPLAY.get(pole, pole)


def _scoring_banner(scoring: str) -> str:
    """One-line provenance banner naming the text-scoring mode used."""
    if scoring == "lowe":
        return (
            f"> **Text scoring: Lowe (2008) log-odds** — `theta_doc = mean over "
            f"in-lexicon lemmas of theta(w)`, `theta(w) = log f_plus(w) − log "
            f"f_minus(w)` on Laplace-smoothed (alpha={_LOWE_ALPHA}) per-pole "
            f"frequencies. Unbounded units (not compressed toward 0). The lexicon "
            f"*calibration* (S(w) + per-word CIs) is unchanged; only the text "
            f"position is recomputed. Each text carries a bootstrap CI on its "
            f"theta_doc (1000 resamples of its in-lexicon tokens, seed "
            f"{_TEXT_BOOTSTRAP_SEED}); `sign_robust` = CI does not cross 0.\n"
        )
    return (
        f"> **Text scoring: Laver-Benoit-Garry (2003)** — `axis_score = mean over "
        f"in-lexicon lemmas of S(w)`, `S(w) ∈ [−1, +1]`. Each text carries a "
        f"bootstrap CI on its axis_score (1000 resamples of its in-lexicon "
        f"tokens, seed {_TEXT_BOOTSTRAP_SEED}); `sign_robust` = CI does not cross 0.\n"
    )


def build_report_generic(
    axis: str,
    rows: List[ScoredText],
    latour: Optional[Dict[str, str]] = None,
    scoring: str = "lbg",
) -> str:
    """Axis-neutral, data-first report (no axis-specific interpretation).

    Used for axes other than Hors-Sol ↔ Terrestre. The narrative interpretation
    lives in the PRD; this report carries the measured numbers + tables only.
    """
    poles = SUPPORTED_AXES[axis]
    minus, plus = poles
    minus_d, plus_d = _disp(minus), _disp(plus)

    in_ok, in_tot = accuracy(rows, "seed")
    in_pct = f"{100 * in_ok / in_tot:.1f}%" if in_tot else "n/a"
    oos_ok, oos_tot = accuracy(rows, "holdout")
    oos_pct = f"{100 * oos_ok / oos_tot:.1f}%" if oos_tot else "n/a"
    inv_md, misclassified, gradable = _inversion_table(rows)
    holdout = [r for r in rows if r.category == "holdout"]
    n_seed = sum(1 for r in rows if r.category == "seed")
    n_inv = sum(1 for r in rows if r.category == "inversion")

    # Per-pole confusion on the seeds (in-sample), purely descriptive.
    seeds = [r for r in rows if r.category == "seed" and r.on_axis]
    confusion_lines = ["| Gold pole | n | Correct | Undecided (0 hits / tie) |", "|---|---|---|---|"]
    for pole in poles:
        sub = [r for r in seeds if r.gold_pole == pole]
        correct = sum(1 for r in sub if r.correct)
        undecided = sum(1 for r in sub if r.predicted_pole is None)
        confusion_lines.append(f"| {_disp(pole)} | {len(sub)} | {correct} | {undecided} |")
    confusion_md = "\n".join(confusion_lines)

    head2head = ""
    if latour:
        h = headtohead(holdout, latour, poles)
        lat_pct = f"{100 * h['lat_ok'] / h['lat_tot']:.1f}%" if h["lat_tot"] else "n/a"
        div_lines = "\n".join(
            f"- `{slug}` (gold **{_disp(g)}**): Wordscores **{_fmt_pole(a)}** "
            f"{'✓' if a == g else '✗'}, Latouromètre **{_fmt_pole(b)}** {'✓' if b == g else '✗'}"
            for slug, g, a, b in sorted(h["div"], key=lambda x: x[0])
        ) or "- _(none on the shared on-axis golds)_"
        head2head = f"""
| Method | Scope | Out-of-sample accuracy |
|---|---|---|
| **Wordscores** (lexical, {minus_d}↔{plus_d}) | {h['ws_tot']} on-axis golds | **{h['ws_ok']}/{h['ws_tot']}** |
| **Latouromètre** (cosine + NLI, 4 poles) | {h['lat_tot']} golds (all poles) | **{h['lat_ok']}/{h['lat_tot']} ({lat_pct})** |

On-axis texts where exactly one method is right:

{div_lines}
"""

    return f"""# Wordscores — {minus_d} ↔ {plus_d} axis (data report)

> Axis-neutral, numbers-only report. Interpretation lives in the PRD
> (`project-management/prds/3_latourometer-corpus-builder-wordscores-latour.md`).

{_scoring_banner(scoring)}

**Method.** Each text is scored as `axis_score = mean over in-lexicon lemmas of
S(w)`, `S(w) ∈ [−1, +1]` (−1 = {minus_d}, +1 = {plus_d}). The sign of `axis_score`
is the predicted pole. CPU-only lexical baseline — no embeddings, no NLI.

**Scope.** The 1-D {minus_d} ↔ {plus_d} axis only. Off-axis texts are excluded
from the headline accuracy; off-axis `inversion-stance` texts are still scored and
reported (not graded). {n_seed} calibration seeds, {n_inv} inversion text(s).

## 1. In-sample accuracy (seeds — circular, reference only)

**Wordscores on the {n_seed} non-inverted seeds: {in_ok}/{in_tot} ({in_pct}).**
In-sample = these texts built the lexicon, so this is circular. Per-pole:

{confusion_md}

## 2. Out-of-sample accuracy (hold-out — the real signal, small N)

**Wordscores hold-out (on-axis golds): {oos_ok}/{oos_tot} ({oos_pct}).** The
hold-out is small — read the point estimate as directional, not conclusive.
{head2head}

## 3. Inversion stress test

`inversion-stance` texts wield a pole's lexicon *against* it. Misclassified
{misclassified} of {gradable} gradable on-axis inversion(s):

{inv_md}

## 4. Hold-out (`tests/`) — per-text

{_holdout_table(rows, latour)}

## Limitations

- **Small reference set.** This axis has few calibration texts per pole; bootstrap
  CIs are wide and many lemmas fall below the `n_docs ≥ 3` CI threshold.
- **Inversion blindness.** A lexical scaler reads vocabulary, not stance.
- **Lexicon coverage.** Low `n_lexicon_hits` texts yield low-confidence scores.
"""


def build_report(
    axis: str,
    rows: List[ScoredText],
    oracle: Optional[ScoredText],
    latour: Optional[Dict[str, str]] = None,
    scoring: str = "lbg",
) -> str:
    # Only the politically-active axis carries the hand-tuned interpretation
    # (greenwashing / adversarial-quotation reading, Latour oracle). Other axes
    # get the neutral data report; their interpretation is authored in the PRD.
    if axis != "hors-sol-terrestre":
        return build_report_generic(axis, rows, latour, scoring)

    in_ok, in_tot = accuracy(rows, "seed")
    in_pct = f"{100 * in_ok / in_tot:.1f}%" if in_tot else "n/a"
    inv_md, misclassified, gradable = _inversion_table(rows)
    holdout = [r for r in rows if r.category == "holdout"]
    n_seed = sum(1 for r in rows if r.category == "seed")
    n_inv = sum(1 for r in rows if r.category == "inversion")

    return f"""# Wordscores vs. the Latouromètre — Hors-Sol ↔ Terrestre comparison

{_scoring_banner(scoring)}
**Method.** Each text is scored as `axis_score = mean over in-lexicon lemmas of
S(w)`, where `S(w) ∈ [−1, +1]` comes from the STORY-183/184 Wordscores lexicon
(−1 = Hors-Sol, +1 = Terrestre). The sign of `axis_score` is the predicted pole.
Out-of-lexicon lemmas carry no axis signal and are skipped. CPU-only; no
embeddings, no NLI — this is the transparent lexical baseline.

**Scope.** This is the **1-D Hors-Sol ↔ Terrestre** sub-problem only. Local /
Global texts are off this axis; full 4-attractor classification needs the second
axis (PRD-3 v2 — the two-axis Wordscores). Accuracy is reported twice: *in-sample*
on the {n_seed} calibration seeds (circular), and *out-of-sample* on the hold-out
(the real signal). The {n_inv} `inversion-stance` texts are a separate stress test.

## 1. In-sample accuracy (seeds — circular, for reference only)

**Wordscores on the {n_seed} non-inverted calibration seeds: {in_ok}/{in_tot}
({in_pct}).** This is **in-sample**: these very texts built the lexicon, so
scoring them back is circular and trivially high — it is *not* evidence of
generalization. The number that matters is §2.

## 2. Out-of-sample accuracy (hold-out — the real signal)

{_headtohead_section(holdout, latour)}

## 3. Inversion stress test

The `inversion-stance` texts wield a pole's lexicon *against* it. A purely lexical
method sees vocabulary, not stance. **Wordscores misclassifies {misclassified} of
the {gradable} gradable on-axis inversions** (Brunel is off-axis, gold *local*,
not graded):

{inv_md}

The result is sharper than "fails on all inversions": Wordscores **catches the
greenwashing** ones (Fink, Pouyanné — a thin sustainability veneer over a
dominant finance/extractive vocabulary that genuinely reads Hors-Sol) but **flips
the adversarial-quotation** one (Damasio speaks « Silicon Valley »/« GAFAM » to
demolish it — and even then only to a near-tie). It reads the words, not the
argument.

## 4. Contrast with the production Latouromètre (cosine + NLI)

The production metric (`{_PROD_DOC}`, seeds v4 + NLI stance blend, run_009)
reaches **36/39 (0.923)**, bootstrap 95% CI **[0.846, 1.000]** on the full 4-pole
calibration corpus. Its NLI stance layer is what recovers the in-corpus inversions
a purely-cosine method misses (Butré terrestre 0.58 → local 0.74; Bruckner
terrestre 0.47 → local 0.63). But the head-to-head in §2 shows the converse: on
the out-of-sample HS↔T axis the lexical baseline is *more robust* on
unambiguously-Hors-Sol texts (techno-imperial, transhumanist) where the NLI layer
over-reasons. **Lexical scaling is not uniformly "necessary but not sufficient":
it is sufficient and even superior on the clean HS↔T axis, and insufficient only
when (a) Global/Local must be told apart or (b) a stance inversion must be read.**

## 5. Latour-oracle (positive control)

{_oracle_section(oracle)}

## 6. Hold-out (`tests/`) — per-text

{_holdout_table(rows, latour)}

## Limitations

- **Global/Local blindness.** Wordscores is 1-D HS↔T; it cannot represent the
  Local↔Global axis (the IMF Article-IV text, gold *global-*, is unrepresentable
  and both methods mis-read it as Hors-Sol — `dérégulation`/`marché` is a lexicon
  *shared* between Global and Hors-Sol). PRD-3 v2's second axis (Local↔Global)
  closes this.
- **Inversion blindness.** By construction it cannot separate stance from topic
  (§3). The cosine+NLI metric closes that specific gap.
- **Lexicon coverage.** Texts with few in-lexicon lemmas yield low-confidence
  scores; the `n_lexicon_hits` column flags them.
"""


def compare(
    axis: str,
    dry_run: bool = False,
    lexicon_path: Optional[Path] = None,
    tests_dir: Optional[Path] = None,
    oracle_path: Optional[Path] = None,
    latourometre_dir: Optional[Path] = None,
    scoring: str = "lbg",
) -> int:
    if axis not in SUPPORTED_AXES:
        print(
            f"ERROR: unknown axis {axis!r}. Available: {list(SUPPORTED_AXES)}",
            file=sys.stderr,
        )
        return 1
    if scoring not in SCORING_MODES:
        print(
            f"ERROR: unknown scoring {scoring!r}. Available: {list(SCORING_MODES)}",
            file=sys.stderr,
        )
        return 1

    out_dir = output_dir(axis)
    lexicon_path = lexicon_path or (out_dir / "lexicon.csv")
    tests_dir = tests_dir or (corpus_dir() / "tests")
    oracle_path = oracle_path or (corpus_base() / _ORACLE_DIRNAME)

    if not lexicon_path.exists():
        print(
            f"ERROR: lexicon not found at {lexicon_path}. Run "
            f"`python -m corpus_builder.calibrate --axis {axis}` first.",
            file=sys.stderr,
        )
        return 1

    # LBG = the calibrated S(w) point estimates (back-compat); Lowe = log-odds
    # theta(w) recomputed from the lexicon's per-pole counts (uncompressed).
    if scoring == "lowe":
        lexicon = compute_lowe_scores(load_pole_counts(lexicon_path))
        logger.info(
            "Computed %d Lowe theta(w) scores from %s (alpha=%s)",
            len(lexicon),
            lexicon_path,
            _LOWE_ALPHA,
        )
    else:
        lexicon = load_lexicon(lexicon_path)
        logger.info("Loaded %d lexicon entries from %s", len(lexicon), lexicon_path)

    rows = score_corpus(axis, lexicon, tests_dir)
    # The Latour oracle (*Où atterrir?*) is a positive control for the new
    # Hors-Sol ↔ Terrestre axis only — the book has no defined position on the
    # old Local ↔ Global axis, so scoring it there would be meaningless.
    oracle = score_oracle(lexicon, oracle_path) if axis == "hors-sol-terrestre" else None
    latour = load_latourometre_preds(latourometre_dir) if latourometre_dir else None

    in_ok, in_tot = accuracy(rows, "seed")
    oos_ok, oos_tot = accuracy(rows, "holdout")
    _inv_md, misclassified, gradable = _inversion_table(rows)

    print("\nComparison stats")
    print(f"  axis:                 {axis}")
    print(f"  scoring:              {scoring}")
    print(f"  lexicon entries:      {len(lexicon)}")
    print(f"  seeds (non-inverted): {sum(1 for r in rows if r.category == 'seed')}")
    print(f"  inversions:           {sum(1 for r in rows if r.category == 'inversion')}")
    print(f"  hold-out:             {sum(1 for r in rows if r.category == 'holdout')}")
    print(f"  in-sample acc:        {in_ok}/{in_tot} (seeds — circular)")
    print(f"  out-of-sample acc:    {oos_ok}/{oos_tot} (hold-out HS/T golds)")
    print(f"  inversions wrong:     {misclassified}/{gradable} (on-axis, gradable)")
    if latour:
        h = headtohead([r for r in rows if r.category == "holdout"], latour)
        print(f"  Latouromètre OOS:     {h['lat_ok']}/{h['lat_tot']} (4 poles, from run dir)")
    if oracle is not None:
        print(f"  oracle axis_score:    {oracle.axis_score:+.3f} -> {oracle.predicted_pole}")
    else:
        print("  oracle:               not found (skipped)")

    if dry_run:
        print("\n(dry-run: nothing written)")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    # LBG keeps the canonical filenames (back-compat); Lowe is suffixed so the
    # two scorings live side by side without clobbering each other.
    suffix = "" if scoring == "lbg" else f"_{scoring}"
    csv_path = out_dir / f"comparison{suffix}.csv"
    report_path = out_dir / f"comparison-report{suffix}.md"
    all_rows = rows + ([oracle] if oracle else [])
    write_comparison_csv(all_rows, csv_path, scoring=scoring)
    report_path.write_text(build_report(axis, rows, oracle, latour, scoring), encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {report_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis",
        default="hors-sol-terrestre",
        choices=list(SUPPORTED_AXES),
        help="Polar axis to compare: hors-sol-terrestre or local-global",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="score + print stats without writing the CSV / report",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help="lexicon.csv path (default axes/<axis>/lexicon.csv under the corpus base)",
    )
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=None,
        help="hold-out dir (default corpus_latourometre/tests)",
    )
    parser.add_argument(
        "--oracle-path",
        type=Path,
        default=None,
        help=f"Latour book path (default {_ORACLE_DIRNAME} under the corpus base); "
        "scored as a positive control, its prose is never written out",
    )
    parser.add_argument(
        "--latourometre-dir",
        type=Path,
        default=None,
        help="optional dir of <slug>.latourometre.json (words-weight run artifacts); "
        "when given, the report adds the out-of-sample Wordscores-vs-Latouromètre "
        "head-to-head. No live words-weight call.",
    )
    parser.add_argument(
        "--scoring",
        default="lbg",
        choices=list(SCORING_MODES),
        help="text-scoring mode: lbg = Laver-Benoit-Garry 2003 mean S(w) "
        "(bounded, default, back-compat); lowe = Lowe 2008 log-odds theta(w) "
        "(unbounded, uncompressed). Lexicon calibration is unchanged either way.",
    )
    args = parser.parse_args(argv)

    return compare(
        axis=args.axis,
        dry_run=args.dry_run,
        lexicon_path=args.lexicon,
        tests_dir=args.tests_dir,
        oracle_path=args.oracle_path,
        latourometre_dir=args.latourometre_dir,
        scoring=args.scoring,
    )


if __name__ == "__main__":
    sys.exit(main())
