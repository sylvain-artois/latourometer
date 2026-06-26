"""Wordfish (Slapin & Proksch 2008) baseline for the Latouromètre.

Wordfish is the canonical **unsupervised** positional-scaling method in
political science: a one-parameter-per-word Poisson item-response model that
recovers latent document positions from word-frequency variation alone, with
**no anchor texts** (unlike Wordscores). For document ``i`` and word ``j`` the
count is

    y_ij ~ Poisson( exp(alpha_i + psi_j + beta_j * theta_i) )

where ``alpha_i`` is the document loquaciousness, ``psi_j`` the word frequency
intercept, ``beta_j`` the word's discrimination on the latent axis, and
``theta_i`` the document position (the quantity of interest). It is fit here by
**alternating Newton-Raphson** (the word block and the document block are each a
2-parameter Poisson regression), pure NumPy, CPU-only — no R, no GPU.

This module mirrors ``compare`` (the Wordscores baseline): same spaCy model +
lemmatizer (so the document-feature matrix is built exactly like the lexicon
was), same corpus + ``audit:`` loading via ``_corpus_loader``, same two-axis
gold convention. Only the *scaling model* differs.

Wordfish is **1-D by design**; the Latouromètre is 4-pole. We fit it **twice**
(once per axis: Hors-Sol↔Terrestre and Local↔Global), then fold the two
positions ``(theta_A, theta_B)`` to a pole by the same quadrant rule
``compare.py`` uses (the axis whose ``|theta|`` is larger, signed). The axis
sign is unidentified in Wordfish; we fix it **deterministically** by orienting
each fitted axis so the mean position of its plus-pole seeds exceeds that of its
minus-pole seeds (uses gold labels for *sign only*, never for the fit).

Usage:

    python -m latourometer.baselines.wordfish [--min-doc-freq 5] [--dry-run]

Outputs under ``CORPUS_BASE_PATH`` (default /data/corpus):

    calibration_latourometre/baselines_20260625/wordfish/
        wordfish_<axis>.csv          -- per-text theta + side + correctness
        wordfish_2d_summary.csv      -- (theta_HS-T, theta_L-G) -> pole
        wordfish-report.md           -- per-axis accuracy + inversion breakdown
    calibration_latourometre/baselines_20260625/comparison.{csv,md}
        -- shared table; this run adds/refreshes the ``wordfish_pred`` column
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Single source of truth for the on-disk corpus format AND the audit-flag
# selection.
from ._corpus_loader import (
    ROLE_INVERSION_HOLD,
    audit_block,
    load_corpus,
    role_of,
    seed_pool,
)

# Same spaCy model + lemmatizer + axis/pole conventions as the Wordscores
# baseline, so the two methods are scored on byte-identical preprocessing.
from .calibrate import (
    SUPPORTED_AXES,
    TokenKey,
    _load_spacy,
    corpus_base,
    corpus_dir,
    lemmatize,
)

logger = logging.getLogger("latourometer.baselines.wordfish")

# The two axes, in the order the 2-D summary lists them.
_AXIS_HS_T = "hors-sol-terrestre"
_AXIS_L_G = "local-global"

# Default document-feature cutoff: drop words appearing in < this many of the
# axis's seed documents. Wordfish has one beta per word, so with ~27-28 docs/axis
# a large vocabulary (V >> N) leaves rarely-seen words quasi-separable and the
# alternating-Newton fit fails to converge. A cutoff of 5 trims the HS↔T vocab
# from ~2765 to ~942 lemmas; BOTH axes then converge and the 4-pole accuracy is
# stable across cutoffs 5-8 (verified 2026-06-25). Lowering it is the first knob
# to revisit if a re-run stops converging (see PRD-9 Risks).
_DEFAULT_MIN_DOC_FREQ = 5

# Alternating-Newton fit controls. Small N + small vocab => fast; these are
# generous. The linear predictor is clipped before exp() to avoid overflow.
_FIT_MAX_SWEEPS = 200
_FIT_TOL = 1e-5            # convergence on max |Δtheta| between sweeps
_INNER_NEWTON = 3         # Newton steps inside each block per sweep
_RIDGE = 1e-6             # Hessian diagonal ridge for numerical stability
_ETA_CLIP = 30.0          # clip linear predictor to [-30, 30] before exp
_PROJ_MAX_ITER = 50       # Newton iters when projecting an out-of-sample doc

# Accuracy bootstrap (resample TEXTS with replacement) — matches the 95% CI
# convention used across the project (compare/calibrate use 2.5/97.5).
_ACC_BOOTSTRAP_N = 1000
_ACC_BOOTSTRAP_SEED = 42
_ACC_BOOTSTRAP_PCT = (2.5, 97.5)

# Where the shared baselines comparison + the wordfish artifacts live.
_BASELINES_DIRNAME = "baselines_20260625"


def baselines_dir() -> Path:
    return corpus_base() / "calibration_latourometre" / _BASELINES_DIRNAME


def wordfish_dir() -> Path:
    return baselines_dir() / "wordfish"


# ===========================================================================
# Pure model core (no spaCy, no corpus, no I/O — unit-testable)
# ===========================================================================


def _safe_exp(eta: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(eta, -_ETA_CLIP, _ETA_CLIP))


def _newton_2param(
    y: np.ndarray,          # [n] counts for the block being solved
    offset: np.ndarray,     # [n] fixed part of the linear predictor
    x1: np.ndarray,         # [n] the slope covariate (theta_i or beta_j)
    b0: float,              # current intercept (psi_j or alpha_i)
    b1: float,              # current slope (beta_j or theta_i)
    n_iter: int = _INNER_NEWTON,
    ridge: float = _RIDGE,
) -> Tuple[float, float]:
    """One block's 2-parameter Poisson-regression Newton solve.

    Model: y ~ Poisson(exp(offset + b0 + b1*x1)). Returns the updated (b0, b1).
    Used for both the word block (b0=psi, b1=beta, x1=theta, offset=alpha) and
    the document block (b0=alpha, b1=theta, x1=beta, offset=psi).
    """
    for _ in range(n_iter):
        mu = _safe_exp(offset + b0 + b1 * x1)
        resid = y - mu
        g0 = resid.sum()
        g1 = (x1 * resid).sum()
        h00 = mu.sum() + ridge
        h01 = (x1 * mu).sum()
        h11 = (x1 * x1 * mu).sum() + ridge
        det = h00 * h11 - h01 * h01
        if not np.isfinite(det) or abs(det) < 1e-12:
            break
        b0 += (h11 * g0 - h01 * g1) / det
        b1 += (-h01 * g0 + h00 * g1) / det
    return float(b0), float(b1)


def _svd_init_theta(Y: np.ndarray) -> np.ndarray:
    """Initialise theta from the first singular vector of the log-rate matrix.

    Row-normalise (with +0.5 smoothing), take logs, column-centre, SVD; the
    first left singular vector is the dominant positional dimension. Returned
    standardised to mean 0 / sd 1.
    """
    d, w = Y.shape
    rowsum = Y.sum(axis=1, keepdims=True)
    p = (Y + 0.5) / (rowsum + 0.5 * w)
    logp = np.log(p)
    logp -= logp.mean(axis=0, keepdims=True)
    try:
        u, _s, _vt = np.linalg.svd(logp, full_matrices=False)
        theta = u[:, 0]
    except np.linalg.LinAlgError:
        theta = np.zeros(d)
    return _standardise(theta)


def _standardise(theta: np.ndarray) -> np.ndarray:
    mean = theta.mean()
    sd = theta.std()
    if sd < 1e-12:
        return theta - mean
    return (theta - mean) / sd


@dataclass
class WordfishFit:
    """Fitted Wordfish parameters for one axis."""

    theta: np.ndarray   # [n_docs] standardised document positions (in-sample)
    alpha: np.ndarray   # [n_docs] document loquaciousness
    psi: np.ndarray     # [n_words] word frequency intercepts
    beta: np.ndarray    # [n_words] word discriminations
    converged: bool
    n_sweeps: int


def fit_wordfish(
    Y: np.ndarray,
    max_sweeps: int = _FIT_MAX_SWEEPS,
    tol: float = _FIT_TOL,
) -> WordfishFit:
    """Fit Wordfish on a document-feature count matrix by alternating Newton.

    ``Y`` is ``[n_docs, n_words]`` non-negative integer counts. Returns a
    :class:`WordfishFit`; theta is standardised (mean 0 / sd 1) but its **sign is
    arbitrary** — call :func:`orient_axis` to fix it. Word parameters are solved
    first each sweep (so beta moves off its zero init given the SVD theta), then
    the document parameters, then theta is re-standardised (folding the location
    shift into psi and the scale into beta so the linear predictor is preserved).
    """
    Y = np.asarray(Y, dtype=float)
    d, w = Y.shape
    rowsum = Y.sum(axis=1)
    colsum = Y.sum(axis=0)

    alpha = np.log(rowsum + 0.5)
    alpha -= alpha.mean()
    psi = np.log(colsum + 0.5) - math.log(max(d, 1))
    beta = np.zeros(w)
    theta = _svd_init_theta(Y)

    converged = False
    sweep = 0
    for sweep in range(1, max_sweeps + 1):
        theta_prev = theta.copy()

        # Word block: for each word, solve (psi_j, beta_j) given theta, alpha.
        for j in range(w):
            psi[j], beta[j] = _newton_2param(
                Y[:, j], offset=alpha, x1=theta, b0=psi[j], b1=beta[j]
            )

        # Document block: for each doc, solve (alpha_i, theta_i) given psi, beta.
        for i in range(d):
            alpha[i], theta[i] = _newton_2param(
                Y[i, :], offset=psi, x1=beta, b0=alpha[i], b1=theta[i]
            )

        # Re-standardise theta, preserving the linear predictor:
        #   theta' = (theta-mu)/sd ; beta' = beta*sd ; psi' = psi + beta*mu
        mu_t = theta.mean()
        sd_t = theta.std()
        if sd_t > 1e-12:
            psi = psi + beta * mu_t
            beta = beta * sd_t
            theta = (theta - mu_t) / sd_t

        if np.max(np.abs(theta - theta_prev)) < tol:
            converged = True
            break

    return WordfishFit(theta=theta, alpha=alpha, psi=psi, beta=beta,
                       converged=converged, n_sweeps=sweep)


def orient_axis(
    theta: np.ndarray,
    beta: np.ndarray,
    plus_mask: np.ndarray,
    minus_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Fix Wordfish's unidentified sign so the plus pole sits at theta > 0.

    Deterministic identification constraint: orient the axis so the mean
    position of the plus-pole seeds exceeds that of the minus-pole seeds. Uses
    gold labels for the *sign only* (the fit itself is label-free). Returns
    ``(theta, beta, flipped)``.
    """
    if plus_mask.any() and minus_mask.any():
        if theta[plus_mask].mean() < theta[minus_mask].mean():
            return -theta, -beta, True
    return theta, beta, False


def project_theta(
    counts: np.ndarray,    # [n_words] count vector over the fitted vocab
    psi: np.ndarray,
    beta: np.ndarray,
    n_iter: int = _PROJ_MAX_ITER,
) -> float:
    """Position an out-of-sample document on a fitted axis (fixed psi, beta).

    Estimates only ``(alpha, theta)`` for the new document by the same
    2-parameter Poisson Newton solve, holding the word parameters fixed — the
    standard Wordfish out-of-sample projection. Returns theta on the fitted,
    oriented scale. (Words absent from the fitted vocab carry no signal, exactly
    like an out-of-lexicon token in Wordscores.)
    """
    counts = np.asarray(counts, dtype=float)
    alpha = math.log(counts.sum() + 0.5)
    theta = 0.0
    alpha, theta = _newton_2param(
        counts, offset=psi, x1=beta, b0=alpha, b1=theta, n_iter=n_iter
    )
    return theta


def fold_pole_2d(
    theta_hs_t: float,
    theta_l_g: float,
) -> str:
    """Quadrant fold: the axis with the larger |theta| picks the pole, signed.

    HS↔T: theta>0 -> terrestre, <0 -> hors_sol. L↔G: theta>0 -> global,
    <0 -> local. Mirrors ``compare.predict_pole`` extended to two axes.
    """
    if abs(theta_hs_t) >= abs(theta_l_g):
        return "terrestre" if theta_hs_t > 0 else "hors_sol"
    return "global" if theta_l_g > 0 else "local"


def bootstrap_accuracy(
    correct: List[bool],
    n_resamples: int = _ACC_BOOTSTRAP_N,
    seed: int = _ACC_BOOTSTRAP_SEED,
) -> Tuple[Optional[float], Optional[float]]:
    """Bootstrap 95% CI on a mean-accuracy by resampling texts with replacement."""
    if not correct:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.asarray(correct, dtype=float)
    draws = arr[rng.integers(0, len(arr), size=(n_resamples, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(draws, _ACC_BOOTSTRAP_PCT)
    return float(lo), float(hi)


# ===========================================================================
# Corpus + spaCy layer
# ===========================================================================


@dataclass
class TextScore:
    """A single text's Wordfish positions and prediction."""

    slug: str
    role: Optional[str]            # SEED / INVERSION-HOLD / TEST / SANITY
    purete: Optional[str]
    gold_pole: Optional[str]       # folded gold, or None (bipolar/off-axis)
    theta_hs_t: float
    theta_l_g: float
    hits_hs_t: int                 # in-vocab lemmas on the HS↔T fit
    hits_l_g: int

    @property
    def predicted_pole(self) -> str:
        return fold_pole_2d(self.theta_hs_t, self.theta_l_g)

    @property
    def correct(self) -> Optional[bool]:
        if self.gold_pole is None:
            return None
        return self.predicted_pole == self.gold_pole


def _purete_of(entry: dict) -> Optional[str]:
    p = audit_block(entry).get("purete")
    return None if p is None else str(p)


def _build_vocab(
    seed_counts: List[Counter], min_doc_freq: int
) -> List[TokenKey]:
    """Vocabulary = lemmas appearing in >= min_doc_freq of the seed documents."""
    doc_freq: Counter = Counter()
    for counts in seed_counts:
        for key in counts:
            doc_freq[key] += 1
    vocab = sorted(k for k, df in doc_freq.items() if df >= min_doc_freq)
    return vocab


def _counts_vector(counts: Counter, vocab_index: Dict[TokenKey, int]) -> np.ndarray:
    vec = np.zeros(len(vocab_index), dtype=float)
    for key, c in counts.items():
        idx = vocab_index.get(key)
        if idx is not None:
            vec[idx] = c
    return vec


@dataclass
class AxisRun:
    axis: str
    vocab_size: int
    converged: bool
    n_sweeps: int
    flipped: bool
    seed_correct: int
    seed_total: int
    theta_by_slug: Dict[str, float]
    hits_by_slug: Dict[str, int]


def run_axis(
    axis: str,
    entries: List[dict],
    counts_by_slug: Dict[str, Counter],
    min_doc_freq: int,
) -> AxisRun:
    """Fit Wordfish on one axis's seeds and position every text on it.

    Seeds (``use_for_seeds: true`` whose gold is one of this axis's two poles)
    define the vocabulary and the fit. Every text — seeds-on-axis (in-sample),
    plus all other texts (projected with fixed word params) — gets a theta on
    the fitted, oriented scale.
    """
    minus_pole, plus_pole = SUPPORTED_AXES[axis]
    on_axis = {minus_pole, plus_pole}

    seed_entries = [
        e
        for e in entries
        if audit_block(e).get("use_for_seeds") is True
        and e["expected_pole"] in on_axis
    ]
    seed_counts = [counts_by_slug[e["slug"]] for e in seed_entries]
    vocab = _build_vocab(seed_counts, min_doc_freq)
    if not vocab:
        raise RuntimeError(f"empty vocabulary on axis {axis} (min_doc_freq too high?)")
    vocab_index = {k: i for i, k in enumerate(vocab)}

    Y = np.vstack([_counts_vector(c, vocab_index) for c in seed_counts])
    fit = fit_wordfish(Y)

    golds = np.array([e["expected_pole"] for e in seed_entries])
    plus_mask = golds == plus_pole
    minus_mask = golds == minus_pole
    theta, beta, flipped = orient_axis(fit.theta, fit.beta, plus_mask, minus_mask)

    theta_by_slug: Dict[str, float] = {}
    hits_by_slug: Dict[str, int] = {}
    seed_slugs = set()
    for e, th in zip(seed_entries, theta):
        theta_by_slug[e["slug"]] = float(th)
        hits_by_slug[e["slug"]] = int((_counts_vector(counts_by_slug[e["slug"]], vocab_index) > 0).sum())
        seed_slugs.add(e["slug"])

    # Project every other text onto this axis with the fitted word params.
    for e in entries:
        slug = e["slug"]
        if slug in seed_slugs:
            continue
        vec = _counts_vector(counts_by_slug[slug], vocab_index)
        theta_by_slug[slug] = project_theta(vec, fit.psi, beta)
        hits_by_slug[slug] = int((vec > 0).sum())

    # In-sample seed accuracy on this 1-D axis (sign of theta vs gold side).
    seed_correct = seed_total = 0
    for e, th in zip(seed_entries, theta):
        seed_total += 1
        pred_side = plus_pole if th > 0 else minus_pole
        if pred_side == e["expected_pole"]:
            seed_correct += 1

    return AxisRun(
        axis=axis,
        vocab_size=len(vocab),
        converged=fit.converged,
        n_sweeps=fit.n_sweeps,
        flipped=flipped,
        seed_correct=seed_correct,
        seed_total=seed_total,
        theta_by_slug=theta_by_slug,
        hits_by_slug=hits_by_slug,
    )


def score_all(min_doc_freq: int) -> Tuple[List[TextScore], AxisRun, AxisRun]:
    """Load the corpus, fit both axes, and fold every text to a pole."""
    nlp = _load_spacy()

    # Root corpus (seeds + INVERSION-HOLD) + the tests/ hold-out (gold may be
    # null for the bipolar rows -> require_pole=False).
    root = load_corpus(corpus_dir())
    holdout = load_corpus(corpus_dir() / "tests", require_pole=False)
    entries = root + holdout

    counts_by_slug: Dict[str, Counter] = {}
    for e in entries:
        counts_by_slug[e["slug"]] = Counter(lemmatize(nlp, e["body"]))

    run_hs_t = run_axis(_AXIS_HS_T, entries, counts_by_slug, min_doc_freq)
    run_l_g = run_axis(_AXIS_L_G, entries, counts_by_slug, min_doc_freq)

    scores: List[TextScore] = []
    for e in entries:
        slug = e["slug"]
        scores.append(
            TextScore(
                slug=slug,
                role=role_of(e),
                purete=_purete_of(e),
                gold_pole=e["expected_pole"],
                theta_hs_t=run_hs_t.theta_by_slug[slug],
                theta_l_g=run_l_g.theta_by_slug[slug],
                hits_hs_t=run_hs_t.hits_by_slug[slug],
                hits_l_g=run_l_g.hits_by_slug[slug],
            )
        )
    return scores, run_hs_t, run_l_g


# ===========================================================================
# Cross-method predictions (for the shared comparison table)
# ===========================================================================


def load_wordscores_pred(axes_dir: Path) -> Dict[str, str]:
    """Wordscores 4-pole prediction by the same quadrant fold, from the two
    per-axis ``comparison.csv`` files (full slugs, LBG ``axis_score``).

    Both LBG axes are bounded [-1, +1], so |score| is comparable across axes
    (a documented caveat: the magnitudes are not identically scaled the way the
    standardised Wordfish thetas are).
    """
    scores: Dict[str, Dict[str, float]] = {}
    for axis, key in ((_AXIS_HS_T, "hs_t"), (_AXIS_L_G, "l_g")):
        path = axes_dir / axis / "comparison.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    val = float(row["axis_score"])
                except (KeyError, ValueError):
                    continue
                scores.setdefault(row["slug"], {})[key] = val
    preds: Dict[str, str] = {}
    for slug, ax in scores.items():
        hs_t = ax.get("hs_t", 0.0)
        l_g = ax.get("l_g", 0.0)
        if hs_t == 0.0 and l_g == 0.0:
            continue
        preds[slug] = fold_pole_2d(hs_t, l_g)
    return preds


def load_cosnli_pred(results_table: Path, slugs: List[str]) -> Dict[str, str]:
    """Best-effort cosNLI v4 per-text prediction from RESULTS_TABLE_20260625.md.

    The table's slug column is truncated; we prefix-match each truncated slug to
    the real slug list (slugs are unique by their numbered prefix). Ambiguous or
    unmatched rows are skipped — the report records the coverage.
    """
    preds: Dict[str, str] = {}
    if not results_table.exists():
        return preds
    valid_poles = {"terrestre", "global", "hors_sol", "local"}
    for line in results_table.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9:
            continue
        table_slug = cells[0].strip("` ")
        pred = cells[8].strip()
        if pred not in valid_poles:
            continue
        matches = [s for s in slugs if s.startswith(table_slug)]
        if len(matches) == 1:
            preds[matches[0]] = pred
    return preds


# ===========================================================================
# Writers
# ===========================================================================

_AXIS_CSV_FIELDS = ["slug", "gold_pole", "audit_role", "theta", "predicted_side", "correct"]
_2D_CSV_FIELDS = [
    "slug", "audit_role", "purete", "gold_pole",
    "theta_hs_t", "theta_l_g", "hits_hs_t", "hits_l_g",
    "predicted_pole", "correct",
]


def _fmt_correct(correct: Optional[bool]) -> str:
    return "" if correct is None else ("1" if correct else "0")


def write_axis_csv(scores: List[TextScore], axis: str, path: Path) -> None:
    """Per-axis 1-D view: theta on this axis + the signed side it implies."""
    minus_pole, plus_pole = SUPPORTED_AXES[axis]
    on_axis = {minus_pole, plus_pole}
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_AXIS_CSV_FIELDS)
        writer.writeheader()
        for s in sorted(scores, key=lambda x: x.slug):
            theta = s.theta_hs_t if axis == _AXIS_HS_T else s.theta_l_g
            side = plus_pole if theta > 0 else minus_pole
            # graded only when the gold is on this 1-D axis
            correct = None
            if s.gold_pole in on_axis:
                correct = side == s.gold_pole
            writer.writerow({
                "slug": s.slug,
                "gold_pole": s.gold_pole or "",
                "audit_role": s.role or "",
                "theta": round(theta, 6),
                "predicted_side": side,
                "correct": _fmt_correct(correct),
            })


def write_2d_summary_csv(scores: List[TextScore], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_2D_CSV_FIELDS)
        writer.writeheader()
        for s in sorted(scores, key=lambda x: x.slug):
            writer.writerow({
                "slug": s.slug,
                "audit_role": s.role or "",
                "purete": s.purete or "",
                "gold_pole": s.gold_pole or "",
                "theta_hs_t": round(s.theta_hs_t, 6),
                "theta_l_g": round(s.theta_l_g, 6),
                "hits_hs_t": s.hits_hs_t,
                "hits_l_g": s.hits_l_g,
                "predicted_pole": s.predicted_pole,
                "correct": _fmt_correct(s.correct),
            })


def _accuracy(scores: List[TextScore]) -> Tuple[int, int]:
    graded = [s for s in scores if s.correct is not None]
    return sum(1 for s in graded if s.correct), len(graded)


def _acc_line(scores: List[TextScore]) -> str:
    ok, tot = _accuracy(scores)
    pct = f"{100 * ok / tot:.1f}%" if tot else "n/a"
    lo, hi = bootstrap_accuracy([bool(s.correct) for s in scores if s.correct is not None])
    ci = f" [{100*lo:.0f}, {100*hi:.0f}]" if lo is not None else ""
    return f"{ok}/{tot} ({pct}){ci}"


def build_report(
    scores: List[TextScore],
    run_hs_t: AxisRun,
    run_l_g: AxisRun,
    wordscores_pred: Dict[str, str],
) -> str:
    """Markdown report: per-axis accuracy + the inversion breakdown (the payload)."""
    by_role = lambda role: [s for s in scores if s.role == role]
    seeds = [s for s in scores if s.role == "SEED"]
    inversions = by_role(ROLE_INVERSION_HOLD)
    holdout = [s for s in scores if s.role in ("TEST", "SANITY")]

    # Inversion breakdown: does Wordfish break on the same texts a lexical
    # scaler (Wordscores) does? Both are distributional -> expected to co-fail.
    inv_lines = [
        "| Text | Gold | Wordfish | Wordscores | Wordfish outcome |",
        "|---|---|---|---|---|",
    ]
    inv_wrong = inv_gradable = 0
    co_fail = 0
    for s in sorted(inversions, key=lambda x: x.slug):
        ws = wordscores_pred.get(s.slug, "—")
        if s.correct is None:
            outcome = "off-axis gold — reported, not graded"
        else:
            inv_gradable += 1
            if s.correct:
                outcome = "✅ correct (rare for a distributional scaler)"
            else:
                inv_wrong += 1
                outcome = "❌ **misclassified** — reads the opposing lexicon"
                if ws != "—" and ws != s.gold_pole:
                    co_fail += 1
        inv_lines.append(
            f"| `{s.slug}` | {s.gold_pole or '—'} | {s.predicted_pole} | {ws} | {outcome} |"
        )
    inv_table = "\n".join(inv_lines)

    def axis_meta(run: AxisRun) -> str:
        conv = "converged" if run.converged else "**did not converge**"
        flip = " (sign flipped to match Latour convention)" if run.flipped else ""
        return (
            f"vocab {run.vocab_size} lemmas · {conv} in {run.n_sweeps} sweeps{flip} · "
            f"in-sample seeds {run.seed_correct}/{run.seed_total}"
        )

    return f"""# Latouromètre — Wordfish baseline (unsupervised positional scaling)

Wordfish (Slapin & Proksch 2008) fit from scratch (alternating-Newton Poisson
IRT, pure NumPy, CPU). Two independent 1-D fits — Hors-Sol↔Terrestre and
Local↔Global — folded to a pole by the larger |θ|, signed (the same quadrant
rule the Wordscores baseline uses). The axis sign is fixed deterministically by
orienting each fit so its plus-pole seeds sit at θ > 0.

## Fit summary

- **Hors-Sol↔Terrestre** — {axis_meta(run_hs_t)}.
- **Local↔Global** — {axis_meta(run_l_g)}.

## Accuracy (2-D quadrant fold, gold = `audit.attracteur`)

| Bucket | Wordfish accuracy |
|---|---|
| **Seeds** (in-sample, circular) | {_acc_line(seeds)} |
| **Inversions** (`INVERSION-HOLD`) | {_acc_line(inversions)} |
| **Hold-out** (`tests/`: TEST+SANITY) | {_acc_line(holdout)} |
| **All gradable** | {_acc_line(scores)} |

Accuracy CIs are bootstrap 95% (resample texts, 1000 draws). Seeds are
in-sample for Wordfish (it is *fit* on them — unlike a zero-shot LLM), so the
hold-out line is the real generalisation signal.

## Inversion stress test (the scientific payload)

The `INVERSION-HOLD` texts argue *for* a pole while speaking the *opposing*
pole's vocabulary (a climate-denier writing dense ecological prose, etc.). A
purely distributional method reads topic, not stance, so it is **expected to
misclassify them** — the same break the unblended cosine shows, and the reason
the production pipeline adds an NLI stance layer.

{inv_table}

**Wordfish misclassifies {inv_wrong}/{inv_gradable} gradable inversions**; of
those, **{co_fail}** are *also* misclassified by Wordscores (the lexical scaler)
— direct evidence that the two distributional methods break on the *same* texts.

## Method note

- Preprocessing is byte-identical to the Wordscores baseline (spaCy
  `fr_core_news_lg`, content-POS lemmas, spaCy French stopwords) — the two
  methods differ only in the scaling model, never in the features.
- Min-doc-frequency cutoff applied to each axis's seed vocabulary.
- Out-of-sample texts (inversions, hold-out) are positioned by the standard
  Wordfish projection: estimate `(α, θ)` for the new document holding the fitted
  word parameters `(ψ, β)` fixed.
"""


# --- shared comparison table (additive merge) ------------------------------

_COMPARISON_FIELDS = [
    "slug", "gold_pole", "audit_role", "purete",
    "cosNLI_pred", "wordscores_pred", "mistral_pred", "wordfish_pred",
    "wordfish_correct",
]


def merge_comparison(
    scores: List[TextScore],
    wordscores_pred: Dict[str, str],
    cosnli_pred: Dict[str, str],
    csv_path: Path,
    md_path: Path,
) -> None:
    """Add/refresh the ``wordfish_pred`` column in the shared comparison table.

    If the CSV already exists (e.g. PRD-8/Mistral ran first), its rows and any
    other method columns are preserved; we only set this run's columns. cosNLI
    and Wordscores predictions are (re)filled from the artifacts we can read;
    ``mistral_pred`` is left to PRD-8.
    """
    existing: Dict[str, Dict[str, str]] = {}
    fieldnames = list(_COMPARISON_FIELDS)
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for fn in reader.fieldnames or []:
                if fn not in fieldnames:
                    fieldnames.append(fn)
            for row in reader:
                existing[row["slug"]] = dict(row)

    for s in scores:
        row = existing.get(s.slug, {})
        row["slug"] = s.slug
        row["gold_pole"] = s.gold_pole or ""
        row["audit_role"] = s.role or ""
        row["purete"] = s.purete or ""
        row["wordfish_pred"] = s.predicted_pole
        row["wordfish_correct"] = _fmt_correct(s.correct)
        # (re)fill the cross-method columns we can read; never clobber mistral.
        if s.slug in cosnli_pred:
            row["cosNLI_pred"] = cosnli_pred[s.slug]
        elif "cosNLI_pred" not in row:
            row["cosNLI_pred"] = ""
        if s.slug in wordscores_pred:
            row["wordscores_pred"] = wordscores_pred[s.slug]
        elif "wordscores_pred" not in row:
            row["wordscores_pred"] = ""
        row.setdefault("mistral_pred", "")
        existing[s.slug] = row

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for slug in sorted(existing):
            writer.writerow(existing[slug])

    _write_comparison_md(existing, fieldnames, md_path)


def _method_accuracy(rows: List[Dict[str, str]], pred_col: str) -> str:
    graded = [r for r in rows if r.get("gold_pole") and r.get(pred_col)]
    if not graded:
        return "n/a"
    ok = sum(1 for r in graded if r[pred_col] == r["gold_pole"])
    pct = f"{100 * ok / len(graded):.1f}%"
    lo, hi = bootstrap_accuracy([r[pred_col] == r["gold_pole"] for r in graded])
    ci = f" [{100*lo:.0f}, {100*hi:.0f}]" if lo is not None else ""
    return f"{ok}/{len(graded)} ({pct}){ci}"


def _write_comparison_md(
    rows_by_slug: Dict[str, Dict[str, str]], fieldnames: List[str], path: Path
) -> None:
    rows = [rows_by_slug[s] for s in sorted(rows_by_slug)]
    methods = [c for c in ("cosNLI_pred", "wordscores_pred", "mistral_pred", "wordfish_pred")
               if any(r.get(c) for r in rows)]

    lines = [
        "# Latouromètre — baseline comparison (shared table)",
        "",
        "Per-text 4-pole predictions, gold = `audit.attracteur`. Methods: "
        "cosine+NLI v4 (prod), Wordscores (PRD-3), Mistral zero-shot (PRD-8), "
        "Wordfish (PRD-9). Accuracy CIs are bootstrap 95% (resample texts).",
        "",
        "## Overall accuracy",
        "",
        "| Method | All gradable | Hold-out (TEST+SANITY) |",
        "|---|---|---|",
    ]
    holdout = [r for r in rows if r.get("audit_role") in ("TEST", "SANITY")]
    for m in methods:
        lines.append(
            f"| {m.replace('_pred','')} | {_method_accuracy(rows, m)} "
            f"| {_method_accuracy(holdout, m)} |"
        )

    # Per-pole and per-purete accuracy for Wordfish (the method this run owns).
    lines += ["", "## Wordfish accuracy by gold pole", "", "| Pole | Accuracy |", "|---|---|"]
    for pole in ("hors_sol", "terrestre", "local", "global"):
        sub = [r for r in rows if r.get("gold_pole") == pole]
        lines.append(f"| {pole} | {_method_accuracy(sub, 'wordfish_pred')} |")

    lines += ["", "## Wordfish accuracy by purity", "", "| Purity | Accuracy |", "|---|---|"]
    puretes = sorted({r.get("purete", "") for r in rows if r.get("purete")})
    for p in puretes:
        sub = [r for r in rows if r.get("purete") == p]
        lines.append(f"| {p} | {_method_accuracy(sub, 'wordfish_pred')} |")

    # Per-text table.
    head = "| slug | gold | role | " + " | ".join(m.replace("_pred", "") for m in methods) + " |"
    sep = "|---|---|---|" + "---|" * len(methods)
    lines += ["", "## Per-text", "", head, sep]
    for r in rows:
        cells = [f"`{r['slug']}`", r.get("gold_pole") or "—", r.get("audit_role") or "—"]
        cells += [r.get(m) or "—" for m in methods]
        lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===========================================================================
# CLI
# ===========================================================================


def run(min_doc_freq: int = _DEFAULT_MIN_DOC_FREQ, dry_run: bool = False) -> int:
    scores, run_hs_t, run_l_g = score_all(min_doc_freq)

    axes_dir = corpus_base() / "axes"
    wordscores_pred = load_wordscores_pred(axes_dir)
    cosnli_pred = load_cosnli_pred(
        corpus_base() / "calibration_latourometre" / "RESULTS_TABLE_20260625.md",
        [s.slug for s in scores],
    )

    seeds = [s for s in scores if s.role == "SEED"]
    inversions = [s for s in scores if s.role == ROLE_INVERSION_HOLD]
    holdout = [s for s in scores if s.role in ("TEST", "SANITY")]

    print("\nWordfish baseline")
    print(f"  texts scored:        {len(scores)}")
    print(f"  HS↔T fit:            vocab {run_hs_t.vocab_size}, "
          f"{'converged' if run_hs_t.converged else 'NO CONVERGE'} ({run_hs_t.n_sweeps} sweeps), "
          f"seeds {run_hs_t.seed_correct}/{run_hs_t.seed_total}")
    print(f"  L↔G fit:             vocab {run_l_g.vocab_size}, "
          f"{'converged' if run_l_g.converged else 'NO CONVERGE'} ({run_l_g.n_sweeps} sweeps), "
          f"seeds {run_l_g.seed_correct}/{run_l_g.seed_total}")
    print(f"  seed acc (2-D):      {_acc_line(seeds)}")
    print(f"  inversion acc:       {_acc_line(inversions)} (expected LOW — the payload)")
    print(f"  hold-out acc:        {_acc_line(holdout)}")
    print(f"  all gradable:        {_acc_line(scores)}")
    print(f"  wordscores preds:    {len(wordscores_pred)} | cosNLI preds: {len(cosnli_pred)}")

    if dry_run:
        print("\n(dry-run: nothing written)")
        return 0

    out = wordfish_dir()
    out.mkdir(parents=True, exist_ok=True)
    write_axis_csv(scores, _AXIS_HS_T, out / "wordfish_hors-sol-terrestre.csv")
    write_axis_csv(scores, _AXIS_L_G, out / "wordfish_local-global.csv")
    write_2d_summary_csv(scores, out / "wordfish_2d_summary.csv")
    (out / "wordfish-report.md").write_text(
        build_report(scores, run_hs_t, run_l_g, wordscores_pred), encoding="utf-8"
    )
    merge_comparison(
        scores, wordscores_pred, cosnli_pred,
        baselines_dir() / "comparison.csv",
        baselines_dir() / "comparison.md",
    )
    print(f"\nWrote {out}/ (axis CSVs, 2-D summary, report)")
    print(f"Wrote {baselines_dir()}/comparison.{{csv,md}}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-doc-freq",
        type=int,
        default=_DEFAULT_MIN_DOC_FREQ,
        help="drop lemmas appearing in fewer than this many seed docs (default 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fit + print stats without writing any artifact",
    )
    args = parser.parse_args(argv)
    return run(min_doc_freq=args.min_doc_freq, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
