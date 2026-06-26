"""Shared corpus loader for Latouromètre calibration scripts.

Two on-disk header formats are supported:

1. Standard ``---``-delimited YAML frontmatter (the migration target)::

    ---
    title: …
    attracteur: global
    license:
      spdx: CC-BY-4.0
      redistribution: allowed
    ---
    # Title
    Body text…

2. The legacy flat-YAML header (no delimiters), kept for unmigrated files::

    source: <url>
    attracteur: <pole-label>
    date: <date>
    auteur:  <author>

    # Title
    Body text…

A file is parsed as frontmatter when its first (BOM-stripped) line is ``---``;
otherwise the legacy parser runs and a single ``WARNING`` names the unmigrated
file. The ``attracteur`` field uses extended labels (``local+``, ``global-``,
etc.); ``fold_pole`` maps them to the four canonical pole keys used by
``LatourometreChart`` and ``LatourometreMetric``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import frontmatter

logger = logging.getLogger(__name__)

# UTF-8 byte-order mark, stripped before header detection.
_BOM = "﻿"

# Maps every attractor label variant to the four canonical pole keys.
_POLE_FOLD: Dict[str, str] = {
    "local+": "local",
    "local-": "local",
    "local": "local",
    "global+": "global",
    "global-": "global",
    "global": "global",
    "horssol": "hors_sol",
    "hors-sol": "hors_sol",
    "hors_sol": "hors_sol",
    "terrestre": "terrestre",
}

# Canonical display labels used by LatourometreChart.
POLE_LABELS: Dict[str, str] = {
    "terrestre": "Terrestre",
    "global": "Global",
    "hors_sol": "Hors-Sol",
    "local": "Local",
}

# Hard-coded pole order required by LatourometreChart.render.
POLE_ORDER = ("terrestre", "global", "hors_sol", "local")


def fold_pole(raw: str) -> str:
    key = raw.strip().lower()
    pole = _POLE_FOLD.get(key)
    if pole is None:
        raise ValueError(f"Unknown attractor label: {raw!r}")
    return pole


def fold_pole_or_none(attracteur: Any) -> str | None:
    """Fold an attractor label to a canonical 4-pole gold key, or ``None``.

    Returns ``None`` for the deliberately-ambiguous bipolar test rows
    (top-level ``attracteur: null`` / ``cross:x/y``) and for any
    unrecognised label, instead of raising the way ``fold_pole`` does.
    """
    if attracteur is None:
        return None
    return _POLE_FOLD.get(str(attracteur).strip().lower())


def _split_header_body(raw: str) -> Tuple[Dict[str, str], str]:
    lines = raw.splitlines()
    header_end: int | None = None
    seen_auteur = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("auteur:"):
            seen_auteur = True
            continue
        if seen_auteur and stripped == "":
            header_end = i
            break
    if header_end is None:
        for i, line in enumerate(lines):
            if line.strip() == "":
                header_end = i
                break
    if header_end is None:
        raise RuntimeError("No header/body boundary found (no blank line)")

    header: Dict[str, str] = {}
    for line in lines[:header_end]:
        if ":" in line:
            k, _, v = line.partition(":")
            header[k.strip().lower()] = v.strip()

    body = "\n".join(lines[header_end + 1 :]).lstrip("\n")
    return header, body


def _parse_frontmatter(raw: str) -> Tuple[Dict[str, Any], str]:
    """Parse standard ``---``-delimited YAML frontmatter.

    Returns the metadata dict (nested ``license`` / ``ingestion`` blocks pass
    through untouched) and the body. ``frontmatter`` strips a single trailing
    newline from the content, matching the legacy parser's ``lstrip``-only
    body handling closely enough that calibration scores are unaffected.
    """
    post = frontmatter.loads(raw)
    return dict(post.metadata), post.content


def parse_corpus_file(path: Path, *, require_pole: bool = True) -> Dict[str, Any]:
    """Parse a single corpus file into a structured dict.

    Routes on the first BOM-stripped line: ``---`` selects the standard
    frontmatter parser, anything else falls back to the legacy flat-YAML
    parser with a one-line WARNING naming the unmigrated file.

    With ``require_pole=True`` (default, preserving the legacy contract) a
    null / empty / ``cross:`` / unparseable ``attracteur`` raises via
    ``fold_pole``. With ``require_pole=False`` it yields
    ``expected_pole = None`` instead — used to load the eval set, which
    contains the bipolar (``attracteur: null``) test rows.
    """
    raw = path.read_text(encoding="utf-8").lstrip(_BOM)
    first_line = raw.split("\n", 1)[0].strip()
    if first_line == "---":
        header, body = _parse_frontmatter(raw)
    else:
        logger.warning(
            "Corpus file uses the legacy flat-YAML header (not migrated to "
            "`---` frontmatter): %s",
            path,
        )
        header, body = _split_header_body(raw)
    raw_attracteur = header.get("attracteur")
    attracteur_raw = "" if raw_attracteur is None else str(raw_attracteur)
    if require_pole:
        expected_pole: str | None = fold_pole(attracteur_raw)
    else:
        expected_pole = fold_pole_or_none(raw_attracteur)
    return {
        "path": path,
        "slug": path.stem,
        "header": header,
        "body": body,
        "attracteur_raw": attracteur_raw,
        "expected_pole": expected_pole,
    }


def load_corpus(corpus_dir: Path, *, require_pole: bool = True) -> List[Dict[str, Any]]:
    """Load all valid corpus files from a directory (non-recursive).

    Skips macOS resource forks (``._*``) and English-original backups
    (``*.en.md``). ``require_pole`` is forwarded to ``parse_corpus_file``;
    the default ``True`` preserves the strict legacy behaviour.

    NOTE: this is a deliberate non-recursive ``glob`` — never switch to
    ``rglob`` or the pools get poisoned by ``_source_artifacts/`` and
    ``_pre_license_backup_STORY182/``.
    """
    entries = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        if path.name.startswith("._") or path.name.endswith(".en.md"):
            continue
        entries.append(parse_corpus_file(path, require_pole=require_pole))
    return entries


# ---------------------------------------------------------------------------
# Audit-flag accessors and calibration pools (Phase 0 anti-circularity wiring)
# ---------------------------------------------------------------------------

# On-disk roles (EXACT casing) from the ``audit:`` frontmatter block.
ROLE_SEED = "SEED"
ROLE_INVERSION_HOLD = "INVERSION-HOLD"
ROLE_TEST = "TEST"
ROLE_SANITY = "SANITY"


def audit_block(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ``audit:`` block of a parsed entry (``{}`` if absent)."""
    return entry["header"].get("audit") or {}


def role_of(entry: Dict[str, Any]) -> str | None:
    """Return the (normalised) ``audit.role`` of a parsed entry, or ``None``.

    Normalises with ``.strip().upper()`` as insurance, but the on-disk
    values already use exact casing (``SEED`` / ``INVERSION-HOLD`` / ``TEST``
    / ``SANITY``).
    """
    role = audit_block(entry).get("role")
    if role is None:
        return None
    return str(role).strip().upper()


def seed_pool(corpus_dir: Path) -> List[Dict[str, Any]]:
    """The extraction / centroid basin: only ``audit.use_for_seeds: true``.

    Auto-excludes the INVERSION-HOLD inversions and the whole tests/ dir.
    """
    return [
        e
        for e in load_corpus(corpus_dir)
        if audit_block(e).get("use_for_seeds") is True
    ]


def eval_pool(root_dir) -> List[Dict[str, Any]]:
    """The out-of-sample evaluation set (SANITY + TEST + INVERSION-HOLD).

    ``tests/`` (SANITY + TEST) is loaded with ``require_pole=False`` so the
    bipolar ``attracteur: null`` rows survive (``expected_pole=None``). The
    8 root INVERSION-HOLD texts (held out of the seed basin) are appended.
    Asserts that NOT ONE entry carries ``use_for_seeds: true``.
    """
    root_path = Path(root_dir)
    pool = load_corpus(root_path / "tests", require_pole=False)
    pool += [
        e
        for e in load_corpus(root_path, require_pole=False)
        if role_of(e) == ROLE_INVERSION_HOLD
    ]
    assert all(
        audit_block(e).get("use_for_seeds") is not True for e in pool
    ), "eval_pool leak: an entry has use_for_seeds:true"
    return pool
