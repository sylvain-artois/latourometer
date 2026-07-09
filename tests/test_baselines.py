"""The Wordscores benchmark regenerates a comparison table offline.

Runs against the tiny synthetic fixture corpus (CC0, committed under
``tests/fixtures``) so it needs no corpus mount and no license-gated prose.
Skipped when spaCy / the FR model are unavailable.
"""

from __future__ import annotations

import pathlib

import pytest

FIXTURE_CORPUS = pathlib.Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture()
def corpus_env(tmp_path, monkeypatch):
    pytest.importorskip("spacy")
    pytest.importorskip("pyarrow")
    pytest.importorskip("frontmatter")
    # The baselines read CORPUS_BASE_PATH/corpus_latourometre and write under
    # CORPUS_BASE_PATH/axes — point reads at the fixture, writes at a tmp copy.
    import shutil

    base = tmp_path / "corpus"
    shutil.copytree(FIXTURE_CORPUS, base)
    monkeypatch.setenv("CORPUS_BASE_PATH", str(base))
    return base


def _skip_if_no_model(func):
    try:
        return func()
    except OSError as exc:
        pytest.skip(str(exc))


def test_wordscores_calibrate_and_compare(corpus_env):
    from latourometer.baselines import calibrate, compare

    rc = _skip_if_no_model(
        lambda: calibrate.calibrate(
            "hors-sol-terrestre", bootstrap_n=0, min_docs_per_pole=1
        )
    )
    assert rc == 0
    lexicon = corpus_env / "axes" / "hors-sol-terrestre" / "lexicon.csv"
    assert lexicon.exists()

    rc = compare.compare("hors-sol-terrestre")
    assert rc == 0
    report = corpus_env / "axes" / "hors-sol-terrestre" / "comparison-report.md"
    assert report.exists()
    assert "Wordscores" in report.read_text(encoding="utf-8")
