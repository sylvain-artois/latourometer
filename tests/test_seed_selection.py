"""Audit-flag seed selection regression tests (Bruckner leak guard).

The bug these guard against: ``inversion_hors-sol_001_pascal-bruckner`` carries
``attracteur: hors_sol`` and ``sub_pole: null`` but ``audit.role: INVERSION-HOLD``
/ ``audit.use_for_seeds: false``. The old ``sub_pole == "inversion-stance"``
filter MISSED him, so he was counted as a Hors-Sol *seed* (training-set
poisoning). The fix routes selection through the shared ``_corpus_loader``
audit-flag helpers (``seed_pool`` / ``role_of`` / ``ROLE_INVERSION_HOLD``).

Two layers of test:

1. A self-contained **fixture** mirroring the on-disk frontmatter shape, so the
   selection contract is asserted with no spaCy model and no samba mount.
2. A **live-corpus** test that runs only inside the container (CORPUS_BASE_PATH
   present + the corpus dir readable); it asserts the real seed counts
   (HS↔T == 28, L↔G == 27) and Bruckner's placement.

Run inside the container:

    docker compose run --rm corpus-builder python -m pytest tests/ -q
"""

import textwrap
from pathlib import Path

import pytest

from latourometer.baselines._corpus_loader import ROLE_INVERSION_HOLD, role_of, seed_pool
from latourometer.baselines.calibrate import load_axis_corpus

_BRUCKNER = "inversion_hors-sol_001_pascal-bruckner_greta-thunberg-ou-la"


# --- Layer 1: fixture (no spaCy, no samba) ---------------------------------


def _write(dirpath: Path, slug: str, attracteur, role, use_for_seeds: bool):
    """Write a minimal frontmatter file mirroring the real corpus shape."""
    seeds = "true" if use_for_seeds else "false"
    body = textwrap.dedent(
        f"""\
        ---
        attracteur: {attracteur}
        audit:
          role: {role}
          use_for_seeds: {seeds}
        ---
        # {slug}
        Texte de calibration pour {slug}.
        """
    )
    (dirpath / f"{slug}.md").write_text(body, encoding="utf-8")


@pytest.fixture
def mini_corpus(tmp_path):
    """A miniature HS↔T corpus: 2 HS seeds, 2 T seeds, Bruckner as INVERSION-HOLD.

    Bruckner mirrors the real leak: attracteur hors_sol, role INVERSION-HOLD,
    use_for_seeds false — exactly the row the old sub_pole filter missed.
    """
    _write(tmp_path, "horssol_a", "hors_sol", "SEED", True)
    _write(tmp_path, "horssol_b", "hors_sol", "SEED", True)
    _write(tmp_path, "terrestre_a", "terrestre", "SEED", True)
    _write(tmp_path, "terrestre_b", "terrestre", "SEED", True)
    _write(tmp_path, _BRUCKNER, "hors_sol", "INVERSION-HOLD", False)
    return tmp_path


def test_fixture_seed_pool_excludes_inversion_hold(mini_corpus):
    slugs = {e["slug"] for e in seed_pool(mini_corpus)}
    assert _BRUCKNER not in slugs  # the leak: must NOT be a seed
    assert slugs == {"horssol_a", "horssol_b", "terrestre_a", "terrestre_b"}


def test_fixture_bruckner_is_an_inversion_hold(mini_corpus):
    from latourometer.baselines._corpus_loader import load_corpus

    by_slug = {e["slug"]: e for e in load_corpus(mini_corpus)}
    assert role_of(by_slug[_BRUCKNER]) == ROLE_INVERSION_HOLD
    # And he still folds to the Hors-Sol pole (on the HS↔T axis).
    assert by_slug[_BRUCKNER]["expected_pole"] == "hors_sol"


def test_fixture_load_axis_corpus_routes_bruckner_to_excluded(mini_corpus, monkeypatch):
    # Point load_axis_corpus at the fixture by patching corpus_dir.
    monkeypatch.setattr("latourometer.baselines.calibrate.corpus_dir", lambda: mini_corpus)
    kept, excluded = load_axis_corpus("hors-sol-terrestre")
    kept_slugs = {e["slug"] for e in kept}
    assert _BRUCKNER not in kept_slugs  # not in the seed basin
    assert _BRUCKNER in excluded  # in the held-out inversion group
    assert len(kept) == 4  # 2 HS + 2 T seeds
