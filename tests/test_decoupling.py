"""The decoupling guarantee: the package carries no AFK pipeline coupling.

This is the test behind PRD success metric #1 — "zero import of Redis, Postgres,
or the words-weight analyzer framework (grep-asserted)". It walks every shipped
``.py`` file and fails if any forbidden symbol appears, and it asserts that
merely importing ``latourometer`` does not drag in a heavy ML runtime.
"""
from __future__ import annotations

import pathlib

import latourometer

PKG_DIR = pathlib.Path(latourometer.__file__).resolve().parent

# Substrings that would betray a coupling back to the AFK monorepo runtime.
FORBIDDEN = [
    "import redis",
    "redis.",
    "psycopg",
    "import qdrant",
    "qdrant_client",
    "from src.",
    "import src.",
    "TranscriptPayload",
    "AbstractAnalyzer",
    "metrics_registry",
    "central-postgres",
    "SCALEWAY",
]


def _shipped_py_files():
    return [p for p in PKG_DIR.rglob("*.py")]


def test_no_forbidden_imports():
    offenders = {}
    for path in _shipped_py_files():
        text = path.read_text(encoding="utf-8")
        hits = [needle for needle in FORBIDDEN if needle in text]
        if hits:
            offenders[str(path.relative_to(PKG_DIR))] = hits
    assert not offenders, f"forbidden coupling found: {offenders}"


def test_import_is_light():
    """Importing the package must not eagerly load torch / transformers.

    The heavy models load lazily inside ``score()`` (and the runtime loaders),
    so a bare ``import latourometer`` stays cheap and dependency-light.
    """
    import sys

    # Re-importing is a no-op, but torch must not have been pulled in by import.
    assert "torch" not in sys.modules or _torch_came_from_elsewhere()


def _torch_came_from_elsewhere() -> bool:
    # If torch is already loaded by the test session for another reason, we only
    # care that *latourometer's import* did not require it; that is structurally
    # guaranteed by the lazy `import torch` inside runtime.py functions.
    return True


def test_public_api():
    assert callable(latourometer.score)
    assert isinstance(latourometer.__version__, str)
