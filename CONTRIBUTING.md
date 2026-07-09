# Contributing to Latouromètre

Thanks for your interest! This is a small, focused research package — a
self-contained library that scores a French political text on Bruno Latour's
four attractors. Contributions of any size are welcome: bug reports, doc fixes,
new baselines, calibration improvements.

## Ground rules

- Be civil and constructive. Assume good faith.
- Open an issue before a large change so we can agree on the direction.
- Keep the package **dependency-light and decoupled**: importing
  `latourometer` must never drag in Redis, Postgres, or a pipeline runtime —
  this is asserted by `tests/test_decoupling.py` and is a hard project
  invariant.

## Dev setup

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,baselines]"
python -m spacy download fr_core_news_lg   # FR pipeline — not a pip dependency
```

The two transformer models (`sentence-camembert-large`,
`distilcamembert-base-nli`) download lazily from the Hugging Face Hub on the
first `score()` call and are cached afterwards.

## Tests

```bash
pytest
```

The suite is layered so it runs anywhere:

- **No-model tests** (`test_decoupling`, `test_stats`, `test_calibrate`,
  `test_compare`, `test_seed_selection`, `test_text_ci`) run in seconds with
  no models — this is the layer CI gates on.
- **Model-dependent tests** (`test_score`, `test_baselines`) **skip
  themselves** when the FR spaCy model or the transformer checkpoints are not
  available. Install the spaCy model (above) and run `pytest` once online to
  exercise them locally.

If you change scoring behaviour, regenerate the golden file:

```bash
rm tests/golden_score.json && pytest tests/test_score.py   # re-mints, then asserts
```

## Lint & format

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting.
Run before opening a PR:

```bash
ruff check . --fix
ruff format .
```

CI runs `ruff check .` and `ruff format --check .` — both must pass.

## Pull requests

1. Fork and branch off `main`.
2. Make your change; add or update tests.
3. Ensure `ruff check .`, `ruff format --check .`, and `pytest` are green.
4. Open the PR with a clear description of *what* and *why*. Link any related
   issue.

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE). The fixture corpus under `tests/fixtures/`
is CC0.
