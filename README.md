# Latouromètre

Score a French political text on **Bruno Latour's two attractor axes** from
*Où atterrir ?* (2017):

```
        Terrestre
            │
   Local ───┼─── Global
            │
        Hors-Sol
```

Each pole is defined by seed phrases. The metric mean-pools the seed embeddings
into a pole vector (a [SemAxis](https://arxiv.org/abs/1806.05201)-style
projection), scores each sentence-chunk by softmax-normalised cosine similarity,
then **blends in a zero-shot NLI stance layer** so a text that *uses* a pole's
vocabulary to *attack* it (a climate-denier writing dense ecological prose) is
not misread as belonging to that pole.

This is a standalone, dependency-light extraction of the metric that runs inside
the [AFK](https://afk.live) observatory's editorial pipeline — repackaged so
anyone can score a single text with no Redis, no Postgres, no pipeline runtime.

## Install

```bash
pip install -e .
python -m spacy download fr_core_news_lg     # FR pipeline (not a pip dep)
```

The two transformer models load lazily on the first `score()` call and are
cached afterwards:

- embedder — [`dangvantuan/sentence-camembert-large`](https://huggingface.co/dangvantuan/sentence-camembert-large)
- NLI head — [`cmarkea/distilcamembert-base-nli`](https://huggingface.co/cmarkea/distilcamembert-base-nli)

## Use

```python
from latourometer import score

result = score("Nous devons habiter la Terre et composer avec le vivant.")
print(result["dominant_pole"])   # 'terrestre'
print(result["scores"])          # {'terrestre': 0.71, 'global': 0.1, 'hors_sol': 0.08, 'local': 0.11}
```

Or from the command line:

```bash
latourometer "Nous croyons en la croissance technologique illimitée."
latourometer --file speech.txt --json
```

`score()` returns the same structure as the production `metrics.json`
`latourometre` block: the four-pole blended `scores`, the pre-blend
`cosine_scores`, the NLI `stance_scores` (in `[-1, +1]`), and a convenience
`dominant_pole`. Pass `use_stance=False` for the pure cosine projection (no NLI
load), or `gamma=` to tune the additive stance blend (γ = 1.0 is the calibrated
default).

## Method

| Layer | What it does |
|---|---|
| **Cosine SemAxis** | mean-pool each pole's seed phrases → pole vector; per-chunk softmax over poles at temperature τ, averaged across chunks. |
| **NLI stance** | per chunk, zero-shot entailment against pro/contra hypotheses per pole; `stance[P] = mean(pro) − mean(contra)` ∈ `[-1, +1]`. |
| **Additive-γ blend** | `blended[P] = max(0, cosine[P] + γ·stance[P])`, renormalised. Neutral stance passes the cosine through; a strong contra stance suppresses a pole. |

The seed phrases and stance hypotheses live in
[`src/latourometer/config/`](src/latourometer/config/). The metric and its
calibration (seeds v4, γ = 1.0) are taken as-is from the AFK research line.

## Baselines & reproducible benchmark

The repo also ships the **Wordscores** (Laver–Benoit–Garry 2003) and
**Wordfish** (Slapin–Proksch 2008) lexical-scaling baselines and a benchmark
that regenerates a method-comparison table — entirely offline, from a tiny
synthetic CC0 fixture corpus (no license-gated prose):

```bash
pip install -e ".[baselines]"
python -m spacy download fr_core_news_lg
bash examples/run_baselines.sh
```

The full labelled calibration corpus is published, non-consumptively, as the
Hugging Face dataset
[`DyePop/latourometer-corpus`](https://huggingface.co/datasets/DyePop/latourometer-corpus);
point `CORPUS_BASE_PATH` at a checkout of it to reproduce the headline accuracy
numbers instead of the fixture's toy table.

## Tests

```bash
pip install -e ".[dev,baselines]"
pytest
```

`tests/test_decoupling.py` runs without any model and asserts the package
carries no Redis / Postgres / pipeline coupling. The functional, golden and
benchmark tests skip themselves when the models or the FR spaCy pipeline are not
installed.

## License

MIT — see [LICENSE](LICENSE). The synthetic fixture corpus under
`tests/fixtures/` is CC0.
