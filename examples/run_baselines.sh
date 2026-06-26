#!/usr/bin/env bash
# Regenerate the Wordscores + Wordfish baseline comparison table from the tiny
# synthetic fixture corpus — fully offline, no samba mount, no license-gated
# prose. Outputs land under $CORPUS_BASE_PATH/axes and .../calibration_latourometre.
#
#   pip install -e ".[baselines]"
#   python -m spacy download fr_core_news_lg
#   bash examples/run_baselines.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export CORPUS_BASE_PATH="${CORPUS_BASE_PATH:-$REPO/out/benchmark-corpus}"

# Seed the working corpus from the committed fixture (read-only source).
mkdir -p "$CORPUS_BASE_PATH"
cp -r "$REPO/tests/fixtures/corpus/corpus_latourometre" "$CORPUS_BASE_PATH/"

echo "== Wordscores: calibrate the Hors-Sol <-> Terrestre lexicon =="
python -m latourometer.baselines.calibrate --axis hors-sol-terrestre --min-docs-per-pole 1 --bootstrap-n 0
python -m latourometer.baselines.calibrate --axis local-global --min-docs-per-pole 1 --bootstrap-n 0

echo "== Wordscores: score texts + build the comparison report =="
python -m latourometer.baselines.compare --axis hors-sol-terrestre

echo "== Wordfish: unsupervised positional scaling (both axes) =="
python -m latourometer.baselines.wordfish --min-doc-freq 1

echo
echo "Done. See:"
echo "  $CORPUS_BASE_PATH/axes/hors-sol-terrestre/comparison-report.md"
echo "  $CORPUS_BASE_PATH/calibration_latourometre/baselines_20260625/wordfish/wordfish-report.md"
echo "  $CORPUS_BASE_PATH/calibration_latourometre/baselines_20260625/comparison.md"
