"""Minimal example: score a French text on Latour's four attractors.

    python examples/score_text.py

The first call downloads / loads the CamemBERT embedder and the distilCamemBERT
NLI head (a few hundred MB, cached afterwards), so it takes a minute cold.
"""

from latourometer import score

TEXTS = {
    "Terrestre": "Nous devons habiter la Terre, défendre le vivant et la biodiversité de la zone critique.",
    "Hors-Sol": "Nous croyons en la croissance technologique illimitée, le marché libre et le capital sans frontières.",
    "Local": "La nation souveraine doit défendre ses frontières, son identité et son peuple enraciné.",
    "Global": "La mondialisation et le multilatéralisme tissent une coopération européenne universaliste.",
}

for expected, text in TEXTS.items():
    result = score(text)
    print(f"\n[{expected}] → dominant: {result['labels'][result['dominant_pole']]}")
    for pole in result["pole_order"]:
        print(f"    {result['labels'][pole]:<12} {result['scores'][pole]:.3f}")
