"""Latouromètre — score a French political text on Bruno Latour's two axes.

    >>> from latourometer import score
    >>> result = score("Nous devons habiter la Terre et composer avec le vivant.")
    >>> result["dominant_pole"]
    'terrestre'

The heavy models (CamemBERT embedder, distilCamemBERT NLI) load lazily on the
first ``score()`` call and are cached for the process lifetime.
"""

from __future__ import annotations

from .score import score

__all__ = ["score", "__version__"]
__version__ = "0.1.0"
