"""``latourometer`` command-line entrypoint: score text on Latour's four poles.

    latourometer "Nous croyons en la croissance technologique illimitée."
    latourometer --file speech.txt --json
    cat speech.txt | latourometer
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .score import score


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="latourometer",
        description="Score a French political text on Bruno Latour's four attractors.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="text to score (omit to read from --file or stdin)",
    )
    parser.add_argument("--file", type=str, default=None, help="read text from this file")
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="stance-blend strength (default 1.0; 0.0 = pure cosine projection)",
    )
    parser.add_argument(
        "--no-stance",
        action="store_true",
        help="skip the NLI stance layer (faster; pure cosine SemAxis projection)",
    )
    parser.add_argument("--json", action="store_true", help="emit the full metric dict as JSON")
    args = parser.parse_args(argv)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            text = fh.read()
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()

    if not text.strip():
        parser.error("no text provided (pass an argument, --file, or pipe via stdin)")

    result = score(text, gamma=args.gamma, use_stance=not args.no_stance)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    labels = result.get("labels", {})
    scores = result.get("scores", {})
    dominant = result.get("dominant_pole")
    print(f"Dominant attractor: {labels.get(dominant, dominant)}\n")
    for pole in result.get("pole_order", list(scores)):
        marker = " ←" if pole == dominant else ""
        print(f"  {labels.get(pole, pole):<12} {scores.get(pole, 0.0):.4f}{marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
