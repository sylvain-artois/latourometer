"""Unit tests for the Wordscores core (STORY-183).

These exercise the pure scoring math only -- no spaCy model and no corpus
volume are needed, so they run anywhere ``pyarrow`` + ``pydantic`` are present.
Run inside the container:

    docker compose run --rm corpus-builder python -m pytest tests/ -q
"""

from collections import Counter

from latourometer.baselines.calibrate import (
    AXIS_REF_SCORE,
    SUPPORTED_AXES,
    bootstrap_cis,
    build_rows,
    compute_wordscores,
)


def test_local_global_axis_is_registered_with_signed_poles():
    assert SUPPORTED_AXES["local-global"] == ("local", "global")
    # Minus pole -1, plus pole +1, per Latour's old modernization front.
    assert AXIS_REF_SCORE["local-global"] == {"local": -1.0, "global": 1.0}


def test_compute_wordscores_honours_an_explicit_axis_ref_score():
    pole_counts = {
        "local": Counter({("terroir", "NOUN"): 10, ("shared", "NOUN"): 5}),
        "global": Counter({("marché", "NOUN"): 10, ("shared", "NOUN"): 5}),
    }
    scores = compute_wordscores(pole_counts, AXIS_REF_SCORE["local-global"])
    assert scores[("terroir", "NOUN")] == -1.0  # local-only -> -1
    assert scores[("marché", "NOUN")] == 1.0  # global-only -> +1


def test_build_rows_freq_columns_follow_the_axis_poles():
    pole_counts = {
        "local": Counter({("terroir", "NOUN"): 3}),
        "global": Counter({("marché", "NOUN"): 4}),
    }
    scores = compute_wordscores(pole_counts, AXIS_REF_SCORE["local-global"])
    doc_freq = Counter({("terroir", "NOUN"): 1, ("marché", "NOUN"): 2})
    rows = {(r.word, r.pos): r for r in build_rows(scores, pole_counts, doc_freq, poles=("local", "global"))}
    # freq_minus is the -1 pole (local), freq_plus the +1 pole (global).
    assert rows[("terroir", "NOUN")].freq_minus == 3
    assert rows[("terroir", "NOUN")].freq_plus == 0


def _toy_docs():
    """6 docs (3 per pole). doc-freq: marche=2, terre=2 (< 3 -> null CI);
    sol=5 (>= 3 -> CI)."""
    return [
        {"slug": "h1", "pole": "hors_sol", "counts": Counter({("marche", "NOUN"): 2})},
        {"slug": "h2", "pole": "hors_sol", "counts": Counter({("marche", "NOUN"): 1, ("sol", "NOUN"): 1})},
        {"slug": "h3", "pole": "hors_sol", "counts": Counter({("sol", "NOUN"): 1})},
        {"slug": "t1", "pole": "terrestre", "counts": Counter({("terre", "NOUN"): 2, ("sol", "NOUN"): 1})},
        {"slug": "t2", "pole": "terrestre", "counts": Counter({("terre", "NOUN"): 1, ("sol", "NOUN"): 1})},
        {"slug": "t3", "pole": "terrestre", "counts": Counter({("sol", "NOUN"): 1})},
    ]


def _doc_freq(docs):
    freq = Counter()
    for d in docs:
        for key in d["counts"]:
            freq[key] += 1
    return freq


def test_pure_pole_words_hit_the_extremes():
    pole_counts = {
        "hors_sol": Counter({("marche", "NOUN"): 10, ("commun", "NOUN"): 5}),
        "terrestre": Counter({("terre", "NOUN"): 10, ("commun", "NOUN"): 5}),
    }
    scores = compute_wordscores(pole_counts)

    # A word seen only in the hors-sol pole scores -1; only-terrestre scores +1.
    assert scores[("marche", "NOUN")] == -1.0
    assert scores[("terre", "NOUN")] == 1.0


def test_shared_word_with_equal_relative_frequency_is_neutral():
    # Equal pole sizes and equal counts -> equal relative frequency -> S ≈ 0.
    pole_counts = {
        "hors_sol": Counter({("sol", "NOUN"): 5, ("x", "NOUN"): 5}),
        "terrestre": Counter({("sol", "NOUN"): 5, ("y", "NOUN"): 5}),
    }
    scores = compute_wordscores(pole_counts)
    assert abs(scores[("sol", "NOUN")]) < 1e-9


def test_relative_frequency_corrects_for_unequal_pole_sizes():
    # hors-sol pole is 10x larger in raw tokens, but a word balanced in
    # *relative* frequency must still land near 0, not be dragged negative.
    pole_counts = {
        "hors_sol": Counter({("w", "NOUN"): 100, ("filler", "NOUN"): 900}),
        "terrestre": Counter({("w", "NOUN"): 10, ("filler2", "NOUN"): 90}),
    }
    scores = compute_wordscores(pole_counts)
    # P_hs(w) = 100/1000 = 0.1 ; P_t(w) = 10/100 = 0.1 -> neutral.
    assert abs(scores[("w", "NOUN")]) < 1e-9


def test_build_rows_carries_counts_and_null_cis():
    pole_counts = {
        "hors_sol": Counter({("marche", "NOUN"): 3}),
        "terrestre": Counter({("terre", "NOUN"): 4}),
    }
    scores = compute_wordscores(pole_counts)
    doc_freq = Counter({("marche", "NOUN"): 1, ("terre", "NOUN"): 2})

    rows = {(r.word, r.pos): r for r in build_rows(scores, pole_counts, doc_freq)}

    marche = rows[("marche", "NOUN")]
    assert marche.score == -1.0
    assert marche.freq_minus == 3  # raw count in the -1 pole
    assert marche.freq_plus == 0  # raw count in the +1 pole
    assert marche.n_docs == 1
    # No CI map passed -> null CIs (the --bootstrap-n 0 fast path).
    assert marche.ci_low is None
    assert marche.ci_high is None


# --- STORY-184: bootstrap CIs (Lowe 2008) ---------------------------------


def test_bootstrap_only_scores_words_seen_in_3plus_docs():
    docs = _toy_docs()
    cis = bootstrap_cis(docs, ("hors_sol", "terrestre"), _doc_freq(docs), n_resamples=200)

    # "sol" is in 5 docs -> gets a CI; "marche"/"terre" are in 2 docs -> omitted.
    assert ("sol", "NOUN") in cis
    assert ("marche", "NOUN") not in cis
    assert ("terre", "NOUN") not in cis


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    docs = _toy_docs()
    freq = _doc_freq(docs)
    a = bootstrap_cis(docs, ("hors_sol", "terrestre"), freq, n_resamples=200)
    b = bootstrap_cis(docs, ("hors_sol", "terrestre"), freq, n_resamples=200)
    assert a == b


def test_bootstrap_ci_low_le_high_and_brackets_a_pure_pole_word():
    # A word that only ever appears in the terrestre pole scores +1 on every
    # resample, so its CI degenerates to [+1, +1].
    docs = [
        {"slug": "h1", "pole": "hors_sol", "counts": Counter({("x", "NOUN"): 1})},
        {"slug": "h2", "pole": "hors_sol", "counts": Counter({("x", "NOUN"): 1})},
        {"slug": "h3", "pole": "hors_sol", "counts": Counter({("x", "NOUN"): 1})},
        {"slug": "t1", "pole": "terrestre", "counts": Counter({("terre", "NOUN"): 1})},
        {"slug": "t2", "pole": "terrestre", "counts": Counter({("terre", "NOUN"): 1})},
        {"slug": "t3", "pole": "terrestre", "counts": Counter({("terre", "NOUN"): 1})},
    ]
    cis = bootstrap_cis(docs, ("hors_sol", "terrestre"), _doc_freq(docs), n_resamples=100)

    lo, hi = cis[("terre", "NOUN")]
    assert lo <= hi
    assert lo == 1.0 and hi == 1.0


def test_build_rows_wires_bootstrap_cis():
    docs = _toy_docs()
    poles = ("hors_sol", "terrestre")
    freq = _doc_freq(docs)
    pole_counts = {"hors_sol": Counter(), "terrestre": Counter()}
    for d in docs:
        pole_counts[d["pole"]].update(d["counts"])
    scores = compute_wordscores(pole_counts)
    cis = bootstrap_cis(docs, poles, freq, n_resamples=200)

    rows = {(r.word, r.pos): r for r in build_rows(scores, pole_counts, freq, cis)}

    sol = rows[("sol", "NOUN")]
    assert sol.ci_low is not None and sol.ci_high is not None
    assert sol.ci_low <= sol.ci_high
    # marche is in 2 docs -> null CI even when a cis map is passed.
    assert rows[("marche", "NOUN")].ci_low is None
