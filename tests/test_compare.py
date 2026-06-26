"""Unit tests for the Wordscores comparison core (STORY-185).

These exercise the pure scoring + grading math only -- no spaCy model and no
corpus volume are needed. Run inside the container:

    docker compose run --rm corpus-builder python -m pytest tests/ -q
"""

from latourometer.baselines.compare import (
    ScoredText,
    accuracy,
    axis_score,
    headtohead,
    _inversion_table,
    load_latourometre_preds,
    load_lexicon,
    predict_pole,
)


def _lexicon():
    # Hors-Sol words score negative, Terrestre words positive, "sol" is neutral.
    return {
        ("marché", "NOUN"): -0.9,
        ("croissance", "NOUN"): -0.8,
        ("terre", "NOUN"): 0.9,
        ("vivant", "NOUN"): 0.8,
        ("sol", "NOUN"): 0.0,
    }


def test_axis_score_means_over_in_lexicon_tokens_only():
    lex = _lexicon()
    # Two in-lexicon hits (-0.9, 0.9) plus one OOV token that must be ignored.
    tokens = [("marché", "NOUN"), ("terre", "NOUN"), ("inconnu", "NOUN")]
    score, n_hits = axis_score(tokens, lex)
    assert n_hits == 2
    assert abs(score - 0.0) < 1e-9  # (-0.9 + 0.9) / 2


def test_axis_score_no_overlap_is_zero_hits():
    score, n_hits = axis_score([("inconnu", "NOUN"), ("autre", "VERB")], _lexicon())
    assert (score, n_hits) == (0.0, 0)


def test_predict_pole_sign_and_undecided():
    assert predict_pole(0.4, 5) == "terrestre"
    assert predict_pole(-0.4, 5) == "hors_sol"
    assert predict_pole(0.0, 5) is None  # exact tie -> undecided
    assert predict_pole(0.4, 0) is None  # no hits -> undecided


def test_predict_pole_honours_axis_specific_pole_labels():
    # Local <-> Global axis: -1 = local, +1 = global.
    assert predict_pole(0.4, 5, "local", "global") == "global"
    assert predict_pole(-0.4, 5, "local", "global") == "local"


def test_scoredtext_uses_its_own_axis_poles():
    # A global-gold hold-out text on the local-global axis, predicted global.
    r = ScoredText("g", "holdout", "global", True, 0.6, 10, 50, pole_minus="local", pole_plus="global")
    assert r.predicted_pole == "global"
    assert r.correct is True


def test_scoredtext_correct_only_grades_on_axis_known_gold():
    # On-axis seed, prediction matches gold -> correct.
    hit = ScoredText("h", "seed", "hors_sol", True, -0.5, 10, 50)
    assert hit.predicted_pole == "hors_sol"
    assert hit.correct is True

    # On-axis seed, prediction flips -> incorrect.
    miss = ScoredText("t", "seed", "terrestre", True, -0.2, 10, 50)
    assert miss.correct is False

    # Off-axis gold (e.g. local) -> not gradable.
    off = ScoredText("b", "inversion", "local", False, -0.3, 10, 50)
    assert off.correct is None

    # Unknown gold (hold-out) -> not gradable.
    unk = ScoredText("x", "holdout", None, False, 0.3, 10, 50)
    assert unk.correct is None


def test_accuracy_counts_only_gradable_seeds():
    rows = [
        ScoredText("s1", "seed", "hors_sol", True, -0.5, 10, 50),  # correct
        ScoredText("s2", "seed", "terrestre", True, 0.5, 10, 50),  # correct
        ScoredText("s3", "seed", "terrestre", True, -0.1, 10, 50),  # wrong
        ScoredText("i1", "inversion", "hors_sol", True, 0.4, 10, 50),  # not a seed
        ScoredText("h1", "holdout", None, False, 0.2, 10, 50),  # ungradable
    ]
    assert accuracy(rows) == (2, 3)


def test_inversion_table_grades_on_axis_and_skips_off_axis():
    rows = [
        # Damasio: gold terrestre, lexicon flips him to hors_sol -> misclassified.
        ScoredText("terrestre011_damasio", "inversion", "terrestre", True, -0.3, 12, 80),
        # Fink: gold hors_sol, lexicon reads sustainability words -> terrestre.
        ScoredText("horssol009_fink", "inversion", "hors_sol", True, 0.4, 15, 90),
        # Brunel: gold local (off-axis) -> reported, not graded.
        ScoredText("local009_brunel", "inversion", "local", False, 0.2, 8, 60),
    ]
    table, misclassified, gradable = _inversion_table(rows)
    assert gradable == 2
    assert misclassified == 2
    assert "local009_brunel" in table  # off-axis row still reported
    assert "off-axis" in table


def test_accuracy_separates_in_sample_seeds_from_out_of_sample_holdout():
    rows = [
        ScoredText("s1", "seed", "hors_sol", True, -0.5, 10, 50),  # in-sample, correct
        ScoredText("s2", "seed", "terrestre", True, -0.1, 10, 50),  # in-sample, wrong
        ScoredText("h1", "holdout", "hors_sol", True, -0.4, 10, 50),  # OOS, correct
        ScoredText("h2", "holdout", "terrestre", True, 0.2, 10, 50),  # OOS, correct
        ScoredText("h3", "holdout", "global", False, -0.3, 10, 50),  # OOS off-axis, ungraded
        ScoredText("h4", "holdout", None, False, 0.1, 10, 50),  # OOS no gold, ungraded
    ]
    assert accuracy(rows, "seed") == (1, 2)
    assert accuracy(rows, "holdout") == (2, 2)  # only the on-axis HS/T golds count


def test_headtohead_grades_each_method_on_its_own_scope():
    holdout = [
        # gold HS, Wordscores right (neg), Latouromètre wrong (global) -> divergence
        ScoredText("a", "holdout", "hors_sol", True, -0.5, 10, 50),
        # gold T, both right
        ScoredText("b", "holdout", "terrestre", True, 0.5, 10, 50),
        # gold global (off WS axis), Latouromètre right
        ScoredText("c", "holdout", "global", False, -0.1, 10, 50),
    ]
    latour = {"a": "global", "b": "terrestre", "c": "global"}
    h = headtohead(holdout, latour)

    assert (h["ws_ok"], h["ws_tot"]) == (2, 2)  # WS graded on HS/T golds only
    assert (h["lat_ok"], h["lat_tot"]) == (2, 3)  # LAT graded on all golds it predicts
    assert [d[0] for d in h["div"]] == ["a"]  # only 'a': WS right, LAT wrong


def test_load_latourometre_preds_reads_predicted_pole(tmp_path):
    (tmp_path / "foo.latourometre.json").write_text(
        '{"slug": "foo", "predicted_pole": "terrestre", "scores": {}}', encoding="utf-8"
    )
    (tmp_path / "bar.latourometre.json").write_text(
        '{"slug": "bar", "predicted_pole": null}', encoding="utf-8"  # null pred -> skipped
    )
    preds = load_latourometre_preds(tmp_path)
    assert preds == {"foo": "terrestre"}


def test_load_lexicon_round_trip(tmp_path):
    csv_path = tmp_path / "lexicon.csv"
    csv_path.write_text(
        "word,pos,score,freq_hors_sol,freq_terrestre,n_docs,ci_low,ci_high\n"
        "terre,NOUN,0.9,1,12,8,0.7,1.0\n"
        "marché,NOUN,-0.8,9,1,6,,\n",  # null CIs -> empty, must not break parsing
        encoding="utf-8",
    )
    lex = load_lexicon(csv_path)
    assert lex[("terre", "NOUN")] == 0.9
    assert lex[("marché", "NOUN")] == -0.8
