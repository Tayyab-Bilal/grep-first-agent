from grep_first_agent.matching import search, tokenize


def names(hits):
    return [r["id"] for r, _ in hits]


def test_exact_phrase_beats_long_fuzzy_text():
    # "plan" is inside "planning" (phrase hit, weak fuzzy); "plam" is a typo (no phrase, better fuzzy).
    recs = [{"id": "typo", "text": "plam"}, {"id": "exact", "text": "planning"}]
    hits = search("plan", recs, ["text"])
    assert hits[0][1].fuzzy < hits[1][1].fuzzy  # the trap: fuzzy alone would pick the typo
    assert names(hits) == ["exact", "typo"]


def test_token_match_across_fields():
    recs = [{"id": "a", "name": "Budget", "owner": "Dana"}, {"id": "b", "name": "Roadmap", "owner": "Sam"}]
    hits = search("dana budget", recs, ["name", "owner"])
    assert names(hits) == ["a"]
    assert hits[0][1].token_hits == 2 and not hits[0][1].phrase_hit


def test_typo_found_by_fuzzy():
    hits = search("proffit margins", [{"id": "e", "body": "the profit margins improved"}], ["body"])
    assert names(hits) == ["e"]
    assert hits[0][1].token_hits == 1 and hits[0][1].fuzzy > 0.9


def test_threshold_filters_noise():
    recs = [{"id": "n", "name": "Team lunch"}]
    assert search("quarterly invoices", recs, ["name"]) == []
    assert names(search("quarterly invoices", recs, ["name"], fuzzy_threshold=0.0)) == ["n"]


def test_prose_capped_at_200_chars():
    hits = search("alpha", [{"id": "x", "name": "alpha", "body": "z" * 500}], ["name", "body"])
    assert len(hits[0][0]["body"]) == 200


def test_arabic_tokens():
    assert tokenize("مرحبا بالعالم، Hello!") == ["مرحبا", "بالعالم", "hello"]
    hits = search("بالعالم", [{"id": "ar", "name": "مرحبا بالعالم"}, {"id": "en", "name": "hello"}], ["name"])
    assert names(hits) == ["ar"]


def test_stopwords_are_not_token_hits():
    recs = [{"id": "x", "name": "All about the team lunch"}]
    assert search("everything about the OPS Manual", recs, ["name"], fuzzy_threshold=1.1) == []
