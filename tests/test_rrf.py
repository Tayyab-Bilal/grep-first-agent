from grep_first_agent.rrf import hybrid_content_search, weighted_rrf


def test_keyword_weighted_3_to_1():
    fused = dict(weighted_rrf({"kw": ["A"], "dense": ["B"]}, {"kw": 3, "dense": 1}))
    assert fused["A"] == 3 * fused["B"]
    # and a keyword rank-2 still beats a dense rank-1
    fused = dict(weighted_rrf({"kw": ["x", "A"], "dense": ["B"]}, {"kw": 3, "dense": 1}))
    assert fused["A"] > fused["B"]


def test_dense_below_floor_dropped():
    dense = [{"doc_id": "low", "snippet": "s", "score": 0.30}, {"doc_id": "ok", "snippet": "s", "score": 0.80}]
    out = hybrid_content_search([], dense)
    assert [r["doc_id"] for r in out] == ["ok"]


def test_per_document_aggregation():
    kw = [{"doc_id": "d1", "snippet": "one"}, {"doc_id": "d1", "snippet": "two"}, {"doc_id": "d1", "snippet": "three"}]
    dense = [{"doc_id": "d1", "snippet": "four", "score": 0.9}, {"doc_id": "d2", "snippet": "x", "score": 0.9}]
    out = hybrid_content_search(kw, dense)
    assert [r["doc_id"] for r in out] == ["d1", "d2"]  # one row per document, lexical first
    assert out[0]["snippets"] == ["one", "two"]  # best 2 only
