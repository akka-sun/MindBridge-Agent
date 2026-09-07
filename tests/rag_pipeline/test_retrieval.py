from app.rag_pipeline.retrieval import reciprocal_rank_fusion


def test_rrf_uses_hand_calculated_rank_scores_and_deduplicates():
    rankings = {"dense": ["a", "b"], "bm25": ["b", "c"]}
    result = reciprocal_rank_fusion(rankings, k=60)
    assert [item[0] for item in result] == ["b", "a", "c"]
    assert abs(result[0][1] - (1 / 61 + 1 / 62)) < 1e-12
