"""
tests/test_retrieval.py
=======================
Offline tests for researchlens.retrieval.
sentence-transformers is NOT loaded in these tests — we test TF-IDF path
and mock the embedding path separately.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from researchlens.retrieval import (
    MAX_CHUNKS_PER_SESSION,
    CorpusIndex,
    _rrf_combine,
    chunk_text,
    select_evidence,
)
from researchlens.schemas import EvidenceChunk, TextCoverage


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_chunk(text: str, paper_id: str = "p1", page: int = 0) -> EvidenceChunk:
    return EvidenceChunk(
        paper_id=paper_id,
        source_type=TextCoverage.FULL_TEXT,
        page_index=page,
        text=text,
    )


SAMPLE_CHUNKS = [
    _make_chunk("Attention mechanisms in transformers improve machine translation quality.", "p1", 0),
    _make_chunk("BERT uses bidirectional transformer encoders for language understanding.", "p1", 1),
    _make_chunk("GPT models are autoregressive and trained on large text corpora.", "p2", 0),
    _make_chunk("Retrieval augmented generation combines search with language models.", "p2", 1),
    _make_chunk("Knowledge graphs represent entities and their relationships.", "p3", 0),
]


class TestChunkText:
    def test_splits_long_text(self):
        text = " ".join([f"word{i}" for i in range(400)])
        chunks = chunk_text(text, "p1", page_index=0, is_abstract=False, target_tokens=180, overlap_tokens=30)
        assert len(chunks) > 1

    def test_short_text_one_chunk(self):
        chunks = chunk_text("Short text here.", "p1", page_index=0, is_abstract=False)
        assert len(chunks) == 1
        assert "Short text" in chunks[0].text

    def test_abstract_chunk_marked(self):
        chunks = chunk_text("Abstract text.", "p1", page_index=None, is_abstract=True)
        assert chunks[0].is_abstract is True
        assert chunks[0].source_type == TextCoverage.ABSTRACT_ONLY

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("", "p1", page_index=0, is_abstract=False)
        assert chunks == []

    def test_paper_id_preserved(self):
        chunks = chunk_text("Some text about AI.", "my_paper_id", page_index=2, is_abstract=False)
        assert all(c.paper_id == "my_paper_id" for c in chunks)

    def test_page_index_preserved(self):
        chunks = chunk_text("Text on page 5.", "p1", page_index=4, is_abstract=False)
        assert chunks[0].page_index == 4


class TestCorpusIndexTfIdf:
    def setup_method(self):
        # Disable semantic to test TF-IDF path cleanly
        self.idx = CorpusIndex(enable_semantic=False)

    def test_empty_search_returns_empty(self):
        results = self.idx.search("query")
        assert results == []

    def test_add_and_search(self):
        self.idx.add_chunks(SAMPLE_CHUNKS)
        results = self.idx.search("transformer attention", top_k=3)
        assert len(results) > 0
        # Results are (chunk_index, score) pairs
        for idx, score in results:
            assert score >= 0.0

    def test_relevant_chunk_ranked_high(self):
        self.idx.add_chunks(SAMPLE_CHUNKS)
        results = self.idx.search("retrieval augmented generation", top_k=5)
        # The RAG chunk should be in top results
        chunk_indices = [idx for idx, _ in results]
        # Find which chunk index corresponds to RAG text
        rag_idx = next(
            i for i, c in enumerate(self.idx._chunks)
            if "retrieval augmented" in c.text.lower()
        )
        assert rag_idx in chunk_indices

    def test_remove_paper_clears_chunks(self):
        self.idx.add_chunks(SAMPLE_CHUNKS)
        count_before = self.idx.chunk_count()
        removed = self.idx.remove_paper("p3")
        assert removed == 1
        assert self.idx.chunk_count() == count_before - 1

    def test_remove_paper_updates_index(self):
        self.idx.add_chunks(SAMPLE_CHUNKS)
        self.idx.search("query")  # build index
        self.idx.remove_paper("p1")
        results = self.idx.search("transformer")
        for idx, _ in results:
            assert self.idx._chunks[idx].paper_id != "p1"

    def test_chunk_limit_enforced(self):
        chunks = [_make_chunk(f"text {i}", "p1") for i in range(MAX_CHUNKS_PER_SESSION + 10)]
        warnings = self.idx.add_chunks(chunks)
        assert self.idx.chunk_count() <= MAX_CHUNKS_PER_SESSION
        assert any("limit" in w.lower() for w in warnings)

    def test_duplicate_content_not_added_twice(self):
        chunk = _make_chunk("Identical content.", "p1")
        self.idx.add_chunks([chunk])
        self.idx.add_chunks([chunk])  # same content_hash
        assert self.idx.chunk_count() == 1

    def test_corpus_hash_changes_on_update(self):
        self.idx.add_chunks(SAMPLE_CHUNKS[:2])
        h1 = self.idx.corpus_hash
        self.idx.add_chunks(SAMPLE_CHUNKS[2:])
        h2 = self.idx.corpus_hash
        assert h1 != h2

    def test_degraded_mode_false_when_semantic_disabled(self):
        # When semantic disabled, degraded_mode starts False
        self.idx.add_chunks(SAMPLE_CHUNKS)
        self.idx.search("query")
        # degraded_mode only set True when semantic is enabled but fails
        assert self.idx.degraded_mode is False


class TestRrfCombine:
    def test_combines_two_lists(self):
        tfidf = [(0, 0.9), (1, 0.7), (2, 0.5)]
        semantic = [(1, 0.95), (0, 0.8), (3, 0.6)]
        combined = _rrf_combine(tfidf, semantic, top_k=4)
        # index 0 and 1 appear in both, should be ranked high
        indices = [idx for idx, _ in combined]
        assert 0 in indices
        assert 1 in indices

    def test_deduplicates_by_index(self):
        tfidf = [(0, 0.9), (0, 0.8)]  # duplicate
        semantic = [(0, 0.95)]
        combined = _rrf_combine(tfidf, semantic, top_k=5)
        indices = [idx for idx, _ in combined]
        assert indices.count(0) == 1

    def test_top_k_respected(self):
        tfidf = [(i, 1.0 / (i + 1)) for i in range(20)]
        semantic = [(i, 1.0 / (i + 2)) for i in range(20)]
        combined = _rrf_combine(tfidf, semantic, top_k=5)
        assert len(combined) <= 5

    def test_scores_are_positive(self):
        tfidf = [(0, 0.5)]
        semantic = [(1, 0.8)]
        combined = _rrf_combine(tfidf, semantic, top_k=3)
        for _, score in combined:
            assert score > 0


class TestSelectEvidence:
    def setup_method(self):
        self.idx = CorpusIndex(enable_semantic=False)
        self.idx.add_chunks(SAMPLE_CHUNKS)

    def test_returns_correct_chunks(self):
        evidence = select_evidence(self.idx, "transformer attention", max_total=3)
        assert len(evidence) <= 3
        assert all(isinstance(c, EvidenceChunk) for c in evidence)

    def test_per_paper_cap(self):
        # Add many chunks from same paper
        chunks = [_make_chunk(f"Transformer text {i}", "p_heavy", i) for i in range(10)]
        self.idx.add_chunks(chunks)
        evidence = select_evidence(self.idx, "transformer", max_total=8, max_per_paper=2)
        from collections import Counter
        counts = Counter(c.paper_id for c in evidence)
        assert all(v <= 2 for v in counts.values())

    def test_empty_index_returns_empty(self):
        empty_idx = CorpusIndex(enable_semantic=False)
        assert select_evidence(empty_idx, "query") == []
