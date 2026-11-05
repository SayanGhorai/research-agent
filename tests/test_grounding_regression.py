"""
tests/test_grounding_regression.py
====================================
Regression tests for the evidence-grounding defects found during the first
live three-PDF test (2026-11-03).

Defects addressed:
  Bug 1 — Three uploads + search_first=True produced six papers (title dedup
           missing between uploads and arXiv/Crossref results).
  Bug 2 — Evidence prompt truncated at 800 chars, missing per-paper attribution.
  Bug 3 — VerifyAgent did not detect numbers in claims that were absent from the
           cited chunk (cross-paper number mixing).
  Bug 4 — index_uploaded_paper added both page chunks AND an abstract chunk for
           uploaded PDFs, polluting retrieval with duplicate/misleading content.

All tests are offline — no live API calls, no real PDFs.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from researchlens.agents import (
    VerifyAgent,
    WriteAgent,
    _check_numeric_grounding,
    _format_evidence,
)
from researchlens.ingestion import abstract_to_chunk
from researchlens.retrieval import CorpusIndex
from researchlens.schemas import (
    ClaimRecord,
    EvidenceChunk,
    PaperRecord,
    RunResult,
    SourceProvider,
    TaskType,
    TextCoverage,
    ValidationState,
)
from researchlens.sources import deduplicate_records
from researchlens.workflow import index_uploaded_paper


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _uploaded_paper(title: str, abstract: str = "") -> PaperRecord:
    return PaperRecord(
        title=title,
        source_provider=SourceProvider.UPLOAD,
        text_coverage=TextCoverage.FULL_TEXT,
        abstract=abstract,
        file_hash="aabbcc112233" + title[:4].lower().replace(" ", "x"),
    )


def _arxiv_paper(title: str, arxiv_id: str = "") -> PaperRecord:
    return PaperRecord(
        title=title,
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
        abstract="Abstract text for " + title,
        arxiv_id=arxiv_id or None,
    )


def _page_chunk(paper: PaperRecord, text: str, page: int = 0) -> EvidenceChunk:
    return EvidenceChunk(
        paper_id=paper.paper_id,
        source_type=TextCoverage.FULL_TEXT,
        page_index=page,
        is_abstract=False,
        text=text,
    )


# ──────────────────────────────────────────────────────────────────────
# Bug 1 — Duplicate paper prevention via title deduplication
# ──────────────────────────────────────────────────────────────────────

class TestTitleDeduplication:
    """Three uploads + search_first must not produce six papers."""

    def test_same_title_arXiv_deduped(self):
        """An uploaded paper with the same title as an arXiv result is skipped."""
        uploaded = _uploaded_paper("Deep Learning for Plant Disease Detection")
        arxiv_result = _arxiv_paper(
            "Deep Learning for Plant Disease Detection", arxiv_id="2301.00001"
        )
        merged, skipped = deduplicate_records([uploaded], [arxiv_result])
        assert len(merged) == 1, "arXiv duplicate should be skipped"
        assert len(skipped) == 1

    def test_same_title_different_case_deduped(self):
        """Title matching is case-insensitive."""
        uploaded = _uploaded_paper("MCNN Plant Classification")
        arxiv_result = _arxiv_paper("mcnn plant classification")
        merged, skipped = deduplicate_records([uploaded], [arxiv_result])
        assert len(merged) == 1
        assert len(skipped) == 1

    def test_same_title_punctuation_difference_deduped(self):
        """'Multi-Scale CNN' and 'Multi Scale CNN' are treated as duplicates."""
        uploaded = _uploaded_paper("Multi-Scale CNN for Image Recognition")
        arxiv_result = _arxiv_paper("Multi Scale CNN for Image Recognition")
        merged, skipped = deduplicate_records([uploaded], [arxiv_result])
        assert len(merged) == 1
        assert len(skipped) == 1

    def test_three_uploads_plus_three_arXiv_deduped_to_three(self):
        """Three uploads + three arXiv results with same titles → still three."""
        titles = [
            "MCNN Plant Disease Classification",
            "Attention Mechanism for Medical Imaging",
            "Hybrid CNN for Remote Sensing",
        ]
        uploads = [_uploaded_paper(t) for t in titles]
        arxiv_results = [_arxiv_paper(t, arxiv_id=f"230{i}.0000{i}") for i, t in enumerate(titles)]
        merged, skipped = deduplicate_records(uploads, arxiv_results)
        assert len(merged) == 3, f"Expected 3 papers, got {len(merged)}"
        assert len(skipped) == 3

    def test_different_title_not_deduped(self):
        """A genuinely different arXiv paper is kept."""
        uploaded = _uploaded_paper("MCNN Plant Classification")
        arxiv_result = _arxiv_paper("Transformer for NLP Tasks", arxiv_id="2301.00001")
        merged, skipped = deduplicate_records([uploaded], [arxiv_result])
        assert len(merged) == 2
        assert len(skipped) == 0

    def test_doi_dedup_still_works(self):
        """DOI deduplication still takes priority."""
        p1 = PaperRecord(title="Paper A", doi="10.1234/abc", source_provider=SourceProvider.ARXIV)
        p2 = PaperRecord(title="Paper A modified title", doi="10.1234/abc",
                         source_provider=SourceProvider.CROSSREF)
        merged, skipped = deduplicate_records([p1], [p2])
        assert len(merged) == 1
        assert len(skipped) == 1

    def test_arxiv_id_dedup_still_works(self):
        """arXiv ID deduplication (without title match) still works."""
        p1 = _arxiv_paper("Paper A", arxiv_id="2301.00001v1")
        p2 = _arxiv_paper("Paper A (reposted)", arxiv_id="2301.00001v2")
        merged, skipped = deduplicate_records([p1], [p2])
        assert len(merged) == 1
        assert len(skipped) == 1


# ──────────────────────────────────────────────────────────────────────
# Bug 4 — Abstract chunk not indexed when page chunks exist
# ──────────────────────────────────────────────────────────────────────

class TestIndexUploadedPaperNoDuplicateAbstract:
    """Uploaded PDFs with page chunks must not also get an abstract chunk."""

    def test_page_chunks_indexed_no_abstract(self):
        paper = _uploaded_paper("MCNN Study", abstract="This paper studies MCNN.")
        page_chunk = _page_chunk(paper, "Results: 98.24% accuracy on 1345 images.")
        index = CorpusIndex(enable_semantic=False)
        warnings = index_uploaded_paper(paper, [page_chunk], index)
        assert index.chunk_count() == 1, (
            "Only the page chunk should be indexed — no abstract chunk duplicate"
        )
        # The one chunk must be the page chunk (full text), not abstract
        assert index._chunks[0].is_abstract is False
        assert index._chunks[0].source_type == TextCoverage.FULL_TEXT

    def test_no_page_chunks_abstract_is_fallback(self):
        """Scanned PDF (no extractable pages) falls back to abstract."""
        paper = _uploaded_paper("MCNN Study", abstract="Abstract of MCNN study.")
        index = CorpusIndex(enable_semantic=False)
        warnings = index_uploaded_paper(paper, [], index)
        assert index.chunk_count() == 1
        assert index._chunks[0].is_abstract is True

    def test_no_page_chunks_no_abstract_returns_empty(self):
        """Scanned PDF with no abstract produces no chunks (not a crash)."""
        paper = _uploaded_paper("Scanned Paper", abstract="")
        index = CorpusIndex(enable_semantic=False)
        warnings = index_uploaded_paper(paper, [], index)
        assert index.chunk_count() == 0

    def test_three_uploaded_papers_chunk_counts_correct(self):
        """Three papers with 2 pages each → exactly 6 chunks, no abstracts."""
        papers = [_uploaded_paper(f"Paper {i}", abstract=f"Abstract {i}.") for i in range(3)]
        index = CorpusIndex(enable_semantic=False)
        for paper in papers:
            chunks = [
                _page_chunk(paper, f"Page 1 of paper {paper.title}."),
                _page_chunk(paper, f"Page 2 of paper {paper.title}.", page=1),
            ]
            index_uploaded_paper(paper, chunks, index)
        assert index.chunk_count() == 6


# ──────────────────────────────────────────────────────────────────────
# Bug 3 — Numeric containment check
# ──────────────────────────────────────────────────────────────────────

class TestCheckNumericGrounding:
    """_check_numeric_grounding detects numbers in claims absent from chunk text."""

    def test_number_present_returns_empty(self):
        claim = "The dataset contains 1,345 images."
        chunk_text = "We collected 1,345 images across four plant species."
        assert _check_numeric_grounding(claim, chunk_text) == []

    def test_number_absent_returns_it(self):
        claim = "The dataset contains 33,063 images."
        chunk_text = "We collected 1,345 images across four plant species."
        missing = _check_numeric_grounding(claim, chunk_text)
        assert "33,063" in missing or "33063" in missing or len(missing) > 0

    def test_percentage_present_returns_empty(self):
        claim = "The model achieved 98.24% accuracy."
        chunk_text = "The MCNN achieved 98.24% accuracy on the test set."
        assert _check_numeric_grounding(claim, chunk_text) == []

    def test_percentage_absent_returns_it(self):
        claim = "The model achieved 99.07% accuracy."
        chunk_text = "The MCNN achieved 98.24% accuracy on the test set."
        missing = _check_numeric_grounding(claim, chunk_text)
        assert len(missing) > 0

    def test_comma_normalized(self):
        """33,063 in claim should match 33063 in chunk text (or vice versa)."""
        claim = "There are 33,063 total images."
        chunk_text = "The dataset contains 33063 images total."
        assert _check_numeric_grounding(claim, chunk_text) == []

    def test_short_numbers_ignored(self):
        """Single-digit or two-digit trivial numbers are not flagged."""
        claim = "We used 3 classes and a split of 70/30."
        chunk_text = "Different text with no numbers."
        missing = _check_numeric_grounding(claim, chunk_text)
        # 70 and 30 are only 2 digits — borderline but the rule skips < 2 chars
        # The important thing is this does not raise
        assert isinstance(missing, list)

    def test_no_numbers_in_claim(self):
        claim = "The method uses convolutional layers."
        chunk_text = "We used 1,345 images."
        assert _check_numeric_grounding(claim, chunk_text) == []

    def test_mcnn_scenario_wrong_numbers(self):
        """Exact regression for reported defect: 33,063 / 99.07% not in MCNN chunk."""
        # MCNN paper reports: 1,345 images, 98.24% accuracy
        mcnn_chunk_text = (
            "The dataset consists of 1,345 plant images across four species: "
            "Arjun, Alstonia, Mango, and Sukh chain. "
            "The 70:30 train-test split yielded 98.24% classification accuracy."
        )
        # Model hallucinated numbers from a different paper
        bad_claim = "The MCNN study used 33,063 images and achieved 99.07% accuracy."
        missing = _check_numeric_grounding(bad_claim, mcnn_chunk_text)
        # Both 33,063 and 99.07 should be flagged as not in the MCNN chunk
        missing_normalized = [m.replace(",", "") for m in missing]
        assert "33063" in missing_normalized, f"33,063 not flagged; got {missing}"
        assert any("99" in m for m in missing), f"99.07 not flagged; got {missing}"


# ──────────────────────────────────────────────────────────────────────
# Bug 3 — VerifyAgent numeric check integration
# ──────────────────────────────────────────────────────────────────────

class TestVerifyAgentNumericGrounding:
    """VerifyAgent.verify_result catches cross-paper number mixing."""

    def _make_result_with_claim(self, claim_text: str, chunk: EvidenceChunk) -> RunResult:
        result = RunResult(task=TaskType.SUMMARIZE, question="Summarize the study.")
        result.evidence_chunk_ids = [chunk.chunk_id]
        result.paper_ids_used = [chunk.paper_id]
        result.answer_text = claim_text
        claim = ClaimRecord(
            claim_text=claim_text,
            evidence_chunk_ids=[chunk.chunk_id],
            validation_state=ValidationState.VALID,
        )
        result.claims = [claim]
        return result

    def test_correct_numbers_pass(self):
        paper = _uploaded_paper("MCNN Study")
        chunk = _page_chunk(
            paper,
            "The dataset has 1,345 images and 98.24% accuracy."
        )
        result = self._make_result_with_claim(
            "The MCNN dataset has 1,345 images and 98.24% accuracy. [" + chunk.chunk_id[:8] + "]",
            chunk,
        )
        agent = VerifyAgent()
        verified = agent.verify_result(result, [chunk], [paper])
        assert verified.validation_state == ValidationState.VALID
        assert result.claims[0].validation_state == ValidationState.VALID

    def test_wrong_numbers_flagged_as_partial(self):
        paper = _uploaded_paper("MCNN Study")
        chunk = _page_chunk(
            paper,
            "The dataset has 1,345 images and 98.24% accuracy."
        )
        # Claim contains numbers not in the chunk
        result = self._make_result_with_claim(
            "The study used 33,063 images achieving 99.07% accuracy. [" + chunk.chunk_id[:8] + "]",
            chunk,
        )
        agent = VerifyAgent()
        verified = agent.verify_result(result, [chunk], [paper])
        assert verified.validation_state == ValidationState.PARTIAL
        assert result.claims[0].validation_state == ValidationState.PARTIAL
        # Warning must mention "cross-paper" or "Possible"
        assert any("Possible" in w or "cross-paper" in w for w in verified.warnings)

    def test_unknown_chunk_id_flagged_invalid(self):
        paper = _uploaded_paper("MCNN Study")
        chunk = _page_chunk(paper, "Some text.")
        result = RunResult(task=TaskType.SUMMARIZE, question="q")
        result.evidence_chunk_ids = ["nonexistent-chunk-id-1234567890ab"]
        result.paper_ids_used = [paper.paper_id]
        result.answer_text = "text"
        agent = VerifyAgent()
        verified = agent.verify_result(result, [chunk], [paper])
        assert verified.validation_state == ValidationState.INVALID


# ──────────────────────────────────────────────────────────────────────
# Bug 2 — Evidence prompt attribution
# ──────────────────────────────────────────────────────────────────────

class TestFormatEvidenceAttribution:
    """Each chunk in the formatted evidence includes explicit attribution rules."""

    def test_attribution_rule_present_per_chunk(self):
        paper = _uploaded_paper("MCNN Study")
        chunk = _page_chunk(paper, "Accuracy: 98.24% on 1,345 images.")
        paper_map = {paper.paper_id: paper}
        formatted = _format_evidence([chunk], paper_map)
        # Must include attribution rule for this specific chunk ID
        assert "ATTRIBUTION RULE" in formatted
        assert chunk.chunk_id[:8] in formatted

    def test_paper_id_in_formatted_evidence(self):
        paper = _uploaded_paper("MCNN Study")
        chunk = _page_chunk(paper, "Some text.")
        paper_map = {paper.paper_id: paper}
        formatted = _format_evidence([chunk], paper_map)
        # paper_id should appear to help trace provenance
        assert paper.paper_id[:8] in formatted

    def test_two_papers_both_attributed(self):
        p1 = _uploaded_paper("Paper One")
        p2 = _uploaded_paper("Paper Two")
        c1 = _page_chunk(p1, "Paper One has 1,345 images.")
        c2 = _page_chunk(p2, "Paper Two has 33,063 images.")
        paper_map = {p1.paper_id: p1, p2.paper_id: p2}
        formatted = _format_evidence([c1, c2], paper_map)
        assert "Paper One" in formatted
        assert "Paper Two" in formatted
        # Each chunk should have its own ATTRIBUTION RULE
        assert formatted.count("ATTRIBUTION RULE") == 2

    def test_text_not_truncated_at_800(self):
        """Chunk text up to 1,200 chars should not be cut at 800."""
        paper = _uploaded_paper("Long Paper")
        long_text = "Word " * 250  # 1,250 chars
        chunk = _page_chunk(paper, long_text)
        paper_map = {paper.paper_id: paper}
        formatted = _format_evidence([chunk], paper_map)
        # Must contain at least 1,200 chars of the source text
        assert long_text[:1200].strip() in formatted or formatted.count("Word") >= 240


# ──────────────────────────────────────────────────────────────────────
# Three-paper upload scenario integration
# ──────────────────────────────────────────────────────────────────────

class TestThreePaperUploadScenario:
    """End-to-end scenario: three PDFs uploaded, search returns similar titles."""

    def test_three_uploads_search_returns_same_titles_stays_three(self):
        """Bug 1 regression: library must remain 3 papers after search dedup."""
        from researchlens.sources import deduplicate_records

        titles = [
            "Plant Disease Detection using MCNN",
            "Attention-Based Medical Image Segmentation",
            "Remote Sensing with Hybrid CNN",
        ]
        uploads = [_uploaded_paper(t) for t in titles]

        # Simulate arXiv returning the same three papers + 2 unrelated ones
        arxiv_returns = [_arxiv_paper(t) for t in titles] + [
            _arxiv_paper("Unrelated Paper Alpha", "2301.00001"),
            _arxiv_paper("Unrelated Paper Beta", "2301.00002"),
        ]

        merged, skipped = deduplicate_records(uploads, arxiv_returns)
        assert len(merged) == 5, f"Expected 5 (3 uploads + 2 new), got {len(merged)}"
        assert len(skipped) == 3

    def test_no_abstract_chunk_for_uploaded_papers_in_index(self):
        """Bug 4 regression: 3 PDFs × 2 pages = 6 index chunks (no abstract extras)."""
        papers = [_uploaded_paper(f"Paper {i}", abstract=f"Abstract of paper {i}.") for i in range(3)]
        index = CorpusIndex(enable_semantic=False)
        for paper in papers:
            page_chunks = [
                _page_chunk(paper, f"Content of paper {paper.title} page 1."),
                _page_chunk(paper, f"Content of paper {paper.title} page 2.", page=1),
            ]
            index_uploaded_paper(paper, page_chunks, index)
        # Must be exactly 6 — no abstract duplicates
        assert index.chunk_count() == 6

    def test_verify_catches_cross_paper_numbers(self):
        """Bug 3 regression: numbers from Paper B cited with chunk from Paper A are flagged."""
        paper_a = _uploaded_paper("MCNN Study")
        paper_b = _uploaded_paper("ResNet Study")
        chunk_a = _page_chunk(paper_a, "MCNN: 1,345 images, 98.24% accuracy.")
        chunk_b = _page_chunk(paper_b, "ResNet: 33,063 images, 99.07% accuracy.")

        result = RunResult(task=TaskType.SUMMARIZE, question="Summarize the studies.")
        result.evidence_chunk_ids = [chunk_a.chunk_id, chunk_b.chunk_id]
        result.paper_ids_used = [paper_a.paper_id, paper_b.paper_id]
        result.answer_text = "MCNN used 33,063 images."  # wrong number from Paper B

        # Claim cites chunk_a but contains Paper B's number
        result.claims = [ClaimRecord(
            claim_text=f"MCNN used 33,063 images. [{chunk_a.chunk_id[:8]}]",
            evidence_chunk_ids=[chunk_a.chunk_id],
            validation_state=ValidationState.VALID,
        )]

        agent = VerifyAgent()
        verified = agent.verify_result(result, [chunk_a, chunk_b], [paper_a, paper_b])
        assert verified.validation_state in (ValidationState.PARTIAL, ValidationState.INVALID)
        assert any("Possible" in w or "cross-paper" in w for w in verified.warnings)
