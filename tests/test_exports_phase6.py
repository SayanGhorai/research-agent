"""
tests/test_exports_phase6.py
============================
Tests for evidence ledger, comparison, and run trace exports (Phase 6).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from researchlens.exports import (
    comparison_to_markdown,
    evidence_ledger_to_markdown,
    run_trace_to_markdown,
)
from researchlens.schemas import (
    ClaimRecord,
    PaperRecord,
    RunResult,
    SourceProvider,
    TaskType,
    TextCoverage,
    ValidationState,
)


def _make_paper(title="Paper A", year=2023, coverage=TextCoverage.ABSTRACT_ONLY):
    return PaperRecord(
        title=title,
        year=year,
        source_provider=SourceProvider.ARXIV,
        text_coverage=coverage,
        abstract=f"Abstract of {title}.",
    )


def _make_result(with_claims=False):
    r = RunResult(task=TaskType.ANSWER_QUESTION, question="What is RAG?")
    r.answer_text = "RAG combines retrieval with generation [abc12345]."
    if with_claims:
        claim = ClaimRecord(
            claim_text="RAG combines retrieval with generation [abc12345].",
            evidence_chunk_ids=[],
            validation_state=ValidationState.VALID,
        )
        r.claims = [claim]
    r.mark_finished()
    return r


class TestEvidenceLedger:
    def test_no_claims_shows_output(self):
        result = _make_result(with_claims=False)
        index_mock = MagicMock()
        index_mock.get_chunks_by_ids.return_value = []
        md = evidence_ledger_to_markdown(result, [], index_mock)
        assert "No citation-linked claims" in md

    def test_with_claims(self):
        result = _make_result(with_claims=True)
        index_mock = MagicMock()
        index_mock.get_chunks_by_ids.return_value = []
        md = evidence_ledger_to_markdown(result, [], index_mock)
        assert "Claim 1" in md

    def test_contains_run_id(self):
        result = _make_result()
        index_mock = MagicMock()
        index_mock.get_chunks_by_ids.return_value = []
        md = evidence_ledger_to_markdown(result, [], index_mock)
        assert result.run_id[:8] in md


class TestComparisonToMarkdown:
    def test_two_papers_comparison(self):
        p1 = _make_paper("Paper A")
        p2 = _make_paper("Paper B")
        ext1 = {"research_objective": "Study RAG", "method": "Experiment"}
        ext2 = {"research_objective": "Study BERT", "method": "Pretraining"}
        md = comparison_to_markdown([p1, p2], [ext1, ext2])
        assert "Paper A" in md
        assert "Paper B" in md
        assert "Study RAG" in md
        assert "Study BERT" in md

    def test_missing_fields_not_fabricated(self):
        p1 = _make_paper("Paper A")
        md = comparison_to_markdown([p1], [{}])
        assert "_not reported_" in md

    def test_not_available_shown(self):
        p1 = _make_paper("Paper A")
        ext = {"research_objective": "not_available", "method": "not_available"}
        md = comparison_to_markdown([p1], [ext])
        assert "_not available_" in md

    def test_disagreements_labeled_tentative(self):
        p1 = _make_paper("A")
        p2 = _make_paper("B")
        md = comparison_to_markdown(
            [p1, p2], [{}, {}],
            possible_disagreements=["Paper A reports 90% accuracy while Paper B reports 70%."]
        )
        assert "Possible Disagreements" in md
        assert "suspected difference" in md.lower() or "not a definitive" in md.lower()

    def test_gap_suggestions_labeled_tentative(self):
        p1 = _make_paper("A")
        md = comparison_to_markdown(
            [p1], [{}],
            gap_suggestions=["Future work: multi-lingual RAG evaluation."]
        )
        assert "Suggested Research Gaps" in md
        assert "not confirmed gaps" in md.lower() or "source set" in md.lower()

    def test_empty_papers_returns_message(self):
        md = comparison_to_markdown([], [])
        assert "No papers" in md


class TestRunTrace:
    def test_trace_contains_run_id(self):
        result = _make_result()
        md = run_trace_to_markdown(result)
        assert result.run_id[:8] in md

    def test_trace_no_model_shows_disabled(self):
        result = _make_result()
        md = run_trace_to_markdown(result)
        assert "No model calls" in md or "disabled" in md.lower()

    def test_trace_with_model(self):
        result = _make_result()
        result.model_id = "ibm/granite-13b-chat-v2"
        result.llm_calls_used = 1
        result.output_tokens_reported = 200
        md = run_trace_to_markdown(result)
        assert "ibm/granite-13b-chat-v2" in md
        assert "200" in md

    def test_trace_with_warnings(self):
        result = _make_result()
        result.warnings = ["Abstract-only source used."]
        md = run_trace_to_markdown(result)
        assert "Warnings" in md
        assert "Abstract-only" in md
