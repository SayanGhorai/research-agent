"""
tests/test_schemas.py
=====================
Offline tests for researchlens.schemas typed data contracts.

No network calls, no credentials, no Streamlit context.
"""

import pytest

from researchlens.schemas import (
    NOT_AVAILABLE,
    NOT_REPORTED,
    ClaimRecord,
    EvidenceChunk,
    PaperRecord,
    ResearchPlan,
    RunResult,
    SourceProvider,
    TaskType,
    TextCoverage,
    ValidationState,
    _content_hash,
    _new_id,
)


class TestPaperRecord:
    def test_creates_with_title(self):
        p = PaperRecord(title="Test paper")
        assert p.title == "Test paper"

    def test_paper_id_is_generated(self):
        p = PaperRecord(title="A")
        assert p.paper_id
        assert len(p.paper_id) == 36  # UUID format

    def test_two_papers_have_different_ids(self):
        a = PaperRecord(title="A")
        b = PaperRecord(title="B")
        assert a.paper_id != b.paper_id

    def test_display_id_arxiv(self):
        p = PaperRecord(title="A", arxiv_id="2301.00001")
        assert p.display_id() == "arXiv:2301.00001"

    def test_display_id_doi(self):
        p = PaperRecord(title="A", doi="10.1234/test")
        assert p.display_id() == "DOI:10.1234/test"

    def test_display_id_arxiv_preferred_over_doi(self):
        p = PaperRecord(title="A", arxiv_id="2301.00001", doi="10.1234/test")
        assert p.display_id().startswith("arXiv:")

    def test_display_id_fallback(self):
        p = PaperRecord(title="A")
        assert p.display_id().startswith("internal:")

    def test_is_metadata_only_abstract(self):
        p = PaperRecord(title="A", text_coverage=TextCoverage.ABSTRACT_ONLY)
        assert p.is_metadata_only() is True

    def test_is_metadata_only_none(self):
        p = PaperRecord(title="A", text_coverage=TextCoverage.NONE)
        assert p.is_metadata_only() is True

    def test_is_not_metadata_only_full(self):
        p = PaperRecord(title="A", text_coverage=TextCoverage.FULL_TEXT)
        assert p.is_metadata_only() is False

    def test_year_valid(self):
        p = PaperRecord(title="A", year=2023)
        assert p.year == 2023

    def test_year_out_of_range_raises(self):
        with pytest.raises(Exception):
            PaperRecord(title="A", year=999)

    def test_default_source_provider(self):
        p = PaperRecord(title="A")
        assert p.source_provider == SourceProvider.UNKNOWN

    def test_retrieval_timestamp_set(self):
        p = PaperRecord(title="A")
        assert p.retrieval_timestamp  # non-empty ISO string


class TestEvidenceChunk:
    def test_creates_basic_chunk(self):
        chunk = EvidenceChunk(
            paper_id="test-paper-id",
            source_type=TextCoverage.ABSTRACT_ONLY,
            text="This is a test excerpt.",
            is_abstract=True,
        )
        assert chunk.paper_id == "test-paper-id"
        assert chunk.text == "This is a test excerpt."

    def test_content_hash_auto_populated(self):
        chunk = EvidenceChunk(
            paper_id="test-paper-id",
            source_type=TextCoverage.ABSTRACT_ONLY,
            text="Hash me.",
        )
        assert chunk.content_hash  # must be non-empty

    def test_content_hash_deterministic(self):
        text = "Deterministic hash test."
        h1 = _content_hash(text)
        h2 = _content_hash(text)
        assert h1 == h2

    def test_content_hash_different_texts(self):
        assert _content_hash("text A") != _content_hash("text B")

    def test_location_label_abstract(self):
        chunk = EvidenceChunk(
            paper_id="p1",
            source_type=TextCoverage.ABSTRACT_ONLY,
            text="x",
            is_abstract=True,
        )
        assert chunk.location_label() == "abstract"

    def test_location_label_page(self):
        chunk = EvidenceChunk(
            paper_id="p1",
            source_type=TextCoverage.FULL_TEXT,
            text="x",
            page_index=4,
        )
        assert chunk.location_label() == "p.5"  # 0-indexed -> 1-based display

    def test_chunk_id_generated(self):
        chunk = EvidenceChunk(
            paper_id="p1",
            source_type=TextCoverage.EXCERPT,
            text="x",
        )
        assert chunk.chunk_id
        assert len(chunk.chunk_id) == 36


class TestClaimRecord:
    def test_creates_claim(self):
        c = ClaimRecord(claim_text="LLMs can hallucinate.")
        assert c.claim_text == "LLMs can hallucinate."
        assert c.validation_state == ValidationState.UNCHECKED

    def test_claim_id_generated(self):
        c = ClaimRecord(claim_text="A")
        assert c.claim_id


class TestResearchPlan:
    def test_creates_plan(self):
        plan = ResearchPlan(
            original_question="What causes hallucination in LLMs?",
            retrieval_query="LLM hallucination causes",
        )
        assert plan.original_question
        assert plan.retrieval_query

    def test_plan_id_generated(self):
        plan = ResearchPlan(original_question="Q", retrieval_query="q")
        assert plan.plan_id


class TestRunResult:
    def test_creates_run(self):
        r = RunResult(task=TaskType.ANSWER_QUESTION, question="What is retrieval-augmented generation?")
        assert r.task == TaskType.ANSWER_QUESTION
        assert r.llm_calls_used == 0

    def test_is_generated_false_without_model(self):
        r = RunResult(task=TaskType.SUMMARIZE, question="Q")
        assert r.is_generated() is False

    def test_is_generated_true_with_model_and_text(self):
        r = RunResult(
            task=TaskType.SUMMARIZE,
            question="Q",
            model_id="ibm/granite-13b-chat-v2",
            answer_text="Summary text.",
        )
        assert r.is_generated() is True

    def test_mark_finished_sets_latency(self):
        r = RunResult(task=TaskType.EXTRACT, question="Q")
        r.mark_finished()
        assert r.finished_at is not None
        assert r.latency_seconds is not None
        assert r.latency_seconds >= 0.0

    def test_run_id_generated(self):
        r = RunResult(task=TaskType.REPORT, question="Q")
        assert r.run_id


class TestSentinels:
    def test_sentinel_values(self):
        assert NOT_REPORTED == "not_reported"
        assert NOT_AVAILABLE == "not_available"
