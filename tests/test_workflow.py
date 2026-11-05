"""
tests/test_workflow.py
======================
Offline tests for researchlens.workflow.
All network and IBM calls are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from researchlens.config import Settings, reset_settings
from researchlens.retrieval import CorpusIndex
from researchlens.schemas import (
    EvidenceChunk,
    PaperRecord,
    RunResult,
    SourceProvider,
    TaskType,
    TextCoverage,
    ValidationState,
)
from researchlens.watsonx_client import (
    MockWatsonxClient,
    UsageRecord,
    WatsonxGenerationDisabled,
)
from researchlens.workflow import ResearchWorkflow, _index_papers, index_uploaded_paper


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _settings_disabled() -> Settings:
    reset_settings()
    return Settings(
        watsonx_api_key="",
        watsonx_project_id="",
        watsonx_url="",
        watsonx_model_id="",
        allow_live_generation=False,
    )


def _settings_enabled() -> Settings:
    reset_settings()
    return Settings(
        watsonx_api_key="fake-key",
        watsonx_project_id="fake-proj",
        watsonx_url="https://us-south.ml.cloud.ibm.com",
        watsonx_model_id="ibm/granite-13b-chat-v2",
        allow_live_generation=True,
        max_llm_calls_per_action=3,
    )


def _make_paper(pid="p1"):
    return PaperRecord(
        paper_id=pid,
        title="Test Paper",
        authors=["Alice"],
        year=2023,
        abstract="This paper studies retrieval augmented generation.",
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
    )


def _make_chunk(paper_id="p1"):
    return EvidenceChunk(
        paper_id=paper_id,
        source_type=TextCoverage.ABSTRACT_ONLY,
        is_abstract=True,
        text="Retrieval augmented generation (RAG) combines search with generation.",
    )


# ------------------------------------------------------------------
# Workflow tests — generation disabled
# ------------------------------------------------------------------

class TestWorkflowGenerationDisabled:
    @patch("researchlens.agents.search_arxiv", return_value=([], None))
    @patch("researchlens.agents.search_crossref", return_value=([], None))
    def test_run_without_generation(self, mock_cr, mock_ax):
        wf = ResearchWorkflow(_settings_disabled())
        index = CorpusIndex(enable_semantic=False)
        result, lib, warnings = wf.run(
            TaskType.ANSWER_QUESTION, "What is RAG?", [], index,
            search_first=True,
        )
        assert isinstance(result, RunResult)
        assert "disabled" in result.answer_text.lower() or "generation" in result.answer_text.lower()

    @patch("researchlens.agents.search_arxiv", return_value=([], None))
    @patch("researchlens.agents.search_crossref", return_value=([], None))
    def test_run_result_has_finished_at(self, mock_cr, mock_ax):
        wf = ResearchWorkflow(_settings_disabled())
        index = CorpusIndex(enable_semantic=False)
        result, _, _ = wf.run(TaskType.SUMMARIZE, "Summarize", [], index)
        assert result.finished_at is not None
        assert result.latency_seconds is not None

    @patch("researchlens.agents.search_arxiv", return_value=([], None))
    @patch("researchlens.agents.search_crossref", return_value=([], None))
    def test_library_updated_with_new_papers(self, mock_cr, mock_ax):
        paper = _make_paper()
        mock_ax.return_value = ([paper], None)
        wf = ResearchWorkflow(_settings_disabled())
        index = CorpusIndex(enable_semantic=False)
        _, lib, _ = wf.run(TaskType.ANSWER_QUESTION, "RAG", [], index)
        assert len(lib) >= 1

    @patch("researchlens.agents.search_arxiv", return_value=([], None))
    @patch("researchlens.agents.search_crossref", return_value=([], None))
    def test_duplicate_papers_not_added_twice(self, mock_cr, mock_ax):
        # Paper with arXiv ID so deduplication can detect it
        paper = PaperRecord(
            paper_id="p1", title="Test Paper", authors=["Alice"], year=2023,
            arxiv_id="2301.99999",
            abstract="This paper studies retrieval augmented generation.",
            source_provider=SourceProvider.ARXIV,
            text_coverage=TextCoverage.ABSTRACT_ONLY,
        )
        mock_ax.return_value = ([paper], None)
        wf = ResearchWorkflow(_settings_disabled())
        index = CorpusIndex(enable_semantic=False)
        # Paper already in library; arXiv dedup should skip it
        _, lib, _ = wf.run(TaskType.ANSWER_QUESTION, "RAG", [paper], index)
        # Should not be added twice
        arxiv_ids = [p.arxiv_id for p in lib if p.arxiv_id]
        assert arxiv_ids.count("2301.99999") == 1

    @patch("researchlens.agents.search_arxiv", return_value=([], None))
    @patch("researchlens.agents.search_crossref", return_value=([], None))
    def test_no_search_when_search_first_false(self, mock_cr, mock_ax):
        wf = ResearchWorkflow(_settings_disabled())
        index = CorpusIndex(enable_semantic=False)
        wf.run(TaskType.SUMMARIZE, "Q", [], index, search_first=False)
        mock_ax.assert_not_called()
        mock_cr.assert_not_called()


# ------------------------------------------------------------------
# Workflow tests — generation mocked
# ------------------------------------------------------------------

class TestWorkflowGenerationMocked:
    @patch("researchlens.agents.search_arxiv", return_value=([], None))
    @patch("researchlens.agents.search_crossref", return_value=([], None))
    @patch("researchlens.workflow.get_client")
    def test_run_with_mocked_generation(self, mock_get_client, mock_cr, mock_ax):
        mock_client = MagicMock()
        mock_client.reset_budget.return_value = None
        mock_client.action_calls_used = 1
        mock_client.chat.return_value = (
            "RAG improves retrieval quality.",
            UsageRecord(input_tokens=30, output_tokens=20, model_id="ibm/test"),
        )
        mock_get_client.return_value = mock_client

        s = _settings_enabled()
        s.allow_live_generation = True
        wf = ResearchWorkflow(s)
        index = CorpusIndex(enable_semantic=False)
        paper = _make_paper()
        chunk = _make_chunk(paper.paper_id)
        index.add_chunks([chunk])

        result, _, _ = wf.run(TaskType.ANSWER_QUESTION, "What is RAG?", [paper], index, search_first=False)
        assert "RAG" in result.answer_text
        assert result.model_id == "ibm/test"

    @patch("researchlens.agents.search_arxiv", return_value=([], None))
    @patch("researchlens.agents.search_crossref", return_value=([], None))
    @patch("researchlens.workflow.get_client")
    def test_ibm_error_captured_in_answer(self, mock_get_client, mock_cr, mock_ax):
        from researchlens.watsonx_client import WatsonxError
        mock_client = MagicMock()
        mock_client.reset_budget.return_value = None
        mock_client.action_calls_used = 1
        mock_client.chat.side_effect = WatsonxError("IBM connection failed")
        mock_get_client.return_value = mock_client

        s = _settings_enabled()
        wf = ResearchWorkflow(s)
        index = CorpusIndex(enable_semantic=False)
        paper = _make_paper()
        chunk = _make_chunk(paper.paper_id)
        index.add_chunks([chunk])

        # Use an answerable query so the safe no-evidence gate does not short-circuit.
        result, _, warnings = wf.run(TaskType.ANSWER_QUESTION, "retrieval", [paper], index, search_first=False)
        assert "error" in result.answer_text.lower() or "IBM" in result.answer_text


# ------------------------------------------------------------------
# Indexing helpers
# ------------------------------------------------------------------

class TestIndexHelpers:
    def test_index_papers_adds_abstract_chunks(self):
        index = CorpusIndex(enable_semantic=False)
        papers = [_make_paper("p1"), _make_paper("p2")]
        _index_papers(papers, index)
        assert index.chunk_count() == 2

    def test_index_papers_no_abstract_skipped(self):
        index = CorpusIndex(enable_semantic=False)
        paper = PaperRecord(title="No Abstract", paper_id="p3")
        _index_papers([paper], index)
        assert index.chunk_count() == 0

    def test_index_uploaded_paper_adds_all_chunks(self):
        index = CorpusIndex(enable_semantic=False)
        paper = _make_paper("upload1")
        page_chunks = [
            EvidenceChunk(paper_id="upload1", source_type=TextCoverage.FULL_TEXT,
                          page_index=0, text="Page one text."),
            EvidenceChunk(paper_id="upload1", source_type=TextCoverage.FULL_TEXT,
                          page_index=1, text="Page two text."),
        ]
        index_uploaded_paper(paper, page_chunks, index)
        # Only page chunks are indexed — abstract is NOT added when page chunks exist
        # (prevents duplicate/misleading retrieval from abstract + full text)
        assert index.chunk_count() == 2
