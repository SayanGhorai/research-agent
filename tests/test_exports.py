"""
tests/test_exports.py
=====================
Offline tests for researchlens.exports.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from researchlens.exports import (
    extraction_to_csv,
    papers_to_bibtex,
    papers_to_csv,
    papers_to_json,
    run_result_to_markdown,
    _csv_safe,
    _bib_escape,
)
from researchlens.schemas import PaperRecord, RunResult, SourceProvider, TaskType, TextCoverage


def _make_paper(**kwargs):
    defaults = dict(
        title="Sample Paper",
        authors=["Alice Smith", "Bob Jones"],
        year=2023,
        doi="10.1234/sample",
        arxiv_id="2301.00001",
        landing_url="https://arxiv.org/abs/2301.00001",
        abstract="This is a sample abstract.",
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
    )
    defaults.update(kwargs)
    return PaperRecord(**defaults)


class TestBibTeX:
    def test_produces_entry(self):
        bib = papers_to_bibtex([_make_paper()])
        assert "@article" in bib or "@misc" in bib
        assert "Sample Paper" in bib

    def test_author_field(self):
        bib = papers_to_bibtex([_make_paper()])
        assert "Alice Smith" in bib

    def test_year_field(self):
        bib = papers_to_bibtex([_make_paper()])
        assert "2023" in bib

    def test_doi_field(self):
        bib = papers_to_bibtex([_make_paper()])
        assert "10.1234/sample" in bib

    def test_empty_list(self):
        assert papers_to_bibtex([]) == ""

    def test_no_doi_uses_misc_type(self):
        p = _make_paper(doi=None)
        bib = papers_to_bibtex([p])
        assert "@misc" in bib

    def test_bib_escape_braces(self):
        assert r"\{" in _bib_escape("a{b}c")


class TestCSV:
    def _parse_csv(self, csv_str: str) -> list[dict]:
        reader = csv.DictReader(io.StringIO(csv_str))
        return list(reader)

    def test_header_present(self):
        csv_str = papers_to_csv([_make_paper()])
        assert "title" in csv_str
        assert "doi" in csv_str

    def test_row_values(self):
        rows = self._parse_csv(papers_to_csv([_make_paper()]))
        assert len(rows) == 1
        assert rows[0]["title"] == "Sample Paper"
        assert rows[0]["doi"] == "10.1234/sample"

    def test_formula_injection_neutralized(self):
        p = _make_paper(title="=SUM(A1:A10)")
        rows = self._parse_csv(papers_to_csv([p]))
        assert not rows[0]["title"].startswith("=")

    def test_csv_safe_prepends_apostrophe(self):
        assert _csv_safe("=dangerous").startswith("'")
        assert _csv_safe("+dangerous").startswith("'")
        assert _csv_safe("-dangerous").startswith("'")
        assert _csv_safe("@dangerous").startswith("'")

    def test_csv_safe_normal_unchanged(self):
        assert _csv_safe("normal text") == "normal text"

    def test_empty_list(self):
        csv_str = papers_to_csv([])
        assert "title" in csv_str  # header still present


class TestJSON:
    def test_valid_json(self):
        json_str = papers_to_json([_make_paper()])
        records = json.loads(json_str)
        assert isinstance(records, list)
        assert len(records) == 1

    def test_fields_present(self):
        records = json.loads(papers_to_json([_make_paper()]))
        assert records[0]["title"] == "Sample Paper"
        assert records[0]["doi"] == "10.1234/sample"
        assert records[0]["source_provider"] == "arxiv"

    def test_empty_list(self):
        records = json.loads(papers_to_json([]))
        assert records == []


class TestMarkdownExport:
    def test_contains_task_and_question(self):
        result = RunResult(task=TaskType.ANSWER_QUESTION, question="What is RAG?")
        result.answer_text = "RAG stands for retrieval-augmented generation."
        md = run_result_to_markdown(result, [])
        assert "What is RAG?" in md
        assert "RAG stands for" in md

    def test_no_answer_shows_placeholder(self):
        result = RunResult(task=TaskType.SUMMARIZE, question="Summarize this.")
        md = run_result_to_markdown(result, [])
        assert "No answer generated" in md

    def test_sources_section_with_papers(self):
        p = _make_paper()
        result = RunResult(task=TaskType.SUMMARIZE, question="Q", paper_ids_used=[p.paper_id])
        md = run_result_to_markdown(result, [p])
        assert "Sources used" in md
        assert "Sample Paper" in md


class TestExtractionCSV:
    def test_basic_extraction(self):
        data = {"objective": "Study RAG", "method": "Experiment", "findings": "Improved recall"}
        csv_str = extraction_to_csv(data, "Paper Title")
        assert "objective" in csv_str
        assert "Study RAG" in csv_str

    def test_none_value_becomes_not_reported(self):
        data = {"metric": None}
        csv_str = extraction_to_csv(data)
        assert "not_reported" in csv_str
