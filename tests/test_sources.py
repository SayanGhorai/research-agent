"""
tests/test_sources.py
=====================
Offline tests for researchlens.sources.
All HTTP calls are mocked — no live network requests.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from researchlens.schemas import SourceProvider, TextCoverage
from researchlens.sources import (
    _extract_arxiv_id,
    _normalize_doi,
    _normalize_arxiv_id,
    deduplicate_records,
    search_arxiv,
    search_crossref,
)
from researchlens.schemas import PaperRecord


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

ARXIV_ATOM_RESPONSE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Test Paper on Retrieval Augmented Generation</title>
    <summary>This is the abstract of the test paper about RAG.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link rel="alternate" href="https://arxiv.org/abs/2301.00001"/>
  </entry>
</feed>"""

CROSSREF_JSON_RESPONSE = {
    "message": {
        "items": [
            {
                "DOI": "10.1234/test.2023",
                "title": ["A Crossref Test Paper"],
                "author": [
                    {"given": "Carol", "family": "White"},
                    {"given": "Dan", "family": "Brown"},
                ],
                "published": {"date-parts": [[2023, 5, 1]]},
                "abstract": "<jats:p>This is the abstract from Crossref.</jats:p>",
                "URL": "https://doi.org/10.1234/test.2023",
                "type": "journal-article",
            }
        ]
    }
}


class TestArxivSearch:
    @patch("researchlens.sources._throttle_arxiv")
    @patch("researchlens.sources.requests.get")
    def test_returns_records_on_success(self, mock_get, mock_throttle):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = ARXIV_ATOM_RESPONSE
        mock_get.return_value = resp

        records, err = search_arxiv("retrieval augmented generation")

        assert err is None
        assert len(records) == 1
        assert records[0].title == "Test Paper on Retrieval Augmented Generation"
        assert records[0].source_provider == SourceProvider.ARXIV
        assert records[0].arxiv_id == "2301.00001v1"
        assert records[0].abstract is not None
        assert records[0].year == 2023

    @patch("researchlens.sources._throttle_arxiv")
    @patch("researchlens.sources.requests.get")
    def test_http_error_returns_empty_with_message(self, mock_get, mock_throttle):
        resp = MagicMock()
        resp.status_code = 503
        mock_get.return_value = resp

        records, err = search_arxiv("query")
        assert records == []
        assert "503" in err

    @patch("researchlens.sources._throttle_arxiv")
    @patch("researchlens.sources.requests.get")
    def test_timeout_returns_empty_with_message(self, mock_get, mock_throttle):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.Timeout()

        records, err = search_arxiv("query")
        assert records == []
        assert "timed out" in err.lower()

    @patch("researchlens.sources._throttle_arxiv")
    @patch("researchlens.sources.requests.get")
    def test_empty_feed_returns_empty_list(self, mock_get, mock_throttle):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        mock_get.return_value = resp

        records, err = search_arxiv("query")
        assert records == []
        assert err is None

    @patch("researchlens.sources._throttle_arxiv")
    @patch("researchlens.sources.requests.get")
    def test_abstract_coverage(self, mock_get, mock_throttle):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = ARXIV_ATOM_RESPONSE
        mock_get.return_value = resp

        records, _ = search_arxiv("rag")
        assert records[0].text_coverage == TextCoverage.ABSTRACT_ONLY


class TestCrossrefSearch:
    @patch("researchlens.sources.requests.get")
    def test_returns_records_on_success(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = CROSSREF_JSON_RESPONSE
        mock_get.return_value = resp

        records, err = search_crossref("test query")
        assert err is None
        assert len(records) == 1
        r = records[0]
        assert r.title == "A Crossref Test Paper"
        assert r.doi == "10.1234/test.2023"
        assert r.source_provider == SourceProvider.CROSSREF
        assert r.year == 2023
        assert "abstract from Crossref" in r.abstract

    @patch("researchlens.sources.requests.get")
    def test_jats_xml_stripped_from_abstract(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = CROSSREF_JSON_RESPONSE
        mock_get.return_value = resp

        records, _ = search_crossref("test")
        assert "<jats:p>" not in records[0].abstract

    @patch("researchlens.sources.requests.get")
    def test_rate_limit_returns_error(self, mock_get):
        resp = MagicMock()
        resp.status_code = 429
        resp.headers = {"Retry-After": "5"}
        mock_get.return_value = resp

        records, err = search_crossref("query")
        assert records == []
        assert "rate-limited" in err.lower()

    @patch("researchlens.sources.requests.get")
    def test_missing_abstract_stays_none(self, mock_get):
        data = {
            "message": {"items": [{"DOI": "10.1/x", "title": ["Paper Without Abstract"]}]}
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = data
        mock_get.return_value = resp

        records, _ = search_crossref("query")
        assert records[0].abstract is None
        assert records[0].text_coverage == TextCoverage.NONE

    @patch("researchlens.sources.requests.get")
    def test_missing_authors_stays_empty(self, mock_get):
        data = {
            "message": {"items": [{"DOI": "10.1/y", "title": ["Paper No Authors"]}]}
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = data
        mock_get.return_value = resp

        records, _ = search_crossref("query")
        assert records[0].authors == []


class TestDeduplication:
    def _make_paper(self, doi=None, arxiv_id=None, title="Paper"):
        return PaperRecord(title=title, doi=doi, arxiv_id=arxiv_id)

    def test_no_duplicates_returns_all(self):
        existing = [self._make_paper(doi="10.1/a", title="Paper Alpha")]
        candidates = [self._make_paper(doi="10.1/b", title="Paper Beta")]
        merged, skipped = deduplicate_records(existing, candidates)
        assert len(merged) == 2
        assert skipped == []

    def test_duplicate_doi_skipped(self):
        existing = [self._make_paper(doi="10.1/same")]
        candidates = [self._make_paper(doi="10.1/same")]
        merged, skipped = deduplicate_records(existing, candidates)
        assert len(merged) == 1
        assert len(skipped) == 1

    def test_duplicate_arxiv_skipped(self):
        existing = [self._make_paper(arxiv_id="2301.00001")]
        candidates = [self._make_paper(arxiv_id="2301.00001v2")]  # version stripped
        merged, skipped = deduplicate_records(existing, candidates)
        assert len(merged) == 1
        assert len(skipped) == 1

    def test_identical_title_deduped(self):
        """Identical normalized titles are deduplicated (regression: upload + arXiv same paper)."""
        existing = [self._make_paper(title="Deep Learning for NLP")]
        candidates = [self._make_paper(title="Deep Learning for NLP")]
        merged, skipped = deduplicate_records(existing, candidates)
        assert len(merged) == 1
        assert len(skipped) == 1

    def test_distinct_titles_not_deduped(self):
        """Genuinely different titles are not merged."""
        existing = [self._make_paper(title="Deep Learning for NLP")]
        candidates = [self._make_paper(title="Convolutional Networks for Vision")]
        merged, skipped = deduplicate_records(existing, candidates)
        assert len(merged) == 2
        assert skipped == []

    def test_empty_existing_adds_all(self):
        candidates = [
            self._make_paper(doi="10.1/a", title="Paper Alpha"),
            self._make_paper(doi="10.1/b", title="Paper Beta"),
        ]
        merged, skipped = deduplicate_records([], candidates)
        assert len(merged) == 2
        assert skipped == []


class TestHelpers:
    def test_extract_arxiv_id_from_url(self):
        assert _extract_arxiv_id("http://arxiv.org/abs/2301.00001v1") == "2301.00001v1"

    def test_extract_arxiv_id_bare(self):
        assert _extract_arxiv_id("2301.00001") == "2301.00001"

    def test_normalize_doi_strips_prefix(self):
        assert _normalize_doi("https://doi.org/10.1/x") == "10.1/x"

    def test_normalize_arxiv_strips_version(self):
        assert _normalize_arxiv_id("2301.00001v3") == "2301.00001"
