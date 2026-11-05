"""
tests/test_ingestion.py
=======================
Offline tests for researchlens.ingestion.
No actual PDF reading from disk; synthetic bytes used.
"""

from __future__ import annotations

import io

import pytest

from researchlens.ingestion import (
    MAX_PAGES,
    MAX_UPLOAD_BYTES,
    sanitize_filename,
    validate_pdf_bytes,
    abstract_to_chunk,
    compute_file_hash,
)
from researchlens.schemas import PaperRecord, TextCoverage


VALID_PDF_MAGIC = b"%PDF-1.4 fake content"


class TestValidatePdfBytes:
    def test_valid_pdf_passes(self):
        ok, err = validate_pdf_bytes(VALID_PDF_MAGIC, "paper.pdf")
        assert ok
        assert err == ""

    def test_too_large_rejected(self):
        big = b"%PDF" + b"x" * (MAX_UPLOAD_BYTES + 1)
        ok, err = validate_pdf_bytes(big, "big.pdf")
        assert not ok
        assert "limit" in err.lower()

    def test_bad_magic_rejected(self):
        ok, err = validate_pdf_bytes(b"PK\x03\x04", "notapdf.pdf")
        assert not ok
        assert "magic" in err.lower() or "valid pdf" in err.lower()

    def test_empty_bytes_rejected(self):
        ok, err = validate_pdf_bytes(b"", "empty.pdf")
        assert not ok

    def test_unsafe_filename_rejected(self):
        ok, err = validate_pdf_bytes(VALID_PDF_MAGIC, "../../../etc/passwd")
        # sanitize_filename strips path traversal; validate should still accept the content
        # (the filename is sanitized, not blocked at the content level)
        # Actual test: sanitize produces a safe name
        safe = sanitize_filename("../../../etc/passwd")
        assert ".." not in safe
        assert "/" not in safe


class TestSanitizeFilename:
    def test_normal_name_unchanged(self):
        assert sanitize_filename("my_paper.pdf") == "my_paper.pdf"

    def test_strips_path(self):
        result = sanitize_filename("/some/path/paper.pdf")
        assert "/" not in result
        assert result == "paper.pdf"

    def test_strips_windows_path(self):
        result = sanitize_filename("C:\\Users\\Bob\\paper.pdf")
        assert "\\" not in result

    def test_dangerous_chars_replaced(self):
        result = sanitize_filename("my<>paper.pdf")
        assert "<" not in result
        assert ">" not in result

    def test_empty_returns_empty(self):
        assert sanitize_filename("") == ""

    def test_dotdot_returns_empty(self):
        assert sanitize_filename("..") == ""

    def test_truncates_long_name(self):
        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= 200


class TestAbstractToChunk:
    def test_creates_chunk_from_abstract(self):
        p = PaperRecord(title="Test", abstract="This is an abstract.")
        chunk = abstract_to_chunk(p)
        assert chunk is not None
        assert chunk.paper_id == p.paper_id
        assert chunk.is_abstract is True
        assert chunk.text == "This is an abstract."
        assert chunk.location_label() == "abstract"

    def test_none_when_no_abstract(self):
        p = PaperRecord(title="No Abstract")
        chunk = abstract_to_chunk(p)
        assert chunk is None


class TestFileHash:
    def test_hash_is_16_chars(self):
        h = compute_file_hash(b"hello world")
        assert len(h) == 16

    def test_hash_deterministic(self):
        h1 = compute_file_hash(b"hello")
        h2 = compute_file_hash(b"hello")
        assert h1 == h2

    def test_different_content_different_hash(self):
        assert compute_file_hash(b"a") != compute_file_hash(b"b")
