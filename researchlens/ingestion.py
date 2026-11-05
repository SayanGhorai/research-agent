"""ResearchLens AI — safe PDF ingestion and page-indexed text extraction.

A successful upload must have readable, attributable PDF text. This module
never sends document content to a model, downloads anything, or writes files.

Limits are validated before reading content where possible. pypdf extraction
runs synchronously: elapsed-time checks are warnings, *not* a preemptive timeout.
"""

from __future__ import annotations

import hashlib
import io
import re
import time
from typing import Optional

from researchlens.schemas import EvidenceChunk, PaperRecord, SourceProvider, TextCoverage

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB
MAX_PAGES = 40
MAX_PAGE_PARSE_SECONDS = 5.0
_PDF_MAGIC = b"%PDF"


def sanitize_filename(name: str) -> str:
    """Return a harmless display filename, handling Unix and Windows separators.

    The string is never used to build a filesystem path. The original name may
    contain a path or unsupported characters; only the final component remains.
    """
    if not isinstance(name, str) or not name:
        return ""
    filename = re.split(r"[/\\]", name)[-1]
    filename = re.sub(r"[^\w\s.\-]", "_", filename, flags=re.UNICODE)
    filename = filename.strip(". \t\r\n")
    if not filename or filename in {".", ".."}:
        return ""
    # Keep the file extension visible even for unusually long upload names.
    if len(filename) > 200:
        if filename.lower().endswith(".pdf"):
            filename = filename[:196] + filename[-4:]
        else:
            filename = filename[:200]
    return filename


def validate_pdf_bytes(
    content: bytes,
    original_filename: str,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> tuple[bool, str]:
    """Check upload size, safe PDF filename and basic signature.

    The %PDF prefix is only an initial filter. PdfReader performs structural
    validation; neither the extension nor prefix proves that a file is safe.
    """
    if not isinstance(content, bytes):
        return False, "Uploaded document must be raw PDF bytes"
    safe_name = sanitize_filename(original_filename)
    if not safe_name or not safe_name.lower().endswith(".pdf"):
        return False, "Upload must have a valid .pdf filename"
    if max_bytes < 4:
        return False, "Invalid PDF upload size limit"
    if len(content) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        return False, f"File exceeds {limit_mb:g} MiB limit"
    if not content.startswith(_PDF_MAGIC):
        return False, "File does not appear to be a valid PDF (bad magic bytes)"
    return True, ""


def compute_file_hash(content: bytes) -> str:
    """Stable short SHA-256 fingerprint compatible with existing paper records."""
    return hashlib.sha256(content).hexdigest()[:16]


def is_duplicate_pdf(content: bytes, papers: list[PaperRecord]) -> bool:
    """Detect repeated uploads by bytes, not sanitized filenames or titles."""
    digest = compute_file_hash(content)
    return any(
        paper.source_provider == SourceProvider.UPLOAD and paper.file_hash == digest
        for paper in papers
    )


def extract_pdf_pages(
    content: bytes,
    paper_id: str,
    max_pages: int = MAX_PAGES,
) -> tuple[list[EvidenceChunk], list[str], TextCoverage]:
    """Extract one EvidenceChunk per readable PDF page (zero-based page_index).

    Refuses documents above the page limit instead of silently indexing only
    the beginning. Unreadable pages make coverage EXCERPT, not FULL_TEXT.
    Text embedded only in figures or scans remains unavailable; text coverage
    is not a guarantee that all tables and figures were extracted correctly.
    """
    if max_pages < 1:
        return [], ["Invalid maximum page count"], TextCoverage.NONE
    try:
        from pypdf import PdfReader
    except ImportError:
        return [], ["pypdf is not installed; PDF extraction unavailable"], TextCoverage.NONE

    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        if reader.is_encrypted:
            # Reject encryption even when a blank password might work; the
            # project accepts only freely readable, unlocked PDFs.
            return [], ["PDF is encrypted; upload an unlocked, text-readable PDF"], TextCoverage.NONE
        page_count = len(reader.pages)
    except Exception:
        return [], ["PDF is malformed or cannot be opened"], TextCoverage.NONE

    if page_count < 1:
        return [], ["PDF has no pages"], TextCoverage.NONE
    if page_count > max_pages:
        return [], [
            f"PDF has {page_count} pages, exceeding the {max_pages}-page limit; "
            "no pages were indexed"
        ], TextCoverage.NONE

    chunks: list[EvidenceChunk] = []
    warnings: list[str] = []
    empty_pages: list[int] = []
    unreadable_pages: list[int] = []

    for page_index in range(page_count):
        start = time.monotonic()
        try:
            text = reader.pages[page_index].extract_text() or ""
        except Exception:
            unreadable_pages.append(page_index + 1)
            warnings.append(f"Page {page_index + 1}: text extraction failed")
            continue

        elapsed = time.monotonic() - start
        slow = elapsed > MAX_PAGE_PARSE_SECONDS
        if slow:
            warnings.append(
                f"Page {page_index + 1}: text extraction took {elapsed:.1f}s; "
                "this is a warning, not an enforced timeout"
            )

        text = text.strip()
        if not text:
            empty_pages.append(page_index + 1)
            continue

        chunk_warnings = ["slow extraction; check page completeness"] if slow else []
        chunks.append(EvidenceChunk(
            paper_id=paper_id,
            source_type=TextCoverage.FULL_TEXT,
            page_index=page_index,
            is_abstract=False,
            text=text,
            extraction_warnings=chunk_warnings,
        ))

    if empty_pages:
        warnings.append(
            f"Pages without extractable text: {empty_pages[:10]}"
            + (" ..." if len(empty_pages) > 10 else "")
        )
    if not chunks:
        return [], warnings + [
            "No readable PDF text found; scanned pages may require OCR outside this app"
        ], TextCoverage.NONE

    if empty_pages or unreadable_pages:
        for chunk in chunks:
            chunk.source_type = TextCoverage.EXCERPT
        return chunks, warnings, TextCoverage.EXCERPT
    return chunks, warnings, TextCoverage.FULL_TEXT


def abstract_to_chunk(paper: PaperRecord) -> Optional[EvidenceChunk]:
    """Convert known abstract metadata into evidence with honest coverage."""
    if not paper.abstract or not paper.abstract.strip():
        return None
    return EvidenceChunk(
        paper_id=paper.paper_id,
        source_type=TextCoverage.ABSTRACT_ONLY,
        page_index=None,
        is_abstract=True,
        text=paper.abstract.strip(),
        extraction_warnings=[],
    )


def ingest_pdf(
    content: bytes,
    original_filename: str,
    title: str = "",
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_pages: int = MAX_PAGES,
) -> tuple[Optional[PaperRecord], list[EvidenceChunk], list[str]]:
    """Validate, parse, and prepare an uploaded paper for indexing.

    Returns (None, [], warnings) for malformed, oversized, encrypted or wholly
    unreadable PDFs. Partial readable PDFs are retained but marked EXCERPT.
    Index capacity is enforced separately by CorpusIndex; callers must check
    that the index accepted every passage before appending the paper to a library.
    """
    ok, err = validate_pdf_bytes(content, original_filename, max_bytes)
    if not ok:
        return None, [], [err]

    safe_name = sanitize_filename(original_filename)
    paper = PaperRecord(
        title=title.strip() or safe_name,
        source_provider=SourceProvider.UPLOAD,
        text_coverage=TextCoverage.NONE,
        license_note="User-uploaded PDF; redistribution requires appropriate permission",
        file_hash=compute_file_hash(content),
    )
    chunks, warnings, coverage = extract_pdf_pages(
        content, paper.paper_id, max_pages=max_pages
    )
    if not chunks:
        return None, [], warnings
    paper.text_coverage = coverage
    return paper, chunks, warnings