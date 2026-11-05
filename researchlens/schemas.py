"""Typed data contracts shared by ResearchLens AI modules.

Models validate structural integrity and preserve source provenance. A valid
schema or evidence ID is NOT proof that a scientific claim is true.
Pydantic v2 is a required application dependency; do not silently substitute
unvalidated stub classes if installation is incomplete.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


NOT_REPORTED = "not_reported"    # source was read but field was not reported
NOT_AVAILABLE = "not_available"  # source text was not available


class SourceProvider(str, Enum):
    ARXIV = "arxiv"
    CROSSREF = "crossref"
    UPLOAD = "upload"
    UNKNOWN = "unknown"


class TextCoverage(str, Enum):
    ABSTRACT_ONLY = "abstract_only"
    EXCERPT = "excerpt"
    FULL_TEXT = "full_text"
    NONE = "none"


class ValidationState(str, Enum):
    VALID = "valid"       # structural checks passed, NOT fact-check certification
    INVALID = "invalid"   # known structural/evidence problem
    PARTIAL = "partial"   # incomplete/uncertain supporting evidence
    UNCHECKED = "unchecked"


class TaskType(str, Enum):
    ANSWER_QUESTION = "answer_question"
    SUMMARIZE = "summarize"
    EXTRACT = "extract"
    COMPARE = "compare"
    REPORT = "report"
    HYPOTHESES = "hypotheses"
    DRAFT_SECTION = "draft_section"


def _new_id() -> str:
    """New run/session-local record identifier; no user identity is encoded."""
    return str(uuid.uuid4())


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(text: str) -> str:
    """Short fingerprint used for duplicate text detection, not authentication."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


class PaperRecord(BaseModel):
    """A scholarly record; missing bibliographic metadata remains missing."""

    paper_id: str = Field(default_factory=_new_id)
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    landing_url: Optional[str] = None
    abstract: Optional[str] = None

    source_provider: SourceProvider = SourceProvider.UNKNOWN
    text_coverage: TextCoverage = TextCoverage.NONE
    license_note: str = ""
    retrieval_timestamp: str = Field(default_factory=_now_utc)
    file_hash: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_range(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not (1000 <= value <= 2100):
            raise ValueError(f"Year {value!r} is outside plausible range 1000–2100")
        return value

    def display_id(self) -> str:
        if self.arxiv_id:
            return f"arXiv:{self.arxiv_id}"
        if self.doi:
            return f"DOI:{self.doi}"
        return f"internal:{self.paper_id[:8]}"

    def is_metadata_only(self) -> bool:
        return self.text_coverage in (TextCoverage.ABSTRACT_ONLY, TextCoverage.NONE)


class EvidenceChunk(BaseModel):
    """A text passage with a paper identifier and optional zero-based PDF page."""

    chunk_id: str = Field(default_factory=_new_id)
    paper_id: str
    source_type: TextCoverage
    page_index: Optional[int] = Field(default=None, ge=0)
    is_abstract: bool = False
    text: str
    content_hash: str = ""
    extraction_warnings: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash:
            self.content_hash = _content_hash(self.text)

    def location_label(self) -> str:
        if self.is_abstract:
            return "abstract"
        if self.page_index is not None:
            return f"p.{self.page_index + 1}"
        return "unknown"


class ClaimRecord(BaseModel):
    """A generated claim with evidence references; truth is not guaranteed."""

    claim_id: str = Field(default_factory=_new_id)
    claim_text: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    validation_state: ValidationState = ValidationState.UNCHECKED
    validation_notes: str = ""


class ResearchPlan(BaseModel):
    """A bounded plan for querying scholarly metadata sources."""

    plan_id: str = Field(default_factory=_new_id)
    original_question: str
    retrieval_query: str
    search_terms: list[str] = Field(default_factory=list)
    providers: list[SourceProvider] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    desired_output: str = ""
    created_at: str = Field(default_factory=_now_utc)


class RunResult(BaseModel):
    """Result and observed usage for one user-initiated research action."""

    run_id: str = Field(default_factory=_new_id)
    task: TaskType
    question: str

    answer_text: str = ""
    claims: list[ClaimRecord] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    paper_ids_used: list[str] = Field(default_factory=list)

    source_coverage: TextCoverage = TextCoverage.NONE
    validation_state: ValidationState = ValidationState.UNCHECKED
    warnings: list[str] = Field(default_factory=list)

    model_id: Optional[str] = None
    input_tokens_estimate: Optional[int] = Field(default=None, ge=0)
    output_tokens_reported: Optional[int] = Field(default=None, ge=0)
    llm_calls_used: int = Field(default=0, ge=0)

    started_at: str = Field(default_factory=_now_utc)
    finished_at: Optional[str] = None
    latency_seconds: Optional[float] = Field(default=None, ge=0)

    def is_generated(self) -> bool:
        """Model identity and nonempty output indicate generation, not correctness."""
        return bool(self.model_id and self.answer_text.strip())

    def mark_finished(self) -> None:
        """Record wall-clock duration; tolerate malformed legacy timestamps."""
        self.finished_at = _now_utc()
        try:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.finished_at)
            self.latency_seconds = max(0.0, (end - start).total_seconds())
        except (ValueError, TypeError, OverflowError):
            self.latency_seconds = None