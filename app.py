"""ResearchLens AI — Streamlit application.

This is the user-interface/orchestration layer. The project's researchlens
modules own PDF parsing, retrieval, IBM inference, evidence validation and
exports. Nothing is sent to IBM merely by opening or navigating the app.

Launch (inside the project's venv): python -m streamlit run app.py
"""

from __future__ import annotations

import csv
import hmac
import inspect
import io
import re
from typing import Any
from urllib.parse import urlsplit

import streamlit as st

from researchlens.config import get_settings
from researchlens.exports import (
    evidence_ledger_to_markdown,
    extraction_to_csv,
    papers_to_bibtex,
    papers_to_csv,
    papers_to_json,
    run_result_to_markdown,
    run_trace_to_markdown,
)
from researchlens.ingestion import compute_file_hash, ingest_pdf, is_duplicate_pdf
from researchlens.retrieval import CorpusIndex
from researchlens.schemas import (
    PaperRecord,
    RunResult,
    SourceProvider,
    TaskType,
    TextCoverage,
    ValidationState,
)
from researchlens.ui import (
    TASK_OPTIONS,
    render_config_status,
    render_empty_library,
    render_empty_results,
    render_generation_disabled_notice,
    render_task_selector,
)
from researchlens.workflow import ResearchWorkflow, index_uploaded_paper, _parsed_grounded_fields


st.set_page_config(
    page_title="ResearchLens AI | Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# An application-level *per-session* guard only. This is not a global usage,
# spending, authentication, or rate limit; see diagnostics and deployment notes.
MAX_UI_MODEL_CALLS_PER_SESSION = 5
UI_REVISION = "prototype-ui-2026-11-05-v1"
EXTRACTION_FIELDS = (
    ("research_objective", "Research Objective"),
    ("method", "Method"),
    ("dataset_sample", "Dataset / Sample"),
    ("key_findings", "Key Findings"),
    ("metrics", "Metrics"),
    ("limitations", "Limitations"),
)


def _init_session() -> None:
    """Session-scoped user data; do not put uploaded text in global caches."""
    defaults: dict[str, Any] = {
        "library": None,
        "corpus_index": None,
        "workflow": None,
        "last_result": None,
        "last_trace": None,
        "last_result_label": "",
        "run_count": 0,
        "ui_model_calls": 0,
        "upload_epoch": 0,
        "selection_epoch": 0,
        "grounded_extractions": None,
        "grounded_comparison": None,
        "workspace_note": "",
        "action_notices": None,
        "latest_attempt": None,
        "authenticated": False,
        # Failed PDFs are memoized only inside this browser session so a rejected
        # upload is not reparsed on every unrelated widget rerun.
        "upload_failures": None,
    }
    for name, default in defaults.items():
        if name not in st.session_state:
            st.session_state[name] = default
    if st.session_state.library is None:
        st.session_state.library = []
    if st.session_state.grounded_extractions is None:
        st.session_state.grounded_extractions = {}
    if st.session_state.upload_failures is None:
        st.session_state.upload_failures = {}
    if st.session_state.action_notices is None:
        st.session_state.action_notices = {}


def _clear_generated() -> None:
    st.session_state.last_result = None
    st.session_state.last_trace = None
    st.session_state.last_result_label = ""
    st.session_state.grounded_extractions = {}
    st.session_state.grounded_comparison = None
    st.session_state.action_notices = {}
    st.session_state.latest_attempt = None


def _clear_session() -> None:
    """Forget local session material, reset uploader and optional selections."""
    st.session_state.library = []
    st.session_state.corpus_index = CorpusIndex(
        enable_semantic=settings.enable_semantic_search
    )
    st.session_state.workflow = ResearchWorkflow(settings)
    _clear_generated()
    st.session_state.run_count = 0
    # Preserve observed inference attempts on Clear. This is still only a
    # browser-session counter, never a global IBM quota or billing safeguard.
    st.session_state.upload_failures = {}
    st.session_state.upload_epoch += 1
    st.session_state.selection_epoch += 1
    st.session_state.workspace_note = ""
    st.session_state.authenticated = False


def _display_paper(pid: str) -> str:
    paper = next(
        (p for p in st.session_state.library if p.paper_id == pid), None
    )
    if paper is None:
        return f"Unlisted source [{pid[:8]}]"
    return f"{paper.title[:95]} [{paper.display_id()}]"


def _paper_ids() -> list[str]:
    return [p.paper_id for p in st.session_state.library]


def _selected_papers(ids: list[str]) -> list[PaperRecord]:
    selected = set(ids)
    return [p for p in st.session_state.library if p.paper_id in selected]


def _update_model_call_counter() -> None:
    """Update the sidebar counter within the same Streamlit script execution.

    Sidebar widgets render before the main form, so without updating the
    existing placeholder the user sees the previous call count until rerun.
    This is still only a browser-session counter, not a provider quota.
    """
    placeholder = globals().get("_model_calls_placeholder")
    if placeholder is not None:
        placeholder.caption(
            f"Model calls recorded in this session: "
            f"{st.session_state.ui_model_calls}/{MAX_UI_MODEL_CALLS_PER_SESSION}"
        )


def _record_action(result: RunResult) -> None:
    """Count both successful and failed model attempts returned by the workflow."""
    st.session_state.run_count += 1
    st.session_state.ui_model_calls += max(0, result.llm_calls_used)
    _update_model_call_counter()


def _save_result(result: RunResult, label: str) -> None:
    st.session_state.last_result = result
    st.session_state.last_result_label = label
    st.session_state.last_trace = run_trace_to_markdown(result)
    _record_action(result)


def _allow_task() -> bool:
    """Check the local UI gate without printing validation warnings."""
    return not _task_block_reason()


def _run_research(
    task: TaskType, question: str, *, paper_ids: list[str] | None,
    search_first: bool, label: str, update_primary: bool = True,
    action_context: str = "workspace",
) -> RunResult | None:
    """Execute only on submission; save validation warnings below the result."""
    _begin_attempt(action_context, label)
    if not question.strip():
        _block_attempt(action_context, label, "Enter a research question or task description.")
        return None
    reason = _task_block_reason()
    if reason:
        _block_attempt(action_context, label, reason)
        return None
    if not search_first and not st.session_state.library:
        _block_attempt(action_context, label, "Upload or discover papers before running without literature search.")
        return None
    if paper_ids is not None and not paper_ids:
        _block_attempt(action_context, label, "Choose at least one source paper.")
        return None
    ui_warnings: list[str] = []
    if paper_ids and search_first:
        ui_warnings.append("Selected-paper scope took priority; external discovery was disabled for this action.")
        search_first = False
    existing_ids = {p.paper_id for p in st.session_state.library}
    try:
        with st.spinner("Retrieving evidence and preparing the research result…"):
            result, new_library, search_warnings = st.session_state.workflow.run(
                task=task, question=question.strip(),
                library=list(st.session_state.library),
                index=st.session_state.corpus_index,
                search_first=search_first, selected_paper_ids=paper_ids,
            )
    except Exception as exc:
        # Unexpected failures can happen after a provider attempt; usage is unknown.
        _block_attempt(
            action_context, label,
            f"Research task failed ({type(exc).__name__}). No new result is available. "
            "Provider usage could not be confirmed; do not retry automatically.", status="error",
        )
        return None
    fresh = [p for p in new_library if p.paper_id not in existing_ids and p.abstract]
    if fresh:
        indexed = {c.paper_id for c in st.session_state.corpus_index._chunks}
        missing = [p for p in fresh if p.paper_id not in indexed]
        if missing:
            ui_warnings.append(
                f"{len(missing)} discovered abstract(s) were not indexed. "
                "These records are bibliography metadata only, not searchable evidence."
            )
    result.warnings = list(dict.fromkeys([
        *ui_warnings, *list(search_warnings or []), *result.warnings,
    ]))
    st.session_state.library = new_library
    if update_primary:
        _save_result(result, label)
    else:
        _record_action(result)
    st.session_state.latest_attempt = {
        "context": action_context, "label": label,
        "status": "completed", "run_id": result.run_id,
    }
    st.session_state.action_notices.pop(action_context, None)
    return result


def _coverage_label(coverage: TextCoverage) -> str:
    return {
        TextCoverage.FULL_TEXT: "🟢 Full text",
        TextCoverage.EXCERPT: "🟡 Partial text",
        TextCoverage.ABSTRACT_ONLY: "🟠 Abstract only",
        TextCoverage.NONE: "⚫ No readable text",
    }.get(coverage, "Unknown coverage")


def _show_evidence(result: RunResult, *, limit: int = 12) -> None:
    """Expose provenance and page IDs for users to inspect themselves."""
    index: CorpusIndex = st.session_state.corpus_index
    paper_map = {p.paper_id: p for p in st.session_state.library}
    chunks = index.get_chunks_by_ids(result.evidence_chunk_ids)
    if not chunks:
        st.info("No retrievable source passages were linked to this result.")
        return
    for chunk in chunks[:limit]:
        paper = paper_map.get(chunk.paper_id)
        title = paper.title if paper else "Unknown source"
        with st.expander(
            f"[{chunk.chunk_id[:8]}] {title[:75]} — {chunk.location_label()}"
        ):
            st.caption(f"Paper ID: {chunk.paper_id} · {_coverage_label(chunk.source_type)}")
            st.text(chunk.text[:2400])
            for warning in chunk.extraction_warnings:
                st.caption(f"Extraction warning: {warning}")
    if len(chunks) > limit:
        st.caption(f"Displaying {limit} of {len(chunks)} evidence passages.")


def _show_result(result: RunResult, *, label: str = "Research result") -> None:
    """Render the draft first, then status and one set of validation details."""
    st.subheader(label)
    if result.model_id:
        st.caption("AI-generated draft — review the original sources before use.")
        st.markdown(result.answer_text or "_No answer was generated._")
    elif _non_generated_message(result):
        st.info(_non_generated_message(result))
    else:
        st.markdown(result.answer_text or "_No answer was generated._")
    st.caption(
        f"{_coverage_label(result.source_coverage)} · "
        f"{len(result.paper_ids_used)} paper(s) · "
        f"{len(result.evidence_chunk_ids)} passage(s) · "
        f"Validation: {result.validation_state.value}"
    )
    st.caption("Validation checks are structural and evidence-scoped; they do not independently establish scientific truth.")
    if result.model_id and result.validation_state in {ValidationState.INVALID, ValidationState.PARTIAL}:
        st.warning("This draft needs review. Inspect the validation details and linked evidence below.")
    if result.warnings:
        with st.expander("Validation warnings — review details", expanded=False):
            for warning in dict.fromkeys(result.warnings):
                st.warning(warning)
    if result.model_id or result.llm_calls_used:
        with st.expander("IBM Granite usage"):
            st.caption(f"Model: `{result.model_id or 'not reported'}`")
            st.caption(f"Calls reported: {result.llm_calls_used}")
            if result.input_tokens_estimate is not None:
                st.caption(f"Input tokens (provider value/estimate): {result.input_tokens_estimate}")
            if result.output_tokens_reported is not None:
                st.caption(f"Output tokens (reported): {result.output_tokens_reported}")
            if result.latency_seconds is not None:
                st.caption(f"Latency: {result.latency_seconds:.2f} seconds")
    if result.evidence_chunk_ids:
        st.markdown("**Source evidence and PDF page references**")
        _show_evidence(result)


def _parse_structured_extraction(
    text: str,
    allowed_chunk_ids: set[str] | None = None,
    verified_claims: list | None = None,
) -> dict[str, str] | None:
    """Accept only labeled fields with citations from the selected paper.

    A structurally plausible response is not proof of factual correctness. Only
    allow a structured CSV when at least one field cites an available passage.
    """
    if not allowed_chunk_ids:
        return None
    if verified_claims is not None:
        # Use the workflow's individual claim validation; a matching passage ID
        # alone cannot make a partially supported extraction safe to export.
        fields = _parsed_grounded_fields(
            text,
            allowed_chunk_ids,
            verified_claims=verified_claims,
        )
        return fields or None
    aliases = {
        "research objective": "research_objective",
        "objective": "research_objective",
        "problem": "research_objective",
        "method": "method",
        "methodology": "method",
        "dataset / sample": "dataset_sample",
        "dataset/sample": "dataset_sample",
        "dataset": "dataset_sample",
        "sample": "dataset_sample",
        "key findings": "key_findings",
        "findings": "key_findings",
        "results": "key_findings",
        "metrics": "metrics",
        "evaluation metrics": "metrics",
        "limitations": "limitations",
        "limitation": "limitations",
    }
    found: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
            if len(cells) != 2:
                continue
            label, value = cells
        else:
            line = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", line)
            line = line.replace("**", "").replace("__", "")
            match = re.match(r"^([A-Za-z /]+?)\s*:\s*(.+)$", line)
            if not match:
                continue
            label, value = match.group(1), match.group(2)
        key = aliases.get(label.strip().lower())
        if not key or key in found:
            continue
        value = value.strip().strip("*")
        if not value or re.sub(r"[\s_*-]+", " ", value).strip().casefold() in {
            "not reported", "not available", "unknown", "n/a", "none"
        }:
            continue
        cited = set(re.findall(r"\[([0-9a-fA-F]{8})\]", value))
        if cited and cited.issubset(allowed_chunk_ids):
            found[key] = value
    return found or None


def _valid_external_url(url: str | None) -> bool:
    """Accept only plain web links, never credentials or control characters."""
    if not url or re.search(r"[\s<>\\]", url):
        return False
    try:
        parsed = urlsplit(url)
        if (parsed.scheme.lower() not in {"http", "https"}
                or not parsed.hostname or parsed.username or parsed.password):
            return False
        _ = parsed.port  # reject invalid port syntax
        return True
    except ValueError:
        return False


def _report_sources(
    result: RunResult,
    library: list[PaperRecord],
    index: CorpusIndex | None = None,
) -> list[PaperRecord]:
    """Return papers explicitly cited by answer claims, not merely retrieved.

    Uncited retrieved passages are useful to inspect but must not appear as
    bibliographic references actually cited in the generated answer.
    """
    if index is None:
        # Without the index we cannot reliably resolve claim IDs to sources.
        return []
    cited_chunk_ids = {
        chunk_id
        for claim in result.claims
        for chunk_id in claim.evidence_chunk_ids
    }
    cited_paper_ids = {
        chunk.paper_id
        for chunk in index.get_chunks_by_ids(sorted(cited_chunk_ids))
    }
    return [paper for paper in library if paper.paper_id in cited_paper_ids]


# ------------------------------------------------------------------
# Current action versus saved output: no automatic retries or new calls
# ------------------------------------------------------------------

def _task_block_reason() -> str:
    if not settings.generation_enabled():
        return ""  # Offline retrieval still works, but no generation occurs.
    if settings.app_env.strip().lower() in {"production", "prod"} and not settings.app_access_password:
        return "Live generation is blocked in production without APP_ACCESS_PASSWORD. Configure an access gate and provider-side usage controls first."
    if settings.app_access_password and not st.session_state.authenticated:
        return "Unlock the Research Agent to use live generation."
    if st.session_state.ui_model_calls >= MAX_UI_MODEL_CALLS_PER_SESSION:
        return (f"This session has reached its {MAX_UI_MODEL_CALLS_PER_SESSION}-call UI budget. "
                "No new model request was sent for this action. This is a session guard, not an IBM account-wide quota.")
    return ""


def _begin_attempt(context: str, label: str) -> None:
    st.session_state.action_notices.clear()
    st.session_state.latest_attempt = {"context": context, "label": label, "status": "pending", "run_id": None}


def _block_attempt(context: str, label: str, message: str, *, status: str = "blocked") -> None:
    """Preserve prior work, but never present it as a response to this attempt."""
    st.session_state.action_notices.clear()
    st.session_state.action_notices[context] = {"label": label, "message": message, "status": status}
    st.session_state.latest_attempt = {"context": context, "label": label, "status": status, "run_id": None}


def _show_action_notice(context: str) -> None:
    notice = st.session_state.action_notices.get(context)
    if notice:
        text = f"{notice['label']}: {notice['message']}"
        if notice["status"] == "error":
            st.error(text)
        else:
            st.warning(text)


def _is_previous_result(result: RunResult) -> bool:
    attempt = st.session_state.latest_attempt or {}
    return bool(attempt and attempt.get("run_id") != result.run_id)


def _show_workspace_output() -> None:
    _show_action_notice("workspace")
    result = st.session_state.last_result
    if result is None:
        if not st.session_state.action_notices.get("workspace"):
            render_empty_results()
        return
    label = st.session_state.last_result_label or "Research result"
    if _is_previous_result(result):
        st.info("The latest attempt did not produce this output. The previous saved result is available below.")
        with st.expander("Previous saved result — not generated for the latest attempt", expanded=False):
            st.caption(f"Previous task: {label} · Validation: {result.validation_state.value}")
            st.markdown(result.answer_text or "_No saved answer._")
            st.caption("This is a previous draft. Its full validation details and downloads remain in Reports & References.")
    else:
        _show_result(result, label=label)


def _non_generated_message(result: RunResult) -> str:
    text = result.answer_text.strip()
    if text.startswith("[Generation disabled]"):
        return ("IBM Granite generation is disabled. No AI answer, extraction, or comparison was generated. "
                "Any retrieved passages are available for inspection below.")
    if text.startswith("[Budget exhausted]"):
        return "The per-action model-call budget was exhausted. No generated answer is available for this action."
    if text.startswith("[Research limit]"):
        return text.replace("[Research limit]", "Local research safeguard:", 1)
    if text.startswith("[IBM error]"):
        return text.replace("[IBM error]", "IBM generation was unsuccessful:", 1)
    return ""


def _upload_size_options() -> dict[str, int]:
    """Set the widget limit on supported versions; backend validation always stays."""
    if "max_upload_size" in inspect.signature(st.file_uploader).parameters:
        return {"max_upload_size": settings.max_upload_mb}
    return {}


# ------------------------------------------------------------------
# Extraction review: an unverified field is NOT a missing fact in a paper
# ------------------------------------------------------------------

def _draft_extraction_fields(text: str) -> dict[str, str]:
    """Parse labeled values for a review worksheet, not for factual acceptance."""
    aliases = {title.casefold(): key for key, title in EXTRACTION_FIELDS}
    aliases.update({"objective": "research_objective", "methodology": "method",
                    "dataset/sample": "dataset_sample", "dataset": "dataset_sample",
                    "sample": "dataset_sample", "findings": "key_findings", "results": "key_findings",
                    "evaluation metrics": "metrics", "limitation": "limitations"})
    fields: dict[str, str] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip("|"))]
            if len(cells) != 2 or re.fullmatch(r"[:\s-]+", cells[0]):
                current = None
                continue
            label, value = cells
        else:
            plain = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+|#{1,6}\s+)", "", line)
            plain = plain.replace("**", "").replace("__", "")
            match = re.match(r"^([A-Za-z /]+?)\s*:\s*(.*)$", plain)
            if not match:
                if current is not None:
                    fields[current] = (fields[current] + " " + line).strip()
                continue
            label, value = match.groups()
        key = aliases.get(label.replace("**", "").strip().casefold())
        if key is None or key in fields:
            current = None
            continue
        fields[key] = value.strip()
        current = key
    return fields


def _model_reports_gap(value: str) -> bool:
    """Only recognize explicit placeholder values, not a finding containing 'not'."""
    cleaned = re.sub(r"\[[^\]\n]*\]", "", value).replace("_", " ")
    cleaned = re.sub(r"[*`]+", "", cleaned).strip().rstrip(".").casefold()
    return bool(re.fullmatch(
        r"(?:not (?:reported|available|stated|applicable)(?: in (?:the )?"
        r"(?:available|provided|retrieved|supplied) (?:evidence|text|passages))?|unknown|n/a|none)", cleaned))


def _extraction_review_rows(result: RunResult, accepted: dict[str, str] | None) -> list[dict[str, str]]:
    draft = _draft_extraction_fields(result.answer_text)
    accepted = accepted or {}
    if not result.model_id or result.validation_state == ValidationState.INVALID:
        accepted = {}
    rows = []
    for key, title in EXTRACTION_FIELDS:
        value, checked = draft.get(key, ""), accepted.get(key, "")
        if checked:
            status = "Structurally checked - human review required"
        elif value and _model_reports_gap(value):
            status = "Model reports missing evidence - not independently verified"
        elif value:
            status = "Needs review - not accepted as a checked field"
        else:
            status = "Not extracted"
        rows.append({"Field": title, "Status": status,
                     "Draft value (unverified)": value, "Structurally checked value": checked})
    return rows


def _extraction_review_csv(rows: list[dict[str, str]], paper_title: str) -> str:
    """Export a labeled draft worksheet, separate from the accepted-fields CSV."""
    buffer = io.StringIO(newline="")
    columns = ["Field", "Status", "Draft value (unverified)", "Structurally checked value", "Paper"]
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        record = {**row, "Paper": paper_title}
        for key, value in record.items():
            text = str(value)
            if re.match(r"^[\s\u200b-\u200d\ufeff]*[=+\-@|%]", text):
                text = "'" + text
            record[key] = text
        writer.writerow(record)
    return buffer.getvalue()


def _review_markdown(result: RunResult) -> str:
    return ("# DRAFT - requires source review\n\n"
            "This is model-generated output, not independently verified research.\n\n"
            + run_result_to_markdown(result, st.session_state.library))


_init_session()
settings = get_settings()
if st.session_state.corpus_index is None:
    st.session_state.corpus_index = CorpusIndex(
        enable_semantic=settings.enable_semantic_search
    )
if st.session_state.workflow is None:
    st.session_state.workflow = ResearchWorkflow(settings)


# ------------------------------------------------------------------
# Sidebar: status, optional private access, uploads, session
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 ResearchLens AI")
    st.caption("Evidence-grounded research assistant · IBM Granite")
    render_config_status(settings)

    if settings.app_access_password:
        st.divider()
        st.subheader("Private access")
        if not st.session_state.authenticated:
            with st.form("auth_form"):
                supplied = st.text_input("Access password", type="password")
                unlock = st.form_submit_button("Unlock", use_container_width=True)
            if unlock:
                st.session_state.authenticated = hmac.compare_digest(
                    supplied.encode("utf-8"), settings.app_access_password.encode("utf-8")
                )
                if st.session_state.authenticated:
                    st.rerun()
                else:
                    st.error("Incorrect access password.")
            if not st.session_state.authenticated:
                st.info("Private Research Agent: sign in before accessing its features.")
                st.stop()
        else:
            st.success("Private access unlocked in this session.")
            if st.button("Lock session", use_container_width=True):
                st.session_state.authenticated = False
                _clear_session()
                st.rerun()
    elif settings.app_env.strip().lower() in {"production", "prod"} and settings.generation_enabled():
        st.warning("Public live generation is blocked until access controls are configured.")

    st.divider()
    st.subheader("📎 Upload research PDFs")
    st.caption(
        f"Up to {settings.max_pdfs_per_session} PDFs · "
        f"{settings.max_upload_mb} MB each · "
        f"{settings.max_pages_per_pdf} pages each."
    )

    upload_count = sum(
        p.source_provider == SourceProvider.UPLOAD
        for p in st.session_state.library
    )
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_upload_{st.session_state.upload_epoch}",
        disabled=upload_count >= settings.max_pdfs_per_session,
        label_visibility="collapsed",
        **_upload_size_options(),
    )
    if uploaded_files and upload_count < settings.max_pdfs_per_session:
        for uf in uploaded_files:
            contents = uf.getvalue()
            digest = compute_file_hash(contents)
            if is_duplicate_pdf(contents, st.session_state.library):
                continue
            if digest in st.session_state.upload_failures:
                st.warning(
                    f"Previously rejected: {uf.name}. "
                    f"{st.session_state.upload_failures[digest]}"
                )
                continue
            if upload_count >= settings.max_pdfs_per_session:
                st.warning("PDF upload limit reached for this session.")
                break
            with st.spinner(f"Reading {uf.name}…"):
                paper, pages, ingest_warnings = ingest_pdf(
                    contents, uf.name,
                    max_bytes=settings.max_upload_mb * 1024 * 1024,
                    max_pages=settings.max_pages_per_pdf,
                )
            if paper is None:
                st.error(f"Unable to ingest {uf.name}.")
                for warning in ingest_warnings:
                    st.warning(warning)
                st.session_state.upload_failures[digest] = (
                    "; ".join(ingest_warnings) or "PDF has no readable text."
                )
                continue
            before = st.session_state.corpus_index.chunk_count()
            try:
                index_warnings = index_uploaded_paper(
                    paper, pages, st.session_state.corpus_index
                )
            except Exception:
                # Do not show exception bodies; roll back possible partial writes.
                st.session_state.corpus_index.remove_paper(paper.paper_id)
                st.session_state.upload_failures[digest] = "Indexing failed."
                st.error(f"Could not index {uf.name}; it was not added to the library.")
                continue
            after = st.session_state.corpus_index.chunk_count()
            if index_warnings or after <= before:
                st.session_state.corpus_index.remove_paper(paper.paper_id)
                st.session_state.upload_failures[digest] = (
                    "; ".join(index_warnings) or "No passages were indexed."
                )
                st.error(f"Could not index {uf.name}; it was not added to the library.")
                for warning in index_warnings:
                    st.warning(warning)
                continue
            st.session_state.library.append(paper)
            upload_count += 1
            _clear_generated()  # old outputs no longer reflect current source set
            st.success(f"Added: {paper.title[:70]}")
            for warning in ingest_warnings:
                st.warning(warning)

    st.divider()
    st.caption(
        f"Library: {len(st.session_state.library)} papers · "
        f"Indexed passages: {st.session_state.corpus_index.chunk_count()}"
    )
    _model_calls_placeholder = st.empty()
    _update_model_call_counter()
    if st.button("🗑️ Clear session", use_container_width=True):
        _clear_session()
        st.rerun()
    st.caption(
        "A new browser session can reset the visible model-call counter. "
        "This is not an IBM account-wide usage or spending limit."
    )
    st.caption(
        "Clearing removes the library and uploaded text held by this session. "
        "Observed model-call counts are retained until this browser session "
        "ends. External provider copies and account usage cannot be cleared here."
    )


st.header("🔬 ResearchLens AI")
st.caption(
    "Search scholarly literature, analyze uploaded research papers, "
    "compare studies, draft reports, and inspect supporting evidence."
)
st.info(
    "Uploaded paper content is sent to IBM watsonx.ai only when you explicitly "
    "run a generation task. Research prototype: generated claims and references "
    "may be incorrect and require source review.",
    icon="ℹ️",
)

workspace_tab, library_tab, compare_tab, reports_tab, about_tab = st.tabs(
    [
        "Research Workspace",
        "Paper Library",
        "Compare & Extract",
        "Reports & References",
        "About & Diagnostics",
    ]
)

# ==================================================================
# 1. Research Workspace
# ==================================================================
with workspace_tab:
    if not settings.generation_enabled():
        render_generation_disabled_notice()
    with st.form("research_form"):
        question = st.text_area(
            "Research question or task instruction",
            placeholder="What methods and evaluation results does the selected paper report?",
            height=115,
        )
        c_task, c_search = st.columns([3, 1])
        with c_task:
            task_key = render_task_selector(key="workspace_task")
        with c_search:
            search_first = st.checkbox(
                "Search literature first", value=False,
                help="When off, uploaded PDFs are preferred; select a paper to guarantee scope.",
            )
        scope = st.multiselect(
            "Restrict evidence to selected papers",
            options=_paper_ids(),
            format_func=_display_paper,
            key=f"workspace_scope_{st.session_state.selection_epoch}",
            help="Select a specific PDF to avoid mixing results from other studies.",
        )
        submitted = st.form_submit_button("▶ Run research task", type="primary")
    if submitted:
        _run_research(
            TaskType(task_key), question,
            paper_ids=list(scope) or None,
            search_first=search_first,
            label=TASK_OPTIONS.get(task_key, "Research task"),
        )
    _workspace_output = st.container()

# ==================================================================
# 2. Paper Library
# ==================================================================
with library_tab:
    lib: list[PaperRecord] = st.session_state.library
    f_left, f_right = st.columns([3, 1])
    with f_left:
        query_filter = st.text_input(
            "Search library titles", placeholder="Filter by title…"
        )
    with f_right:
        provider_filter = st.selectbox(
            "Source", ["All", "upload", "arxiv", "crossref", "unknown"]
        )
    display_lib = [
        paper for paper in lib
        if query_filter.casefold() in paper.title.casefold()
        and (provider_filter == "All" or paper.source_provider.value == provider_filter)
    ]
    st.caption(f"{len(display_lib)} / {len(lib)} papers shown")
    if not lib:
        render_empty_library()
    elif not display_lib:
        st.info("No papers match the current filters.")
    else:
        for paper in display_lib:
            with st.expander(
                f"{paper.title[:90]} — {paper.source_provider.value} · "
                f"{_coverage_label(paper.text_coverage)} · {paper.year or 'year unknown'}"
            ):
                st.caption(f"Paper ID: `{paper.display_id()}`")
                if paper.authors:
                    st.write("Authors:", ", ".join(paper.authors[:8]))
                if paper.abstract:
                    st.text(paper.abstract[:1000])
                if _valid_external_url(paper.landing_url):
                    st.link_button("Open source record", paper.landing_url)
                if paper.is_metadata_only():
                    st.warning("Metadata/abstract only; full-paper conclusions are not available.")
                if paper.source_provider == SourceProvider.UPLOAD:
                    st.caption("PDF title/author/year are not necessarily extracted automatically.")
                with st.form(f"paper_metadata_{paper.paper_id}"):
                    title_edit = st.text_input(
                        "Correct title", value=paper.title,
                        key=f"paper_title_{paper.paper_id}",
                    )
                    authors_edit = st.text_input(
                        "Authors (comma-separated)",
                        value=", ".join(paper.authors),
                        key=f"paper_authors_{paper.paper_id}",
                    )
                    year_edit = st.text_input(
                        "Publication year (leave blank if unknown)",
                        value=str(paper.year or ""),
                        key=f"paper_year_{paper.paper_id}",
                    )
                    doi_edit = st.text_input(
                        "DOI (optional)", value=paper.doi or "",
                        key=f"paper_doi_{paper.paper_id}",
                    )
                    url_edit = st.text_input(
                        "Source URL (optional)", value=paper.landing_url or "",
                        key=f"paper_url_{paper.paper_id}",
                    )
                    save_metadata = st.form_submit_button("Save bibliography details")
                if save_metadata:
                    parsed_year: int | None = None
                    year_text = year_edit.strip()
                    if year_text:
                        if not year_text.isdigit() or not 1000 <= int(year_text) <= 2100:
                            st.error("Enter a valid four-digit publication year.")
                        else:
                            parsed_year = int(year_text)
                    if not year_text or parsed_year is not None:
                        if not title_edit.strip():
                            st.error("A paper title cannot be empty.")
                        elif url_edit.strip() and not _valid_external_url(url_edit.strip()):
                            st.error("The source URL must start with http:// or https://.")
                        else:
                            paper.title = title_edit.strip()
                            paper.authors = [a.strip() for a in authors_edit.split(",") if a.strip()]
                            paper.year = parsed_year
                            paper.doi = doi_edit.strip() or None
                            paper.landing_url = url_edit.strip() or None
                            _clear_generated()
                            st.success("Bibliographic metadata updated. Recheck any generated output.")
                            st.rerun()
                if st.button(
                    "Remove this paper from session",
                    key=f"remove_paper_{paper.paper_id}",
                ):
                    st.session_state.corpus_index.remove_paper(paper.paper_id)
                    st.session_state.library = [
                        p for p in lib if p.paper_id != paper.paper_id
                    ]
                    _clear_generated()
                    # A rejected upload may now fit after this paper is removed.
                    st.session_state.upload_failures = {}
                    st.session_state.upload_epoch += 1
                    st.session_state.selection_epoch += 1
                    st.rerun()

    if lib:
        st.divider()
        st.subheader("Export library")
        j, c, b = st.columns(3)
        with j:
            st.download_button(
                "⬇ JSON", papers_to_json(lib), "research_library.json",
                "application/json", use_container_width=True,
            )
        with c:
            st.download_button(
                "⬇ CSV", papers_to_csv(lib), "research_library.csv",
                "text/csv", use_container_width=True,
            )
        with b:
            st.download_button(
                "⬇ BibTeX", papers_to_bibtex(lib), "research_library.bib",
                "text/plain", use_container_width=True,
            )

# ==================================================================
# 3. Compare & Extract
# ==================================================================
with compare_tab:
    lib = st.session_state.library
    st.subheader("Structured paper extraction")
    st.caption(
        "Extract objective, method, sample, findings, metrics and limitations "
        "from one selected paper at a time. Each new extraction uses one bounded "
        "IBM Granite workflow action when generation is enabled."
    )
    if not lib:
        st.info("Upload or discover research papers to enable extraction.")
    else:
        with st.form("extract_form"):
            extract_pid = st.selectbox(
                "Paper to extract", options=_paper_ids(), format_func=_display_paper,
                key=f"extract_paper_{st.session_state.selection_epoch}",
            )
            pressed_extract = st.form_submit_button(
                "Extract with IBM Granite", type="primary"
            )
        if pressed_extract:
            # Do not retain a previously cached result after a failed new attempt.
            st.session_state.grounded_extractions.pop(extract_pid, None)
            paper = _selected_papers([extract_pid])[0]
            prompt = (
                f"Extract structured information from the selected paper '{paper.title}'. "
                "Use only THIS paper's own research findings; distinguish its "
                "literature-review citations from experiments it actually conducted. "
                "Return exactly six labeled lines, each with supporting [CHUNK-ID] "
                "citations where evidence exists. The labels must be: "
                "Research Objective:, Method:, Dataset / Sample:, "
                "Key Findings:, Metrics:, Limitations:. "
                "For any field not supported by the selected excerpts, write 'not reported'."
            )
            result = _run_research(
                TaskType.EXTRACT, prompt, paper_ids=[extract_pid],
                search_first=False, label="Structured Extraction",
                update_primary=True, action_context="extraction",
            )
            if result:
                parsed = None
                if result.model_id and result.validation_state != ValidationState.INVALID:
                    scoped_chunks = st.session_state.corpus_index.get_chunks_by_ids(result.evidence_chunk_ids)
                    allowed = {c.chunk_id[:8] for c in scoped_chunks if c.paper_id == extract_pid}
                    parsed = _parse_structured_extraction(result.answer_text, allowed, verified_claims=result.claims)
                st.session_state.grounded_extractions[extract_pid] = (result, parsed)

        _show_action_notice("extraction")
        cached_extraction = st.session_state.grounded_extractions.get(extract_pid)
        if cached_extraction:
            result, fields = cached_extraction
            paper = next((p for p in lib if p.paper_id == extract_pid), None)
            if paper is not None:
                with st.container(border=True):
                    st.caption(f"Selected paper: {paper.title[:95]}")
                    _show_result(result, label="Extracted findings")
                    if result.model_id:
                        rows = _extraction_review_rows(result, fields)
                        st.markdown("**Structured field review**")
                        st.caption(
                            "Rejected fields are not necessarily absent from the paper. "
                            "Draft values are unverified. Only individually accepted fields "
                            "are included in the checked-fields CSV."
                        )
                        st.table(rows)
                        accepted = (fields or {}) if result.validation_state != ValidationState.INVALID else {}
                        st.caption(f"{len(accepted)} / {len(EXTRACTION_FIELDS)} fields passed the current structural acceptance checks.")
                        if accepted:
                            st.download_button(
                                "⬇ Structurally checked fields CSV", extraction_to_csv(accepted, paper.title),
                                f"extraction_checked_{extract_pid[:8]}.csv", "text/csv",
                                key=f"download_extract_csv_{extract_pid}",
                            )
                        st.download_button(
                            "⬇ Review worksheet CSV (includes unverified draft)",
                            _extraction_review_csv(rows, paper.title),
                            f"extraction_review_{extract_pid[:8]}.csv", "text/csv",
                            key=f"download_extract_review_{extract_pid}",
                        )
                        st.download_button(
                            "⬇ Draft extraction Markdown", _review_markdown(result),
                            f"extraction_{extract_pid[:8]}.md", "text/markdown",
                            key=f"download_extract_md_{extract_pid}",
                        )

    st.divider()
    st.subheader("Compare two or three research papers")
    if len(lib) < 2:
        st.info("Add at least two papers to enable comparison.")
    else:
        with st.form("paper_comparison_form"):
            comparison_ids = st.multiselect(
                "Select papers to compare",
                options=_paper_ids(), format_func=_display_paper,
                max_selections=3,
                key=f"comparison_scope_{st.session_state.selection_epoch}",
            )
            compare_pressed = st.form_submit_button(
                "Compare selected papers", type="primary",
            )
        if compare_pressed and not 2 <= len(comparison_ids) <= 3:
            st.session_state.grounded_comparison = None
            _block_attempt("comparison", "Multi-paper comparison", "Select two or three papers before comparing.")
        if compare_pressed and 2 <= len(comparison_ids) <= 3:
            st.session_state.grounded_comparison = None
            comparison_question = (
                "Compare ONLY the selected source papers using their own study findings. "
                "Write a Markdown comparison table with one column for each "
                "paper and rows for Research Objective, Method, Dataset / Sample, "
                "Metrics, Key Findings, and Limitations. Cite [CHUNK-ID] for each "
                "supported factual finding; otherwise write 'not reported'. "
                "Differentiate each study's own experiment from studies described "
                "in its literature review. State which results cannot be compared "
                "directly because the datasets, tasks or protocols differ. "
                "Do not guess publication years or fabricate missing facts."
            )
            result = _run_research(
                TaskType.COMPARE, comparison_question,
                paper_ids=list(comparison_ids), search_first=False,
                label="Multi-paper comparison", update_primary=True, action_context="comparison",
            )
            if result:
                if set(comparison_ids) - set(result.paper_ids_used):
                    result.warnings = list(dict.fromkeys([
                        *result.warnings,
                        "One or more selected papers contributed no evidence; the comparison is incomplete.",
                    ]))
                st.session_state.grounded_comparison = (tuple(comparison_ids), result)

        _show_action_notice("comparison")

        cached = st.session_state.grounded_comparison
        if cached:
            compared_ids, comparison_result = cached
            if set(compared_ids) == set(comparison_ids):
                _show_result(comparison_result, label="Research comparison")
                if comparison_result.model_id:
                    st.download_button(
                        "⬇ Draft comparison Markdown", _review_markdown(comparison_result),
                        "research_comparison.md", "text/markdown", key="download_comparison_md",
                    )
            else:
                st.info("The selected papers changed. Press Compare to generate a new result.")

# ==================================================================
# 4. Reports & References
# ==================================================================
with reports_tab:
    st.subheader("Create a research report")
    with st.form("report_generation_form"):
        report_topic = st.text_area(
            "Report question / scope",
            placeholder="Summarize the methods, findings and research gaps within these selected studies.",
            height=90,
        )
        report_ids = st.multiselect(
            "Report sources (leave empty to use uploaded PDFs)",
            options=_paper_ids(), format_func=_display_paper,
            key=f"report_source_scope_{st.session_state.selection_epoch}",
        )
        create_report = st.form_submit_button("Generate report", type="primary")
    if create_report:
        if not report_ids and not any(
            p.source_provider == SourceProvider.UPLOAD for p in st.session_state.library
        ):
            _block_attempt("report", "Research report", "Choose at least one research paper to avoid an ungrounded report.")
        else:
            _run_research(
                TaskType.REPORT, report_topic,
                paper_ids=report_ids or None,
                search_first=False,
                label="Research report", action_context="report",
            )

    _show_action_notice("report")
    result = st.session_state.last_result
    if result is None:
        st.info("Run a research task or generate a report to enable result exports.")
    else:
        previous = _is_previous_result(result)
        st.caption(f"Saved task: {result.task.value} · Run: {result.run_id[:8]} · Started: {result.started_at}")
        if previous:
            st.info("The latest attempt did not produce this output. Downloads below refer to the previous saved run.")
        _show_result(result, label="Previous saved research output" if previous else "Latest research output")
        sources = _report_sources(
            result, st.session_state.library, st.session_state.corpus_index
        )
        st.divider()
        st.subheader("Download research artifacts")
        if result.paper_ids_used and not sources:
            st.warning(
                "The answer has no resolvable citation-linked references. "
                "Retrieved papers are visible in the evidence panel, but "
                "uncited papers are not exported as cited references."
            )
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button(
                "⬇ Report / answer",
                run_result_to_markdown(result, sources),
                f"report_{result.run_id[:8]}.md", "text/markdown",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇ Cited references",
                papers_to_bibtex(sources),
                "cited_references.bib", "text/plain",
                use_container_width=True, disabled=not bool(sources),
            )
        with d3:
            st.download_button(
                "⬇ Evidence ledger",
                evidence_ledger_to_markdown(
                    result, st.session_state.library, st.session_state.corpus_index
                ),
                f"evidence_ledger_{result.run_id[:8]}.md", "text/markdown",
                use_container_width=True,
            )
        with d4:
            st.download_button(
                "⬇ Run trace",
                st.session_state.last_trace or run_trace_to_markdown(result),
                f"run_trace_{result.run_id[:8]}.md", "text/markdown",
                use_container_width=True,
            )
        if result.claims:
            with st.expander("Claim validation details"):
                for n, claim in enumerate(result.claims, 1):
                    st.write(f"**Claim {n}: {claim.validation_state.value}**")
                    st.text(claim.claim_text)
                    if claim.validation_notes:
                        st.caption(claim.validation_notes)

    if st.session_state.library:
        st.divider()
        st.subheader("All library references")
        st.download_button(
            "⬇ Whole library BibTeX",
            papers_to_bibtex(st.session_state.library),
            "all_research_references.bib", "text/plain",
        )

# ==================================================================
# 5. About & Diagnostics
# ==================================================================
with about_tab:
    st.subheader("Architecture")
    st.markdown(
        """| Component | Implementation |
|---|---|
| User interface | Python / Streamlit |
| Language model | IBM Granite via hosted watsonx.ai (configured externally) |
| Search | arXiv and Crossref bibliographic metadata APIs |
| PDF ingestion | Page-indexed pypdf text extraction |
| Retrieval | TF-IDF; optional MiniLM semantic retrieval; RRF when enabled |
| Agent roles | Search, Read/Extract, Verify, Write |
| Claim checks | Source IDs and numerical consistency checks (not scientific proof) |
| Export formats | Markdown, JSON, CSV, BibTeX |"""
    )
    st.subheader("Current session diagnostics")
    st.caption(f"UI revision: {UI_REVISION}")
    index = st.session_state.corpus_index
    st.write(f"**IBM configured:** {'Yes' if settings.ibm_configured() else 'No'}")
    st.write(f"**Live generation setting:** {'Enabled' if settings.generation_enabled() else 'Disabled'}")
    st.write(f"**Configured Granite model:** `{settings.watsonx_model_id or 'not set'}`")
    st.write(f"**Streamlit environment:** `{settings.app_env}`")
    st.write(f"**Library records:** {len(st.session_state.library)}")
    st.write(f"**Indexed evidence passages:** {index.chunk_count()}")
    st.write(f"**Task actions in this session:** {st.session_state.run_count}")
    st.write(
        f"**Recorded model calls in this session:** "
        f"{st.session_state.ui_model_calls} / {MAX_UI_MODEL_CALLS_PER_SESSION}"
    )
    if index.degraded_mode:
        st.warning("Semantic search unavailable: retrieval used lexical-only mode.")
    elif settings.enable_semantic_search:
        st.caption("Semantic retrieval was requested; successful model loading depends on runtime availability.")
    else:
        st.caption("Semantic retrieval disabled in settings; lexical TF-IDF remains available.")
    if st.session_state.last_result:
        st.caption(
            f"Last completed task: {st.session_state.last_result.task.value} · "
            f"Validation: {st.session_state.last_result.validation_state.value} · "
            f"Model used: {st.session_state.last_result.model_id or 'none'}"
        )
    else:
        st.caption("No AI research task has been completed in this browser session.")

    st.subheader("Original Research Agent scope")
    st.markdown(
        """- Literature discovery and reference export
- Paper summaries and research-question answering
- Structured evidence extraction
- Comparisons across research papers
- Research report and related-work drafting
- Untested hypothesis suggestions
- IBM Granite generation when the account and model are configured

These are supported application routes, **not a claim that every feature has passed
an independent live research-quality evaluation**."""
    )

    st.subheader("Known limitations and responsible use")
    st.markdown(
        """- Source identifiers and numerical checks do not establish scientific truth.
- PDFs with scanned pages, images and tables may have missing or imperfect text.
- Research results are limited to the retrieved passages, not automatically the entire PDF.
- Uploaded PDFs may lack verified author, publication-year and DOI metadata; edit those fields manually before export.
- Cross-study accuracy metrics are not directly comparable without matching tasks and datasets.
- Hypotheses and research-gap suggestions are untested and must be reviewed.
- Private session data is not guaranteed to persist after reloads or server restarts.
- **Public deployment warning:** model-call counts are per browser session (and persist through the Clear button), not an account-wide token/spending cap. A new browser connection can reset the count. An access password is not enterprise authentication. Configure provider-side limits or keep public live generation disabled to protect the free allowance.
- Langflow `app.json` submission is a separate companion workflow and must be actually built/tested before claiming completion."""
    )


# Render after all action handlers so tab changes cannot display stale run state.
with _workspace_output:
    _show_workspace_output()
