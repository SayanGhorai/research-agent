"""Reusable, display-only Streamlit helpers for ResearchLens AI.

These functions do not call IBM, perform retrieval, or process uploaded PDFs.
Visible status reflects application settings, not a verified IBM connection.
"""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from researchlens.config import Settings


# ----------------------------------------------------------
# Safe Markdown formatting
# ----------------------------------------------------------

def _safe_inline_markdown(value: object) -> str:
    """Render external metadata as text, not clickable Markdown or HTML."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = html.escape(text, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", text)


# ----------------------------------------------------------
# Configuration status
# ----------------------------------------------------------

def render_config_status(settings: "Settings") -> None:
    """Show configured/disabled status and every non-duplicate safety warning."""
    warnings = settings.validate()

    if settings.generation_enabled():
        st.success(
            "IBM Granite generation is enabled in settings. "
            "This does not confirm current IBM availability or remaining quota."
        )

    elif settings.ibm_configured():
        st.warning(
            "IBM Granite credentials are configured, but live generation is "
            "disabled. Enable it only after an authorized smoke test and "
            "appropriate access and usage safeguards."
        )

    else:
        st.info(
            "IBM Granite is not fully configured. Add credentials privately "
            "through local environment settings or Streamlit Secrets. "
            "Paper discovery, PDF ingestion, and reference export can "
            "still operate without IBM generation."
        )

    # Do not suppress important warnings merely because they
    # contain phrases such as "live generation".
    duplicate_status = (
        settings.ibm_status_label()
        if not settings.ibm_configured()
        else None
    )

    seen: set[str] = set()

    for warning in warnings:
        if warning == duplicate_status or warning in seen:
            continue

        seen.add(warning)
        st.warning(warning)


# ----------------------------------------------------------
# Empty states
# ----------------------------------------------------------

def render_empty_library() -> None:
    st.info(
        "Your paper library is empty. Search for papers or upload "
        "text-readable PDFs using the sidebar."
    )


def render_empty_results() -> None:
    st.info(
        "No research result yet. Choose a task, enter your question, "
        "and select Run."
    )


# ----------------------------------------------------------
# Generation-disabled notice
# ----------------------------------------------------------

def render_generation_disabled_notice() -> None:
    st.warning(
        "AI generation is currently disabled. Paper search, PDF ingestion "
        "and non-AI exports remain available. Enable IBM generation only "
        "after an authorized access check and suitable usage safeguards."
    )


# ----------------------------------------------------------
# Task selector
# ----------------------------------------------------------

TASK_OPTIONS: dict[str, str] = {
    "answer_question": "Answer a research question",
    "summarize": "Summarize a paper",
    "extract": "Extract structured data",
    "compare": "Compare papers",
    "report": "Generate a research report",
    "hypotheses": "Suggest hypotheses",
    "draft_section": "Draft a paper section",
}


def render_task_selector(key: str = "task_selector") -> str:
    """Return the task identifier selected in the Streamlit widget."""

    label_to_key = {
        label: task
        for task, label in TASK_OPTIONS.items()
    }

    label = st.selectbox(
        "Task",
        options=list(TASK_OPTIONS.values()),
        key=key,
        help="Choose what you want ResearchLens AI to do.",
    )

    if label not in label_to_key:
        raise ValueError("No valid research task was selected")

    return label_to_key[label]


# ----------------------------------------------------------
# Paper library card
# ----------------------------------------------------------

def render_paper_card_stub(
    paper_id: str,
    title: str,
    year: int | None,
    provider: str,
) -> None:
    """Render compact paper metadata while treating all strings as data."""

    with st.container(border=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            st.markdown(
                f"**{_safe_inline_markdown(title)}**"
            )

            display_year = (
                str(year) if year is not None else "?"
            )

            st.caption(
                f"{_safe_inline_markdown(display_year)} · "
                f"{_safe_inline_markdown(provider)}"
            )

        with col2:
            st.caption(
                f"`{_safe_inline_markdown(paper_id[:8])}`"
            )