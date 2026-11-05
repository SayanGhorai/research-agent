"""Safe, evidence-aware exports for ResearchLens AI.

Exports preserve available bibliographic metadata and distinguish an evidence
reference from an independently verified scientific fact. No model/API calls
or filesystem writes occur in this module.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
from typing import Any
from urllib.parse import quote, urlsplit

from researchlens.schemas import (
    NOT_AVAILABLE,
    NOT_REPORTED,
    PaperRecord,
    RunResult,
    TextCoverage,
    ValidationState,
)


# ---------------------------------------------------------------------------
# Safe output helpers
# ---------------------------------------------------------------------------

def _csv_safe(value: str) -> str:
    """Prefix dangerous spreadsheet formula starts, even after whitespace.

    This reduces formula-injection risk but cannot guarantee safe behavior in
    every spreadsheet application/import mode. csv.writer handles quoting.
    """
    value = str(value)
    if re.match(r"^[\s\u200b-\u200d\ufeff]*[=+\-@|%]", value):
        return "'" + value
    return value


def _md_inline(value: Any) -> str:
    """Escape externally supplied content used as a Markdown inline field."""
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    normalized = html.escape(normalized, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+|>])", r"\\\1", normalized)


def _md_table(value: Any) -> str:
    """Escape a user-supplied Markdown table cell without truncating it."""
    return _md_inline(value)


def _safe_link(url: str | None) -> str:
    """Generate a link only for ordinary, well-formed HTTP(S) destinations."""
    if not url or re.search(r"[\s<>\\]", url):
        return ""
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in ("https", "http") or not parts.hostname:
            return ""
        if parts.username or parts.password:
            return ""
        return "[Source](<" + quote(
            url, safe=":/?#[]@!$&'()*+,;=%-._~"
        ) + ">)"
    except ValueError:
        return ""


def _bib_escape(value: str) -> str:
    """Escape common TeX special characters in literal BibTeX field data."""
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    compact = re.sub(r"\s+", " ", value).strip()
    return "".join(replacements.get(char, char) for char in compact)


def _bibtex_key(p: PaperRecord) -> str:
    """Create a repeatable ASCII key from author, year and record ID prefix."""
    surname = "Unknown"
    if p.authors:
        first = p.authors[0].strip()
        family = (
            first.split(",", 1)[0]
            if "," in first
            else first.split()[-1] if first else ""
        )
        cleaned = re.sub(r"[^A-Za-z0-9]", "", family)
        if cleaned:
            surname = cleaned
    year = str(p.year) if p.year is not None else "XXXX"
    ident = re.sub(r"[^A-Za-z0-9]", "", p.paper_id)[:8] or "record"
    return f"{surname}{year}_{ident}"


def _coverage_note(coverage: TextCoverage) -> str:
    return {
        TextCoverage.ABSTRACT_ONLY: "abstract only",
        TextCoverage.EXCERPT: "partial extracted text",
        TextCoverage.FULL_TEXT: (
            "PDF text extracted; figures/tables may be incomplete"
        ),
        TextCoverage.NONE: "no readable text",
    }[coverage]


# ---------------------------------------------------------------------------
# Paper library formats
# ---------------------------------------------------------------------------

def papers_to_bibtex(papers: list[PaperRecord]) -> str:
    """Export available citations; do not infer publication type or venue."""
    entries: list[str] = []
    used_keys: set[str] = set()

    for paper in papers:
        base_key = _bibtex_key(paper)
        key = base_key
        suffix = 2

        while key in used_keys:
            key = f"{base_key}_{suffix}"
            suffix += 1
        used_keys.add(key)

        fields = [f"  title = {{{_bib_escape(paper.title)}}}"]

        if paper.authors:
            author = " and ".join(_bib_escape(a) for a in paper.authors)
            fields.append(f"  author = {{{author}}}")

        if paper.year is not None:
            fields.append(f"  year = {{{paper.year}}}")

        if paper.doi:
            fields.append(f"  doi = {{{_bib_escape(paper.doi)}}}")

        if paper.landing_url:
            fields.append(f"  url = {{{_bib_escape(paper.landing_url)}}}")

        if paper.arxiv_id:
            fields.append(f"  eprint = {{{_bib_escape(paper.arxiv_id)}}}")
            fields.append("  archivePrefix = {arXiv}")

        if paper.abstract:
            snippet = paper.abstract[:300]
            if len(paper.abstract) > 300:
                snippet += "..."
            fields.append(f"  abstract = {{{_bib_escape(snippet)}}}")

        # DOI alone does not prove that a work is a journal article.
        entries.append(
            f"@misc{{{key},\n" + ",\n".join(fields) + "\n}"
        )

    return "\n\n".join(entries)


_CSV_COLUMNS = [
    "paper_id", "title", "authors", "year", "doi", "arxiv_id",
    "source_provider", "text_coverage", "landing_url", "abstract_snippet",
    "retrieval_timestamp",
]


def papers_to_csv(papers: list[PaperRecord]) -> str:
    """Export paper metadata as CSV, with formula-risk mitigation."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_CSV_COLUMNS)
    writer.writeheader()

    for paper in papers:
        row = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": "; ".join(paper.authors),
            "year": str(paper.year) if paper.year is not None else "",
            "doi": paper.doi or "",
            "arxiv_id": paper.arxiv_id or "",
            "source_provider": paper.source_provider.value,
            "text_coverage": paper.text_coverage.value,
            "landing_url": paper.landing_url or "",
            "abstract_snippet": re.sub(
                r"\s+", " ", (paper.abstract or "")[:200]
            ),
            "retrieval_timestamp": paper.retrieval_timestamp,
        }
        writer.writerow({
            key: _csv_safe(val) for key, val in row.items()
        })

    return buffer.getvalue()


def papers_to_json(papers: list[PaperRecord]) -> str:
    """JSON includes only available public-facing library metadata."""
    return json.dumps([
        {
            "paper_id": p.paper_id,
            "title": p.title,
            "authors": p.authors,
            "year": p.year,
            "doi": p.doi,
            "arxiv_id": p.arxiv_id,
            "source_provider": p.source_provider.value,
            "text_coverage": p.text_coverage.value,
            "landing_url": p.landing_url,
            "abstract": p.abstract,
            "license_note": p.license_note,
            "retrieval_timestamp": p.retrieval_timestamp,
        }
        for p in papers
    ], ensure_ascii=False, indent=2)


def extraction_to_csv(extraction: dict, paper_title: str = "") -> str:
    """Preserve extraction values as CSV; None means explicitly not reported."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["field", "value", "paper"])

    for field, value in extraction.items():
        if value is None:
            printable = NOT_REPORTED
        elif isinstance(value, (dict, list)):
            printable = json.dumps(value, ensure_ascii=False)
        else:
            printable = str(value)

        writer.writerow([
            _csv_safe(str(field)),
            _csv_safe(printable),
            _csv_safe(paper_title),
        ])

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Research report, ledger, comparison and trace
# ---------------------------------------------------------------------------

def run_result_to_markdown(
    result: RunResult,
    papers: list[PaperRecord],
) -> str:
    """Export a research response with source coverage and explicit caveats."""
    paper_map = {p.paper_id: p for p in papers}

    lines = [
        "# ResearchLens AI — Research Brief",
        "",
        f"**Task:** {result.task.value}",
        f"**Question:** {_md_inline(result.question)}",
        f"**Run ID:** `{_md_inline(result.run_id)}`",
        f"**Started:** {_md_inline(result.started_at)}",
        "",
        "---",
        "",
        "## Answer / Output",
        "",
        result.answer_text or "_No answer generated._",
        "",
        f"**Evidence coverage:** {_coverage_note(result.source_coverage)}",
        (
            f"**Validation state:** {result.validation_state.value} "
            "(not independent fact-checking)"
        ),
        "",
    ]

    if result.paper_ids_used:
        lines.extend(["---", "", "## Sources used", ""])
        seen: set[str] = set()

        for pid in result.paper_ids_used:
            if pid in seen:
                continue
            seen.add(pid)

            paper = paper_map.get(pid)
            if paper is None:
                lines.append(
                    f"- Unresolved paper ID: `{_md_inline(pid)}`"
                )
                continue

            detail = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                detail += " et al."
            if paper.year is not None:
                detail += f" ({paper.year})"

            link = _safe_link(paper.landing_url)

            lines.extend([
                (
                    f"- **{_md_inline(paper.title)}** — "
                    f"{_coverage_note(paper.text_coverage)}"
                ),
                (
                    f"  {_md_inline(paper.display_id())}"
                    + (f" — {link}" if link else "")
                ),
            ])

            if detail:
                lines.append("  " + _md_inline(detail))

        lines.append("")

    if result.warnings:
        lines.extend(["---", "", "## Warnings", ""])
        for warning in result.warnings:
            lines.append("- " + _md_inline(warning))
        lines.append("")

    if result.model_id or result.llm_calls_used:
        lines.extend(["---", "", "## Model usage", ""])
        lines.append(
            "- Model: "
            + (_md_inline(result.model_id) if result.model_id else "not reported")
        )
        lines.append(
            f"- LLM calls (local counter): {result.llm_calls_used}"
        )

        if result.output_tokens_reported is not None:
            lines.append(
                f"- Output tokens (reported): {result.output_tokens_reported}"
            )

        if result.input_tokens_estimate is not None:
            lines.append(
                f"- Input tokens (estimate): {result.input_tokens_estimate}"
            )

        if result.latency_seconds is not None:
            lines.append(
                f"- Latency: {result.latency_seconds:.2f}s"
            )

    lines.extend([
        "",
        "---",
        "",
        "_Generated by ResearchLens AI. Verify claims and citations before use._",
    ])
    return "\n".join(lines)


def evidence_ledger_to_markdown(
    result: RunResult,
    papers: list[PaperRecord],
    index: Any,
) -> str:
    """Show claim-to-excerpt links without claiming independent fact-checking."""
    paper_map = {p.paper_id: p for p in papers}

    lines = [
        "# Evidence Ledger",
        "",
        f"**Run ID:** `{_md_inline(result.run_id)}`  ",
        f"**Task:** {result.task.value}  ",
        f"**Question:** {_md_inline(result.question)}",
        "",
        "---",
        "",
        (
            "**Interpretation:** A linked excerpt is not independent "
            "verification of a claim."
        ),
        "",
    ]

    if not result.claims:
        lines.extend([
            "No citation-linked claims were extracted from this output.",
            "",
            "**Full output:**",
            result.answer_text or "_empty_",
        ])
        return "\n".join(lines)

    for number, claim in enumerate(result.claims, 1):
        ids = list(dict.fromkeys(claim.evidence_chunk_ids))

        try:
            found = index.get_chunks_by_ids(ids) if ids else []
        except Exception:
            found = []

        found_by_id = {
            chunk.chunk_id: chunk
            for chunk in found
            if getattr(chunk, "chunk_id", None) in ids
        }

        resolved = [
            found_by_id[cid]
            for cid in ids
            if cid in found_by_id
        ]
        missing = [
            cid for cid in ids
            if cid not in found_by_id
        ]

        all_known = (
            bool(ids)
            and not missing
            and all(c.paper_id in paper_map for c in resolved)
        )

        # A VALID state is structural, not proof of scientific truth.
        if all_known and claim.validation_state == ValidationState.VALID:
            status = "evidence linked (not fact-checked)"
        else:
            status = "unverified linkage"

        lines.extend([
            f"### Claim {number} [{status}]",
            "",
            "> " + _md_inline(claim.claim_text),
            "",
            f"**Validation state:** {claim.validation_state.value}",
            "",
        ])

        if not ids:
            lines.extend([
                "No supporting passage IDs are recorded.",
                "",
            ])

        for chunk in resolved:
            paper = paper_map.get(chunk.paper_id)
            title = (
                _md_inline(paper.title)
                if paper
                else "Unknown paper (not in library)"
            )

            excerpt = chunk.text[:550]
            if len(chunk.text) > 550:
                excerpt += "..."

            lines.extend([
                (
                    f"**Source:** {title} — "
                    f"{_md_inline(chunk.location_label())}"
                ),
                (
                    f"**Paper ID:** `{_md_inline(chunk.paper_id)}`; "
                    f"**Chunk ID:** `{_md_inline(chunk.chunk_id)}`"
                ),
                f"**Excerpt:** {_md_inline(excerpt)}",
                "",
            ])

        if missing:
            lines.extend([
                "**Unresolved evidence IDs:** "
                + ", ".join(
                    f"`{_md_inline(cid)}`"
                    for cid in missing
                ),
                "",
            ])

        if claim.validation_notes:
            lines.extend([
                "**Validation notes:** "
                + _md_inline(claim.validation_notes),
                "",
            ])

        lines.extend(["---", ""])

    return "\n".join(lines)


_COMPARISON_FIELDS = [
    "problem / objective",
    "method",
    "dataset / sample",
    "metrics",
    "key findings",
    "limitations",
]


def comparison_to_markdown(
    papers: list[PaperRecord],
    extractions: list[dict],
    possible_disagreements: list[str] | None = None,
    gap_suggestions: list[str] | None = None,
) -> str:
    """Render comparative evidence without clipping metrics or inventing blanks."""
    if not papers:
        return "No papers selected for comparison."

    lines = [
        "# Paper Comparison",
        "",
        (
            "| Field | "
            + " | ".join(
                f"Paper {i + 1}" for i in range(len(papers))
            )
            + " |"
        ),
        "|---|" + "---|" * len(papers),
    ]

    aliases = {
        "Problem / Objective": "research_objective",
        "Method": "method",
        "Dataset / Sample": "dataset_sample",
        "Metrics": "metrics",
        "Key Findings": "key_findings",
        "Limitations": "limitations",
    }

    def append_row(label: str, values: list[str]) -> None:
        lines.append(
            "| **" + label + "** | "
            + " | ".join(values) + " |"
        )

    append_row("Title", [
        _md_table(p.title)
        + (
            f" ({_safe_link(p.landing_url)})"
            if _safe_link(p.landing_url)
            else ""
        )
        for p in papers
    ])

    append_row("Year", [
        str(p.year) if p.year is not None else "—"
        for p in papers
    ])

    append_row("Authors", [
        _md_table(", ".join(p.authors))
        if p.authors else "_not reported_"
        for p in papers
    ])

    append_row("Identifier", [
        _md_table(p.display_id())
        for p in papers
    ])

    append_row("Text coverage", [
        _md_table(p.text_coverage.value)
        for p in papers
    ])

    for label, field in aliases.items():
        values: list[str] = []

        for index, paper in enumerate(papers):
            extraction = (
                extractions[index]
                if index < len(extractions)
                and isinstance(extractions[index], dict)
                else {}
            )

            value = extraction.get(field, NOT_REPORTED)

            if value is None or value == NOT_REPORTED:
                values.append("_not reported_")
            elif value == NOT_AVAILABLE:
                values.append("_not available_")
            else:
                if isinstance(value, (list, dict)):
                    rendered = json.dumps(
                        value, ensure_ascii=False
                    )
                else:
                    rendered = str(value)

                values.append(_md_table(rendered))

        append_row(label, values)

    lines.extend(["", "---", ""])

    if possible_disagreements:
        lines.extend([
            "## ⚠️ Possible Disagreements",
            "",
            (
                "_Label: suspected difference — "
                "not a definitive contradiction._"
            ),
            "",
        ])
        lines.extend(
            "- " + _md_inline(d)
            for d in possible_disagreements
        )
        lines.append("")

    if gap_suggestions:
        lines.extend([
            "## 💡 Suggested Research Gaps",
            "",
            (
                "_Label: suggestions based on this source set — "
                "not confirmed gaps in the literature._"
            ),
            "",
        ])
        lines.extend(
            "- " + _md_inline(g)
            for g in gap_suggestions
        )
        lines.append("")

    lines.extend([
        "---",
        (
            "_Comparison generated by ResearchLens AI. "
            "Review all claims before use._"
        ),
    ])

    return "\n".join(lines)


def run_trace_to_markdown(
    result: RunResult,
    plan: Any = None,
) -> str:
    """Describe observed workflow metadata without asserting unseen steps."""
    lines = [
        "# Run Trace",
        "",
        f"**Run ID:** `{_md_inline(result.run_id)}`",
        f"**Task:** {result.task.value}",
        f"**Started:** {_md_inline(result.started_at)}",
        (
            f"**Finished:** "
            f"{_md_inline(result.finished_at or 'in progress')}"
        ),
        (
            f"**Latency:** {result.latency_seconds:.2f}s"
            if result.latency_seconds is not None
            else "**Latency:** unknown"
        ),
        "",
        "## Steps",
        "",
    ]

    if plan is not None:
        providers = [
            getattr(p, "value", str(p))
            for p in getattr(plan, "providers", [])
        ]
        lines.append(
            f"- **Search plan:** "
            f"query={_md_inline(getattr(plan, 'retrieval_query', '?'))}; "
            f"providers={_md_inline(', '.join(providers))}"
        )

    lines.extend([
        (
            f"- **Evidence assembled:** "
            f"{len(result.evidence_chunk_ids)} chunks from "
            f"{len(result.paper_ids_used)} paper(s)"
        ),
        f"- **Source coverage:** {result.source_coverage.value}",
        (
            f"- **Validation state:** "
            f"{result.validation_state.value} "
            "(structural, not fact-checking)"
        ),
        "",
        "## Model usage",
        "",
    ])

    if result.model_id or result.llm_calls_used:
        lines.extend([
            (
                f"- Model: "
                f"{_md_inline(result.model_id) if result.model_id else 'not reported'}"
            ),
            f"- LLM calls (local counter): {result.llm_calls_used}",
        ])

        if result.input_tokens_estimate is not None:
            lines.append(
                f"- Input tokens (estimate): {result.input_tokens_estimate}"
            )

        if result.output_tokens_reported is not None:
            lines.append(
                f"- Output tokens (reported): {result.output_tokens_reported}"
            )
    else:
        lines.append("- No model calls recorded in this run.")

    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(
            "- ⚠️ " + _md_inline(w)
            for w in result.warnings
        )

    return "\n".join(lines)