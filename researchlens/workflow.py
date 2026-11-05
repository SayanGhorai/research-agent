"""ResearchLens AI — bounded, document-scoped research orchestration.

Workflow: optional scholarly discovery -> scoped retrieval -> one bounded Granite
synthesis -> evidence validation -> reusable RunResult. This module makes no
network or foundation-model requests on import or when indexing PDFs.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from researchlens.agents import ReadAgent, SearchAgent, VerifyAgent, WriteAgent
from researchlens.config import Settings
from researchlens.ingestion import abstract_to_chunk
from researchlens.retrieval import CorpusIndex, chunk_text
from researchlens.schemas import (
    NOT_REPORTED,
    EvidenceChunk,
    PaperRecord,
    RunResult,
    SourceProvider,
    TaskType,
    TextCoverage,
    ValidationState,
)
from researchlens.sources import deduplicate_records
from researchlens.watsonx_client import (
    WatsonxBudgetExhausted,
    WatsonxError,
    WatsonxGenerationDisabled,
    get_client,
)


_EXTRACTION_FIELDS = (
    ("research_objective", "Research Objective"),
    ("method", "Method"),
    ("dataset_sample", "Dataset / Sample"),
    ("key_findings", "Key Findings"),
    ("metrics", "Metrics"),
    ("limitations", "Limitations"),
)


def _unique_warnings(warnings: list[str]) -> list[str]:
    """Retain warning order without reporting the same message multiple times."""
    return list(dict.fromkeys(warnings))


def _provider_error_message(exc: Exception) -> str:
    """Provide safe diagnostics without echoing credentials or source text."""
    detail = str(exc).lower()

    if any(phrase in detail for phrase in (
        "no model request was made",
        "local prompt-size guard",
        "local character guard",
    )):
        return (
            "Local evidence or prompt-size safeguard blocked generation. "
            "Narrow the question or selected source passages; "
            "no IBM request was made."
        )

    if "429" in detail or "consumption_limit_reached" in detail:
        return (
            "IBM Granite rate or capacity limit (HTTP 429); "
            "request not completed."
        )

    if "401" in detail or "403" in detail or "unauthorized" in detail:
        return (
            "IBM Granite authentication or permission rejected the request."
        )

    if "timeout" in detail or "timed out" in detail:
        return "IBM Granite request timed out."

    return (
        "IBM Granite request failed; "
        "review connectivity and service configuration."
    )


def _field_citations(value: str) -> set[str]:
    """Recognize one or more exact eight-hex passage IDs in bracket groups."""
    found: set[str] = set()

    for group in re.findall(r"\[([^\]\n]+)\]", value):
        parts = re.split(r"[,;\s]+", group.strip())

        if parts and all(
            re.fullmatch(r"[0-9a-fA-F]{8}", part)
            for part in parts
        ):
            found.update(part.lower() for part in parts)

    return found


def _parsed_grounded_fields(
    answer_text: str,
    allowed_chunk_ids: set[str],
    verified_claims: Optional[list] = None,
) -> dict[str, str]:
    """Accept individually verified, cited extraction fields.

    A provider citation points to evidence; it is not automatically
    an adequate scientific justification.

    If verified claims are available, only fields whose exact value
    or labeled line was checked VALID are exported as findings.
    """
    labels = {
        label.casefold(): key
        for key, label in _EXTRACTION_FIELDS
    }

    labels.update({
        "objective": "research_objective",
        "dataset": "dataset_sample",
        "sample": "dataset_sample",
        "results": "key_findings",
        "findings": "key_findings",
        "evaluation metrics": "metrics",
        "limitation": "limitations",
        "methodology": "method",
    })

    allowed = {
        cid[:8].lower()
        for cid in allowed_chunk_ids
    }

    valid_claims = None

    if verified_claims is not None:
        valid_claims = []

        for claim in verified_claims:
            if claim.validation_state != ValidationState.VALID:
                continue

            linked = {
                cid[:8].lower()
                for cid in claim.evidence_chunk_ids
            }

            if linked and linked.issubset(allowed):
                valid_claims.append(claim.claim_text)

    def normalize(value: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            value.strip().replace("**", "").replace("__", ""),
        ).strip()

    fields: dict[str, str] = {}

    for raw_line in answer_text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("|") and line.endswith("|"):
            cells = [
                cell.strip()
                for cell in line.strip("|").split("|")
            ]

            if len(cells) != 2:
                continue

            label, value = cells

        else:
            simple = re.sub(
                r"^(?:[-*•]\s+|\d+[.)]\s+)",
                "",
                line,
            )

            match = re.match(
                r"^([A-Za-z /]+?)\s*:\s*(.*)$",
                normalize(simple),
            )

            if not match:
                continue

            label, value = match.groups()

        label = normalize(label).casefold()
        key = labels.get(label)

        if key is None or key in fields:
            continue

        value = value.strip()

        bare = re.sub(
            r"\[[^]\n]+\]",
            "",
            value,
        ).strip().strip("*_ ").casefold()

        if not bare or bare.replace("_", " ") in {
            "not reported",
            "not available",
            "unknown",
            "n/a",
            "none",
            "not stated in the supplied evidence",
            "not reported in the supplied evidence",
        }:
            continue

        citations = _field_citations(value)

        if not citations or not citations.issubset(allowed):
            continue

        if valid_claims is not None and not any(
            normalize(claim) in {
                normalize(line),
                normalize(value),
            }
            for claim in valid_claims
        ):
            continue

        fields[key] = value

    return fields


class ResearchWorkflow:
    """An orchestration object with no shared private corpus state.

    The caller supplies a session-scoped index and library.

    Only run() or explicit extract(generate=True) can request
    IBM Granite inference.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._read_agent = ReadAgent()
        self._verify_agent = VerifyAgent()
        self._write_agent = WriteAgent()

    def run(
        self,
        task: TaskType,
        question: str,
        library: list[PaperRecord],
        index: CorpusIndex,
        *,
        search_first: bool = True,
        selected_paper_ids: Optional[list[str]] = None,
    ) -> tuple[RunResult, list[PaperRecord], list[str]]:
        """Run one bounded research task.

        Returns:
            result, updated_library, warnings

        Selected paper IDs are a hard scope constraint.
        """
        result = RunResult(
            task=task,
            question=question,
        )

        warnings: list[str] = []
        working_library = list(library)

        if selected_paper_ids is not None and search_first:
            warnings.append(
                "Selected-paper scope disables online discovery for this run."
            )
            search_first = False

        client = get_client(self._settings)
        client.reset_budget()

        # --------------------------------------------------
        # Optional academic source discovery
        # --------------------------------------------------

        if search_first:
            search_agent = SearchAgent(
                self._settings,
                client,
            )

            _, discovered, search_warnings = (
                search_agent.plan_and_search(
                    question,
                    working_library,
                )
            )

            warnings.extend(search_warnings)

            merged, skipped = deduplicate_records(
                working_library,
                discovered,
            )

            previous_ids = {
                p.paper_id
                for p in working_library
            }

            fresh_papers = [
                p for p in merged
                if p.paper_id not in previous_ids
            ]

            working_library = merged

            if skipped:
                warnings.append(
                    f"Skipped {len(skipped)} duplicate records."
                )

            index_warnings = _index_papers(
                fresh_papers,
                index,
            )

            warnings.extend(index_warnings)

            if index_warnings:
                warnings.append(
                    "Some discovered abstracts were not indexed. "
                    "Their metadata may be listed, but rejected "
                    "abstracts cannot support research answers."
                )

        # --------------------------------------------------
        # Hard document scope
        # --------------------------------------------------

        if selected_paper_ids is not None:
            chosen = list(
                dict.fromkeys(selected_paper_ids)
            )

            lookup = {
                p.paper_id: p
                for p in working_library
            }

            eligible_papers = [
                lookup[pid]
                for pid in chosen
                if pid in lookup
            ]

            missing = (
                set(chosen)
                - {p.paper_id for p in eligible_papers}
            )

            if missing:
                warnings.append(
                    f"{len(missing)} selected paper(s) "
                    "are not in this session library."
                )

        elif not search_first and any(
            p.source_provider == SourceProvider.UPLOAD
            for p in working_library
        ):
            eligible_papers = [
                p for p in working_library
                if p.source_provider == SourceProvider.UPLOAD
            ]

        else:
            eligible_papers = working_library

        # --------------------------------------------------
        # Task-aware retrieval
        # --------------------------------------------------

        evidence, evidence_warnings = (
            self._read_agent.assemble_evidence(
                question,
                index,
                eligible_papers,
                max_chunks=(
                    self._settings.max_pdfs_per_session * 3
                ),
                task=task,
            )
        )

        warnings.extend(evidence_warnings)

        result.evidence_chunk_ids = list(
            dict.fromkeys(
                c.chunk_id for c in evidence
            )
        )

        result.paper_ids_used = list(
            dict.fromkeys(
                c.paper_id for c in evidence
            )
        )

        # --------------------------------------------------
        # Text coverage
        # --------------------------------------------------

        if evidence:
            coverages = {
                chunk.source_type
                for chunk in evidence
            }

            if coverages == {TextCoverage.FULL_TEXT}:
                result.source_coverage = TextCoverage.FULL_TEXT

            elif coverages == {TextCoverage.ABSTRACT_ONLY}:
                result.source_coverage = TextCoverage.ABSTRACT_ONLY

            else:
                result.source_coverage = TextCoverage.EXCERPT

        # --------------------------------------------------
        # Generation disabled
        # --------------------------------------------------

        if not self._settings.generation_enabled():
            result.answer_text = (
                "[Generation disabled] Configure IBM credentials "
                "and set ALLOW_LIVE_GENERATION=true to enable "
                "AI synthesis. "
                f"Found {len(evidence)} evidence chunks from "
                f"{len(result.paper_ids_used)} paper(s)."
            )

            result.warnings = _unique_warnings(warnings)
            result.mark_finished()

            return (
                result,
                working_library,
                result.warnings,
            )

        # --------------------------------------------------
        # Incomplete comparison guard
        # --------------------------------------------------

        if task == TaskType.COMPARE and (
            len(eligible_papers) < 2
            or (
                {p.paper_id for p in eligible_papers}
                - {chunk.paper_id for chunk in evidence}
            )
        ):
            warnings.append(
                "Comparison stopped before IBM generation: "
                "at least two selected papers, each with "
                "retrievable evidence, are required."
            )

            result.answer_text = (
                "Incomplete comparison: one or more selected "
                "papers has no retrievable evidence. "
                "Choose readable papers before comparing."
            )

            result.warnings = _unique_warnings(warnings)
            result.mark_finished()

            return (
                result,
                working_library,
                result.warnings,
            )

        # --------------------------------------------------
        # No evidence = no model request
        # --------------------------------------------------

        if not evidence:
            result.answer_text = (
                "No relevant evidence was found within the "
                "selected document scope. Choose different "
                "papers or broaden the source selection."
            )

            result.warnings = _unique_warnings(warnings)
            result.mark_finished()

            return (
                result,
                working_library,
                result.warnings,
            )

        # --------------------------------------------------
        # Single bounded IBM synthesis
        # --------------------------------------------------

        try:
            answer, claims, usage = (
                self._write_agent.synthesize(
                    task=task,
                    question=question,
                    chunks=evidence,
                    papers=eligible_papers,
                    client=client,
                )
            )

            result.answer_text = answer
            result.claims = claims
            result.model_id = usage.model_id

            result.output_tokens_reported = (
                usage.output_tokens
            )

            result.input_tokens_estimate = (
                usage.input_tokens
            )

        except WatsonxGenerationDisabled:
            result.answer_text = (
                "[Generation disabled] IBM model access "
                "is not enabled."
            )

            warnings.append(
                "IBM generation is disabled or "
                "credentials are unavailable."
            )

        except WatsonxBudgetExhausted:
            result.answer_text = (
                "[Budget exhausted] The per-action "
                "Granite call limit was reached."
            )

            warnings.append(
                "Granite call budget exhausted for this action."
            )

        except WatsonxError as exc:
            summary = _provider_error_message(exc)

            if "no IBM request was made" in summary:
                result.answer_text = (
                    f"[Research limit] {summary}"
                )

            else:
                result.answer_text = (
                    f"[IBM error] {summary}"
                )

            warnings.append(summary)

        finally:
            # Count attempted requests, including HTTP 429.
            result.llm_calls_used = client.action_calls_used

        # --------------------------------------------------
        # Evidence validation
        # --------------------------------------------------

        if result.model_id:
            result = self._verify_agent.verify_result(
                result,
                evidence,
                eligible_papers,
            )

        else:
            result.validation_state = (
                ValidationState.UNCHECKED
            )

        result.warnings = _unique_warnings(
            [*warnings, *result.warnings]
        )

        result.mark_finished()

        return (
            result,
            working_library,
            result.warnings,
        )

    def extract(
        self,
        paper: PaperRecord,
        index: CorpusIndex,
        *,
        generate: bool = False,
    ) -> dict[str, Any]:
        """Return metadata or explicitly requested grounded extraction.

        generate=False never invokes IBM.
        generate=True uses the normal bounded workflow.
        """
        chunks = [
            c for c in index._chunks
            if c.paper_id == paper.paper_id
        ]

        data = self._read_agent.extract_structured(
            paper,
            chunks,
        )

        data["generation_status"] = "not_requested"
        data["validation_state"] = ValidationState.UNCHECKED.value
        data["evidence_chunk_ids"] = []
        data["warnings"] = []

        if not generate:
            return data

        if not self._settings.generation_enabled():
            data["generation_status"] = "disabled"
            data["warnings"] = [
                "Granite generation is disabled or not configured."
            ]
            return data

        if not chunks:
            data["generation_status"] = "no_indexed_evidence"
            data["warnings"] = [
                "The selected paper has no indexed evidence. "
                "Add readable text or index its available abstract "
                "before requesting Granite."
            ]
            return data

        question = (
            f"Extract research findings only from the "
            f"selected paper '{paper.title}'. "
            "Return exactly six labeled fields: "
            "Research Objective:, Method:, Dataset / Sample:, "
            "Key Findings:, Metrics:, Limitations:. "
            "Distinguish the paper's own experiments from "
            "studies it cites. Attach a valid [CHUNK-ID] "
            "citation to every supported field, otherwise "
            "write 'not reported'. Do not infer missing metrics."
        )

        run_result, _, _ = self.run(
            task=TaskType.EXTRACT,
            question=question,
            library=[paper],
            index=index,
            search_first=False,
            selected_paper_ids=[paper.paper_id],
        )

        data["validation_state"] = (
            run_result.validation_state.value
        )

        data["evidence_chunk_ids"] = list(
            run_result.evidence_chunk_ids
        )

        data["warnings"] = list(
            run_result.warnings
        )

        if (
            not run_result.model_id
            or run_result.validation_state == ValidationState.INVALID
        ):
            data["generation_status"] = "failed_or_unverified"
            return data

        # Only evidence from this selected paper.
        valid_ids = {
            chunk.chunk_id[:8]
            for chunk in index.get_chunks_by_ids(
                run_result.evidence_chunk_ids
            )
            if chunk.paper_id == paper.paper_id
        }

        grounded_fields = _parsed_grounded_fields(
            run_result.answer_text,
            valid_ids,
            verified_claims=run_result.claims,
        )

        for field, _ in _EXTRACTION_FIELDS:
            if field in grounded_fields:
                data[field] = grounded_fields[field]

        data["generation_status"] = "partial" if (
            len(grounded_fields) < len(_EXTRACTION_FIELDS)
            or run_result.validation_state != ValidationState.VALID
        ) else "completed_review_required"

        if not grounded_fields:
            data["warnings"].append(
                "The model returned no individually cited "
                "fields; no extracted findings were accepted."
            )

        return data


def _index_papers(
    papers: list[PaperRecord],
    index: CorpusIndex,
) -> list[str]:
    """Index discovered abstracts without treating them as full text."""
    additions: list[EvidenceChunk] = []

    for paper in papers:
        chunk = abstract_to_chunk(paper)

        if chunk:
            additions.append(chunk)

    return (
        index.add_chunks(additions)
        if additions else []
    )


def index_uploaded_paper(
    paper: PaperRecord,
    page_chunks: list[EvidenceChunk],
    index: CorpusIndex,
) -> list[str]:
    """Index page-preserving passages, with abstract-only fallback."""
    if page_chunks:
        passages: list[EvidenceChunk] = []

        for page in page_chunks:
            if page.is_abstract:
                passages.append(page)
                continue

            for passage in chunk_text(
                page.text,
                paper.paper_id,
                page.page_index,
                is_abstract=False,
                target_tokens=105,
                overlap_tokens=18,
            ):
                passage.source_type = page.source_type

                passage.extraction_warnings = list(
                    page.extraction_warnings
                )

                passages.append(passage)

        return index.add_chunks(passages) if passages else [
            "No readable passages were produced; "
            "this PDF was not indexed."
        ]

    abstract_chunk = abstract_to_chunk(paper)

    return (
        index.add_chunks([abstract_chunk])
        if abstract_chunk else [
            "No page passages or abstract available; "
            "this paper was not indexed."
        ])