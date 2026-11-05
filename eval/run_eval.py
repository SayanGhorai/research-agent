"""Offline retrieval evaluation with honest relevance and abstention reporting.

Usage: .venv\\Scripts\\python.exe eval/run_eval.py

The included corpus contains ten illustrative abstracts, not independently
verified bibliographic records. Relevance labels derived from keywords are
PROXY labels, not manual judgments. An empty retrieval result is not proof
that an LLM would abstain.

No Granite calls or semantic-model downloads occur in this script.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from researchlens.ingestion import abstract_to_chunk
from researchlens.retrieval import CorpusIndex
from researchlens.schemas import (
    PaperRecord,
    SourceProvider,
    TextCoverage,
)

EVAL_FILE = ROOT / "eval" / "results.json"


# ============================================================
# Illustrative offline corpus
# ============================================================

SYNTHETIC_PAPERS = [
    PaperRecord(
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Lewis, P.", "Perez, E."],
        year=2020,
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
        abstract=(
            "We explore a general-purpose fine-tuning recipe for "
            "retrieval-augmented generation (RAG) that combines parametric "
            "and non-parametric memory for language generation. RAG models "
            "retrieve relevant documents and then attend over them "
            "to generate answers."
        ),
    ),
    PaperRecord(
        title="Survey of Hallucination in Natural Language Generation",
        authors=["Ji, Z.", "Lee, N."],
        year=2022,
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
        abstract=(
            "This survey reviews hallucination in NLG systems including "
            "neural language models. We categorize factual errors, "
            "intrinsic and extrinsic hallucinations, and discuss "
            "mitigation strategies including grounding and "
            "retrieval-based corrections."
        ),
    ),
    PaperRecord(
        title="Evaluating Question Answering Systems: Metrics and Benchmarks",
        authors=["Rajpurkar, P.", "Jia, R."],
        year=2018,
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
        abstract=(
            "We discuss standard evaluation metrics for question answering "
            "including F1, Exact Match, BLEU, and ROUGE. We compare "
            "these metrics on SQuAD and other benchmark datasets."
        ),
    ),
    PaperRecord(
        title="Dense Passage Retrieval for Open-Domain Question Answering",
        authors=["Karpukhin, V.", "Oguz, B."],
        year=2020,
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
        abstract=(
            "We propose DPR, a dense retrieval method that outperforms "
            "TF-IDF based retrieval for open-domain QA. Dense encoders "
            "trained with BERT embeddings enable semantic matching "
            "beyond keyword overlap."
        ),
    ),
    PaperRecord(
        title=(
            "BERT: Pre-training of Deep Bidirectional Transformers "
            "for Language Understanding"
        ),
        authors=["Devlin, J.", "Chang, M."],
        year=2019,
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
        abstract=(
            "BERT introduces deep bidirectional transformer pretraining "
            "on masked language modeling and next-sentence prediction. "
            "Contextual embeddings from BERT achieve state-of-the-art "
            "results on a range of NLP tasks including question "
            "answering and NLI."
        ),
    ),
    PaperRecord(
        title="Mitigating Hallucinations in LLMs with Retrieval Grounding",
        authors=["Shuster, K.", "Poff, S."],
        year=2021,
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
        abstract=(
            "We show that grounding language model generation with "
            "retrieved documents significantly reduces hallucination "
            "rates. Factual grounding via retrieval-augmented prompting "
            "improves faithfulness across dialogue and QA tasks."
        ),
    ),
    PaperRecord(
        title=(
            "Reciprocal Rank Fusion Outperforms Condorcet "
            "and Individual Rank Learning Methods"
        ),
        authors=["Cormack, G.", "Clarke, C."],
        year=2009,
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
        abstract=(
            "Reciprocal rank fusion (RRF) is a simple rank aggregation "
            "method that combines the results of hybrid retrieval "
            "systems. RRF outperforms more complex learning-to-rank "
            "methods on standard TREC collections."
        ),
    ),
    PaperRecord(
        title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N."],
        year=2017,
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
        abstract=(
            "We propose the Transformer architecture based on multi-head "
            "self-attention mechanisms. The model uses attention to capture "
            "long-range dependencies without recurrence or convolution. "
            "Transformers achieve state-of-the-art results in machine "
            "translation."
        ),
    ),
    PaperRecord(
        title="Extractive Summarization Using Sentence Scoring",
        authors=["Nallapati, R.", "Zhou, B."],
        year=2017,
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
        abstract=(
            "Extractive summarization selects key sentences directly "
            "from the source document. We compare sentence scoring "
            "methods and show that neural models outperform TF-IDF "
            "based selection for news summarization."
        ),
    ),
    PaperRecord(
        title="Abstractive Text Summarization with Sequence-to-Sequence Models",
        authors=["Rush, A.", "Chopra, S."],
        year=2015,
        source_provider=SourceProvider.ARXIV,
        text_coverage=TextCoverage.ABSTRACT_ONLY,
        abstract=(
            "Abstractive summarization generates novel text capturing "
            "the meaning of the input. Sequence-to-sequence models with "
            "attention produce fluent and informative summaries "
            "that differ from the extractive approach."
        ),
    ),
]


# ============================================================
# Evaluation helpers
# ============================================================

def _load_questions() -> list[dict[str, Any]]:
    """Load questions without inventing relevance labels."""
    with EVAL_FILE.open(encoding="utf-8") as handle:
        data = json.load(handle)

    questions = data.get("questions")

    if not isinstance(questions, list):
        raise ValueError(
            "eval/results.json must contain a 'questions' array"
        )

    for position, item in enumerate(questions, 1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Question {position} must be an object"
            )

        if (
            not isinstance(item.get("question"), str)
            or not item["question"].strip()
        ):
            raise ValueError(
                f"Question {position} has no valid question text"
            )

        if item.get("type") not in {
            "answerable",
            "unanswerable",
            "comparison",
        }:
            raise ValueError(
                f"Question {position} has an unsupported type"
            )

    return questions


def _hit_at_k(
    ranked_results: list[tuple[int, float]],
    relevant_paper_ids: set[str],
    chunks: list,
    k: int = 5,
) -> int:
    """Return 1 when a labeled relevant paper appears in top k."""
    return int(
        any(
            0 <= idx < len(chunks)
            and chunks[idx].paper_id in relevant_paper_ids
            for idx, _ in ranked_results[:max(0, k)]
        )
    )


def _mrr_at_k(
    ranked_results: list[tuple[int, float]],
    relevant_paper_ids: set[str],
    chunks: list,
    k: int = 5,
) -> float:
    """Reciprocal rank of the first labeled relevant paper."""
    for rank, (idx, _) in enumerate(
        ranked_results[:max(0, k)],
        1,
    ):
        if (
            0 <= idx < len(chunks)
            and chunks[idx].paper_id in relevant_paper_ids
        ):
            return 1.0 / rank

    return 0.0


def _normal_title(title: str) -> str:
    return " ".join(title.split()).casefold()


def _resolve_labels(
    question: dict[str, Any],
    papers: list[PaperRecord],
) -> tuple[set[str], str, list[str]]:
    """Resolve explicit relevance labels or a weaker keyword proxy.

    Explicit labels:
      expected_relevant_paper_titles
      expected_relevant_paper_ids

    Proxy labels:
      expected_relevant_paper_keywords

    Keyword matches in the evaluated corpus are not independent
    human judgments of relevance.
    """
    problems: list[str] = []

    titles = question.get(
        "expected_relevant_paper_titles"
    )

    ids = question.get(
        "expected_relevant_paper_ids"
    )

    if titles is not None or ids is not None:
        if titles is not None and (
            not isinstance(titles, list)
            or any(
                not isinstance(title, str)
                or not title.strip()
                for title in titles
            )
        ):
            return (
                set(),
                "invalid",
                ["Manual title labels must be nonempty strings in a list"],
            )

        if ids is not None and (
            not isinstance(ids, list)
            or any(
                not isinstance(item, str)
                or not item.strip()
                for item in ids
            )
        ):
            return (
                set(),
                "invalid",
                ["Manual ID labels must be nonempty strings in a list"],
            )

        names = {
            _normal_title(p.title): p.paper_id
            for p in papers
        }

        known_ids = {
            p.paper_id for p in papers
        }

        selected: set[str] = set()

        for title in titles or []:
            match = names.get(
                _normal_title(title)
            )

            if match:
                selected.add(match)
            else:
                problems.append(
                    f"Unknown manually labeled title: {title}"
                )

        for paper_id in ids or []:
            if paper_id in known_ids:
                selected.add(paper_id)
            else:
                problems.append(
                    "An explicitly labeled paper ID "
                    "does not exist in this corpus"
                )

        return (
            selected,
            "invalid" if problems else "manual",
            problems,
        )

    keywords = question.get(
        "expected_relevant_paper_keywords"
    )

    if keywords is None:
        return set(), "unlabelled", []

    if (
        not isinstance(keywords, list)
        or any(
            not isinstance(keyword, str)
            for keyword in keywords
        )
    ):
        return (
            set(),
            "invalid",
            ["Keyword proxy labels must be a list of strings"],
        )

    keywords = [
        keyword.strip().casefold()
        for keyword in keywords
        if keyword.strip()
    ]

    if not keywords:
        return set(), "unlabelled", []

    relevant = {
        paper.paper_id
        for paper in papers
        if any(
            keyword in (
                paper.title + " " + (paper.abstract or "")
            ).casefold()
            for keyword in keywords
        )
    }

    return relevant, "keyword_proxy", []


def _relevant_paper_ids(
    question: dict[str, Any],
    papers: list[PaperRecord],
) -> set[str]:
    """Compatibility helper; missing labels never imply abstention."""
    relevant, _, _ = _resolve_labels(
        question,
        papers,
    )

    return relevant


def _make_stable_corpus() -> list[PaperRecord]:
    """Give synthetic records reproducible identifiers."""
    copies = [
        paper.model_copy(deep=True)
        for paper in SYNTHETIC_PAPERS
    ]

    for paper in copies:
        paper.paper_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "researchlens-offline-eval:"
                + _normal_title(paper.title),
            )
        )

    return copies


def _mean(values: list[float]) -> float | None:
    if not values:
        return None

    return round(
        sum(values) / len(values),
        3,
    )


# ============================================================
# Main offline evaluation
# ============================================================

def run_eval() -> dict[str, Any]:
    """Evaluate TF-IDF retrieval, not generated answer correctness."""

    print(
        "ResearchLens AI — Offline Retrieval Evaluation (TF-IDF only)"
    )
    print("=" * 65)

    index = CorpusIndex(
        enable_semantic=False
    )

    papers = _make_stable_corpus()

    chunks = [
        chunk
        for paper in papers
        if (chunk := abstract_to_chunk(paper))
    ]

    warnings = index.add_chunks(chunks)

    if warnings:
        raise RuntimeError(
            "Evaluation corpus could not be fully indexed: "
            + "; ".join(warnings)
        )

    if len(index._chunks) != len(chunks):
        raise RuntimeError(
            "Evaluation corpus indexing was incomplete"
        )

    questions = _load_questions()

    manual_hits: list[float] = []
    manual_mrrs: list[float] = []

    proxy_hits: list[float] = []
    proxy_mrrs: list[float] = []

    comparison_coverages: list[float] = []
    comparison_proxy_coverages: list[float] = []

    skipped: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    unanswerable_count = 0

    print(
        f"Illustrative corpus: {len(papers)} papers, "
        f"{len(chunks)} passages"
    )

    for index_num, question in enumerate(
        questions,
        start=1,
    ):
        qid = str(
            question.get("id", index_num)
        )

        start = time.perf_counter()

        ranked = index.search(
            question["question"],
            top_k=5,
        )

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        relevant, source, problems = _resolve_labels(
            question,
            papers,
        )

        retrieved_ids = [
            index._chunks[idx].paper_id
            for idx, _ in ranked
            if 0 <= idx < index.chunk_count()
        ]

        observation: dict[str, Any] = {
            "id": qid,
            "type": question["type"],
            "label_source": source,
            "retrieved_candidates": len(ranked),
            "retrieved_titles": [
                next(
                    p.title
                    for p in papers
                    if p.paper_id == pid
                )
                for pid in dict.fromkeys(retrieved_ids)
            ],
            "latency_ms": latency_ms,
            "evaluation_warning": problems,
        }

        # --------------------------------------------------
        # Unanswerable questions
        # --------------------------------------------------

        if question["type"] == "unanswerable":
            unanswerable_count += 1

            observation["abstention_tested"] = False

            observation["note"] = (
                "Retrieval was observed; no language-model "
                "abstention test was performed. Missing relevance "
                "labels do not prove that a question is unanswerable."
            )

            print(
                f"  [UNTESTED ABSTENTION] q{qid}: "
                f"{len(ranked)} retrieval candidates"
            )

        # --------------------------------------------------
        # Questions without usable relevance labels
        # --------------------------------------------------

        elif source not in {
            "manual",
            "keyword_proxy",
        } or not relevant:
            observation["scored"] = False

            if source == "unlabelled":
                reason = (
                    "No independently labeled relevance target"
                )

            elif (
                not relevant
                and source == "keyword_proxy"
            ):
                reason = (
                    "No matching proxy target in the corpus"
                )

            else:
                reason = (
                    "; ".join(problems)
                    or "No eligible labeled target"
                )

            observation["note"] = reason

            skipped.append({
                "id": qid,
                "reason": reason,
            })

            print(
                f"  [UNSCORED] q{qid}: {reason}"
            )

        # --------------------------------------------------
        # Measurable retrieval relevance
        # --------------------------------------------------

        else:
            hit = _hit_at_k(
                ranked,
                relevant,
                index._chunks,
                k=5,
            )

            mrr = _mrr_at_k(
                ranked,
                relevant,
                index._chunks,
                k=5,
            )

            observation.update({
                "scored": True,
                "hit_at_5": hit,
                "mrr_at_5": mrr,
                "relevant_papers": len(relevant),
            })

            if source == "manual":
                manual_hits.append(float(hit))
                manual_mrrs.append(mrr)

            else:
                proxy_hits.append(float(hit))
                proxy_mrrs.append(mrr)

            if question["type"] == "comparison":
                coverage = round(
                    len(
                        set(retrieved_ids) & relevant
                    ) / len(relevant),
                    3,
                )

                observation[
                    "paper_coverage_at_5"
                ] = coverage

                if source == "manual":
                    comparison_coverages.append(
                        coverage
                    )

                else:
                    comparison_proxy_coverages.append(
                        coverage
                    )

            print(
                f"  [{source.upper()}] q{qid}: "
                f"Hit@5={hit}, MRR@5={mrr:.3f}"
            )

        observations.append(observation)

    # ------------------------------------------------------
    # Honest evaluation result
    # ------------------------------------------------------

    result = {
        "hit_at_5_tfidf": _mean(manual_hits),
        "mrr_at_5_tfidf": _mean(manual_mrrs),

        "hit_at_5_tfidf_proxy": _mean(proxy_hits),
        "mrr_at_5_tfidf_proxy": _mean(proxy_mrrs),

        "comparison_paper_coverage_at_5": _mean(
            comparison_coverages
        ),

        "comparison_paper_coverage_at_5_proxy": _mean(
            comparison_proxy_coverages
        ),

        "manual_scored_questions": len(manual_hits),
        "proxy_scored_questions": len(proxy_hits),
        "unscored_questions": len(skipped),

        "unanswerable_questions_observed": (
            unanswerable_count
        ),

        "unanswerable_abstention_accuracy": None,

        "hit_at_5_hybrid": None,
        "mrr_at_5_hybrid": None,
        "hybrid_evaluated": False,

        "model_answer_quality_evaluated": False,

        "corpus_size_papers": len(papers),
        "corpus_size_chunks": len(chunks),
        "questions_total": len(questions),

        "per_question_results": observations,
        "skipped_questions": skipped,

        "evaluation_basis": (
            "Explicit title/ID labels are user-provided "
            "relevance annotations; keyword_proxy labels "
            "are heuristics derived from the evaluated corpus, "
            "not independently judged ground truth."
        ),

        "notes": (
            "Illustrative 10-paper synthetic abstract corpus; "
            "TF-IDF only. No semantic/hybrid retrieval was run, "
            "no IBM/LLM generation was run, and no answer correctness "
            "or model abstention accuracy was measured."
        ),

        "status": (
            "LABELLED_TFIDF_MEASURED"
            if manual_hits
            else "PROXY_ONLY"
            if proxy_hits
            else "NO_SCORED_RELEVANCE_LABELS"
        ),
    }

    print(
        "\nManual-label Hit@5:",
        result["hit_at_5_tfidf"],
    )

    print(
        "Keyword-proxy Hit@5 (not gold):",
        result["hit_at_5_tfidf_proxy"],
    )

    print(
        "Model abstention/answer accuracy: NOT TESTED"
    )

    return result


def main() -> None:
    """Update evaluation results while preserving question definitions.

    Running this entry point intentionally updates eval/results.json.
    Verification tests should call run_eval() with mocked questions instead.
    """
    if not EVAL_FILE.is_file():
        raise FileNotFoundError(
            "eval/results.json was not found; supply question "
            "definitions before running. "
            "No evaluation file was created or overwritten."
        )

    with EVAL_FILE.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "eval/results.json must be a JSON object"
        )

    results = run_eval()

    # Replace stale scores, including potentially misleading
    # abstention claims from older evaluator versions.
    # Preserve questions and unrelated top-level fields.
    data["offline_retrieval_results"] = results

    with EVAL_FILE.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Results saved to eval/results.json "
        "(question definitions preserved)"
    )


if __name__ == "__main__":
    main()