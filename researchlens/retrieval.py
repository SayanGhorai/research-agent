"""researchlens.retrieval

Small, session-scoped, CPU-friendly research retrieval index.

- Page-preserving, overlapping text passages (approximate word counts).
- TF-IDF retrieval and optional MiniLM dense retrieval with RRF.
- Document-scope filtering before ranking in either branch.
- Fail closed on a full corpus: never silently index half a new PDF.
- Reuse previously computed dense vectors within the same session/index.
- If dense inference fails, report lexical-only degraded mode.

No IBM foundation-model calls occur here. Optional SentenceTransformer loading
can download model weights if they are not already installed or cached.
Uploaded text and vectors are not globally cached.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from researchlens.schemas import EvidenceChunk, TextCoverage


MAX_CHUNKS_PER_SESSION = 1_500
CHUNK_TOKEN_TARGET = 180
CHUNK_TOKEN_OVERLAP = 30
RRF_K = 60
DEFAULT_TOP_K = 20
DEFAULT_ANSWER_CHUNKS = 8
MAX_PER_PAPER = 3
_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _rough_tokens(text: str) -> list[str]:
    """Whitespace-delimited words, not actual model tokens."""
    return re.split(r"\s+", text.strip()) if text.strip() else []


def chunk_text(
    text: str,
    paper_id: str,
    page_index: Optional[int],
    is_abstract: bool,
    target_tokens: int = CHUNK_TOKEN_TARGET,
    overlap_tokens: int = CHUNK_TOKEN_OVERLAP,
) -> list[EvidenceChunk]:
    """Produce overlapping, page-linked passages.

    Limits use approximate word counts. Model tokenization may differ.
    """
    if (
        target_tokens < 1
        or overlap_tokens < 0
        or overlap_tokens >= target_tokens
    ):
        raise ValueError(
            "Require target_tokens >= 1 and "
            "0 <= overlap_tokens < target_tokens"
        )

    words = _rough_tokens(text)

    if not words:
        return []

    result: list[EvidenceChunk] = []
    step = target_tokens - overlap_tokens

    for start in range(0, len(words), step):
        end = min(start + target_tokens, len(words))

        result.append(
            EvidenceChunk(
                paper_id=paper_id,
                source_type=(
                    TextCoverage.ABSTRACT_ONLY
                    if is_abstract
                    else TextCoverage.FULL_TEXT
                ),
                page_index=page_index,
                is_abstract=is_abstract,
                text=" ".join(words[start:end]),
            )
        )

        if end == len(words):
            break

    return result


class CorpusIndex:
    """A session-local corpus index with no disk-based document cache."""

    def __init__(self, enable_semantic: bool = True) -> None:
        self.enable_semantic = enable_semantic

        self._chunks: list[EvidenceChunk] = []
        self._texts: list[str] = []

        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._tfidf_attempted = False

        self._embeddings: Optional[np.ndarray] = None
        self._embed_model = None
        self._embed_model_version = ""

        self._degraded_mode = False
        self._semantic_unavailable = False

        # Exact-text embedding reuse is confined to this CorpusIndex.
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._corpus_hash = ""

    def add_chunks(
        self,
        chunks: list[EvidenceChunk],
    ) -> list[str]:
        """Add an entire batch or reject it without partial indexing.

        Identical passages from different papers retain separate provenance.
        """
        if not chunks:
            return []

        seen = {
            (c.paper_id, c.page_index, c.content_hash)
            for c in self._chunks
        }

        new_chunks: list[EvidenceChunk] = []

        for chunk in chunks:
            key = (
                chunk.paper_id,
                chunk.page_index,
                chunk.content_hash,
            )

            if key not in seen:
                new_chunks.append(chunk)
                seen.add(key)

        available = (
            MAX_CHUNKS_PER_SESSION - len(self._chunks)
        )

        if len(new_chunks) > available:
            return [
                "Session chunk limit (1,500) would be exceeded; "
                "the entire new batch was rejected. "
                "Remove a paper or upload a smaller PDF."
            ]

        if not new_chunks:
            return []

        self._chunks.extend(new_chunks)
        self._texts.extend(
            chunk.text for chunk in new_chunks
        )

        self._corpus_hash = _corpus_hash(self._chunks)
        self._invalidate_index()

        return []

    def remove_paper(self, paper_id: str) -> int:
        """Remove one paper's passages and private cached vectors."""
        before = len(self._chunks)

        self._chunks = [
            chunk
            for chunk in self._chunks
            if chunk.paper_id != paper_id
        ]

        removed = before - len(self._chunks)

        if removed:
            self._texts = [
                chunk.text for chunk in self._chunks
            ]

            self._corpus_hash = _corpus_hash(
                self._chunks
            )

            remaining_hashes = {
                _embedding_key(chunk.text)
                for chunk in self._chunks
            }

            self._embedding_cache = {
                key: vector
                for key, vector in self._embedding_cache.items()
                if key in remaining_hashes
            }

            self._invalidate_index()

        return removed

    def chunk_count(self) -> int:
        return len(self._chunks)

    def is_empty(self) -> bool:
        return not self._chunks

    def _invalidate_index(self) -> None:
        """Invalidate indexed matrices after changing the corpus."""
        self._tfidf = None
        self._tfidf_matrix = None
        self._tfidf_attempted = False
        self._embeddings = None

        # Keep a successfully loaded model and valid session vectors.
        # If the semantic model failed, do not retry every search.

    def _build_tfidf(self) -> None:
        self._tfidf_attempted = True

        if not self._texts:
            return

        vectorizer = TfidfVectorizer(
            min_df=1,
            max_features=50_000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )

        try:
            matrix = vectorizer.fit_transform(
                self._texts
            )

        except ValueError as exc:
            if "empty vocabulary" in str(exc).lower():
                return
            raise

        self._tfidf = vectorizer
        self._tfidf_matrix = matrix

    def _build_embeddings(self) -> None:
        """Build dense vectors lazily, degrading safely on failures."""
        if self._semantic_unavailable or not self._texts:
            return

        try:
            if self._embed_model is None:
                from sentence_transformers import (  # pyright: ignore[reportMissingImports]
                    SentenceTransformer,
                )

                self._embed_model = SentenceTransformer(
                    _MODEL_ID,
                    device="cpu",
                )

                self._embed_model_version = _MODEL_ID

            keys = [
                _embedding_key(text)
                for text in self._texts
            ]

            missing: dict[str, str] = {}

            for key, text in zip(keys, self._texts):
                if key not in self._embedding_cache:
                    missing.setdefault(key, text)

            if missing:
                raw = np.asarray(
                    self._embed_model.encode(
                        list(missing.values()),
                        batch_size=16,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                )

                if raw.ndim == 1:
                    raw = raw.reshape(1, -1)

                if (
                    raw.ndim != 2
                    or raw.shape[0] != len(missing)
                ):
                    raise ValueError(
                        "Embedding model returned an invalid shape"
                    )

                norms = np.linalg.norm(
                    raw,
                    axis=1,
                    keepdims=True,
                )

                if (
                    not np.all(np.isfinite(raw))
                    or not np.all(np.isfinite(norms))
                    or np.any(norms <= 0)
                ):
                    raise ValueError(
                        "Embedding model returned invalid or zero vectors"
                    )

                raw = raw / norms

                for key, vector in zip(missing, raw):
                    self._embedding_cache[key] = vector

            self._embeddings = np.stack([
                self._embedding_cache[key]
                for key in keys
            ]).astype(np.float32)

            self._degraded_mode = False

        except Exception:
            # No cloud inference fallback and no automatic retry loop.
            self._embeddings = None
            self._degraded_mode = True
            self._semantic_unavailable = True

    def _ensure_index(self) -> None:
        if not self._tfidf_attempted:
            self._build_tfidf()

        if (
            self.enable_semantic
            and self._embeddings is None
            and not self._semantic_unavailable
        ):
            self._build_embeddings()

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_ANSWER_CHUNKS,
        top_candidates: int = DEFAULT_TOP_K,
        allowed_paper_ids: Optional[set[str]] = None,
    ) -> list[tuple[int, float]]:
        """Rank eligible passages only.

        A failed semantic branch cannot prevent lexical retrieval.
        """
        if (
            self.is_empty()
            or not query.strip()
            or top_k <= 0
            or top_candidates <= 0
        ):
            return []

        eligible = (
            None
            if allowed_paper_ids is None
            else np.array([
                i
                for i, chunk in enumerate(self._chunks)
                if chunk.paper_id in allowed_paper_ids
            ], dtype=int)
        )

        if eligible is not None and len(eligible) == 0:
            return []

        self._ensure_index()

        tfidf_ranks = self._tfidf_search(
            query,
            top_candidates,
            eligible,
        )

        if (
            self.enable_semantic
            and self._embeddings is not None
            and not self._degraded_mode
        ):
            dense_ranks = self._semantic_search(
                query,
                top_candidates,
                eligible,
            )

            # A failed or empty dense branch is not an RRF vote.
            if not self._degraded_mode and dense_ranks:
                return _rrf_combine(
                    tfidf_ranks,
                    dense_ranks,
                    top_k,
                )

        return _ranks_to_scored(
            tfidf_ranks,
            top_k,
        )

    def _tfidf_search(
        self,
        query: str,
        top_k: int,
        eligible: Optional[np.ndarray] = None,
    ) -> list[tuple[int, float]]:
        if (
            self._tfidf is None
            or self._tfidf_matrix is None
            or top_k <= 0
        ):
            return []

        try:
            query_vector = self._tfidf.transform(
                [query]
            )

            scores = cosine_similarity(
                query_vector,
                self._tfidf_matrix,
            ).ravel()

            candidates = (
                eligible
                if eligible is not None
                else np.arange(len(scores))
            )

            ranked = sorted(
                (
                    int(i)
                    for i in candidates
                    if scores[i] > 0
                ),
                key=lambda i: (-float(scores[i]), i),
            )[:top_k]

            return [
                (i, float(scores[i]))
                for i in ranked
            ]

        except ValueError:
            return []

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        eligible: Optional[np.ndarray] = None,
    ) -> list[tuple[int, float]]:
        """Rank dense evidence or disable a failed semantic branch."""
        if (
            self._embeddings is None
            or self._embed_model is None
            or top_k <= 0
        ):
            return []

        try:
            q_raw = np.asarray(
                self._embed_model.encode(
                    [query],
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            ).reshape(-1)

            if (
                q_raw.size != self._embeddings.shape[1]
                or not np.all(np.isfinite(q_raw))
            ):
                raise ValueError(
                    "Semantic query vector has an "
                    "invalid shape or value"
                )

            norm = float(np.linalg.norm(q_raw))

            if not np.isfinite(norm) or norm <= 0:
                raise ValueError(
                    "Semantic query vector has zero "
                    "or non-finite norm"
                )

            scores = self._embeddings @ (
                q_raw / norm
            )

            candidates = (
                eligible
                if eligible is not None
                else np.arange(len(scores))
            )

            ranked = sorted(
                (
                    int(i)
                    for i in candidates
                    if (
                        np.isfinite(scores[i])
                        and scores[i] > 0
                    )
                ),
                key=lambda i: (-float(scores[i]), i),
            )[:top_k]

            return [
                (i, float(scores[i]))
                for i in ranked
            ]

        except Exception:
            # Runtime failures may occur after a successful model load.
            # Never crash a research task or repeatedly retry inference.
            self._degraded_mode = True
            self._semantic_unavailable = True
            self._embeddings = None
            return []

    def get_chunks_by_ids(
        self,
        chunk_ids: list[str],
    ) -> list[EvidenceChunk]:
        """Return evidence in requested order for stable citations."""
        chunk_map = {
            chunk.chunk_id: chunk
            for chunk in self._chunks
        }

        return [
            chunk_map[cid]
            for cid in dict.fromkeys(chunk_ids)
            if cid in chunk_map
        ]

    @property
    def degraded_mode(self) -> bool:
        return self._degraded_mode

    @property
    def corpus_hash(self) -> str:
        return self._corpus_hash


def _rrf_combine(
    tfidf_ranks: list[tuple[int, float]],
    semantic_ranks: list[tuple[int, float]],
    top_k: int,
) -> list[tuple[int, float]]:
    """Combine lexical and dense ranks with reciprocal-rank fusion."""
    if top_k <= 0:
        return []

    scores: dict[int, float] = {}

    for ranks in (tfidf_ranks, semantic_ranks):
        seen: set[int] = set()

        for rank, (idx, _) in enumerate(
            ranks,
            start=1,
        ):
            if idx in seen:
                continue

            seen.add(idx)

            scores[idx] = (
                scores.get(idx, 0.0)
                + 1.0 / (RRF_K + rank)
            )

    return sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )[:top_k]


def _ranks_to_scored(
    ranks: list[tuple[int, float]],
    top_k: int,
) -> list[tuple[int, float]]:
    return ranks[:top_k] if top_k > 0 else []


def select_evidence(
    index: CorpusIndex,
    query: str,
    max_total: int = DEFAULT_ANSWER_CHUNKS,
    max_per_paper: int = MAX_PER_PAPER,
    top_candidates: int = DEFAULT_TOP_K,
    allowed_paper_ids: Optional[set[str]] = None,
) -> list[EvidenceChunk]:
    """Select scoped passages with a per-paper cap."""
    if (
        index.is_empty()
        or max_total <= 0
        or max_per_paper <= 0
        or top_candidates <= 0
    ):
        return []

    candidates = index.search(
        query,
        top_k=max(top_candidates, max_total),
        top_candidates=max(top_candidates, max_total),
        allowed_paper_ids=allowed_paper_ids,
    )

    chosen: list[EvidenceChunk] = []
    per_paper: dict[str, int] = {}

    for idx, _score in candidates:
        if not 0 <= idx < len(index._chunks):
            continue

        chunk = index._chunks[idx]

        if (
            allowed_paper_ids is not None
            and chunk.paper_id not in allowed_paper_ids
        ):
            continue

        if per_paper.get(chunk.paper_id, 0) >= max_per_paper:
            continue

        chosen.append(chunk)

        per_paper[chunk.paper_id] = (
            per_paper.get(chunk.paper_id, 0) + 1
        )

        if len(chosen) == max_total:
            break

    return chosen


def _embedding_key(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8", errors="replace")
    ).hexdigest()


def _corpus_hash(chunks: list[EvidenceChunk]) -> str:
    digest = hashlib.sha256()

    for chunk in chunks:
        for field in (
            chunk.paper_id,
            str(chunk.page_index),
            chunk.content_hash,
        ):
            digest.update(
                field.encode(
                    "utf-8",
                    errors="replace",
                )
            )
            digest.update(b"\0")

    return digest.hexdigest()[:12]