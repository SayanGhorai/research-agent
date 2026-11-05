"""Scholarly metadata adapters for arXiv and Crossref.

These APIs provide bibliographic records and, when available, abstracts—not
verified access to complete research papers. No AI model calls are made here.
"""

from __future__ import annotations

import hashlib
import html
import math
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional
from urllib.parse import unquote, urlsplit

import requests

from researchlens.schemas import PaperRecord, SourceProvider, TextCoverage

_ARXIV_API = "https://export.arxiv.org/api/query"
_CROSSREF_API = "https://api.crossref.org/works"
_CROSSREF_USER_AGENT = (
    "ResearchLensAI/0.1 "
    "(academic research assistant; project documentation)"
)

_DEFAULT_TIMEOUT = 15
_MAX_RESULTS = 5
_ARXIV_MIN_INTERVAL = 3.0
_ARXIV_LAST_CALL = 0.0
_ARXIV_LOCK = threading.Lock()


def _throttle_arxiv() -> None:
    """Space arXiv requests from threads in this process at least 3s apart."""
    global _ARXIV_LAST_CALL

    with _ARXIV_LOCK:
        delay = _ARXIV_MIN_INTERVAL - (
            time.monotonic() - _ARXIV_LAST_CALL
        )

        if delay > 0:
            time.sleep(delay)

        _ARXIV_LAST_CALL = time.monotonic()


def _safe_year(value: Any) -> Optional[int]:
    try:
        year = int(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return year if 1000 <= year <= 2100 else None


def _normalize_authors(raw: Any) -> list[str]:
    """Preserve only author names actually supplied by the metadata source."""
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []

    if not isinstance(raw, list):
        return []

    authors: list[str] = []

    for item in raw:
        if isinstance(item, str):
            name = item.strip()

        elif isinstance(item, dict):
            parts = [
                str(item[key]).strip()
                for key in ("given", "family")
                if item.get(key) is not None
                and str(item[key]).strip()
            ]

            name = (
                " ".join(parts)
                or str(item.get("name") or "").strip()
            )

        else:
            continue

        if name:
            authors.append(name)

    return authors


def _file_hash_prefix(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def _normalize_doi(doi: str) -> str:
    """Remove exact DOI prefixes without deleting valid identifier characters."""
    value = doi.strip()

    value = re.sub(
        r"^doi:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"^(?:https?://)?(?:dx\.)?doi\.org/",
        "",
        value,
        flags=re.IGNORECASE,
    )

    return unquote(value).strip().casefold()


def _extract_arxiv_id(url_or_id: str) -> str:
    """Recognize modern and legacy arXiv identifiers."""

    value = str(url_or_id or "").strip()

    if not value:
        return ""

    value = re.sub(
        r"^arxiv:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    if "://" in value:
        parts = urlsplit(value)

        if parts.hostname not in {
            "arxiv.org",
            "www.arxiv.org",
            "export.arxiv.org",
        }:
            return ""

        path = parts.path

        if not (
            path.startswith("/abs/")
            or path.startswith("/pdf/")
        ):
            return ""

        value = path[5:]

    value = value.removesuffix(".pdf")

    # Modern identifier: 2301.00001v2
    if re.fullmatch(
        r"\d{4}\.\d{4,5}(?:v\d+)?",
        value,
    ):
        return value

    # Legacy identifier: hep-th/9901001v2
    if re.fullmatch(
        r"[A-Za-z][A-Za-z.\-]+/\d{7}(?:v\d+)?",
        value,
    ):
        return value

    return ""


def _normalize_arxiv_id(arxiv_id: str) -> str:
    """Remove version suffixes for deduplication."""

    bare = _extract_arxiv_id(arxiv_id)

    return re.sub(
        r"v\d+$",
        "",
        bare,
        flags=re.IGNORECASE,
    )


def _normalize_title(title: str) -> str:
    """Normalize paper titles for conservative duplicate detection."""

    title = str(title or "").casefold().strip()
    title = re.sub(r"\.pdf$", "", title)
    title = re.sub(r"[-\u2013\u2014_]+", " ", title)
    title = re.sub(r"[^\w\s]", "", title)

    return re.sub(r"\s+", " ", title).strip()


def _safe_max_results(value: int) -> int:
    try:
        return max(
            1,
            min(int(value), _MAX_RESULTS),
        )
    except (ValueError, TypeError, OverflowError):
        return _MAX_RESULTS


# ==================================================================
# arXiv adapter
# ==================================================================

def search_arxiv(
    query: str,
    max_results: int = 5,
    timeout: int = _DEFAULT_TIMEOUT,
) -> tuple[list[PaperRecord], str | None]:
    """Search arXiv's Atom feed.

    Returns (records, error_message).
    Never invents missing authors, publication years, or abstracts.
    """

    if not query or not query.strip():
        return [], "arXiv query is empty"

    try:
        import feedparser  # type: ignore
    except ImportError:
        return [], (
            "arXiv search unavailable: "
            "feedparser is not installed"
        )

    _throttle_arxiv()

    try:
        response = requests.get(
            _ARXIV_API,
            params={
                "search_query": f"all:{query.strip()}",
                "start": 0,
                "max_results": _safe_max_results(max_results),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=timeout,
        )

    except requests.exceptions.Timeout:
        return [], (
            f"arXiv request timed out after {timeout}s"
        )

    except requests.exceptions.RequestException:
        return [], "arXiv network request failed"

    if response.status_code != 200:
        return [], f"arXiv HTTP {response.status_code}"

    try:
        feed = feedparser.parse(response.text)
        entries = feed.get("entries", [])

        if getattr(feed, "bozo", False) and not entries:
            return [], (
                "arXiv returned an unreadable Atom feed"
            )

    except Exception:
        return [], "arXiv feed could not be parsed"

    records: list[PaperRecord] = []

    for entry in entries[:_safe_max_results(max_results)]:
        try:
            title = re.sub(
                r"\s+",
                " ",
                str(entry.get("title") or ""),
            ).strip()

            if not title:
                continue

            arxiv_id = _extract_arxiv_id(
                entry.get("id", "")
            )

            authors = _normalize_authors(
                entry.get("authors")
            )

            published = str(
                entry.get("published") or ""
            )

            year = (
                _safe_year(published[:4])
                if re.match(r"^\d{4}", published)
                else None
            )

            abstract = re.sub(
                r"\s+",
                " ",
                str(entry.get("summary") or ""),
            ).strip() or None

            landing_url = None

            for link in entry.get("links", []) or []:
                if (
                    isinstance(link, dict)
                    and link.get("rel") == "alternate"
                ):
                    candidate = str(
                        link.get("href") or ""
                    )

                    if candidate.startswith(
                        ("http://", "https://")
                    ):
                        landing_url = candidate
                        break

            if not landing_url and arxiv_id:
                landing_url = (
                    f"https://arxiv.org/abs/{arxiv_id}"
                )

            records.append(
                PaperRecord(
                    title=title,
                    authors=authors,
                    year=year,
                    arxiv_id=arxiv_id or None,
                    landing_url=landing_url,
                    abstract=abstract,
                    source_provider=SourceProvider.ARXIV,
                    text_coverage=(
                        TextCoverage.ABSTRACT_ONLY
                        if abstract
                        else TextCoverage.NONE
                    ),
                    license_note=(
                        "arXiv metadata/abstract; "
                        "full-text rights are paper-specific"
                    ),
                )
            )

        except (ValueError, TypeError, AttributeError):
            continue

    return records, None


# ==================================================================
# Crossref adapter
# ==================================================================

def _retry_after_seconds(
    value: str | None,
) -> int | None:
    """Interpret Retry-After seconds or HTTP date.

    Do not shorten a provider's requested delay.
    This function does not sleep or retry automatically.
    """

    if not value:
        return None

    try:
        return max(
            0,
            int(value.strip()),
        )

    except (TypeError, ValueError):
        pass

    try:
        date = parsedate_to_datetime(value)

        if date.tzinfo is None:
            date = date.replace(
                tzinfo=timezone.utc
            )

        seconds = (
            date - datetime.now(timezone.utc)
        ).total_seconds()

        return max(
            0,
            math.ceil(seconds),
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
        IndexError,
    ):
        return None


def search_crossref(
    query: str,
    max_results: int = 5,
    mailto: str = "",
    timeout: int = _DEFAULT_TIMEOUT,
) -> tuple[list[PaperRecord], str | None]:
    """Search Crossref metadata.

    Missing bibliographic values remain missing.
    HTTP 429 does not trigger an automatic retry.
    """

    if not query or not query.strip():
        return [], "Crossref query is empty"

    params: dict[str, Any] = {
        "query": query.strip(),
        "rows": _safe_max_results(max_results),
        "select": (
            "DOI,title,author,published,"
            "published-print,published-online,"
            "abstract,URL,type"
        ),
    }

    if mailto and mailto.strip():
        params["mailto"] = mailto.strip()

    try:
        response = requests.get(
            _CROSSREF_API,
            params=params,
            headers={
                "User-Agent": _CROSSREF_USER_AGENT
            },
            timeout=timeout,
        )

    except requests.exceptions.Timeout:
        return [], (
            f"Crossref request timed out after {timeout}s"
        )

    except requests.exceptions.RequestException:
        return [], "Crossref network request failed"

    if response.status_code == 429:
        delay = _retry_after_seconds(
            response.headers.get("Retry-After")
        )

        if delay is None:
            return [], (
                "Crossref rate-limited; "
                "check Retry-After and try later"
            )

        return [], (
            f"Crossref rate-limited; "
            f"retry after {delay}s"
        )

    if response.status_code != 200:
        return [], (
            f"Crossref HTTP {response.status_code}"
        )

    try:
        payload = response.json()

        items = payload.get(
            "message", {}
        ).get("items", [])

        if not isinstance(items, list):
            return [], (
                "Crossref response has invalid items format"
            )

    except (ValueError, TypeError, AttributeError):
        return [], (
            "Crossref response could not be parsed"
        )

    records: list[PaperRecord] = []

    for item in items[:_safe_max_results(max_results)]:
        if not isinstance(item, dict):
            continue

        try:
            titles = item.get("title") or []

            title = (
                titles[0]
                if isinstance(titles, list) and titles
                else titles
                if isinstance(titles, str)
                else ""
            )

            title = re.sub(
                r"\s+",
                " ",
                str(title),
            ).strip()

            if not title:
                continue

            doi = str(
                item.get("DOI") or ""
            ).strip() or None

            landing_url = item.get("URL") or (
                f"https://doi.org/{doi}"
                if doi
                else None
            )

            year = None

            for key in (
                "published",
                "published-print",
                "published-online",
            ):
                date_info = item.get(key) or {}

                if not isinstance(date_info, dict):
                    continue

                parts = (
                    date_info.get("date-parts") or []
                )

                if (
                    parts
                    and isinstance(
                        parts[0],
                        (tuple, list),
                    )
                    and parts[0]
                ):
                    year = _safe_year(
                        parts[0][0]
                    )

                    if year is not None:
                        break

            raw_abstract = item.get("abstract")
            abstract = None

            if isinstance(raw_abstract, str):
                clean = html.unescape(
                    re.sub(
                        r"<[^>]*>",
                        " ",
                        raw_abstract,
                    )
                )

                abstract = re.sub(
                    r"\s+",
                    " ",
                    clean,
                ).strip() or None

            records.append(
                PaperRecord(
                    title=title,
                    authors=_normalize_authors(
                        item.get("author")
                    ),
                    year=year,
                    doi=doi,
                    landing_url=landing_url,
                    abstract=abstract,
                    source_provider=SourceProvider.CROSSREF,
                    text_coverage=(
                        TextCoverage.ABSTRACT_ONLY
                        if abstract
                        else TextCoverage.NONE
                    ),
                    license_note=(
                        "Crossref metadata; abstract "
                        "may be publisher copyrighted"
                    ),
                )
            )

        except (
            ValueError,
            TypeError,
            AttributeError,
        ):
            continue

    return records, None


# ==================================================================
# Deduplication
# ==================================================================

def _same_bibliographic_item(
    a: PaperRecord,
    b: PaperRecord,
) -> bool:
    """Conservatively determine whether two records describe the same work."""

    a_doi = (
        _normalize_doi(a.doi)
        if a.doi else ""
    )

    b_doi = (
        _normalize_doi(b.doi)
        if b.doi else ""
    )

    if a_doi and b_doi and a_doi == b_doi:
        return True

    a_arxiv = (
        _normalize_arxiv_id(a.arxiv_id)
        if a.arxiv_id else ""
    )

    b_arxiv = (
        _normalize_arxiv_id(b.arxiv_id)
        if b.arxiv_id else ""
    )

    if a_arxiv and b_arxiv and a_arxiv == b_arxiv:
        return True

    if (
        a.file_hash
        and b.file_hash
        and a.file_hash == b.file_hash
    ):
        return True

    # Title-only matching is weaker than DOI/arXiv/hash identity.
    name = _normalize_title(a.title)

    if (
        not name
        or len(name) < 12
        or name != _normalize_title(b.title)
    ):
        return False

    if a_doi and b_doi and a_doi != b_doi:
        return False

    if a_arxiv and b_arxiv and a_arxiv != b_arxiv:
        return False

    if (
        a.file_hash
        and b.file_hash
        and a.file_hash != b.file_hash
    ):
        return False

    if (
        a.year is not None
        and b.year is not None
        and a.year != b.year
    ):
        return False

    if a.authors and b.authors:
        def lead_family(
            paper: PaperRecord,
        ) -> str:
            first = paper.authors[0].strip()

            # A blank metadata author must not crash deduplication.
            if not first:
                return ""

            family = (
                first.split(",", 1)[0]
                if "," in first
                else first.split()[-1]
            )

            return re.sub(
                r"[^\w]",
                "",
                family,
            ).casefold()

        if lead_family(a) != lead_family(b):
            return False

    return True


def deduplicate_records(
    existing: list[PaperRecord],
    candidates: list[PaperRecord],
) -> tuple[list[PaperRecord], list[str]]:
    """Append unique papers without silently merging conflicting studies."""

    merged = list(existing)
    skipped: list[str] = []

    for candidate in candidates:
        if any(
            _same_bibliographic_item(
                current,
                candidate,
            )
            for current in merged
        ):
            skipped.append(
                candidate.display_id()
            )
        else:
            merged.append(candidate)

    return merged, skipped