"""Safe configuration for ResearchLens AI.

Priority:
1. Explicit environment variables
2. Streamlit Secrets
3. Local .env
4. Safe defaults

Never print or commit credentials.
Restart Streamlit after changing configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------
# Local dotenv loading
# ----------------------------------------------------------

# Track values injected by dotenv so Streamlit Secrets
# can override them without overriding real environment
# variables.

_DOTENV_ADDED: dict[str, str] = {}

try:
    from dotenv import load_dotenv

    _before = set(os.environ)

    _dotenv_path = (
        Path(__file__).resolve().parent.parent / ".env"
    )

    if _dotenv_path.is_file():
        load_dotenv(
            dotenv_path=_dotenv_path,
            override=False,
        )

        _DOTENV_ADDED = {
            key: os.environ[key]
            for key in set(os.environ) - _before
        }

except ImportError:
    pass


# ----------------------------------------------------------
# Streamlit Secrets mapping
# ----------------------------------------------------------

_STREAMLIT_KEY_MAP: dict[str, tuple[str, str]] = {
    "WATSONX_API_KEY": ("watsonx", "api_key"),
    "WATSONX_PROJECT_ID": ("watsonx", "project_id"),
    "WATSONX_URL": ("watsonx", "url"),
    "WATSONX_MODEL_ID": ("watsonx", "model_id"),

    "ALLOW_LIVE_GENERATION": (
        "app", "allow_live_generation"
    ),
    "APP_ACCESS_PASSWORD": (
        "app", "access_password"
    ),
    "APP_ENV": ("app", "env"),
    "ENABLE_SEMANTIC_SEARCH": (
        "app", "enable_semantic_search"
    ),

    "MAX_LLM_CALLS_PER_ACTION": (
        "app", "max_llm_calls_per_action"
    ),
    "MAX_OUTPUT_TOKENS": (
        "app", "max_output_tokens"
    ),
    "MAX_UPLOAD_MB": (
        "app", "max_upload_mb"
    ),
    "MAX_PDFS_PER_SESSION": (
        "app", "max_pdfs_per_session"
    ),
    "MAX_PAGES_PER_PDF": (
        "app", "max_pages_per_pdf"
    ),

    "CROSSREF_MAILTO": ("crossref", "mailto"),
}


def _secret_to_string(value: object) -> str:
    """Convert secret values safely, including booleans."""

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def _streamlit_secret(key: str) -> str | None:
    """Read top-level or nested Streamlit Secrets."""

    try:
        import streamlit as st

        secrets = st.secrets

        # Support flat secrets:
        # WATSONX_API_KEY = "..."
        direct = secrets.get(key)

        if direct is not None:
            return _secret_to_string(direct)

        # Support nested secrets:
        # [watsonx]
        # api_key = "..."

        section_name, item_name = (
            _STREAMLIT_KEY_MAP.get(key, ("", ""))
        )

        if section_name:
            section = secrets.get(section_name)

            if section is not None:
                value = section.get(item_name)

                if value is not None:
                    return _secret_to_string(value)

    except Exception:
        # No Streamlit context or unavailable secrets.
        pass

    return None


def _env(key: str, default: str = "") -> str:
    """Resolve configuration using the correct priority."""

    current = os.environ.get(key)

    injected = (
        key in _DOTENV_ADDED
        and current == _DOTENV_ADDED[key]
    )

    # Real environment variables have highest priority.
    if current is not None and not injected:
        return current

    # Streamlit Secrets override dotenv-injected values.
    secret = _streamlit_secret(key)

    if secret is not None:
        return secret

    # Local dotenv fallback.
    if current is not None:
        return current

    return default


# ----------------------------------------------------------
# Safe conversion helpers
# ----------------------------------------------------------

def _flag(key: str, default: bool) -> bool:
    """Require an explicit 'true' to enable a feature."""

    raw = _env(
        key,
        "true" if default else "false",
    )

    return raw.strip().casefold() == "true"


def _bounded_int(
    key: str,
    default: int,
    lower: int,
    upper: int,
) -> int:
    """Use safe defaults for invalid resource limits."""

    try:
        value = int(
            _env(key, str(default)).strip()
        )

    except (ValueError, TypeError, OverflowError):
        return default

    if lower <= value <= upper:
        return value

    return default


def _call_limit() -> int:
    """Read model-call limit; validation handles range."""

    try:
        return int(
            _env(
                "MAX_LLM_CALLS_PER_ACTION",
                "3",
            ).strip()
        )

    except (ValueError, TypeError, OverflowError):
        return 3


# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------

@dataclass(repr=False)
class Settings:
    """Research Agent application configuration."""

    # IBM watsonx.ai

    watsonx_api_key: str = field(
        default_factory=lambda: _env("WATSONX_API_KEY"),
        repr=False,
    )

    watsonx_project_id: str = field(
        default_factory=lambda: _env("WATSONX_PROJECT_ID")
    )

    watsonx_url: str = field(
        default_factory=lambda: _env(
            "WATSONX_URL",
            "https://us-south.ml.cloud.ibm.com",
        )
    )

    watsonx_model_id: str = field(
        default_factory=lambda: _env("WATSONX_MODEL_ID")
    )

    # Live generation must be explicitly enabled.

    allow_live_generation: bool = field(
        default_factory=lambda: _flag(
            "ALLOW_LIVE_GENERATION",
            False,
        )
    )

    # Optional private access gate.

    app_access_password: str = field(
        default_factory=lambda: _env(
            "APP_ACCESS_PASSWORD"
        ),
        repr=False,
    )

    # Crossref contact.

    crossref_mailto: str = field(
        default_factory=lambda: _env(
            "CROSSREF_MAILTO"
        )
    )

    # Environment.

    app_env: str = field(
        default_factory=lambda: _env(
            "APP_ENV",
            "local",
        )
    )

    # Retrieval setting.

    enable_semantic_search: bool = field(
        default_factory=lambda: _flag(
            "ENABLE_SEMANTIC_SEARCH",
            True,
        )
    )

    # Operational limits.

    max_llm_calls_per_action: int = field(
        default_factory=_call_limit
    )

    max_output_tokens: int = field(
        default_factory=lambda: _bounded_int(
            "MAX_OUTPUT_TOKENS",
            900,
            1,
            2048,
        )
    )

    max_upload_mb: int = field(
        default_factory=lambda: _bounded_int(
            "MAX_UPLOAD_MB",
            10,
            1,
            10,
        )
    )

    max_pdfs_per_session: int = field(
        default_factory=lambda: _bounded_int(
            "MAX_PDFS_PER_SESSION",
            3,
            1,
            3,
        )
    )

    max_pages_per_pdf: int = field(
        default_factory=lambda: _bounded_int(
            "MAX_PAGES_PER_PDF",
            40,
            1,
            40,
        )
    )

    # ------------------------------------------------------
    # IBM status
    # ------------------------------------------------------

    def ibm_configured(self) -> bool:
        """Check credentials without exposing their values."""

        return all((
            self.watsonx_api_key.strip(),
            self.watsonx_project_id.strip(),
            self.watsonx_url.strip(),
            self.watsonx_model_id.strip(),
        ))

    def generation_enabled(self) -> bool:
        """Return whether IBM generation is configured and enabled."""

        return (
            self.ibm_configured()
            and self.allow_live_generation
        )

    def ibm_status_label(self) -> str:
        """Return safe, human-readable IBM configuration status."""

        if self.generation_enabled():
            return "IBM Granite: live generation enabled"

        if self.ibm_configured():
            return (
                "IBM Granite: credentials configured — "
                "live generation disabled "
                "(ALLOW_LIVE_GENERATION=false)"
            )

        missing = [
            key
            for key, present in (
                ("WATSONX_API_KEY", self.watsonx_api_key),
                ("WATSONX_PROJECT_ID", self.watsonx_project_id),
                ("WATSONX_URL", self.watsonx_url),
                ("WATSONX_MODEL_ID", self.watsonx_model_id),
            )
            if not present.strip()
        ]

        return (
            "IBM Granite: NOT CONFIGURED — missing: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate settings without displaying secret values."""

        warnings: list[str] = []

        if not self.ibm_configured():
            warnings.append(self.ibm_status_label())

        if (
            self.allow_live_generation
            and not self.ibm_configured()
        ):
            warnings.append(
                "ALLOW_LIVE_GENERATION=true but IBM "
                "credentials are incomplete — generation "
                "will remain disabled."
            )

        if not 1 <= self.max_llm_calls_per_action <= 10:
            warnings.append(
                f"MAX_LLM_CALLS_PER_ACTION="
                f"{self.max_llm_calls_per_action} "
                "is outside the expected range 1–10; "
                "defaulting to 3."
            )

            self.max_llm_calls_per_action = 3

        # Production access warning.
        if (
            self.generation_enabled()
            and self.app_env.strip().lower()
            in {"production", "prod"}
            and not self.app_access_password
        ):
            warnings.append(
                "Public live generation must remain disabled "
                "without a private access gate. "
                "A per-session limit is not an "
                "account-wide quota."
            )

        return warnings


# ----------------------------------------------------------
# Settings singleton
# ----------------------------------------------------------

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return cached settings."""

    global _settings

    if _settings is None:
        _settings = Settings()

    return _settings


def reset_settings() -> None:
    """Clear cached settings for tests or configuration reload."""

    global _settings
    _settings = None