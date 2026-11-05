"""
tests/test_config.py
====================
Offline tests for researchlens.config.

All tests run without live credentials, network calls, or Streamlit context.
"""

import os
import pytest

from researchlens.config import Settings, get_settings, reset_settings


class TestSettingsDefaults:
    """Settings with no environment variables set use safe defaults."""

    def setup_method(self):
        reset_settings()

    def test_ibm_not_configured_when_no_env(self, monkeypatch):
        """IBM is not configured when credentials are absent."""
        for key in ("WATSONX_API_KEY", "WATSONX_PROJECT_ID", "WATSONX_URL", "WATSONX_MODEL_ID"):
            monkeypatch.delenv(key, raising=False)
        s = Settings()
        assert s.ibm_configured() is False

    def test_generation_disabled_by_default(self, monkeypatch):
        """Live generation is disabled unless explicitly enabled."""
        monkeypatch.delenv("ALLOW_LIVE_GENERATION", raising=False)
        s = Settings()
        assert s.generation_enabled() is False

    def test_generation_disabled_even_with_credentials(self, monkeypatch):
        """Generation gate requires both credentials AND the flag."""
        monkeypatch.setenv("WATSONX_API_KEY", "fake-key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "fake-project")
        monkeypatch.setenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        monkeypatch.setenv("WATSONX_MODEL_ID", "ibm/granite-13b-chat-v2")
        monkeypatch.setenv("ALLOW_LIVE_GENERATION", "false")
        s = Settings()
        assert s.ibm_configured() is True
        assert s.generation_enabled() is False

    def test_generation_enabled_with_credentials_and_flag(self, monkeypatch):
        """Generation is enabled only when all four credentials + flag are set."""
        monkeypatch.setenv("WATSONX_API_KEY", "fake-key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "fake-project")
        monkeypatch.setenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        monkeypatch.setenv("WATSONX_MODEL_ID", "ibm/granite-13b-chat-v2")
        monkeypatch.setenv("ALLOW_LIVE_GENERATION", "true")
        s = Settings()
        assert s.generation_enabled() is True

    def test_ibm_status_label_no_credentials(self, monkeypatch):
        """Status label lists missing keys without exposing values."""
        for key in ("WATSONX_API_KEY", "WATSONX_PROJECT_ID", "WATSONX_URL", "WATSONX_MODEL_ID"):
            monkeypatch.delenv(key, raising=False)
        s = Settings()
        label = s.ibm_status_label()
        assert "NOT CONFIGURED" in label
        assert "WATSONX_API_KEY" in label
        # Must NOT expose any secret value
        assert "fake" not in label.lower()

    def test_ibm_status_label_configured_disabled(self, monkeypatch):
        """Status label shows configured + disabled state."""
        monkeypatch.setenv("WATSONX_API_KEY", "fake-key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "fake-project")
        monkeypatch.setenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        monkeypatch.setenv("WATSONX_MODEL_ID", "ibm/granite-13b-chat-v2")
        monkeypatch.setenv("ALLOW_LIVE_GENERATION", "false")
        s = Settings()
        label = s.ibm_status_label()
        assert "configured" in label.lower()
        assert "disabled" in label.lower()
        # Must NOT expose the actual key value
        assert "fake-key" not in label

    def test_default_limits(self, monkeypatch):
        """Operational limits fall back to safe handoff defaults."""
        for key in ("MAX_LLM_CALLS_PER_ACTION", "MAX_OUTPUT_TOKENS",
                    "MAX_UPLOAD_MB", "MAX_PDFS_PER_SESSION", "MAX_PAGES_PER_PDF"):
            monkeypatch.delenv(key, raising=False)
        s = Settings()
        assert s.max_llm_calls_per_action == 3
        assert s.max_output_tokens == 900
        assert s.max_upload_mb == 10
        assert s.max_pdfs_per_session == 3
        assert s.max_pages_per_pdf == 40

    def test_validate_returns_warning_when_not_configured(self, monkeypatch):
        """validate() returns at least one warning when IBM is not configured."""
        for key in ("WATSONX_API_KEY", "WATSONX_PROJECT_ID", "WATSONX_URL", "WATSONX_MODEL_ID"):
            monkeypatch.delenv(key, raising=False)
        s = Settings()
        warnings = s.validate()
        assert len(warnings) >= 1
        assert any("NOT CONFIGURED" in w or "missing" in w.lower() for w in warnings)

    def test_validate_no_warnings_when_fully_configured(self, monkeypatch):
        """validate() returns no warnings when all credentials are present."""
        monkeypatch.setenv("WATSONX_API_KEY", "fake-key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "fake-project")
        monkeypatch.setenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        monkeypatch.setenv("WATSONX_MODEL_ID", "ibm/granite-13b-chat-v2")
        monkeypatch.setenv("ALLOW_LIVE_GENERATION", "false")
        s = Settings()
        assert s.validate() == []

    def test_singleton_returns_same_object(self):
        """get_settings() returns the same instance on repeated calls."""
        reset_settings()
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_reset_settings_creates_new_instance(self, monkeypatch):
        """reset_settings() forces a fresh Settings on next call."""
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        reset_settings()
        a = get_settings()
        reset_settings()
        b = get_settings()
        assert a is not b

    def test_environment_precedence(self, monkeypatch):
        """Real environment variable takes precedence over defaults."""
        monkeypatch.setenv("APP_ENV", "staging")
        s = Settings()
        assert s.app_env == "staging"

    def test_invalid_llm_call_budget_clamped(self, monkeypatch):
        """Out-of-range MAX_LLM_CALLS_PER_ACTION is clamped to 3 with a warning."""
        monkeypatch.setenv("MAX_LLM_CALLS_PER_ACTION", "99")
        s = Settings()
        warnings = s.validate()
        assert s.max_llm_calls_per_action == 3
        assert any("MAX_LLM_CALLS_PER_ACTION" in w for w in warnings)
