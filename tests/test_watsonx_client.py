"""
tests/test_watsonx_client.py
============================
Offline tests for researchlens.watsonx_client.
No live IBM calls. All SDK interactions are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from researchlens.config import Settings, reset_settings
from researchlens.watsonx_client import (
    MockWatsonxClient,
    UsageRecord,
    WatsonxBudgetExhausted,
    WatsonxClient,
    WatsonxError,
    WatsonxGenerationDisabled,
    _redact,
    get_client,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _settings_generation_disabled(**overrides) -> Settings:
    reset_settings()
    s = Settings(
        watsonx_api_key="fake-key",
        watsonx_project_id="fake-project",
        watsonx_url="https://us-south.ml.cloud.ibm.com",
        watsonx_model_id="ibm/granite-13b-chat-v2",
        allow_live_generation=False,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _settings_generation_enabled(**overrides) -> Settings:
    reset_settings()
    s = Settings(
        watsonx_api_key="fake-key",
        watsonx_project_id="fake-project",
        watsonx_url="https://us-south.ml.cloud.ibm.com",
        watsonx_model_id="ibm/granite-13b-chat-v2",
        allow_live_generation=True,
        max_llm_calls_per_action=3,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class TestWatsonxClientGenerationDisabled:
    def test_chat_raises_when_disabled(self):
        s = _settings_generation_disabled()
        client = WatsonxClient(s)
        with pytest.raises(WatsonxGenerationDisabled):
            client.chat([{"role": "user", "content": "Hello"}])

    def test_smoke_test_returns_disabled_status(self):
        s = _settings_generation_disabled()
        client = WatsonxClient(s)
        result = client.smoke_test()
        assert result["status"] == "disabled"

    def test_budget_starts_at_max(self):
        s = _settings_generation_disabled()
        client = WatsonxClient(s)
        assert client.budget_remaining() == 3

    def test_reset_budget_resets_counter(self):
        s = _settings_generation_enabled()
        client = WatsonxClient(s)
        client._action_calls = 2
        client.reset_budget()
        assert client.action_calls_used == 0


class TestWatsonxClientBudget:
    def test_budget_exhausted_after_max_calls(self):
        s = _settings_generation_enabled(max_llm_calls_per_action=2)
        client = WatsonxClient(s)

        mock_response = {
            "choices": [{"message": {"content": "response text"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch.object(client, "_do_chat", return_value=("response text", UsageRecord(model_id="ibm/test"))):
            client.chat([{"role": "user", "content": "Q1"}])
            client.chat([{"role": "user", "content": "Q2"}])
            with pytest.raises(WatsonxBudgetExhausted):
                client.chat([{"role": "user", "content": "Q3"}])

    def test_reset_budget_allows_new_action(self):
        s = _settings_generation_enabled(max_llm_calls_per_action=1)
        client = WatsonxClient(s)

        with patch.object(client, "_do_chat", return_value=("ok", UsageRecord(model_id="ibm/test"))):
            client.chat([{"role": "user", "content": "Q1"}])
            with pytest.raises(WatsonxBudgetExhausted):
                client.chat([{"role": "user", "content": "Q2"}])
            client.reset_budget()
            # After reset, should succeed
            client.chat([{"role": "user", "content": "Q3"}])


class TestWatsonxClientAuthErrors:
    def test_401_not_retried(self):
        s = _settings_generation_enabled()
        client = WatsonxClient(s)

        def raise_401(_):
            raise Exception("401 Unauthorized")

        with patch.object(client, "_get_model") as mock_model:
            mock_model.return_value.chat.side_effect = Exception("401 Unauthorized")
            with pytest.raises(WatsonxError) as exc_info:
                client._do_chat([{"role": "user", "content": "test"}])
            assert "authentication" in str(exc_info.value).lower() or "401" in str(exc_info.value)

    def test_403_not_retried(self):
        s = _settings_generation_enabled()
        client = WatsonxClient(s)

        with patch.object(client, "_get_model") as mock_model:
            mock_model.return_value.chat.side_effect = Exception("403 Forbidden")
            with pytest.raises(WatsonxError):
                client._do_chat([{"role": "user", "content": "test"}])


class TestWatsonxClientMockedResponse:
    def test_chat_returns_text_and_usage(self):
        s = _settings_generation_enabled()
        client = WatsonxClient(s)

        mock_usage = UsageRecord(
            input_tokens=50, output_tokens=30,
            model_id="ibm/granite-13b-chat-v2", latency_ms=500.0
        )
        with patch.object(client, "_do_chat", return_value=("Hello from IBM", mock_usage)):
            text, usage = client.chat([{"role": "user", "content": "Hi"}])
            assert text == "Hello from IBM"
            assert usage.model_id == "ibm/granite-13b-chat-v2"
            assert client.action_calls_used == 1

    def test_system_prompt_prepended(self):
        s = _settings_generation_enabled()
        client = WatsonxClient(s)

        captured = []

        def capture(msgs):
            captured.extend(msgs)
            return ("ok", UsageRecord())

        with patch.object(client, "_do_chat", side_effect=capture):
            client.chat(
                [{"role": "user", "content": "Question"}],
                system_prompt="You are a research assistant."
            )

        assert captured[0]["role"] == "system"
        assert "research assistant" in captured[0]["content"]


class TestMockWatsonxClient:
    def test_chat_raises_disabled(self):
        client = MockWatsonxClient()
        with pytest.raises(WatsonxGenerationDisabled):
            client.chat([{"role": "user", "content": "test"}])

    def test_smoke_test_returns_disabled(self):
        client = MockWatsonxClient()
        result = client.smoke_test()
        assert result["status"] == "disabled"

    def test_sdk_not_available(self):
        client = MockWatsonxClient()
        assert client.sdk_available is False


class TestGetClient:
    def test_returns_mock_when_disabled(self):
        s = _settings_generation_disabled()
        client = get_client(s)
        assert isinstance(client, MockWatsonxClient)

    def test_returns_real_when_enabled(self):
        s = _settings_generation_enabled()
        client = get_client(s)
        assert isinstance(client, WatsonxClient)


class TestRedact:
    def test_redacts_long_tokens(self):
        text = "Error: api_key=abcdefghijklmnopqrstuvwxyz0123456789"
        result = _redact(text)
        assert "abcdefghijklmnopqrstuvwxyz0123456789" not in result
        assert "[REDACTED]" in result

    def test_short_tokens_unchanged(self):
        result = _redact("Error: code=404")
        assert "404" in result

    def test_empty_string(self):
        assert _redact("") == ""
