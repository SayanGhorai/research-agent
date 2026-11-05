"""IBM watsonx.ai adapter for ResearchLens AI.

One bounded, explicit chat request per user action invocation. The IBM SDK's
internal retry mechanism is disabled, and failed requests count toward the
per-action limit. This is NOT an account-wide spending or usage limit.

Credentials are read from Settings and must never appear in logs or errors.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

from researchlens.config import Settings


class WatsonxError(Exception):
    """Controlled IBM service or response error."""


class WatsonxBudgetExhausted(WatsonxError):
    """The configured local, per-action call allowance was exhausted."""


class WatsonxGenerationDisabled(WatsonxError):
    """Live inference is disabled or required settings are missing."""


class UsageRecord:
    """Usage reported by the provider, when available; never invented."""

    def __init__(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        model_id: str = "",
        latency_ms: float = 0.0,
        is_mock: bool = False,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model_id = model_id
        self.latency_ms = latency_ms
        self.is_mock = is_mock

    def __repr__(self) -> str:
        return (
            f"UsageRecord(model={self.model_id!r}, "
            f"in={self.input_tokens}, out={self.output_tokens}, "
            f"mock={self.is_mock})"
        )


class WatsonxClient:
    """Thin, mockable SDK wrapper with explicit per-action request accounting."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._action_calls = 0
        self._total_session_calls = 0
        self._model: Any = None
        self._sdk_available = False
        self._check_sdk()

    def _check_sdk(self) -> None:
        try:
            import ibm_watsonx_ai  # type: ignore  # noqa: F401
        except ImportError:
            self._sdk_available = False
        else:
            self._sdk_available = True

    def _get_model(self) -> Any:
        """Initialize on first use; do not log raw SDK errors or credentials."""
        if self._model is not None:
            return self._model

        if not self._sdk_available:
            raise WatsonxError(
                "IBM watsonx.ai SDK is unavailable in this environment."
            )

        if not self._settings.ibm_configured():
            raise WatsonxGenerationDisabled(
                "IBM configuration is incomplete. Set WATSONX_API_KEY, "
                "WATSONX_PROJECT_ID, WATSONX_URL and WATSONX_MODEL_ID privately."
            )

        try:
            from ibm_watsonx_ai import Credentials  # type: ignore
            from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore

            credentials = Credentials(
                url=self._settings.watsonx_url,
                api_key=self._settings.watsonx_api_key,
            )

            self._model = ModelInference(
                model_id=self._settings.watsonx_model_id,
                credentials=credentials,
                project_id=self._settings.watsonx_project_id,
                max_retries=0,
                params={
                    "max_tokens": self._settings.max_output_tokens,
                    "temperature": 0.1,
                },
            )

        except Exception:
            raise WatsonxError(
                "Unable to initialize the IBM model. Verify the SDK, project, "
                "region, model availability and private credentials."
            ) from None

        return self._model

    def reset_budget(self) -> None:
        """Reset local action allowance, not any provider/account quota."""
        self._action_calls = 0

    def budget_remaining(self) -> int:
        return max(
            0,
            self._settings.max_llm_calls_per_action - self._action_calls,
        )

    def _check_budget(self) -> None:
        if self._action_calls >= self._settings.max_llm_calls_per_action:
            raise WatsonxBudgetExhausted(
                f"Per-action model-call limit "
                f"({self._settings.max_llm_calls_per_action}) reached."
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
    ) -> tuple[str, UsageRecord]:
        """Send one provider request, reserving its budget before execution.

        A failed provider request still counts against the local action budget.
        No hidden/manual retry is performed by this adapter.
        """
        if not self._settings.generation_enabled():
            raise WatsonxGenerationDisabled(
                "Live IBM generation is disabled or configuration is missing. "
                "Enable it only after an authorized access check."
            )

        self._check_budget()

        # Reject invalid requests locally, before reserving an IBM attempt.
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(item, dict)
                or item.get("role") not in {"user", "assistant", "system"}
                or not isinstance(item.get("content"), str)
                or not item["content"].strip()
                for item in messages
            )
            or not isinstance(system_prompt, str)
        ):
            raise WatsonxError(
                "Invalid or empty chat messages; no IBM request was made."
            )

        full_messages: list[dict[str, str]] = []

        if system_prompt:
            full_messages.append({
                "role": "system",
                "content": system_prompt,
            })

        full_messages.extend(messages)

        if not full_messages:
            raise WatsonxError(
                "An IBM chat request must contain a message."
            )

        # Reserve before calling. _do_chat is replaceable in offline tests.
        self._action_calls += 1
        self._total_session_calls += 1

        return self._do_chat(full_messages)

    def _do_chat(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, UsageRecord]:
        """One SDK request. Never forward provider exception bodies to the UI."""
        model = self._get_model()
        start = time.monotonic()

        try:
            response = model.chat(messages=messages)

        except Exception as exc:
            error_text = str(exc).lower()
            status_code = getattr(exc, "status_code", None)

            if (
                status_code == 429
                or "429" in error_text
                or "consumption_limit_reached" in error_text
            ):
                raise WatsonxError(
                    "IBM free-tier capacity or rate limit (HTTP 429). "
                    "No retry was performed."
                ) from None

            if status_code in {401, 403} or any(
                marker in error_text
                for marker in (
                    "401",
                    "403",
                    "unauthorized",
                    "forbidden",
                )
            ):
                raise WatsonxError(
                    "IBM authentication or permission failure (HTTP 401/403). "
                    "No retry was performed."
                ) from None

            if status_code in {500, 502, 503, 504} or any(
                marker in error_text
                for marker in (
                    "500",
                    "502",
                    "503",
                    "504",
                    "timeout",
                    "timed out",
                )
            ):
                raise WatsonxError(
                    "IBM service unavailable or request timed out. "
                    "No automatic retry was performed."
                ) from None

            raise WatsonxError(
                "IBM inference request failed. Check regional service status "
                "and application configuration; no retry was performed."
            ) from None

        elapsed_ms = (time.monotonic() - start) * 1000.0

        text = _extract_text(response)

        if not text.strip():
            raise WatsonxError(
                "IBM returned an empty or unsupported chat response. "
                "No automatic retry was performed."
            )

        usage = _extract_usage(
            response,
            self._settings.watsonx_model_id,
            elapsed_ms,
        )

        return text, usage

    def smoke_test(self) -> dict[str, Any]:
        """One explicitly authorized smoke request; never run automatically."""
        self.reset_budget()

        try:
            text, usage = self.chat(
                messages=[{
                    "role": "user",
                    "content": "Reply with the single word: ready",
                }],
                system_prompt="You are a helpful assistant.",
            )

        except WatsonxGenerationDisabled as exc:
            return {
                "status": "disabled",
                "reason": str(exc),
            }

        except WatsonxBudgetExhausted as exc:
            return {
                "status": "budget_exhausted",
                "reason": str(exc),
            }

        except WatsonxError as exc:
            return {
                "status": "error",
                "reason": str(exc),
            }

        return {
            "status": "ok",
            "model_id": usage.model_id,
            "response_preview": text[:100],
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": round(usage.latency_ms, 1),
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }

    @property
    def sdk_available(self) -> bool:
        return self._sdk_available

    @property
    def action_calls_used(self) -> int:
        return self._action_calls

    @property
    def total_session_calls(self) -> int:
        return self._total_session_calls


def _extract_text(response: Any) -> str:
    """Extract actual chat text; never stringify raw provider responses.

    Normal watsonx.ai chat responses contain choices[0].message.content.
    Some structured chat responses return content as typed text parts. Ignore
    tool calls and non-text blocks instead of converting them into answer text.
    """
    if not isinstance(response, dict):
        return ""

    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    part["text"]
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type", "text") == "text"
                    and isinstance(part.get("text"), str)
                ]
                if parts:
                    return "\n".join(parts)

    # Older text-generation responses sometimes have results[] instead.
    results = response.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        generated = results[0].get("generated_text")
        if isinstance(generated, str):
            return generated
    return ""


def _nonnegative_token_count(value: Any) -> Optional[int]:
    """Accept genuine nonnegative integer usage, including zero, never guess."""
    return value if type(value) is int and value >= 0 else None


def _extract_usage(
    response: Any,
    model_id: str,
    latency_ms: float,
) -> UsageRecord:
    """Read only reported counters, preserving genuine zeros and unknowns.

    Standard chat responses report token usage under 'usage'. Legacy text
    generation responses may report counts in their first 'results' entry.
    Inconsistent, negative or malformed counters are left unknown (None).
    """
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        usage = {}

    # Never use 'or' here: zero tokens is a meaningful reported value.
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = usage.get("input_token_count")
    output_tokens = usage.get("completion_tokens")
    if output_tokens is None:
        output_tokens = usage.get("generated_token_count")

    if isinstance(response, dict):
        results = response.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            legacy_usage = results[0]
            if prompt_tokens is None:
                prompt_tokens = legacy_usage.get("input_token_count")
            if output_tokens is None:
                output_tokens = legacy_usage.get("generated_token_count")

    return UsageRecord(
        input_tokens=_nonnegative_token_count(prompt_tokens),
        output_tokens=_nonnegative_token_count(output_tokens),
        model_id=model_id,
        latency_ms=latency_ms,
        is_mock=False,
    )


def _redact(text: str) -> str:
    """Legacy redaction utility; raw provider errors are not exposed by adapter."""
    return re.sub(
        r"[A-Za-z0-9_\-]{32,}",
        "[REDACTED]",
        text or "",
    )


class MockWatsonxClient:
    """Disabled-generation client with the same public methods."""

    def __init__(self) -> None:
        self._action_calls = 0
        self._total_session_calls = 0

    def reset_budget(self) -> None:
        self._action_calls = 0

    def budget_remaining(self) -> int:
        return 0

    def chat(
        self,
        messages: list[dict],
        system_prompt: str = "",
    ) -> tuple[str, UsageRecord]:
        raise WatsonxGenerationDisabled(
            "[MOCK] IBM generation disabled. "
            "Configure the service and enable it explicitly."
        )

    def smoke_test(self) -> dict[str, Any]:
        return {
            "status": "disabled",
            "reason": "[MOCK] Generation disabled",
        }

    @property
    def sdk_available(self) -> bool:
        return False

    @property
    def action_calls_used(self) -> int:
        return self._action_calls

    @property
    def total_session_calls(self) -> int:
        return self._total_session_calls


def get_client(
    settings: Settings,
) -> WatsonxClient | MockWatsonxClient:
    """Select real or disabled client based on the current configuration."""
    if settings.generation_enabled():
        return WatsonxClient(settings)

    return MockWatsonxClient()