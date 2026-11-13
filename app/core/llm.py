"""Thin wrapper around any OpenAI-compatible endpoint (OpenAI, LiteLLM proxy, ...)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from app.core.config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM backend cannot be reached or returns unusable output."""


class ChatLLM(Protocol):
    """Interface the rest of the app depends on (easy to fake in tests)."""

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        model: str | None = None,
        timeout: float | None = None,
    ) -> str: ...

    def chat_json(
        self, messages: list[dict], *, temperature: float = 0.0, model: str | None = None, timeout: float | None = None
    ) -> dict: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model reply (tolerates code fences / chatter)."""
    text = text.strip()
    candidates = [text, re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMError(f"Model did not return valid JSON: {text[:200]!r}")


class LLMClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Any = None

    @property
    def client(self):
        if self._client is None:
            if not self._settings.api_key:
                raise LLMError("OPENAI_API_KEY is not set (see .env.example).")
            from openai import OpenAI  # imported lazily so tests do not need the SDK

            self._client = OpenAI(
                api_key=self._settings.api_key,
                base_url=self._settings.base_url,
                timeout=self._settings.llm_timeout,
                max_retries=0,
            )
        return self._client

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        model: str | None = None,
        timeout: float | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model or self._settings.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            return self._create(kwargs)
        except LLMError:
            raise
        except Exception as exc:  # network, auth, quota...
            raise LLMError(f"LLM call failed: {exc}") from exc

    def _create(self, kwargs: dict[str, Any]) -> str:
        from openai import BadRequestError

        for _ in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except BadRequestError as exc:
                # Some models reject temperature / response_format: drop the option and retry.
                msg = str(exc).lower()
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                elif "response_format" in msg and "response_format" in kwargs:
                    kwargs.pop("response_format")
                else:
                    raise
        raise LLMError("LLM rejected the request parameters.")

    def chat_json(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        model: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        return extract_json(self.chat(messages, temperature=temperature, json_mode=True, model=model, timeout=timeout))

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = self.client.embeddings.create(model=self._settings.embedding_model, input=texts)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Embedding call failed: {exc}") from exc
        return [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]
