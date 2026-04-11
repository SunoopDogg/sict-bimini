"""Minimal vLLM client (OpenAI-compatible /v1/chat/completions + guided_json).

Only ``generate_json`` is exposed — free-text completion would invite
parsing drift. Retry policy matches TEIClient style but caps at 1 retry
for vLLM (generation is expensive; transient 5xx / timeouts retry,
4xx does not).

Requires vLLM ≥ 0.5.x that accepts ``extra_body.guided_json`` with a
JSON-schema dict (outlines backend).
"""
from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)


class VLLMError(RuntimeError):
    """Non-retryable or exhausted-retry vLLM failure."""


class VLLMSchemaError(VLLMError):
    """4xx from vLLM — usually bad guided_json schema or bad request."""


class VLLMTimeoutError(VLLMError):
    """Network timeout after retries."""


class VLLMClient:
    def __init__(
        self,
        *,
        url: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 1,
        retry_backoff_s: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._client = httpx.Client(
            base_url=self._url, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> VLLMClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def generate_json(
        self,
        *,
        prompt: str,
        response_schema: dict,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"guided_json": response_schema},
        }
        resp = self._post_with_retry("/v1/chat/completions", body)
        return self._extract_content(resp)

    def _post_with_retry(self, path: str, body: dict) -> httpx.Response:
        last_exc: Exception | None = None
        saw_timeout = False
        attempts = self._max_retries + 1  # 1 initial + N retries

        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.post(path, json=body)
            except httpx.TimeoutException as exc:
                last_exc = exc
                saw_timeout = True
                self._maybe_sleep(attempt, attempts, exc)
                continue
            except httpx.HTTPError as exc:
                last_exc = exc
                self._maybe_sleep(attempt, attempts, exc)
                continue

            if resp.status_code < 400:
                return resp
            if 400 <= resp.status_code < 500:
                raise VLLMSchemaError(
                    f"vLLM {resp.status_code}: {resp.text[:200]}"
                )
            # 5xx — retryable
            last_exc = VLLMError(f"vLLM 5xx: {resp.status_code}")
            self._maybe_sleep(attempt, attempts, last_exc)

        if saw_timeout:
            raise VLLMTimeoutError(
                f"vLLM timeout after {attempts} attempts"
            ) from last_exc
        raise VLLMError(
            f"vLLM exhausted {attempts} attempts. Last: {last_exc}"
        ) from last_exc

    def _maybe_sleep(self, attempt: int, total: int, exc: Exception) -> None:
        if attempt >= total:
            return
        backoff = self._retry_backoff_s * (2 ** (attempt - 1))
        logger.warning(
            "vLLM attempt %d/%d failed (%s); retrying in %.2fs",
            attempt, total, exc, backoff,
        )
        if backoff > 0:
            time.sleep(backoff)

    @staticmethod
    def _extract_content(resp: httpx.Response) -> str:
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLLMError(
                f"vLLM returned unexpected response shape: {data!r}"
            ) from exc
