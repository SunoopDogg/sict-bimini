"""Minimal vLLM embedding client (OpenAI-compatible /v1/embeddings).

Performs MRL truncation + L2 renormalization client-side because vLLM
returns native-dim embeddings.
"""

from __future__ import annotations

import logging
import math
import time

import httpx

logger = logging.getLogger(__name__)


class VLLMEmbedError(RuntimeError):
    """Raised on non-retryable error or when retries are exhausted."""


class VLLMEmbedClient:
    def __init__(
        self,
        url: str,
        model: str,
        dim: int,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._dim = dim
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._client = httpx.Client(
            base_url=self._url, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> VLLMEmbedClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []

        body = {"model": self._model, "input": inputs}
        resp = self._post_with_retry("/v1/embeddings", body)

        raw = resp.json()
        try:
            items = raw["data"]
            items_sorted = sorted(items, key=lambda it: it["index"])
            vectors = [it["embedding"] for it in items_sorted]
        except (KeyError, TypeError) as exc:
            raise VLLMEmbedError(
                f"vLLM embeddings returned unexpected shape: {raw!r}"
            ) from exc

        return [self._prepare_vector(v) for v in vectors]

    def _prepare_vector(self, vec: list[float]) -> list[float]:
        native = len(vec)
        if self._dim > native:
            raise VLLMEmbedError(
                f"Configured dim={self._dim} exceeds native dim={native}. "
                "Pick a smaller dim or use a larger model."
            )
        if self._dim == native:
            return vec

        truncated = vec[: self._dim]
        norm = math.hypot(*truncated)
        if norm == 0.0:
            return truncated
        return [x / norm for x in truncated]

    def _post_with_retry(self, path: str, body: dict) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.post(path, json=body)
                if resp.status_code < 500:
                    if resp.status_code >= 400:
                        raise VLLMEmbedError(
                            f"vLLM embeddings error {resp.status_code}: "
                            f"{resp.text[:200]}"
                        )
                    return resp
                last_exc = VLLMEmbedError(f"vLLM embeddings 5xx: {resp.status_code}")
            except httpx.HTTPError as exc:
                last_exc = exc

            if attempt < self._max_retries:
                backoff = self._retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "vLLM embeddings attempt %d/%d failed (%s); retrying in %.2fs",
                    attempt,
                    self._max_retries,
                    last_exc,
                    backoff,
                )
                if backoff > 0:
                    time.sleep(backoff)

        raise VLLMEmbedError(
            f"vLLM embeddings exhausted {self._max_retries} retries. Last: {last_exc}"
        )
