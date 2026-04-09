"""Minimal TEI (Text Embeddings Inference) HTTP client.

TEI's ``POST /embed`` returns a 2D float list. For Qwen3-Embedding-8B with
MRL truncation, callers configure ``dim`` smaller than native; the client
then truncates+L2-renormalizes so cosine distance remains valid.
"""

from __future__ import annotations

import logging
import math
import time

import httpx

logger = logging.getLogger(__name__)


class TEIError(RuntimeError):
    """Raised when TEI returns a non-retryable error or retries are exhausted."""


class TEIClient:
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

    def __enter__(self) -> TEIClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []

        body = {"inputs": inputs, "model": self._model}
        resp = self._post_with_retry("/embed", body)

        raw = resp.json()
        if not isinstance(raw, list) or not all(isinstance(v, list) for v in raw):
            raise TEIError(f"TEI returned unexpected shape: {type(raw).__name__}")

        return [self._prepare_vector(v) for v in raw]

    def _prepare_vector(self, vec: list[float]) -> list[float]:
        native = len(vec)
        if self._dim > native:
            raise TEIError(
                f"Configured dim={self._dim} exceeds native dim={native}. "
                "Pick a smaller dim or use a larger model."
            )
        if self._dim == native:
            return vec

        truncated = vec[: self._dim]
        norm = math.sqrt(sum(x * x for x in truncated))
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
                        raise TEIError(
                            f"TEI error {resp.status_code}: {resp.text[:200]}"
                        )
                    return resp
                last_exc = TEIError(f"TEI 5xx: {resp.status_code}")
            except httpx.HTTPError as exc:
                last_exc = exc

            if attempt < self._max_retries:
                backoff = self._retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "TEI attempt %d/%d failed (%s); retrying in %.2fs",
                    attempt,
                    self._max_retries,
                    last_exc,
                    backoff,
                )
                if backoff > 0:
                    time.sleep(backoff)

        raise TEIError(
            f"TEI exhausted {self._max_retries} retries. Last: {last_exc}"
        )
