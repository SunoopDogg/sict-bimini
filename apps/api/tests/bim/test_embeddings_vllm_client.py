import json as _json

import httpx
import pytest

from api.bim.clients.embeddings_vllm import VLLMEmbedClient, VLLMEmbedError


def _mock_transport(handler):
    return httpx.MockTransport(handler)


class TestVLLMEmbedClient:
    def test_embed_returns_vectors(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["json"] = request.read().decode()
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "m",
                    "data": [
                        {
                            "index": 0,
                            "object": "embedding",
                            "embedding": [0.1, 0.2, 0.3],
                        },
                        {
                            "index": 1,
                            "object": "embedding",
                            "embedding": [0.4, 0.5, 0.6],
                        },
                    ],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        client = VLLMEmbedClient(
            url="http://embed.local",
            model="Qwen/Qwen3-Embedding-4B",
            dim=3,
            transport=_mock_transport(handler),
        )
        vecs = client.embed(["hello", "world"])

        assert vecs == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert captured["method"] == "POST"
        assert captured["url"].endswith("/v1/embeddings")
        body = _json.loads(captured["json"])
        assert body == {"model": "Qwen/Qwen3-Embedding-4B", "input": ["hello", "world"]}

    def test_embed_sorts_by_index_when_response_unordered(self):
        """vLLM index 역순 응답이어도 클라이언트가 오름차순 정렬 후 반환."""

        def handler(_req):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "m",
                    "data": [
                        {"index": 1, "object": "embedding", "embedding": [0.4, 0.5]},
                        {"index": 0, "object": "embedding", "embedding": [0.1, 0.2]},
                    ],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        client = VLLMEmbedClient(
            url="http://embed.local", model="m", dim=2,
            transport=_mock_transport(handler),
        )
        vecs = client.embed(["a", "b"])
        assert vecs == [[0.1, 0.2], [0.4, 0.5]]

    def test_embed_truncates_to_configured_dim(self):
        """vLLM이 native 4-D를 반환할 때 dim=2로 truncate + L2 renorm."""

        def handler(_req):
            return httpx.Response(
                200,
                json={
                    "object": "list", "model": "m",
                    "data": [{
                        "index": 0,
                        "object": "embedding",
                        "embedding": [3.0, 4.0, 0.0, 0.0],
                    }],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        client = VLLMEmbedClient(
            url="http://embed.local", model="m", dim=2,
            transport=_mock_transport(handler),
        )
        [vec] = client.embed(["x"])
        assert len(vec) == 2
        assert abs(vec[0] - 0.6) < 1e-6
        assert abs(vec[1] - 0.8) < 1e-6

    def test_embed_dim_matches_when_native(self):
        """dim==native이면 패스스루(정규화 없음)."""

        def handler(_req):
            return httpx.Response(
                200,
                json={
                    "object": "list", "model": "m",
                    "data": [{
                        "index": 0,
                        "object": "embedding",
                        "embedding": [0.6, 0.8],
                    }],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        client = VLLMEmbedClient(
            url="http://embed.local", model="m", dim=2,
            transport=_mock_transport(handler),
        )
        [vec] = client.embed(["x"])
        assert vec == [0.6, 0.8]

    def test_embed_raises_when_dim_exceeds_native(self):
        """dim > native는 misconfig → VLLMEmbedError."""

        def handler(_req):
            return httpx.Response(
                200,
                json={
                    "object": "list", "model": "m",
                    "data": [{
                        "index": 0,
                        "object": "embedding",
                        "embedding": [0.1, 0.2],
                    }],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        client = VLLMEmbedClient(
            url="http://embed.local", model="m", dim=4,
            transport=_mock_transport(handler),
        )
        with pytest.raises(VLLMEmbedError):
            client.embed(["x"])

    def test_retries_on_500_then_succeeds(self):
        attempts = {"n": 0}

        def handler(_req):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(500, json={"error": "transient"})
            return httpx.Response(
                200,
                json={
                    "object": "list", "model": "m",
                    "data": [{
                        "index": 0,
                        "object": "embedding",
                        "embedding": [0.1, 0.2],
                    }],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        client = VLLMEmbedClient(
            url="http://embed.local", model="m", dim=2,
            transport=_mock_transport(handler),
            max_retries=3, retry_backoff_s=0.0,
        )
        [vec] = client.embed(["x"])
        assert vec == [0.1, 0.2]
        assert attempts["n"] == 3

    def test_raises_after_max_retries(self):
        def handler(_req):
            return httpx.Response(503, json={"error": "down"})

        client = VLLMEmbedClient(
            url="http://embed.local", model="m", dim=2,
            transport=_mock_transport(handler),
            max_retries=2, retry_backoff_s=0.0,
        )
        with pytest.raises(VLLMEmbedError):
            client.embed(["x"])

    def test_retries_on_network_error_then_succeeds(self):
        """httpx.HTTPError(ConnectError 등)는 5xx와 동일하게 재시도."""
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise httpx.ConnectError("boom")
            return httpx.Response(
                200,
                json={
                    "object": "list", "model": "m",
                    "data": [{
                        "index": 0,
                        "object": "embedding",
                        "embedding": [0.1, 0.2],
                    }],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        client = VLLMEmbedClient(
            url="http://embed.local", model="m", dim=2,
            transport=_mock_transport(handler),
            max_retries=3, retry_backoff_s=0.0,
        )
        [vec] = client.embed(["x"])
        assert vec == [0.1, 0.2]
        assert attempts["n"] == 2

    def test_empty_input_returns_empty(self):
        def handler(_req):
            raise AssertionError("HTTP should not be called for empty input")

        client = VLLMEmbedClient(
            url="http://embed.local", model="m", dim=2,
            transport=_mock_transport(handler),
        )
        assert client.embed([]) == []

    def test_raises_on_unexpected_response_shape(self):
        """응답에 'data' 키가 없거나 구조가 깨졌을 때 VLLMEmbedError."""

        def handler(_req):
            return httpx.Response(200, json={"object": "list", "wrong_key": []})

        client = VLLMEmbedClient(
            url="http://embed.local", model="m", dim=2,
            transport=_mock_transport(handler),
        )
        with pytest.raises(VLLMEmbedError):
            client.embed(["x"])
