import httpx
import pytest

from api.bim.clients.tei import TEIClient, TEIError


def _mock_transport(handler):
    return httpx.MockTransport(handler)


class TestTEIClient:
    def test_embed_returns_vectors(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["json"] = request.read().decode()
            return httpx.Response(
                200,
                json=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            )

        client = TEIClient(
            url="http://tei.local",
            model="Qwen/Qwen3-Embedding-8B",
            dim=3,
            transport=_mock_transport(handler),
        )
        vecs = client.embed(["hello", "world"])

        assert vecs == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert captured["method"] == "POST"
        assert captured["url"].endswith("/embed")
        import json as _json
        body = _json.loads(captured["json"])
        assert body["inputs"] == ["hello", "world"]

    def test_embed_truncates_to_configured_dim(self):
        """TEI returns native 4096-D; MRL truncation to 2048-D happens client-side.

        Truncated vectors must be L2-renormalized so cosine distance stays valid.
        """

        def handler(_req):
            return httpx.Response(200, json=[[3.0, 4.0, 0.0, 0.0]])

        client = TEIClient(
            url="http://tei.local",
            model="m",
            dim=2,
            transport=_mock_transport(handler),
        )
        [vec] = client.embed(["x"])
        assert len(vec) == 2
        assert abs(vec[0] - 0.6) < 1e-6
        assert abs(vec[1] - 0.8) < 1e-6

    def test_embed_dim_matches_when_native(self):
        """If TEI returns exactly ``dim`` floats, no truncation/renorm."""

        def handler(_req):
            return httpx.Response(200, json=[[0.6, 0.8]])

        client = TEIClient(
            url="http://tei.local",
            model="m",
            dim=2,
            transport=_mock_transport(handler),
        )
        [vec] = client.embed(["x"])
        assert vec == [0.6, 0.8]

    def test_embed_raises_when_dim_exceeds_native(self):
        """If configured dim > native dim, this is a misconfiguration."""

        def handler(_req):
            return httpx.Response(200, json=[[0.1, 0.2]])

        client = TEIClient(
            url="http://tei.local",
            model="m",
            dim=4,
            transport=_mock_transport(handler),
        )
        with pytest.raises(TEIError):
            client.embed(["x"])

    def test_retries_on_500_then_succeeds(self):
        attempts = {"n": 0}

        def handler(_req):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(500, json={"error": "transient"})
            return httpx.Response(200, json=[[0.1, 0.2]])

        client = TEIClient(
            url="http://tei.local",
            model="m",
            dim=2,
            transport=_mock_transport(handler),
            max_retries=3,
            retry_backoff_s=0.0,
        )
        [vec] = client.embed(["x"])
        assert vec == [0.1, 0.2]
        assert attempts["n"] == 3

    def test_raises_after_max_retries(self):
        def handler(_req):
            return httpx.Response(503, json={"error": "down"})

        client = TEIClient(
            url="http://tei.local",
            model="m",
            dim=2,
            transport=_mock_transport(handler),
            max_retries=2,
            retry_backoff_s=0.0,
        )
        with pytest.raises(TEIError):
            client.embed(["x"])

    def test_empty_input_returns_empty(self):
        def handler(_req):
            raise AssertionError("HTTP should not be called for empty input")

        client = TEIClient(
            url="http://tei.local",
            model="m",
            dim=2,
            transport=_mock_transport(handler),
        )
        assert client.embed([]) == []
