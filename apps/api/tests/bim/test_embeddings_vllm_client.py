import json as _json

import httpx

from api.bim.clients.embeddings_vllm import VLLMEmbedClient


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
