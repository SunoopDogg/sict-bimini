import json

import httpx
import pytest

from api.bim.clients.vllm import (
    VLLMClient,
    VLLMError,
    VLLMSchemaError,
    VLLMTimeoutError,
)


def _mock_transport(handler):
    return httpx.MockTransport(handler)


_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


class TestVLLMClient:
    def test_generate_json_returns_content_string(self):
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            captured["method"] = req.method
            captured["body"] = json.loads(req.read().decode())
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"ok": true}'}}
                    ]
                },
            )

        client = VLLMClient(
            url="http://vllm.local",
            model="m-x",
            timeout=5.0,
            transport=_mock_transport(handler),
        )
        raw = client.generate_json(
            prompt="hello",
            response_schema=_SCHEMA,
        )

        assert raw == '{"ok": true}'
        assert captured["method"] == "POST"
        assert captured["url"].endswith("/v1/chat/completions")
        body = captured["body"]
        assert body["model"] == "m-x"
        assert body["messages"][0]["content"] == "hello"
        assert body["temperature"] == 0.2  # default
        assert body["max_tokens"] == 2048  # default
        # vLLM ≥0.10 / OpenAI standard: response_format={"type":"json_schema",...}.
        # extra_body.guided_json is silently ignored when sent over raw HTTP.
        rf = body["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] == _SCHEMA
        assert "extra_body" not in body

    def test_custom_temperature_and_max_tokens(self):
        def handler(req):
            body = json.loads(req.read().decode())
            assert body["temperature"] == 0.7
            assert body["max_tokens"] == 512
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "{}"}}]},
            )

        client = VLLMClient(
            url="http://vllm.local",
            model="m",
            transport=_mock_transport(handler),
        )
        client.generate_json(
            prompt="p",
            response_schema=_SCHEMA,
            temperature=0.7,
            max_tokens=512,
        )

    def test_retries_once_on_5xx_then_succeeds(self):
        attempts = {"n": 0}

        def handler(_req):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(503, text="busy")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "{}"}}]},
            )

        client = VLLMClient(
            url="http://vllm.local",
            model="m",
            transport=_mock_transport(handler),
            retry_backoff_s=0.0,
        )
        raw = client.generate_json(prompt="p", response_schema=_SCHEMA)
        assert raw == "{}"
        assert attempts["n"] == 2

    def test_raises_vllm_timeout_after_retries_exhausted(self):
        attempts = {"n": 0}

        def handler(_req):
            attempts["n"] += 1
            raise httpx.TimeoutException("slow")

        client = VLLMClient(
            url="http://vllm.local",
            model="m",
            transport=_mock_transport(handler),
            retry_backoff_s=0.0,
        )
        with pytest.raises(VLLMTimeoutError):
            client.generate_json(prompt="p", response_schema=_SCHEMA)
        assert attempts["n"] == 2  # 1 initial + 1 retry

    def test_400_schema_error_is_not_retried(self):
        attempts = {"n": 0}

        def handler(_req):
            attempts["n"] += 1
            return httpx.Response(
                400,
                json={"error": "guided_decoding failed"},
            )

        client = VLLMClient(
            url="http://vllm.local",
            model="m",
            transport=_mock_transport(handler),
            retry_backoff_s=0.0,
        )
        with pytest.raises(VLLMSchemaError):
            client.generate_json(prompt="p", response_schema=_SCHEMA)
        assert attempts["n"] == 1

    def test_unexpected_shape_raises_vllm_error(self):
        def handler(_req):
            return httpx.Response(200, json={"no_choices_here": True})

        client = VLLMClient(
            url="http://vllm.local",
            model="m",
            transport=_mock_transport(handler),
        )
        with pytest.raises(VLLMError):
            client.generate_json(prompt="p", response_schema=_SCHEMA)

    def test_context_manager_closes_client(self):
        def handler(_req):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "{}"}}]},
            )

        with VLLMClient(
            url="http://vllm.local",
            model="m",
            transport=_mock_transport(handler),
        ) as client:
            assert client.generate_json(prompt="p", response_schema=_SCHEMA) == "{}"

    def test_raises_vllm_error_after_5xx_exhausted(self):
        attempts = {"n": 0}

        def handler(_req):
            attempts["n"] += 1
            return httpx.Response(503, text="busy")

        client = VLLMClient(
            url="http://vllm.local",
            model="m",
            transport=_mock_transport(handler),
            retry_backoff_s=0.0,
        )
        with pytest.raises(VLLMError) as excinfo:
            client.generate_json(prompt="p", response_schema=_SCHEMA)
        # Must NOT be classified as timeout — no httpx.TimeoutException ever raised
        assert not isinstance(excinfo.value, VLLMTimeoutError)
        assert attempts["n"] == 2

    def test_timeout_then_5xx_classified_as_timeout(self):
        """If any attempt timed out and we exhaust, raise VLLMTimeoutError
        even if the final attempt was a 5xx — caller should treat the
        transient timeout as the dominant failure mode."""
        attempts = {"n": 0}

        def handler(_req):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.TimeoutException("first slow")
            return httpx.Response(503, text="then busy")

        client = VLLMClient(
            url="http://vllm.local",
            model="m",
            transport=_mock_transport(handler),
            retry_backoff_s=0.0,
        )
        with pytest.raises(VLLMTimeoutError):
            client.generate_json(prompt="p", response_schema=_SCHEMA)
        assert attempts["n"] == 2
