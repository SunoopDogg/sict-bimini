"""Exception hierarchy for the predict module.

``VLLMError`` and friends live in ``api.bim.clients.vllm`` (infra layer).
Everything raised inside the orchestration layer inherits ``PredictError``.
"""


class PredictError(Exception):
    """Base class for all prediction-pipeline errors."""


class EmptyRetrievalError(PredictError):
    """Qdrant returned zero neighbors with the given code filter.

    Distinct from generic Qdrant failures: this is a meaningful
    domain signal (no prior object has this code in the collection).
    """


class LLMGenerationError(PredictError):
    """vLLM call failed or returned malformed output.

    Always chained from the original infra exception via __cause__ — the
    underlying failure is one of:
      - api.bim.clients.vllm.VLLMError (transport / 5xx exhausted)
      - api.bim.clients.vllm.VLLMSchemaError (4xx from vLLM)
      - api.bim.clients.vllm.VLLMTimeoutError (timeout after retries)
      - pydantic.ValidationError (guided_json echoed valid JSON but
        failed our stricter schema)
    """
