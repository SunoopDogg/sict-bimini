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
