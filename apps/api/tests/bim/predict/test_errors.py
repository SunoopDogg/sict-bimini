import pytest

from api.bim.predict.errors import EmptyRetrievalError, PredictError


def test_predict_error_is_base_exception():
    assert issubclass(PredictError, Exception)


def test_empty_retrieval_error_is_predict_error():
    assert issubclass(EmptyRetrievalError, PredictError)


def test_raise_and_catch_via_base():
    with pytest.raises(PredictError):
        raise EmptyRetrievalError("no neighbors")


def test_llm_generation_error_is_predict_error():
    """LLMGenerationError must be catchable via the PredictError root."""
    from api.bim.predict.errors import LLMGenerationError, PredictError
    assert issubclass(LLMGenerationError, PredictError)
