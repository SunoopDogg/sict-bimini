from api.bim.predict.catalog import CatalogSource, NoOpCatalog
from api.bim.predict.schemas import PredictionCandidate


def _candidate() -> PredictionCandidate:
    return PredictionCandidate(
        code="KM001",
        llm_confidence=0.9,
        retrieval_score=0.8,
        source="neighbor",
    )


def test_noop_returns_candidate_unchanged():
    cat = NoOpCatalog()
    c = _candidate()
    out = cat.validate(c)
    assert out is c or out == c


def test_noop_satisfies_protocol():
    cat: CatalogSource = NoOpCatalog()
    out = cat.validate(_candidate())
    assert out.code == "KM001"
