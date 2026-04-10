import pytest

from api.bim.predict.prompt import PromptBuilder
from api.bim.predict.schemas import CandidatePool, PredictionMode


def test_build_strong_substitutes_all_placeholders(sample_attribute):
    pool = CandidatePool(
        code_to_max_score={"KM001": 0.85, "KM002": 0.72},
        top1_score=0.85,
        unique_count=2,
    )
    text = PromptBuilder().build(
        target="kbims_code",
        mode=PredictionMode.STRONG,
        attribute=sample_attribute,
        pool=pool,
        n=5,
    )
    for ph in ("{attribute_block}", "{candidates_block}", "{n}"):
        assert ph not in text
    assert "IfcColumn" in text
    assert "KM001" in text
    assert "KM002" in text
    assert "5개" in text


def test_build_weak_for_pps(sample_attribute):
    pool = CandidatePool(
        code_to_max_score={"A-1": 0.4},
        top1_score=0.4,
        unique_count=1,
    )
    text = PromptBuilder().build(
        target="pps_code",
        mode=PredictionMode.WEAK,
        attribute=sample_attribute,
        pool=pool,
        n=3,
    )
    assert "조달청" in text
    assert "A-1" in text
    assert "3개" in text
    assert "참고용" in text or "신뢰도가 낮" in text


def test_build_includes_similarity_in_candidates_block(sample_attribute):
    pool = CandidatePool(
        code_to_max_score={"KM001": 0.85},
        top1_score=0.85,
        unique_count=1,
    )
    text = PromptBuilder().build(
        target="kbims_code",
        mode=PredictionMode.WEAK,
        attribute=sample_attribute,
        pool=pool,
        n=1,
    )
    assert "0.85" in text


def test_build_with_empty_pool_still_works_in_weak(sample_attribute):
    pool = CandidatePool(code_to_max_score={}, top1_score=0.0, unique_count=0)
    text = PromptBuilder().build(
        target="kbims_code",
        mode=PredictionMode.WEAK,
        attribute=sample_attribute,
        pool=pool,
        n=3,
    )
    assert "{candidates_block}" not in text


def test_missing_placeholder_in_template_raises(sample_attribute, monkeypatch):
    bad_template = "Missing placeholder.\n{attribute_block}\n{candidates_block}\n"
    builder = PromptBuilder()
    monkeypatch.setattr(builder, "_load_template", lambda *a, **kw: bad_template)

    pool = CandidatePool(
        code_to_max_score={"KM001": 0.8},
        top1_score=0.8,
        unique_count=1,
    )
    with pytest.raises(KeyError):
        builder.build(
            target="kbims_code",
            mode=PredictionMode.STRONG,
            attribute=sample_attribute,
            pool=pool,
            n=5,
        )
