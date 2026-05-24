import pytest

from api.bim.predict.prompt import PromptBuilder
from api.bim.predict.schemas import CandidatePool, PredictionMode
from api.bim.schemas import BIMAttribute


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
        code_to_max_score={"AMB161A": 0.4},
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
    assert "AMB161A" in text
    assert "3개" in text
    assert "similarity" in text


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


def test_build_excludes_target_label_to_prevent_leak():
    """eval이 payload로 attribute를 재구성할 때 target 필드의 ground truth가
    프롬프트에 그대로 들어가면 LLM이 답을 보게 됨. target은 반드시 제외."""
    attr = BIMAttribute(
        ifc_type="IfcColumn",
        category="건축",
        family_name="RC기둥",
        family="기둥",
        type="T1",
        type_id="X",
        kbims_code="KM999",
        pps_code="A-99",
    )
    pool = CandidatePool(
        code_to_max_score={"KM001": 0.8},
        top1_score=0.8,
        unique_count=1,
    )

    kbims_prompt = PromptBuilder().build(
        target="kbims_code", mode=PredictionMode.STRONG,
        attribute=attr, pool=pool, n=3,
    )
    assert "KM999" not in kbims_prompt
    assert "A-99" in kbims_prompt  # cross-target hint stays

    pps_prompt = PromptBuilder().build(
        target="pps_code", mode=PredictionMode.STRONG,
        attribute=attr, pool=pool, n=3,
    )
    assert "A-99" not in pps_prompt
    assert "KM999" in pps_prompt


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
