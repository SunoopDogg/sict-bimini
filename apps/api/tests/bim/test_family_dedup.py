from api.bim.family_dedup import dedup_raw_by_family
from api.bim.schemas import BIMObjectRaw


def _raw(family="벽-기본", name="obj", ifc_type="IfcWall", extra=None):
    other = {"Family Name": family} if family is not None else {}
    if extra:
        other.update(extra)
    return BIMObjectRaw(
        source_file="test.xlsx",
        object_name=name,
        ifc_type=ifc_type,
        properties={"Other": other},
    )


def test_collapses_same_family_first_wins():
    raws = [
        _raw(family="벽-기본", name="A"),
        _raw(family="벽-기본", name="B"),
        _raw(family="문-외부", name="C"),
    ]
    out = dedup_raw_by_family(raws)
    assert [r.object_name for r in out] == ["A", "C"]


def test_empty_family_name_kept_individually():
    raws = [
        _raw(family="", name="D"),
        _raw(family="", name="E"),
    ]
    out = dedup_raw_by_family(raws)
    assert [r.object_name for r in out] == ["D", "E"]


def test_keeps_codeless_rows():
    # 코드(kbims/pps) 전혀 없는 행도 유지 (validity 게이트 없음)
    raws = [_raw(family="문-외부", name="C")]
    out = dedup_raw_by_family(raws)
    assert len(out) == 1
    assert out[0].object_name == "C"


def test_preserves_first_appearance_order():
    raws = [
        _raw(family="z", name="A"),
        _raw(family="a", name="B"),
        _raw(family="z", name="A2"),
        _raw(family="m", name="C"),
    ]
    out = dedup_raw_by_family(raws)
    assert [r.object_name for r in out] == ["A", "B", "C"]


def test_extracts_korean_family_key():
    raw_ko = BIMObjectRaw(
        source_file="t.xlsx",
        object_name="ko",
        ifc_type="IfcWall",
        properties={"기타": {"패밀리 이름": "벽-기본"}},
    )
    raw_en = _raw(family="벽-기본", name="en")
    out = dedup_raw_by_family([raw_ko, raw_en])
    assert [r.object_name for r in out] == ["ko"]


def test_family_name_stripped():
    out = dedup_raw_by_family(
        [_raw(family=" 벽-기본 ", name="A"), _raw(family="벽-기본", name="B")]
    )
    assert [r.object_name for r in out] == ["A"]


def _raw_other(other: dict, name: str) -> BIMObjectRaw:
    return BIMObjectRaw(
        source_file="test.xlsx",
        object_name=name,
        ifc_type="IfcCurtainWall",
        properties={"Other": other},
    )


def test_falls_back_to_family_when_no_family_name():
    # 커튼월/문/창 등: Family Name 키 없음 → Family로 dedup
    raws = [
        _raw_other({"Family": "커튼월-CW110"}, "A"),
        _raw_other({"Family": "커튼월-CW110"}, "B"),
        _raw_other({"Family": "커튼월-CW200"}, "C"),
    ]
    out = dedup_raw_by_family(raws)
    assert [r.object_name for r in out] == ["A", "C"]


def test_family_name_preferred_over_family():
    # Family Name 있으면 그게 키 (Family는 무시)
    raws = [
        _raw_other({"Family Name": "벽-기본", "Family": "X"}, "A"),
        _raw_other({"Family Name": "벽-기본", "Family": "Y"}, "B"),
    ]
    out = dedup_raw_by_family(raws)
    assert [r.object_name for r in out] == ["A"]


def test_neither_family_name_nor_family_kept_individually():
    raws = [
        _raw_other({}, "A"),
        _raw_other({}, "B"),
    ]
    out = dedup_raw_by_family(raws)
    assert [r.object_name for r in out] == ["A", "B"]
