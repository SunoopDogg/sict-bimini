from api.bim.normalizer import normalize_raw_objects
from api.bim.schemas import BIMAttribute, BIMObjectRaw


def _raw(ifc_type="IfcColumn", other=None, name="기둥-1", source="test.xlsx"):
    return BIMObjectRaw(
        source_file=source,
        object_name=name,
        ifc_type=ifc_type,
        properties={"Other": other or {}},
    )


class TestNormalizeRawObjects:
    def test_extracts_english_keys(self):
        raw = _raw(
            other={
                "Category": "건축",
                "Family Name": "RC기둥",
                "Family": "기둥",
                "Type": "T1",
                "Type Id": "X",
                "KBIMS-부위코드": "AR-C-001",
                "조달청표준공사코드": "",
            }
        )
        out = normalize_raw_objects([raw])
        assert len(out) == 1
        attr = out[0]
        assert isinstance(attr, BIMAttribute)
        assert attr.ifc_type == "IfcColumn"
        assert attr.category == "건축"
        assert attr.family_name == "RC기둥"
        assert attr.family == "기둥"
        assert attr.type == "T1"
        assert attr.type_id == "X"
        assert attr.kbims_code == "AR-C-001"
        assert attr.pps_code == ""

    def test_extracts_korean_keys_as_fallback(self):
        raw = _raw(
            other={
                "카테고리": "건축",
                "패밀리 이름": "RC기둥",
                "패밀리": "기둥",
                "유형": "T1",
                "유형 ID": "X",
                "KBIMS-부위코드": "AR-C-001",
            }
        )
        out = normalize_raw_objects([raw])
        assert len(out) == 1
        attr = out[0]
        assert attr.category == "건축"
        assert attr.family_name == "RC기둥"

    def test_english_takes_precedence_over_korean(self):
        raw = _raw(other={"Category": "EN", "카테고리": "KO", "KBIMS-부위코드": "X"})
        out = normalize_raw_objects([raw])
        assert out[0].category == "EN"

    def test_drops_when_both_codes_empty(self):
        raw = _raw(other={"Category": "건축"})
        assert normalize_raw_objects([raw]) == []

    def test_keeps_when_only_pps_present(self):
        raw = _raw(other={"Category": "건축", "조달청표준공사코드": "P1"})
        out = normalize_raw_objects([raw])
        assert len(out) == 1
        assert out[0].pps_code == "P1"
        assert out[0].kbims_code == ""

    def test_drops_when_other_missing(self):
        raw = BIMObjectRaw(
            source_file="x.xlsx",
            object_name="a",
            ifc_type="IfcColumn",
            properties={},
        )
        assert normalize_raw_objects([raw]) == []

    def test_accepts_korean_other_bucket(self):
        raw = BIMObjectRaw(
            source_file="x.xlsx",
            object_name="a",
            ifc_type="IfcColumn",
            properties={"기타": {"Category": "건축", "KBIMS-부위코드": "AR-C-001"}},
        )
        out = normalize_raw_objects([raw])
        assert len(out) == 1
        assert out[0].category == "건축"

    def test_dedups_identical_attributes(self):
        raw1 = _raw(other={"Category": "건축", "KBIMS-부위코드": "AR-C-001"})
        raw2 = _raw(other={"Category": "건축", "KBIMS-부위코드": "AR-C-001"})
        out = normalize_raw_objects([raw1, raw2])
        assert len(out) == 1

    def test_dedup_by_stable_id_ignores_label_diff(self):
        """두 BIMAttribute가 identity 동일하고 label만 다르면 dedup 되어 1개 남음.

        dedup 정책: stable_id 기준 (label 제외). 둘 중 마지막 것이 남는다.
        """
        raw1 = _raw(other={"Category": "건축", "KBIMS-부위코드": "AR-C-001"})
        raw2 = _raw(other={"Category": "건축", "KBIMS-부위코드": "AR-C-999"})
        out = normalize_raw_objects([raw1, raw2])
        assert len(out) == 1
        assert out[0].kbims_code == "AR-C-999"

    def test_whitespace_trimmed(self):
        raw = _raw(other={"Category": "  건축  ", "KBIMS-부위코드": "  AR-C-001  "})
        out = normalize_raw_objects([raw])
        assert out[0].category == "건축"
        assert out[0].kbims_code == "AR-C-001"

    def test_ifc_type_none_becomes_empty_string(self):
        raw = BIMObjectRaw(
            source_file="x.xlsx",
            object_name="a",
            ifc_type=None,
            properties={"Other": {"Category": "건축", "KBIMS-부위코드": "AR-C-001"}},
        )
        out = normalize_raw_objects([raw])
        assert out[0].ifc_type == ""
