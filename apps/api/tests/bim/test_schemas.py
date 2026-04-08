from api.bim.schemas import BIMAttribute, BIMObjectRaw


class TestBIMObjectRaw:
    def test_minimal_construction(self):
        raw = BIMObjectRaw(
            source_file="test.xlsx",
            object_name="기둥-1",
            properties={},
        )
        assert raw.source_file == "test.xlsx"
        assert raw.object_name == "기둥-1"
        assert raw.ifc_type is None
        assert raw.global_id is None
        assert raw.properties == {}

    def test_full_construction(self):
        raw = BIMObjectRaw(
            source_file="속성테이블(10층).xlsx",
            object_name="기둥-1",
            ifc_type="IfcColumn",
            global_id="abc123",
            properties={"Other": {"Category": "건축", "Family Name": "RC기둥"}},
        )
        assert raw.ifc_type == "IfcColumn"
        assert raw.global_id == "abc123"
        assert raw.properties["Other"]["Category"] == "건축"

    def test_roundtrip_json(self):
        raw = BIMObjectRaw(
            source_file="test.xlsx",
            object_name="기둥-1",
            ifc_type="IfcColumn",
            properties={"Other": {"k": "v"}},
        )
        dumped = raw.model_dump_json()
        restored = BIMObjectRaw.model_validate_json(dumped)
        assert restored == raw


class TestBIMAttribute:
    def _make(self, **overrides):
        defaults = dict(
            ifc_type="IfcColumn",
            category="건축",
            family_name="RC기둥-직사각형",
            family="기둥",
            type="RC-300x500",
            type_id="T001",
            kbims_code="AR-C-RC-001",
            pps_code="",
        )
        defaults.update(overrides)
        return BIMAttribute(**defaults)

    def test_default_labels_empty(self):
        attr = BIMAttribute(
            ifc_type="IfcColumn",
            category="건축",
            family_name="F1",
            family="기둥",
            type="T1",
            type_id="X",
        )
        assert attr.kbims_code == ""
        assert attr.pps_code == ""

    def test_stable_id_is_deterministic(self):
        a = self._make()
        b = self._make()
        assert a.stable_id == b.stable_id

    def test_stable_id_is_hex_32_chars(self):
        sid = self._make().stable_id
        assert len(sid) == 32
        assert all(c in "0123456789abcdef" for c in sid)

    def test_stable_id_ignores_labels(self):
        """Changing kbims_code or pps_code must NOT change stable_id."""
        base = self._make(kbims_code="AR-C-001", pps_code="")
        changed = self._make(kbims_code="AR-C-999", pps_code="P1")
        assert base.stable_id == changed.stable_id

    def test_stable_id_changes_with_identity(self):
        base = self._make(ifc_type="IfcColumn")
        other = self._make(ifc_type="IfcWall")
        assert base.stable_id != other.stable_id

    def test_embed_text_excludes_labels(self):
        attr = self._make(kbims_code="AR-C-001", pps_code="P1")
        text = attr.embed_text()
        assert "AR-C-001" not in text
        assert "P1" not in text

    def test_embed_text_includes_identity_fields(self):
        attr = self._make(
            ifc_type="IfcColumn",
            category="건축",
            family_name="RC기둥",
            family="기둥",
            type="T1",
            type_id="X",
        )
        text = attr.embed_text()
        assert "IfcColumn" in text
        assert "건축" in text
        assert "RC기둥" in text
        assert "기둥" in text
        assert "T1" in text
        assert "X" in text

    def test_is_valid_true_when_kbims_only(self):
        attr = self._make(kbims_code="AR-C-001", pps_code="")
        assert attr.is_valid() is True

    def test_is_valid_true_when_pps_only(self):
        attr = self._make(kbims_code="", pps_code="P1")
        assert attr.is_valid() is True

    def test_is_valid_false_when_both_empty(self):
        attr = self._make(kbims_code="", pps_code="")
        assert attr.is_valid() is False
