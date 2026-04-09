
import pytest

from api.bim.schemas import BIMObjectRaw
from api.bim.xlsx_parser import MissingColumnsError, parse_xlsx_to_raw


class TestParseXlsxToRaw:
    def test_empty_sheet_returns_empty_list(self, make_xlsx):
        path = make_xlsx([])
        result = parse_xlsx_to_raw(path)
        assert result == []

    def test_single_object(self, make_xlsx):
        rows = [
            (None, None, None, None),
            ("객체유형:Column", None, None, None),
            ("GlobalID:abc123", None, None, None),
            ("기둥-1", "Other", "Category", "건축"),
            ("기둥-1", "Other", "Family Name", "RC기둥"),
        ]
        path = make_xlsx(rows)
        result = parse_xlsx_to_raw(path)
        assert len(result) == 1
        obj = result[0]
        assert isinstance(obj, BIMObjectRaw)
        assert obj.source_file == "fixture.xlsx"
        assert obj.ifc_type == "IfcColumn"
        assert obj.global_id == "abc123"
        assert obj.object_name == "기둥-1"
        assert obj.properties["Other"]["Category"] == "건축"
        assert obj.properties["Other"]["Family Name"] == "RC기둥"

    def test_two_objects(self, make_xlsx):
        rows = [
            (None, None, None, None),
            ("객체유형:Column", None, None, None),
            ("GlobalID:id1", None, None, None),
            ("기둥-1", "Other", "Category", "건축"),
            (None, None, None, None),
            ("객체유형:Wall", None, None, None),
            ("GlobalID:id2", None, None, None),
            ("벽-1", "Other", "Category", "건축"),
        ]
        result = parse_xlsx_to_raw(make_xlsx(rows))
        assert len(result) == 2
        assert result[0].ifc_type == "IfcColumn"
        assert result[0].global_id == "id1"
        assert result[1].ifc_type == "IfcWall"
        assert result[1].global_id == "id2"

    def test_nan_value_becomes_empty_string(self, make_xlsx):
        rows = [
            (None, None, None, None),
            ("객체유형:Column", None, None, None),
            ("GlobalID:id1", None, None, None),
            ("기둥-1", "Other", "Description", None),
        ]
        result = parse_xlsx_to_raw(make_xlsx(rows))
        assert result[0].properties["Other"]["Description"] == ""

    def test_missing_columns_raises(self, tmp_path):
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.append(["객체명", "속성값"])
        ws.append(["x", "y"])
        path = tmp_path / "bad.xlsx"
        wb.save(path)

        with pytest.raises(MissingColumnsError) as exc:
            parse_xlsx_to_raw(path)
        msg = str(exc.value)
        assert "속성세트" in msg
        assert "속성명" in msg

    def test_source_file_is_basename(self, make_xlsx):
        rows = [
            (None, None, None, None),
            ("객체유형:Column", None, None, None),
            ("GlobalID:id1", None, None, None),
            ("기둥-1", "Other", "Category", "건축"),
        ]
        path = make_xlsx(rows, filename="속성테이블(10층).xlsx")
        result = parse_xlsx_to_raw(path)
        assert result[0].source_file == "속성테이블(10층).xlsx"
