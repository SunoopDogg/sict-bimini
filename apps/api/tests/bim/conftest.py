from pathlib import Path

import pytest
from openpyxl import Workbook


@pytest.fixture
def make_xlsx(tmp_path: Path):
    """Programmatically create a BIM-style xlsx file for parser tests.

    ``rows`` is a list of 4-tuples (객체명, 속성세트, 속성명, 속성값).
    ``None`` values become empty cells (which the parser treats as NaN).
    """

    def _make(rows: list[tuple], filename: str = "fixture.xlsx") -> Path:
        wb = Workbook()
        ws = wb.active
        ws.append(["객체명", "속성세트", "속성명", "속성값"])
        for row in rows:
            ws.append(list(row))
        path = tmp_path / filename
        wb.save(path)
        return path

    return _make
