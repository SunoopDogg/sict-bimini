"""xlsx → list[BIMObjectRaw] parser (state-machine, openpyxl-based).

Ported from legacy ``sict-bimini-python/src/converters/xlsx2json.py`` with:
- pandas/NaN removed → openpyxl + ``None`` tracking
- returns typed ``BIMObjectRaw`` instead of raw dicts
"""

from __future__ import annotations

import logging
from enum import IntEnum
from pathlib import Path

from openpyxl import load_workbook

from api.bim.schemas import BIMObjectRaw

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("객체명", "속성세트", "속성명", "속성값")


class MissingColumnsError(ValueError):
    """Required columns missing from the xlsx header."""


class _ParseState(IntEnum):
    OBJECT_NAME = 1
    OBJECT_INFO = 2
    PROPERTIES = 3


def _cell(value: object) -> str | None:
    """Normalize a cell value: ``None`` stays ``None``, others become str."""
    if value is None:
        return None
    return str(value)


def parse_xlsx_to_raw(path: Path) -> list[BIMObjectRaw]:
    """Parse a BIM xlsx file into a list of :class:`BIMObjectRaw`.

    The xlsx follows the legacy state-machine convention:
    blank row → "객체유형:<type>" row → "GlobalID:<id>" row → property rows.
    """
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    row_iter = ws.iter_rows(values_only=True)

    try:
        header = next(row_iter)
    except StopIteration:
        return []

    header_index = {name: i for i, name in enumerate(header) if name}
    missing = [c for c in REQUIRED_COLUMNS if c not in header_index]
    if missing:
        raise MissingColumnsError(
            f"Missing required columns: {missing}. Found: {list(header)}"
        )

    source_file = path.name
    bim_objects: list[BIMObjectRaw] = []
    current: dict[str, object] = {}
    state = _ParseState.PROPERTIES
    ifc_type: str | None = None
    global_id: str | None = None

    col_obj = header_index["객체명"]
    col_set = header_index["속성세트"]
    col_prop = header_index["속성명"]
    col_val = header_index["속성값"]

    for row in row_iter:
        obj_name = _cell(row[col_obj])
        prop_set = _cell(row[col_set])
        prop_name = _cell(row[col_prop])
        prop_val = _cell(row[col_val])

        # NaN separator: finish current object, start new
        is_separator = (
            state == _ParseState.PROPERTIES
            and prop_set is None
            and prop_name is None
            and prop_val is None
        )
        if is_separator:
            if current:
                bim_objects.append(_finalize(current, source_file, ifc_type, global_id))
            current = {}
            state = _ParseState.OBJECT_NAME
            ifc_type = None
            global_id = None
            # If the separator row itself does NOT carry 객체유형, skip to next row.
            # If it does carry 객체유형, fall through to the OBJECT_NAME handler below.
            if not (
                obj_name
                and (
                    obj_name.startswith("객체유형") or obj_name.startswith("객체 유형")
                )
            ):
                continue

        if state == _ParseState.OBJECT_NAME:
            if obj_name and (
                obj_name.startswith("객체유형") or obj_name.startswith("객체 유형")
            ):
                after = obj_name.split(":", 1)[1].strip() if ":" in obj_name else ""
                ifc_type = f"Ifc{after}" if after else None
                continue  # stay in OBJECT_NAME; next row is GlobalID
            # This row is the GlobalID row
            state = _ParseState.OBJECT_INFO
            if obj_name and ":" in obj_name:
                global_id = obj_name.split(":", 1)[1].strip()
            continue

        if state == _ParseState.OBJECT_INFO:
            state = _ParseState.PROPERTIES
            current["object_name"] = obj_name or ""
            current["properties"] = {}

        if state == _ParseState.PROPERTIES:
            props: dict[str, dict[str, str]] = current.setdefault("properties", {})
            if prop_set is None:
                continue
            bucket = props.setdefault(prop_set, {})
            if prop_name is not None:
                bucket[prop_name] = prop_val if prop_val is not None else ""

    if current:
        bim_objects.append(_finalize(current, source_file, ifc_type, global_id))

    logger.info("Parsed %d objects from %s", len(bim_objects), source_file)
    return bim_objects


def _finalize(
    current: dict[str, object],
    source_file: str,
    ifc_type: str | None,
    global_id: str | None,
) -> BIMObjectRaw:
    return BIMObjectRaw(
        source_file=source_file,
        object_name=str(current.get("object_name", "")),
        ifc_type=ifc_type,
        global_id=global_id,
        properties=current.get("properties", {}) or {},
    )
