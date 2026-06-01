"""Collapse raw BIM objects by family_name for the predict-input object list.

Independent of normalize: keeps records with no codes (they are exactly the
objects needing prediction) and preserves the raw ``BIMObjectRaw``
representative (object_name + all fields) rather than flattening to
``BIMAttribute``.
"""

from __future__ import annotations

from api.bim.schemas import BIMObjectRaw


def _family_name(raw: BIMObjectRaw) -> str:
    other = raw.properties.get("Other") or raw.properties.get("기타") or {}
    value = other.get("Family Name") or other.get("패밀리 이름") or ""
    return str(value).strip()


def dedup_raw_by_family(raws: list[BIMObjectRaw]) -> list[BIMObjectRaw]:
    """Keep one representative per non-empty family_name (first wins).

    Rows with an empty family_name are kept individually (never merged).
    Order follows first appearance. No validity gate — code-less rows survive.
    """
    seen: set[str] = set()
    out: list[BIMObjectRaw] = []
    for raw in raws:
        family = _family_name(raw)
        if family == "":
            out.append(raw)
            continue
        if family in seen:
            continue
        seen.add(family)
        out.append(raw)
    return out
