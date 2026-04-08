"""BIMObjectRaw → BIMAttribute normalization with stable_id dedup.

Mirrors the legacy ``extract_bim_attribute_from_json`` logic: bilingual
(English/Korean) key fallback, label validity gate, and dedup by
``stable_id`` (identity-only hash). "Last wins" within a batch so that a
newer label replaces an older one for the same identity.
"""

from __future__ import annotations

import logging

from api.bim.schemas import BIMAttribute, BIMObjectRaw

logger = logging.getLogger(__name__)


def _get_bilingual(data: dict[str, str], en: str, ko: str) -> str:
    value = data.get(en) or data.get(ko) or ""
    return str(value).strip()


def _extract_attribute(raw: BIMObjectRaw) -> BIMAttribute | None:
    other = raw.properties.get("Other") or raw.properties.get("기타") or {}
    if not other:
        return None

    attr = BIMAttribute(
        ifc_type=(raw.ifc_type or "").strip(),
        category=_get_bilingual(other, "Category", "카테고리"),
        family_name=_get_bilingual(other, "Family Name", "패밀리 이름"),
        family=_get_bilingual(other, "Family", "패밀리"),
        type=_get_bilingual(other, "Type", "유형"),
        type_id=_get_bilingual(other, "Type Id", "유형 ID"),
        kbims_code=str(other.get("KBIMS-부위코드") or "").strip(),
        pps_code=str(other.get("조달청표준공사코드") or "").strip(),
    )
    return attr if attr.is_valid() else None


def normalize_raw_objects(raws: list[BIMObjectRaw]) -> list[BIMAttribute]:
    """Normalize raw BIM objects into BIMAttributes with dedup.

    Invalid records (``is_valid()`` False) are dropped. Duplicates by
    ``stable_id`` collapse with "last wins" semantics so a later record's
    label replaces an earlier one with the same identity.
    """
    by_id: dict[str, BIMAttribute] = {}
    dropped = 0

    for raw in raws:
        attr = _extract_attribute(raw)
        if attr is None:
            dropped += 1
            continue
        by_id[attr.stable_id] = attr

    logger.info(
        "Normalized %d raw → %d unique (dropped %d invalid)",
        len(raws),
        len(by_id),
        dropped,
    )
    return list(by_id.values())
