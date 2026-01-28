from typing import Any

from .bim_attribute import BIMAttribute


def _get_bilingual_value(data: dict[str, Any], en_key: str, ko_key: str) -> str:
    """Get value from dictionary using English key with Korean fallback."""
    return str(data.get(en_key, '') or data.get(ko_key, '')).strip()


def extract_bim_attribute_from_json(obj: dict[str, Any]) -> BIMAttribute | None:
    """
    Extract BIMAttribute from a JSON BIM object.

    Supports both English and Korean property keys for bilingual compatibility.

    Args:
        obj: BIM object dictionary (typically from JSON file)

    Returns:
        BIMAttribute instance, or None if no valid data (no KBIMS/PPS code)
    """
    other_properties = obj.get('Other', {}) or obj.get('기타', {})

    if not other_properties:
        return None

    attribute = BIMAttribute(
        ifc_type=str(obj.get('IFCType', '')).strip(),
        category=_get_bilingual_value(other_properties, 'Category', '카테고리'),
        family_name=_get_bilingual_value(other_properties, 'Family Name', '패밀리 이름'),
        kbims_code=str(other_properties.get('KBIMS-부위코드', '')).strip(),
        pps_code=str(other_properties.get('조달청표준공사코드', '')).strip(),
        family=_get_bilingual_value(other_properties, 'Family', '패밀리'),
        type=_get_bilingual_value(other_properties, 'Type', '유형'),
        type_id=_get_bilingual_value(other_properties, 'Type Id', '유형 ID'),
    )

    return attribute if attribute.is_valid() else None


def bim_attribute_from_csv_row(row: dict[str, str]) -> BIMAttribute:
    """
    Create BIMAttribute from a CSV row dictionary.

    Strips whitespace from keys and values to handle CSV files with
    padded column names.

    Args:
        row: Dictionary from csv.DictReader

    Returns:
        BIMAttribute instance
    """
    stripped = {k.strip(): (v.strip() if v else "") for k, v in row.items() if k}
    return BIMAttribute(
        ifc_type=stripped.get("ifc_type", ""),
        category=stripped.get("category", ""),
        family_name=stripped.get("family_name", ""),
        kbims_code=stripped.get("kbims_code", ""),
        pps_code=stripped.get("pps_code", ""),
        family=stripped.get("family", ""),
        type=stripped.get("type", ""),
        type_id=stripped.get("type_id", ""),
    )


def format_bim_object_for_prediction(obj: dict[str, Any]) -> str:
    """
    Convert a BIM JSON object to a string for prediction/vector search.

    Uses extract_bim_attribute_from_json internally and produces text
    in the same format as the stored embeddings (to_search_text).

    Args:
        obj: BIM object dictionary (loaded from JSON)

    Returns:
        Formatted string matching the embedding format for accurate search
    """
    attr = extract_bim_attribute_from_json(obj)
    if attr:
        return attr.to_search_text()

    # Fallback: build minimal search text from raw object
    parts = []
    if obj.get("ObjectType"):
        parts.append(f"IFC Type: {obj['ObjectType']}")
    return " | ".join(parts) if parts else str(obj)
