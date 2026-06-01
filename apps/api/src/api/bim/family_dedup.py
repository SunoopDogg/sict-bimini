"""Collapse raw BIM objects by family identity for the predict-input list.

Independent of normalize: keeps records with no codes (they are exactly the
objects needing prediction) and preserves the raw ``BIMObjectRaw``
representative (object_name + all fields) rather than flattening to
``BIMAttribute``.
"""

from __future__ import annotations

from api.bim.schemas import BIMObjectRaw


def _dedup_key(raw: BIMObjectRaw) -> str:
    """Family identity for dedup: ``Family Name`` with fallback to ``Family``.

    ``Family Name`` exists only on structural elements (beams etc.); most object
    types (spaces, curtain walls, doors, windows) carry the discriminator in
    ``Family`` instead. Prefer the finer ``Family Name`` where present, else
    ``Family``. Empty string when neither is set.
    """
    other = raw.properties.get("Other") or raw.properties.get("기타") or {}
    name = str(other.get("Family Name") or other.get("패밀리 이름") or "").strip()
    if name:
        return name
    return str(other.get("Family") or other.get("패밀리") or "").strip()


def dedup_raw_by_family(raws: list[BIMObjectRaw]) -> list[BIMObjectRaw]:
    """Keep one representative per non-empty family key (first wins).

    Key = ``Family Name`` with fallback to ``Family`` (see ``_dedup_key``).
    Rows with an empty key are kept individually (never merged). Order follows
    first appearance. No validity gate — code-less rows survive.
    """
    seen: set[str] = set()
    out: list[BIMObjectRaw] = []
    for raw in raws:
        key = _dedup_key(raw)
        if key == "":
            out.append(raw)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out
