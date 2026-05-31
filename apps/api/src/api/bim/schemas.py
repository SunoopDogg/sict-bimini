import logging
from hashlib import blake2b

from pydantic import BaseModel, ValidationError

_logger = logging.getLogger(__name__)


class BIMObjectRaw(BaseModel):
    """Raw BIM object parsed from an xlsx row block.

    Preserves the raw nested structure so that re-normalization is possible
    without re-parsing the xlsx (schema migrations, new field extraction).
    """

    source_file: str
    object_name: str
    ifc_type: str | None = None
    global_id: str | None = None
    properties: dict[str, dict[str, str]]


class BIMAttribute(BaseModel):
    """Flattened, Qdrant-indexable BIM attribute record.

    Identity fields determine ``stable_id`` (for idempotent upsert).
    Label fields are prediction targets and excluded from ``embed_text``
    so query-time and index-time embeddings use the same schema.
    """

    # Identity fields (non-label)
    ifc_type: str
    category: str
    family_name: str
    family: str
    type: str
    type_id: str
    # Label fields (prediction targets)
    kbims_code: str = ""
    pps_code: str = ""

    @property
    def stable_id(self) -> str:
        key = (
            f"{self.ifc_type}|{self.category}|{self.family_name}|"
            f"{self.family}|{self.type}|{self.type_id}"
        )
        return blake2b(key.encode("utf-8"), digest_size=16).hexdigest()

    def embed_text(self) -> str:
        return (
            f"IFC Type: {self.ifc_type} | Category: {self.category} | "
            f"Family Name: {self.family_name} | Family: {self.family} | "
            f"Type: {self.type} | Type ID: {self.type_id}"
        )

    def is_valid(self) -> bool:
        return bool(self.kbims_code or self.pps_code)


def bim_attr_from_payload(payload: dict) -> BIMAttribute | None:
    """Parse a BIMAttribute from a Qdrant point payload; None on invalid data."""
    try:
        return BIMAttribute.model_validate(payload)
    except ValidationError:
        _logger.warning(
            "Skipping point with invalid payload: stable_id=%s",
            payload.get("stable_id"),
        )
        return None
