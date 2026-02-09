from dataclasses import dataclass, asdict

# Shared path constants
CSV_PATH = "./data/csv/bim_attributes.csv"

# BIM attribute field names used across the codebase
BIM_ATTRIBUTE_FIELDS: tuple[str, ...] = (
    "ifc_type", "category", "family_name", "kbims_code",
    "pps_code", "family", "type", "type_id",
)

# Display labels for text formatting
_FIELD_LABELS: dict[str, str] = {
    "ifc_type": "IFC Type",
    "category": "Category",
    "family_name": "Family Name",
    "kbims_code": "KBIMS Code",
    "pps_code": "PPS Code",
    "family": "Family",
    "type": "Type",
    "type_id": "Type ID",
}


@dataclass(frozen=True)
class BIMAttribute:
    """
    Represents a single BIM object with its attributes.

    This is a frozen (immutable) dataclass to support:
    - Hashability for set-based deduplication
    - Consistent representation across the codebase
    """

    ifc_type: str
    category: str
    family_name: str
    kbims_code: str
    pps_code: str
    family: str
    type: str
    type_id: str

    def _format_parts(self, *, include_kbims: bool = True, separator: str = " | ") -> str:
        """Format attribute fields as labeled text."""
        parts = []
        for field in BIM_ATTRIBUTE_FIELDS:
            if not include_kbims and field == "kbims_code":
                continue
            parts.append(f"{_FIELD_LABELS[field]}: {getattr(self, field)}")
        return separator.join(parts)

    def to_text(self, separator: str = " | ") -> str:
        """Convert attributes to formatted text for embedding generation."""
        return self._format_parts(include_kbims=True, separator=separator)

    def to_search_text(self, separator: str = " | ") -> str:
        """Convert to search text excluding kbims_code (unknown at query time)."""
        return self._format_parts(include_kbims=False, separator=separator)

    def to_dict(self) -> dict[str, str]:
        """Convert attributes to dictionary."""
        return asdict(self)

    def is_valid(self) -> bool:
        """Check if at least one code (KBIMS or PPS) has a value."""
        return bool(self.kbims_code or self.pps_code)


def load_bim_attributes_from_csv(csv_path: str) -> list["BIMAttribute"]:
    """Load BIM attributes from a CSV file.

    Args:
        csv_path: Path to CSV file with BIM attribute columns.

    Returns:
        List of BIMAttribute instances.
    """
    import csv

    from .bim_converter import bim_attribute_from_csv_row

    attributes: list[BIMAttribute] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            attributes.append(bim_attribute_from_csv_row(row))
    return attributes
