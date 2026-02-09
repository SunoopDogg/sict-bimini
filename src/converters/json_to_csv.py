import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.utils import BIM_ATTRIBUTE_FIELDS, BIMAttribute, extract_bim_attribute_from_json

# Constants
JSON_DIRECTORY = "./data/json"
OUTPUT_DIRECTORY = "./data/csv"
OUTPUT_FILENAME = "bim_attributes.csv"

logger = logging.getLogger(__name__)


class BIMAttributeExporter:
    """
    Extracts BIM object attributes from JSON files and exports to CSV.

    Focuses on extracting these 8 attributes from the "Other" field:
    - IFCType
    - Category
    - Family Name
    - KBIMS-부위코드 (KBIMS code)
    - 조달청표준공사코드 (PPS code)
    - Family
    - Type
    - Type Id
    """

    def __init__(self):
        """Initialize the extractor with default directories."""
        self.json_directory = Path(JSON_DIRECTORY)
        self.output_directory = Path(OUTPUT_DIRECTORY)
        self._cached_attributes: list[BIMAttribute] | None = None

        # Ensure directories exist
        if not self.json_directory.exists():
            raise FileNotFoundError(f"JSON directory not found: {JSON_DIRECTORY}")

        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Find JSON files
        self.json_files = list(self.json_directory.glob("*.json"))
        if not self.json_files:
            raise FileNotFoundError(f"No JSON files found in {JSON_DIRECTORY}")

        logger.info(f"Found {len(self.json_files)} JSON files in {JSON_DIRECTORY}")

    def load_json_file(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Load and parse a JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of BIM objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both list and single object
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                logger.warning(f"Unexpected data type in {file_path.name}")
                return []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {file_path.name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            return []

    def extract_all(self) -> list[BIMAttribute]:
        """
        Extract attributes from all JSON files with deduplication.

        Returns:
            List of unique BIMAttribute instances (cached after first call)
        """
        if self._cached_attributes is not None:
            return self._cached_attributes

        all_attributes = []
        total_objects = 0

        for json_file in self.json_files:
            logger.info(f"Processing: {json_file.name}")

            objects = self.load_json_file(json_file)
            total_objects += len(objects)

            for obj in objects:
                attribute = extract_bim_attribute_from_json(obj)
                if attribute:
                    all_attributes.append(attribute)

        logger.info(
            f"Processed {total_objects} objects, extracted {len(all_attributes)} valid attributes")

        # Always deduplicate
        unique_attributes = list(set(all_attributes))
        logger.info(
            f"After deduplication: {len(unique_attributes)} unique attribute combinations")

        self._cached_attributes = unique_attributes
        return unique_attributes

    def to_csv(self) -> Path:
        """
        Extract attributes and save to CSV file.

        Returns:
            Path to created CSV file
        """
        # Extract all attributes
        attributes = self.extract_all()

        if not attributes:
            logger.warning("No attributes extracted, CSV file will be empty")

        # Create output path
        output_path = self.output_directory / OUTPUT_FILENAME

        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(BIM_ATTRIBUTE_FIELDS))
            writer.writeheader()

            for attr in attributes:
                writer.writerow(attr.to_dict())

        logger.info(f"CSV file created: {output_path}")
        logger.info(f"Total rows: {len(attributes)}")

        return output_path

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the extracted data.

        Returns:
            Dictionary with statistics
        """
        attributes = self.extract_all()

        # Single-pass collection of unique values per field
        field_sets: dict[str, set[str]] = {field: set() for field in BIM_ATTRIBUTE_FIELDS}
        for attr in attributes:
            for field in BIM_ATTRIBUTE_FIELDS:
                value = getattr(attr, field)
                if value:
                    field_sets[field].add(value)

        return {
            'unique_combinations': len(attributes),
            'unique_ifc_types': len(field_sets['ifc_type']),
            'unique_categories': len(field_sets['category']),
            'unique_family_names': len(field_sets['family_name']),
            'unique_kbims_codes': len(field_sets['kbims_code']),
            'unique_pps_codes': len(field_sets['pps_code']),
            'unique_families': len(field_sets['family']),
            'unique_types': len(field_sets['type']),
            'unique_type_ids': len(field_sets['type_id']),
            'ifc_types': sorted(field_sets['ifc_type']),
            'categories': sorted(field_sets['category']),
            'kbims_codes': sorted(field_sets['kbims_code']),
            'pps_codes': sorted(field_sets['pps_code']),
        }


def convert_json_to_csv(show_stats: bool = True) -> Path:
    """
    Convert BIM JSON files to CSV.

    Args:
        show_stats: Print statistics after conversion

    Returns:
        Path to created CSV file
    """
    extractor = BIMAttributeExporter()

    # Convert to CSV
    output_path = extractor.to_csv()

    # Show statistics
    if show_stats:
        stats = extractor.get_statistics()
        print("\n" + "=" * 60)
        print("BIM Attribute Extraction Statistics")
        print("=" * 60)
        print(f"Unique attribute combinations: {stats['unique_combinations']}")
        print(f"\nUnique values per field:")
        print(f"  - IFC Types: {stats['unique_ifc_types']}")
        print(f"  - Categories: {stats['unique_categories']}")
        print(f"  - Family Names: {stats['unique_family_names']}")
        print(f"  - KBIMS Codes: {stats['unique_kbims_codes']}")
        print(f"  - PPS Codes: {stats['unique_pps_codes']}")
        print(f"  - Families: {stats['unique_families']}")
        print(f"  - Types: {stats['unique_types']}")
        print(f"  - Type IDs: {stats['unique_type_ids']}")
        print(f"\nIFC Types found: {stats['ifc_types']}")
        print(f"Categories found: {stats['categories']}")
        print(f"KBIMS Codes found: {stats['kbims_codes']}")
        print(f"PPS Codes found: {stats['pps_codes']}")
        print("=" * 60)

    return output_path


if __name__ == "__main__":
    try:
        output_path = convert_json_to_csv()
        print(f"\nCSV file created successfully: {output_path}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        exit(1)
