# -*- coding: utf-8 -*-
"""
JSON to CSV Converter for BIM Object Attributes

Extracts specific attributes from BIM object JSON files and converts them to CSV format
for use with vector embedding systems.
"""

import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Constants
JSON_DIRECTORY = "/root/sict-bimini/data/json"
OUTPUT_DIRECTORY = "/root/sict-bimini/data/csv"
OUTPUT_FILENAME = "bim_attributes.csv"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BIMAttribute:
    """Data class representing extracted BIM object attributes."""
    ifc_type: str
    category: str
    family_name: str
    kbims_code: str
    pps_code: str
    family: str
    type: str
    type_id: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return asdict(self)

    def is_valid(self) -> bool:
        """Check if at least one code (KBIMS or PPS) has a value."""
        return bool(self.kbims_code or self.pps_code)


class BIMAttributeExtractor:
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
        self._cached_attributes: Optional[List[BIMAttribute]] = None

        # Ensure directories exist
        if not self.json_directory.exists():
            raise FileNotFoundError(f"JSON directory not found: {JSON_DIRECTORY}")

        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Find JSON files
        self.json_files = list(self.json_directory.glob("*.json"))
        if not self.json_files:
            raise FileNotFoundError(f"No JSON files found in {JSON_DIRECTORY}")

        logger.info(f"Found {len(self.json_files)} JSON files in {JSON_DIRECTORY}")

    def extract_from_object(self, obj: Dict[str, Any]) -> Optional[BIMAttribute]:
        """
        Extract attributes from a single BIM object.

        Args:
            obj: BIM object dictionary

        Returns:
            BIMAttribute instance or None if no valid data
        """
        # Get the "Other" field
        other = obj.get('Other', {}) or obj.get('기타', {})

        if not other:
            return None

        # Extract each attribute
        attribute = BIMAttribute(
            ifc_type=str(obj.get('IFCType', '')).strip(),
            category=str(other.get('Category', '') or other.get('카테고리', '')).strip(),
            family_name=str(other.get('Family Name', '') or other.get('패밀리 이름', '')).strip(),
            kbims_code=str(other.get('KBIMS-부위코드', '')).strip(),
            pps_code=str(other.get('조달청표준공사코드', '')).strip(),
            family=str(other.get('Family', '') or other.get('패밀리', '')).strip(),
            type=str(other.get('Type', '') or other.get('유형', '')).strip(),
            type_id=str(other.get('Type Id', '') or other.get('유형 ID', '')).strip()
        )

        return attribute if attribute.is_valid() else None

    def load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
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

    def extract_all(self) -> List[BIMAttribute]:
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
                attribute = self.extract_from_object(obj)
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
        fieldnames = ['ifc_type', 'category', 'family_name',
                      'kbims_code', 'pps_code', 'family', 'type', 'type_id']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for attr in attributes:
                writer.writerow(attr.to_dict())

        logger.info(f"CSV file created: {output_path}")
        logger.info(f"Total rows: {len(attributes)}")

        return output_path

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the extracted data.

        Returns:
            Dictionary with statistics
        """
        attributes = self.extract_all()

        # Count unique values for each field
        ifc_types = set(attr.ifc_type for attr in attributes if attr.ifc_type)
        categories = set(attr.category for attr in attributes if attr.category)
        family_names = set(attr.family_name for attr in attributes if attr.family_name)
        kbims_codes = set(attr.kbims_code for attr in attributes if attr.kbims_code)
        pps_codes = set(attr.pps_code for attr in attributes if attr.pps_code)
        families = set(attr.family for attr in attributes if attr.family)
        types = set(attr.type for attr in attributes if attr.type)
        type_ids = set(attr.type_id for attr in attributes if attr.type_id)

        return {
            'unique_combinations': len(attributes),
            'unique_ifc_types': len(ifc_types),
            'unique_categories': len(categories),
            'unique_family_names': len(family_names),
            'unique_kbims_codes': len(kbims_codes),
            'unique_pps_codes': len(pps_codes),
            'unique_families': len(families),
            'unique_types': len(types),
            'unique_type_ids': len(type_ids),
            'ifc_types': sorted(ifc_types),
            'categories': sorted(categories),
            'kbims_codes': sorted(kbims_codes),
            'pps_codes': sorted(pps_codes)
        }


def convert_json_to_csv(show_stats: bool = True) -> Path:
    """
    Convert BIM JSON files to CSV.

    Args:
        show_stats: Print statistics after conversion

    Returns:
        Path to created CSV file
    """
    extractor = BIMAttributeExtractor()

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
