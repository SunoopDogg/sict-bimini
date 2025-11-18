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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BIMAttribute:
    """Data class representing extracted BIM object attributes."""
    category: str
    family_name: str
    kbims_code: str
    family: str
    type: str
    type_id: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return asdict(self)

    def is_valid(self) -> bool:
        """Check if at least one attribute has a value."""
        return any([
            self.category,
            self.family_name,
            self.kbims_code,
            self.family,
            self.type,
            self.type_id
        ])

    def __hash__(self):
        """Make hashable for deduplication."""
        return hash((
            self.category,
            self.family_name,
            self.kbims_code,
            self.family,
            self.type,
            self.type_id
        ))

    def __eq__(self, other):
        """Equality comparison for deduplication."""
        if not isinstance(other, BIMAttribute):
            return False
        return (
            self.category == other.category and
            self.family_name == other.family_name and
            self.kbims_code == other.kbims_code and
            self.family == other.family and
            self.type == other.type and
            self.type_id == other.type_id
        )


class BIMAttributeExtractor:
    """
    Extracts BIM object attributes from JSON files and exports to CSV.

    Focuses on extracting these 6 attributes from the "Other" field:
    - Category
    - Family Name
    - KBIMS-부위코드 (KBIMS code)
    - Family
    - Type
    - Type Id
    """

    # Mapping from JSON field names to CSV column names
    FIELD_MAPPING = {
        'Category': 'category',
        'Family Name': 'family_name',
        'KBIMS-부위코드': 'kbims_code',
        'Family': 'family',
        'Type': 'type',
        'Type Id': 'type_id'
    }

    def __init__(self, json_directory: str, output_directory: str):
        """
        Initialize the extractor.

        Args:
            json_directory: Directory containing JSON files
            output_directory: Directory for CSV output
        """
        self.json_directory = Path(json_directory)
        self.output_directory = Path(output_directory)

        # Ensure directories exist
        if not self.json_directory.exists():
            raise FileNotFoundError(f"JSON directory not found: {json_directory}")

        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Find JSON files
        self.json_files = list(self.json_directory.glob("*.json"))
        if not self.json_files:
            raise FileNotFoundError(f"No JSON files found in {json_directory}")

        logger.info(f"Found {len(self.json_files)} JSON files in {json_directory}")

    def extract_from_object(self, obj: Dict[str, Any]) -> Optional[BIMAttribute]:
        """
        Extract attributes from a single BIM object.

        Args:
            obj: BIM object dictionary

        Returns:
            BIMAttribute instance or None if no valid data
        """
        # Get the "Other" field
        other = obj.get('Other', {})

        if not other:
            return None

        # Extract each attribute
        attribute = BIMAttribute(
            category=str(other.get('Category', '')).strip(),
            family_name=str(other.get('Family Name', '')).strip(),
            kbims_code=str(other.get('KBIMS-부위코드', '')).strip(),
            family=str(other.get('Family', '')).strip(),
            type=str(other.get('Type', '')).strip(),
            type_id=str(other.get('Type Id', '')).strip()
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

    def extract_all(self, deduplicate: bool = True) -> List[BIMAttribute]:
        """
        Extract attributes from all JSON files.

        Args:
            deduplicate: Remove duplicate attribute combinations

        Returns:
            List of BIMAttribute instances
        """
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

        logger.info(f"Processed {total_objects} objects, extracted {len(all_attributes)} valid attributes")

        if deduplicate:
            unique_attributes = list(set(all_attributes))
            logger.info(f"After deduplication: {len(unique_attributes)} unique attribute combinations")
            return unique_attributes

        return all_attributes

    def to_csv(self,
               output_filename: str = "bim_attributes.csv",
               deduplicate: bool = True) -> Path:
        """
        Extract attributes and save to CSV file.

        Args:
            output_filename: Name of output CSV file
            deduplicate: Remove duplicate attribute combinations

        Returns:
            Path to created CSV file
        """
        # Extract all attributes
        attributes = self.extract_all(deduplicate=deduplicate)

        if not attributes:
            logger.warning("No attributes extracted, CSV file will be empty")

        # Create output path
        output_path = self.output_directory / output_filename

        # Write to CSV
        fieldnames = ['category', 'family_name', 'kbims_code', 'family', 'type', 'type_id']

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
        attributes = self.extract_all(deduplicate=False)
        unique_attributes = list(set(attributes))

        # Count unique values for each field
        categories = set(attr.category for attr in attributes if attr.category)
        family_names = set(attr.family_name for attr in attributes if attr.family_name)
        kbims_codes = set(attr.kbims_code for attr in attributes if attr.kbims_code)
        families = set(attr.family for attr in attributes if attr.family)
        types = set(attr.type for attr in attributes if attr.type)
        type_ids = set(attr.type_id for attr in attributes if attr.type_id)

        return {
            'total_objects': len(attributes),
            'unique_combinations': len(unique_attributes),
            'unique_categories': len(categories),
            'unique_family_names': len(family_names),
            'unique_kbims_codes': len(kbims_codes),
            'unique_families': len(families),
            'unique_types': len(types),
            'unique_type_ids': len(type_ids),
            'categories': sorted(categories),
            'kbims_codes': sorted(kbims_codes)
        }


def convert_json_to_csv(
    json_directory: str = "/root/sict-bimini/data/json",
    output_directory: str = "/root/sict-bimini/data/csv",
    output_filename: str = "bim_attributes.csv",
    deduplicate: bool = True,
    show_stats: bool = True
) -> Path:
    """
    Convenience function to convert BIM JSON files to CSV.

    Args:
        json_directory: Directory containing JSON files
        output_directory: Directory for CSV output
        output_filename: Name of output CSV file
        deduplicate: Remove duplicate attribute combinations
        show_stats: Print statistics after conversion

    Returns:
        Path to created CSV file
    """
    extractor = BIMAttributeExtractor(json_directory, output_directory)

    # Convert to CSV
    output_path = extractor.to_csv(output_filename, deduplicate)

    # Show statistics
    if show_stats:
        stats = extractor.get_statistics()
        print("\n" + "=" * 60)
        print("BIM Attribute Extraction Statistics")
        print("=" * 60)
        print(f"Total objects processed: {stats['total_objects']}")
        print(f"Unique attribute combinations: {stats['unique_combinations']}")
        print(f"\nUnique values per field:")
        print(f"  - Categories: {stats['unique_categories']}")
        print(f"  - Family Names: {stats['unique_family_names']}")
        print(f"  - KBIMS Codes: {stats['unique_kbims_codes']}")
        print(f"  - Families: {stats['unique_families']}")
        print(f"  - Types: {stats['unique_types']}")
        print(f"  - Type IDs: {stats['unique_type_ids']}")
        print(f"\nCategories found: {stats['categories']}")
        print(f"KBIMS Codes found: {stats['kbims_codes']}")
        print("=" * 60)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BIM JSON files to CSV with attribute extraction"
    )
    parser.add_argument(
        "--json-dir",
        default="/root/sict-bimini/data/json",
        help="Directory containing JSON files"
    )
    parser.add_argument(
        "--output-dir",
        default="/root/sict-bimini/data/csv",
        help="Directory for CSV output"
    )
    parser.add_argument(
        "--output-file",
        default="bim_attributes.csv",
        help="Name of output CSV file"
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Keep duplicate attribute combinations"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress statistics output"
    )

    args = parser.parse_args()

    try:
        output_path = convert_json_to_csv(
            json_directory=args.json_dir,
            output_directory=args.output_dir,
            output_filename=args.output_file,
            deduplicate=not args.no_deduplicate,
            show_stats=not args.quiet
        )
        print(f"\nCSV file created successfully: {output_path}")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
