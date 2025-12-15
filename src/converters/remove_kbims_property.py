"""
BIM 객체에서 KBIMS-부위코드 프로퍼티를 제거하는 스크립트

Usage:
    python remove_kbims_property.py [input_json] [output_json]
"""

import json
import argparse
from pathlib import Path


def remove_kbims_property(input_path: str, output_path: str) -> dict:
    """
    모든 객체에서 KBIMS-부위코드 프로퍼티를 제거

    Args:
        input_path: 입력 JSON 파일 경로
        output_path: 출력 JSON 파일 경로

    Returns:
        통계 정보 딕셔너리
    """
    print(f"Loading JSON file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total objects: {len(data)}")

    processed_objects = []
    removed_count = 0

    for obj in data:
        new_obj = obj.copy()
        other = new_obj.get("Other", {})

        if "KBIMS-부위코드" in other:
            del other["KBIMS-부위코드"]
            removed_count += 1

        processed_objects.append(new_obj)

    print(f"Properties removed: {removed_count}")
    print(f"Objects unchanged: {len(data) - removed_count}")

    print(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_objects, f, ensure_ascii=False, indent=2)

    stats = {
        "total": len(data),
        "properties_removed": removed_count,
        "objects_unchanged": len(data) - removed_count,
        "input_file": str(input_path),
        "output_file": str(output_path)
    }

    return stats


def extract_objects_without_kbims_property(input_path: str, output_path: str) -> dict:
    """
    KBIMS-부위코드가 없거나 비어있는 객체들을 추출

    Args:
        input_path: 입력 JSON 파일 경로
        output_path: 출력 JSON 파일 경로 (추출된 객체들 저장)

    Returns:
        통계 정보 딕셔너리
    """
    print(f"Loading JSON file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total objects: {len(data)}")

    extracted_objects = []

    for obj in data:
        other = obj.get("Other", {})
        kbims_value = other.get("KBIMS-부위코드")

        if kbims_value is None or kbims_value == "":
            extracted_objects.append(obj)

    extracted_count = len(extracted_objects)
    valid_count = len(data) - extracted_count

    print(f"Objects without valid KBIMS-부위코드: {extracted_count}")
    print(f"Objects with valid KBIMS-부위코드: {valid_count}")

    print(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_objects, f, ensure_ascii=False, indent=2)

    stats = {
        "total": len(data),
        "extracted": extracted_count,
        "with_valid_kbims": valid_count,
        "input_file": str(input_path),
        "output_file": str(output_path)
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Remove KBIMS-부위코드 property from BIM objects"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(Path(__file__).parent.parent.parent / "data" / "json" / "속성테이블(경희대).json"),
        help="Input JSON file path"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=str(Path(__file__).parent.parent.parent / "data" /
                    "json" / "속성테이블(경희대)_no_kbims.json"),
        help="Output JSON file path"
    )
    parser.add_argument(
        "--extract-empty",
        action="store_true",
        help="Extract objects without KBIMS-부위코드 instead of removing the property"
    )

    args = parser.parse_args()

    if args.extract_empty:
        stats = extract_objects_without_kbims_property(args.input, args.output)

        print("\n=== Extraction Complete ===")
        print(f"Total objects: {stats['total']}")
        print(f"Extracted (without valid KBIMS-부위코드): {stats['extracted']}")
        print(f"Skipped (with valid KBIMS-부위코드): {stats['with_valid_kbims']}")
        print(f"Output saved to: {stats['output_file']}")
    else:
        stats = remove_kbims_property(args.input, args.output)

        print("\n=== Processing Complete ===")
        print(f"Total objects: {stats['total']}")
        print(f"Properties removed: {stats['properties_removed']}")
        print(f"Objects unchanged: {stats['objects_unchanged']}")
        print(f"Output saved to: {stats['output_file']}")


if __name__ == "__main__":
    main()
