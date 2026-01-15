import io
import json
import logging
import os
import time
from typing import BinaryIO

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False

REQUIRED_COLUMNS = ["객체명", "속성세트", "속성명", "속성값"]

FILE_NAMES = [
    "속성테이블(10층)",
    # "속성테이블(경희대)",
    # "속성테이블(법규검토)",
    # "속성테이블(법규검토용)",
]


def convert_bim_xlsx_from_bytes(
    file_content: bytes | BinaryIO,
    filename: str = "uploaded_file.xlsx",
    log_interval: int = 1000,
) -> list[dict]:
    """
    Convert Excel file content (bytes or file-like object) to a list of BIM objects.

    Args:
        file_content: Bytes or file-like object containing Excel data
        filename: Original filename (used for logging)
        log_interval: Progress logging interval (default: 1000 rows)

    Returns:
        list[dict]: List of BIM objects as dictionaries

    Raises:
        ValueError: If required columns are missing or file cannot be parsed
    """
    # Convert bytes to file-like object if needed
    if isinstance(file_content, bytes):
        file_content = io.BytesIO(file_content)

    # Read Excel file
    logger.info(f"Loading Excel file: {filename}")
    start_time = time.time()
    try:
        df = pd.read_excel(file_content)
    except Exception as e:
        raise ValueError(f"Failed to parse Excel file: {e}")

    load_time = time.time() - start_time
    total_rows = len(df)
    logger.info(f"Loaded {total_rows} rows in {load_time:.2f}s")

    # Validate required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    result = []
    item = {}
    step = 3
    ifc_type = None
    global_id = None
    objects_completed = 0

    # Process each row
    logger.info("Starting row processing...")
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        if DEBUG:
            print(json.dumps(row_dict, ensure_ascii=False, indent=4))
            print(type(row_dict["속성세트"]), type(row_dict["속성명"]), type(row_dict["속성값"]))

        # step 3: When all of 속성세트, 속성명, 속성값 are NaN
        if step == 3 and pd.isna(row_dict["속성세트"]) and pd.isna(row_dict["속성명"]) and pd.isna(row_dict["속성값"]):
            result.append(item)
            objects_completed += 1
            item = {}
            step = 1

            obj_name = row_dict["객체명"]
            if obj_name.startswith("객체유형") or obj_name.startswith("객체 유형"):
                ifc_type = obj_name.split(":")[1].strip()
                continue

        # Periodic progress logging
        if (idx + 1) % log_interval == 0:
            progress = (idx + 1) / total_rows * 100
            logger.info(
                f"Progress: {idx + 1}/{total_rows} rows ({progress:.1f}%), {objects_completed} objects completed")

        # step 1: Set object name
        if step == 1:
            step = 2
            global_id = row_dict["객체명"].split(":")[1].strip()
        # step 2: Set object info
        elif step == 2:
            step = 3
            item["IFCType"] = "Ifc{}".format(ifc_type)
            item["GlobalID"] = global_id
            item["Name"] = row_dict["객체명"]

        # step 3: Set property set and property name
        if step == 3:
            if DEBUG:
                print(json.dumps(item, ensure_ascii=False, indent=4))

            if item.get(row_dict["속성세트"]) is None:
                if pd.isna(row_dict["속성세트"]):
                    item[row_dict["속성명"]] = row_dict["속성값"] if not pd.isna(row_dict["속성값"]) else ""
                    continue
                else:
                    item[row_dict["속성세트"]] = {}

            item[row_dict["속성세트"]][row_dict["속성명"]
                                   ] = row_dict["속성값"] if not pd.isna(row_dict["속성값"]) else ""

    result.append(item)
    result = result[1:]

    # Log completion summary
    total_time = time.time() - start_time
    logger.info(
        f"Conversion complete: {len(result)} objects from {total_rows} rows "
        f"in {total_time:.2f}s ({len(result)/total_time:.1f} obj/s)"
    )

    return result


def bim_xlsx_to_json(file_name: str, log_interval: int = 1000) -> list:
    """
    Convert Excel file to JSON and save to disk.

    Args:
        file_name: File name (without extension)
        log_interval: Progress logging interval (default: 1000 rows)

    Returns:
        list: Converted JSON data
    """
    # Set up file paths
    data_dir = os.path.join(os.getcwd(), "data")
    xlsx_dir = os.path.join(data_dir, "xlsx")
    file_path = os.path.join(xlsx_dir, f"{file_name}.xlsx")

    # Read file and convert using the bytes function
    with open(file_path, "rb") as f:
        file_content = f.read()

    result = convert_bim_xlsx_from_bytes(file_content, file_path, log_interval)

    # Save JSON file
    save_path = os.path.join(data_dir, "json", f"{file_name}.json")
    logger.info(f"Saving JSON file: {save_path}")
    save_start = time.time()
    with open(save_path, "w") as f:
        f.write(json.dumps(result, indent=4, ensure_ascii=False))
    save_time = time.time() - save_start
    logger.info(f"Saved {len(result)} objects in {save_time:.2f}s")

    return result


if __name__ == '__main__':
    for file_name in FILE_NAMES:
        json_data = bim_xlsx_to_json(file_name)
