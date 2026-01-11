import os
import json
import time
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False


FILE_NAMES = [
    "속성테이블(10층)",
    "속성테이블(경희대)",
    "속성테이블(법규검토)",
    "속성테이블(법규검토용)",
]


def bim_xlsx_to_json(file_name: str, log_interval: int = 1000) -> list:
    """
    엑셀 파일을 JSON 파일로 변환하는 함수

    Args:
        file_name (str): 파일명
        log_interval (int): 진행 상황 로깅 간격 (기본값: 1000행)

    Returns:
        list: JSON 데이터
    """

    # 데이터 디렉토리와 엑셀 파일 경로 설정
    data_dir = os.path.join(os.getcwd(), "data")
    xlsx_dir = os.path.join(data_dir, "xlsx")
    file_path = os.path.join(xlsx_dir, f"{file_name}.xlsx")

    # 엑셀 파일을 읽어 데이터프레임으로 변환
    logger.info(f"Loading Excel file: {file_path}")
    start_time = time.time()
    df = pd.read_excel(file_path)
    load_time = time.time() - start_time
    total_rows = len(df)
    logger.info(f"Loaded {total_rows} rows in {load_time:.2f}s")

    result = []
    item = {}
    step = 3
    ifc_type = None
    global_id = None
    objects_completed = 0

    # 데이터프레임의 각 행을 순회하며 JSON 형식으로 변환
    logger.info("Starting row processing...")
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        if DEBUG:
            print(json.dumps(row_dict, ensure_ascii=False, indent=4))
            print(type(row_dict["속성세트"]), type(row_dict["속성명"]), type(row_dict["속성값"]))

        # step 3: 속성세트, 속성명, 속성값이 모두 NaN인 경우
        if step == 3 and pd.isna(row_dict["속성세트"]) and pd.isna(row_dict["속성명"]) and pd.isna(row_dict["속성값"]):
            result.append(item)
            objects_completed += 1
            item = {}
            step = 1

            obj_name = row_dict["객체명"]
            if obj_name.startswith("객체유형") or obj_name.startswith("객체 유형"):
                ifc_type = obj_name.split(":")[1].strip()
                continue

        # 주기적 진행 상황 로깅
        if (idx + 1) % log_interval == 0:
            progress = (idx + 1) / total_rows * 100
            logger.info(
                f"Progress: {idx + 1}/{total_rows} rows ({progress:.1f}%), {objects_completed} objects completed")

        # step 1: 객체명을 설정
        if step == 1:
            step = 2
            global_id = row_dict["객체명"].split(":")[1].strip()
        # step 2: 객체 정보를 설정
        elif step == 2:
            step = 3
            item["IFCType"] = "Ifc{}".format(ifc_type)
            item["GlobalID"] = global_id
            item["Name"] = row_dict["객체명"]

        # step 3: 속성세트와 속성명을 설정
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

    # JSON 파일로 저장
    save_path = os.path.join(data_dir, "json", f"{file_name}.json")
    logger.info(f"Saving JSON file: {save_path}")
    save_start = time.time()
    with open(save_path, "w") as f:
        f.write(json.dumps(result, indent=4, ensure_ascii=False))
    save_time = time.time() - save_start
    logger.info(f"Saved {len(result)} objects in {save_time:.2f}s")

    # 완료 요약
    total_time = time.time() - start_time
    logger.info(
        f"Conversion complete: {len(result)} objects from {total_rows} rows "
        f"in {total_time:.2f}s ({len(result)/total_time:.1f} obj/s)"
    )

    return result


if __name__ == '__main__':
    for file_name in FILE_NAMES:
        json_data = bim_xlsx_to_json(file_name)
