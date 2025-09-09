import os
import json
import pandas as pd


FILE_NAMES = [
    "속성테이블(경희대)",
    "속성테이블(법규검토)",
    "속성테이블(법규검토용)",
]

def bim_xlsx_to_json(file_name: str):
    """
    엑셀 파일을 JSON 파일로 변환하는 함수

    Args:
        file_name (str): 파일명

    Returns:
        list: JSON 데이터
    """

    # 데이터 디렉토리와 엑셀 파일 경로 설정
    data_dir = os.path.join(os.getcwd(), "data")
    xlsx_dir = os.path.join(data_dir, "xlsx")
    file_path = os.path.join(xlsx_dir, f"{file_name}.xlsx")

    # 엑셀 파일을 읽어 데이터프레임으로 변환
    df = pd.read_excel(file_path)

    result = []
    item = {}
    step = 3
    object_type = None
    global_id = None

    # 데이터프레임의 각 행을 순회하며 JSON 형식으로 변환
    for idx, row in df.iterrows():
        row_dict = row.to_dict()

        # step 3: 속성세트, 속성명, 속성값이 모두 NaN인 경우
        if step == 3 and pd.isna(row_dict["속성세트"]) and pd.isna(row_dict["속성명"]) and pd.isna(row_dict["속성값"]):
            result.append(item)
            item = {}
            step = 1

            # 객체유형을 설정
            if row_dict["객체명"].startswith("객체유형"):
                object_type = row_dict["객체명"].split(":")[1].strip()
                continue

        # step 1: 객체명을 설정
        if step == 1:
            step = 2
            global_id = row_dict["객체명"].split(":")[1].strip()
        # step 2: 객체 정보를 설정
        elif step == 2:
            step = 3
            item["ObjectType"] = object_type
            item["GlobalID"] = global_id
            item["Name"] = row_dict["객체명"]

        # step 3: 속성세트와 속성명을 설정
        if step == 3:
            if item.get(row_dict["속성세트"]) is None:
                if pd.isna(row_dict["속성세트"]):
                    item[row_dict["속성명"]] = row_dict["속성값"] if not pd.isna(
                        row_dict["속성값"]) else ""
                    continue
                else:
                    item[row_dict["속성세트"]] = {}

            item[row_dict["속성세트"]][row_dict["속성명"]
                                   ] = row_dict["속성값"] if not pd.isna(row_dict["속성값"]) else ""

    result.append(item)
    result = result[1:]

    # JSON 파일로 저장
    save_path = os.path.join(data_dir, "json", f"{file_name}.json")
    with open(save_path, "w") as f:
        f.write(json.dumps(result, indent=4, ensure_ascii=False))

    return result


if __name__ == '__main__':
    for file_name in FILE_NAMES:
        json_data = bim_xlsx_to_json(file_name)
