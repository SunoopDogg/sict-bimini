# filepath: /root/sict-mvp/src/main.py
import os
import pandas as pd
from utils.change_extension import json_to_dict
from utils.xlsx_list import load_filenames_from_txt

# 예측에 사용할 특성(features) 목록 정의 - 타입 정보 포함
# cSpell: disable
FEATURES = {
    'PredefinedType': ('', 'PredefinedType', 'string'),
    'Category': ('Other', 'Category', 'string'),
    'Family': ('Other', 'Family', 'string'),
    'StructuralUsage': ('Structural', 'Structural Usage', 'string'),
    'StructuralMaterial': ('Materials and Finishes', 'Structural Material', 'string'),
    'Width': ('Dimensions', '가로치수', 'number'),
    'Height': ('Dimensions', '세로치수', 'number'),
    'Length': ('Dimensions', 'Length', 'number'),
    'Volume': ('Dimensions', 'Volume', 'number'),
    'CrossSectionArea': ('Dimensions', '단면적', 'number'),
    'IsExternal': ('Pset_BeamCommon', 'IsExternal', 'boolean'),
    'LoadBearing': ('Pset_BeamCommon', 'LoadBearing', 'boolean')
}
# cSpell: enable

# 기본값 정의
DEFAULT_VALUES = {
    'string': '',
    'number': 0,
    'boolean': False
}


def extract_features_from_json(json_data):
    """
    JSON 데이터에서 부위코드와 예측에 사용할 특성(features)을 추출하는 함수

    Args:
        json_data (list): JSON 데이터 리스트

    Returns:
        tuple: (추출된 특성들의 리스트 (각 항목은 딕셔너리 형태), 부위코드가 없어 스킵된 항목 수)
    """
    dataset = []
    skipped_count = 0

    for item in json_data:
        # 부위코드 추출 (타겟 변수)
        # cSpell: disable-next-line
        part_code = item.get('Other', {}).get('KBIMS-부위코드', '')

        # 부위코드가 없는 경우 스킵
        if not part_code:
            skipped_count += 1
            continue

        # 특성(features) 추출 - 지정된 feature들만 사용
        features = {}

        # 정의된 특성 목록을 순회하며 데이터 추출
        for feature_name, (section, key, data_type) in FEATURES.items():
            default_value = DEFAULT_VALUES[data_type]
            if section:
                features[feature_name] = item.get(
                    section, {}).get(key, default_value)
            else:
                features[feature_name] = item.get(key, default_value)

        # 타겟 변수 추가
        features['PartCode'] = part_code

        dataset.append(features)

    return dataset, skipped_count


def build_dataset():
    """
    데이터셋을 구축하고 CSV 파일로 저장하는 함수
    """
    file_names = load_filenames_from_txt()
    all_data = []

    # 각 JSON 파일에서 데이터 추출
    total_skipped = 0

    for file_name in file_names:
        try:
            json_data = json_to_dict(file_name)
            extracted_data, skipped_count = extract_features_from_json(
                json_data)
            all_data.extend(extracted_data)
            total_skipped += skipped_count
            print(
                f"파일 '{file_name}' 처리 완료: {len(extracted_data)}개 항목 추출, {skipped_count}개 항목 스킵")
        except Exception as e:
            print(f"파일 '{file_name}' 처리 중 오류 발생: {e}")

    # 데이터프레임 생성
    df = pd.DataFrame(all_data)

    # 데이터셋 저장 경로
    data_dir = os.path.join(os.getcwd(), "data")
    dataset_dir = os.path.join(data_dir, "dataset")

    # dataset 디렉토리가 없으면 생성
    os.makedirs(dataset_dir, exist_ok=True)

    # CSV 파일로 저장
    output_path = os.path.join(dataset_dir, "part_code_dataset.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n데이터셋 구축 완료: {len(df)}개 항목, 부위코드 없는 항목 {total_skipped}개 스킵")
    print(f"부위코드 종류: {df['PartCode'].nunique()}개")
    print(f"데이터셋 저장 경로: {output_path}")

    # 데이터 분포 확인
    print("\n부위코드 분포:")
    print(df['PartCode'].value_counts())

    return df


if __name__ == '__main__':
    df = build_dataset()
