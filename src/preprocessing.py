import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from build_dataset import FEATURES, DEFAULT_VALUES


class PartCodeDataPreprocessor:
    """
    부위코드 예측 모델을 위한 데이터 전처리 클래스

    이 클래스는 다음과 같은 전처리 작업을 수행합니다:
    1. 범주형 변수 처리: OneHotEncoding 적용
    2. 수치형 변수 처리: 표준화(StandardScaler) 적용
    3. 이진형 변수 처리: 0, 1 변환
    4. 타겟 변수(부위코드) 처리: LabelEncoder 적용
    """

    def __init__(self):
        """전처리기 초기화"""
        # build_dataset.py의 FEATURES에서 특성 타입별로 분류
        self.categorical_features = []
        self.numerical_features = []
        self.binary_features = []

        for feature_name, (_, _, data_type) in FEATURES.items():
            if data_type == 'string':
                self.categorical_features.append(feature_name)
            elif data_type == 'number':
                self.numerical_features.append(feature_name)
            elif data_type == 'boolean':
                self.binary_features.append(feature_name)

        self.encoders = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def fit(self, df):
        """
        데이터프레임을 기반으로 전처리기를 학습시킵니다.

        Args:
            df (pd.DataFrame): 학습 데이터 프레임

        Returns:
            self: 학습된 전처리기
        """
        # 범주형 변수 인코더 학습
        for feature in self.categorical_features:
            encoder = OneHotEncoder(
                sparse_output=False, handle_unknown='ignore')
            # 결측값 처리
            df[feature] = df[feature].fillna('')
            encoder.fit(df[[feature]])
            self.encoders[feature] = encoder

        # 수치형 변수 스케일러 학습
        numeric_data = df[self.numerical_features].fillna(0)
        self.scaler.fit(numeric_data)

        # 타겟 변수(부위코드) 인코더 학습
        self.label_encoder.fit(df['PartCode'])

        return self

    def transform(self, df):
        """
        학습된 전처리기로 데이터를 변환합니다.

        Args:
            df (pd.DataFrame): 변환할 데이터 프레임

        Returns:
            tuple: (X, y) - 변환된 특성과 타겟 변수
        """
        # 전처리된 데이터를 저장할 리스트
        transformed_features = []

        # 1. 범주형 변수 처리
        for feature in self.categorical_features:
            # 결측값 채우기
            df[feature] = df[feature].fillna('')

            # One-Hot Encoding 적용
            encoded = self.encoders[feature].transform(df[[feature]])
            transformed_features.append(encoded)

        # 2. 수치형 변수 처리
        numeric_data = df[self.numerical_features].fillna(0)
        scaled_numeric = self.scaler.transform(numeric_data)
        transformed_features.append(scaled_numeric)

        # 3. 이진형 변수 처리
        for feature in self.binary_features:
            # boolean이나 문자열을 숫자로 변환
            binary_vals = df[feature].fillna(0)
            if binary_vals.dtype == bool:
                binary_vals = binary_vals.astype(int)
            else:
                # 문자열이나 다른 타입인 경우 처리
                binary_vals = binary_vals.apply(lambda x: 1 if str(
                    x).lower() in ['true', '1', 'yes'] else 0)

            # 차원 확장 (n, 1) 형태로
            transformed_features.append(binary_vals.values.reshape(-1, 1))

        # 4. 모든 특성 결합
        # cspell: disable-next-line
        X = np.hstack(transformed_features)

        # 5. 타겟 변수 처리 (있는 경우에만)
        if 'PartCode' in df.columns:
            y = self.label_encoder.transform(df['PartCode'])
            return X, y
        else:
            return X

    def fit_transform(self, df):
        """
        학습과 변환을 한번에 수행합니다.

        Args:
            df (pd.DataFrame): 학습 및 변환할 데이터 프레임

        Returns:
            tuple: (X, y) - 변환된 특성과 타겟 변수
        """
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self):
        """
        변환 후 특성 이름을 가져옵니다.

        Returns:
            list: 변환 후 특성 이름 목록
        """
        feature_names = []

        # 범주형 변수 특성 이름
        for feature in self.categorical_features:
            encoder = self.encoders[feature]
            categories = encoder.categories_[0]
            for category in categories:
                if category == '':
                    feature_names.append(f"{feature}_MISSING")
                else:
                    feature_names.append(f"{feature}_{category}")

        # 수치형 변수 특성 이름
        feature_names.extend(self.numerical_features)

        # 이진형 변수 특성 이름
        feature_names.extend(self.binary_features)

        return feature_names

    def get_class_names(self):
        """
        클래스(부위코드) 이름을 가져옵니다.

        Returns:
            list: 클래스 이름 목록
        """
        return list(self.label_encoder.classes_)

    def get_num_features(self):
        """
        변환 후 특성 수를 반환합니다.

        Returns:
            int: 특성 수
        """
        num_features = 0

        # 범주형 변수 차원 계산
        for feature in self.categorical_features:
            encoder = self.encoders[feature]
            num_features += len(encoder.categories_[0])

        # 수치형 변수 및 이진형 변수 차원 추가
        num_features += len(self.numerical_features)
        num_features += len(self.binary_features)

        return num_features

    def get_num_classes(self):
        """
        부위코드 클래스 수를 반환합니다.

        Returns:
            int: 클래스 수
        """
        return len(self.label_encoder.classes_)


class PartCodeDataset(Dataset):
    """파이토치 데이터셋 클래스"""

    def __init__(self, X, y):
        """
        데이터셋 초기화

        Args:
            cSpell:ignore ndarray
            X (numpy.ndarray): 특성 데이터
            y (numpy.ndarray): 타겟 데이터
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        """데이터셋 길이 반환"""
        return len(self.X)

    def __getitem__(self, idx):
        """인덱스별 샘플 반환"""
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(csv_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    CSV 파일에서 데이터를 로드하고 전처리하는 함수

    Args:
        csv_path (str): CSV 파일 경로
        test_size (float): 테스트 데이터 비율 (0~1)
        val_size (float): 검증 데이터 비율 (0~1)
        random_state (int): 난수 시드

    Returns:
        tuple: (preprocessor, train_dataset, val_dataset, test_dataset) - 전처리기와 데이터셋
    """
    # 데이터 로드
    df = pd.read_csv(csv_path)

    # 데이터 확인 및 정보 출력
    print(f"데이터 로드 완료: {len(df)}개 샘플")
    print(f"부위코드 종류: {df['PartCode'].nunique()}개")
    print(f"부위코드 분포:\n{df['PartCode'].value_counts()}")

    # 수치형 특성의 기본 통계 확인
    numeric_features = ['Width', 'Height',
                        'Length', 'Volume', 'CrossSectionArea']
    print("\n수치형 특성 통계:")
    print(df[numeric_features].describe())

    # 결측치 확인
    print("\n결측치 개수:")
    print(df.isnull().sum())

    # 데이터 전처리기 생성 및 학습
    preprocessor = PartCodeDataPreprocessor()

    # 학습/검증/테스트 분할
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state,
                                             stratify=df['PartCode'])
    train_df, val_df = train_test_split(train_val_df, test_size=val_size/(1-test_size),
                                        random_state=random_state, stratify=train_val_df['PartCode'])

    print(
        f"\n데이터 분할: 학습({len(train_df)}), 검증({len(val_df)}), 테스트({len(test_df)})")

    # 전처리기를 학습 데이터로만 학습
    preprocessor.fit(train_df)

    # 학습/검증/테스트 데이터 변환
    X_train, y_train = preprocessor.transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)

    print(f"\n변환된 특성 차원: {X_train.shape[1]}")
    print(f"클래스 수: {preprocessor.get_num_classes()}")

    # PyTorch 데이터셋 생성
    train_dataset = PartCodeDataset(X_train, y_train)
    val_dataset = PartCodeDataset(X_val, y_val)
    test_dataset = PartCodeDataset(X_test, y_test)

    return preprocessor, train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    데이터셋으로부터 DataLoader를 생성하는 함수

    Args:
        train_dataset (PartCodeDataset): 학습 데이터셋
        val_dataset (PartCodeDataset): 검증 데이터셋
        test_dataset (PartCodeDataset): 테스트 데이터셋
        batch_size (int): 배치 크기

    Returns:
        tuple: (train_loader, val_loader, test_loader) - 데이터 로더들
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 데이터셋 경로
    data_dir = os.path.join(os.getcwd(), "data")
    dataset_path = os.path.join(data_dir, "dataset", "part_code_dataset.csv")

    # 데이터 로드 및 전처리
    preprocessor, train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
        dataset_path, test_size=0.2, val_size=0.1
    )

    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=64
    )

    print(
        f"\n전처리 완료: 학습({len(train_dataset)}), 검증({len(val_dataset)}), 테스트({len(test_dataset)})")
    print(
        f"배치 수: 학습({len(train_loader)}), 검증({len(val_loader)}), 테스트({len(test_loader)})")

    # 샘플 데이터 확인
    for X_batch, y_batch in train_loader:
        print(f"\n배치 샘플 형태: X={X_batch.shape}, y={y_batch.shape}")
        print(f"클래스 분포: {torch.bincount(y_batch)}")
        break
