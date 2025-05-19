import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt


class PartCodeClassifier(nn.Module):
    """
    부위코드 분류를 위한 신경망 모델

    다층 퍼셉트론(MLP) 모델로 구성되어 있습니다.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        """
        모델 초기화

        Args:
            input_dim (int): 입력 특성 차원
            hidden_dims (list): 은닉층 차원 리스트, 예: [128, 64]
            output_dim (int): 출력 클래스 수
            dropout_rate (float): 드롭아웃 비율 (0~1)
        """
        super(PartCodeClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # 은닉층 구성
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 출력층
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        순전파 수행

        Args:
            x (torch.Tensor): 입력 텐서

        Returns:
            torch.Tensor: 예측값
        """
        return self.model(x)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                device, num_epochs=50, early_stopping_patience=5, save_path=None):
    """
    모델 학습 함수

    Args:
        model (nn.Module): 학습할 모델
        train_loader (DataLoader): 학습 데이터 로더
        val_loader (DataLoader): 검증 데이터 로더
        criterion: 손실 함수
        optimizer: 최적화 알고리즘
        device (torch.device): 학습 장치 (CPU/GPU)
        num_epochs (int): 학습할 에폭 수
        early_stopping_patience (int): 조기 종료 인내심
        save_path (str): 모델 저장 경로

    Returns:
        dict: 학습 이력
    """
    # 학습 이력 저장용 딕셔너리
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # 조기 종료 설정
    best_val_loss = float('inf')
    patience_counter = 0

    model.to(device)

    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 순전파 및 역전파
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # 학습 통계 업데이트
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        # 검증 모드
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        # 이력 업데이트
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 모델 저장 (검증 손실이 개선된 경우에만)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            if save_path:
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f'모델 저장: {save_path}')
        else:
            patience_counter += 1

        # 조기 종료 검사
        if patience_counter >= early_stopping_patience:
            print(f'{early_stopping_patience}회 연속 검증 손실 개선이 없어 학습을 조기 종료합니다.')
            break

    return history


def evaluate_model(model, test_loader, criterion, device, class_names):
    """
    모델 평가 함수

    Args:
        model (nn.Module): 평가할 모델
        test_loader (DataLoader): 테스트 데이터 로더
        criterion: 손실 함수
        device (torch.device): 평가 장치 (CPU/GPU)
        class_names (list): 클래스 이름 목록

    Returns:
        tuple: (test_loss, test_acc, classification_report, confusion_matrix)
    """
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = accuracy_score(all_targets, all_predictions)

    # 분류 보고서 및 혼동 행렬 생성
    report = classification_report(all_targets, all_predictions,
                                   target_names=class_names, zero_division=0)
    conf_mat = confusion_matrix(all_targets, all_predictions)

    print(f'테스트 손실: {test_loss:.4f}, 정확도: {test_acc:.4f}')
    print('\n분류 보고서:')
    print(report)

    return test_loss, test_acc, report, conf_mat


def plot_training_history(history, save_path=None):
    """
    학습 이력 시각화 함수

    Args:
        history (dict): 학습 이력
        save_path (str): 이미지 저장 경로
    """
    plt.figure(figsize=(12, 5))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)
        print(f'학습 그래프 저장: {save_path}')

    plt.show()


def plot_confusion_matrix(conf_mat, class_names, save_path=None):
    """
    혼동 행렬 시각화 함수

    Args:
        conf_mat (np.ndarray): 혼동 행렬
        class_names (list): 클래스 이름 목록
        save_path (str): 이미지 저장 경로
    """
    plt.figure(figsize=(10, 8))

    # 정규화된 혼동 행렬 계산
    conf_mat_norm = conf_mat.astype(
        'float') / conf_mat.sum(axis=1)[:, np.newaxis]
    conf_mat_norm = np.nan_to_num(conf_mat_norm)  # NaN 처리

    plt.imshow(conf_mat_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()

    # 축 레이블 설정
    num_classes = len(class_names)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # 격자 추가
    plt.grid(False)

    # 값 표시
    thresh = conf_mat_norm.max() / 2.0
    for i in range(conf_mat_norm.shape[0]):
        for j in range(conf_mat_norm.shape[1]):
            if conf_mat_norm[i, j] > 0.01:  # 1% 이상인 경우에만 표시
                plt.text(j, i, f'{conf_mat_norm[i, j]:.2f}',
                         horizontalalignment='center',
                         color='white' if conf_mat_norm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)
        print(f'혼동 행렬 저장: {save_path}')

    plt.show()
