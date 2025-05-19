import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from preprocessing import load_and_preprocess_data, create_dataloaders
from model import PartCodeClassifier, train_model, evaluate_model, plot_training_history, plot_confusion_matrix


def main(args):
    """메인 함수"""
    print("부위코드 예측 모델 학습을 시작합니다.")

    # CUDA 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    # 데이터 로드 및 전처리
    print("\n데이터 로드 및 전처리 중...")
    preprocessor, train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
        args.dataset_path, test_size=args.test_size, val_size=args.val_size, random_state=args.seed
    )

    # 데이터로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
    )

    # 모델 설정
    input_dim = preprocessor.get_num_features()
    output_dim = preprocessor.get_num_classes()
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]

    print(f"\n모델 구성:")
    print(f"- 입력 차원: {input_dim}")
    print(f"- 은닉층 차원: {hidden_dims}")
    print(f"- 출력 차원: {output_dim}")
    print(f"- 드롭아웃 비율: {args.dropout}")

    # 모델 초기화
    model = PartCodeClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=args.dropout
    )

    # 손실 함수 및 최적화기 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 모델 저장 경로 설정
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'part_code_model.pth')

    # 모델 학습
    print("\n모델 학습 시작...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_path=model_path
    )

    # 학습 결과 시각화
    print("\n학습 이력 시각화...")
    history_plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, save_path=history_plot_path)

    # 최종 모델 로드 (early stopping으로 저장된 최적 모델)
    print("\n최적 모델 로드...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 테스트 데이터로 모델 평가
    print("\n테스트 데이터에서 모델 평가...")
    class_names = preprocessor.get_class_names()
    test_loss, test_acc, report, conf_mat = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names
    )

    # 혼동 행렬 시각화
    print("\n혼동 행렬 시각화...")
    conf_mat_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(conf_mat, class_names, save_path=conf_mat_path)

    # 평가 결과 저장
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"테스트 손실: {test_loss:.4f}\n")
        f.write(f"테스트 정확도: {test_acc:.4f}\n\n")
        f.write("분류 보고서:\n")
        f.write(report)

    print(f"\n평가 결과 저장: {report_path}")
    print(f"모델 저장: {model_path}")

    print("\n학습이 완료되었습니다!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="부위코드 예측 모델 학습")

    # 데이터 관련 인자
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.join(
                            os.getcwd(), "data", "dataset", "part_code_dataset.csv"),
                        help='데이터셋 CSV 파일 경로')
    parser.add_argument('--test_size', type=float,
                        default=0.2, help='테스트 데이터 비율')
    parser.add_argument('--val_size', type=float,
                        default=0.1, help='검증 데이터 비율')

    # 모델 관련 인자
    parser.add_argument('--hidden_dims', type=str,
                        default='256,128', help='은닉층 차원 (쉼표로 구분)')
    parser.add_argument('--dropout', type=float, default=0.5, help='드롭아웃 비율')

    # 학습 관련 인자
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=50, help='학습 에폭 수')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='학습률')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5, help='가중치 감소')
    parser.add_argument('--patience', type=int, default=10, help='조기 종료 인내심')
    parser.add_argument('--seed', type=int, default=42, help='난수 시드')

    # 출력 관련 인자
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.getcwd(), "data", "models"),
                        help='출력 디렉토리')

    args = parser.parse_args()

    main(args)
