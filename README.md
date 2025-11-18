# SICT-BIMINI: Korean BIM Standard Part Code Prediction System

한국 BIM 표준(KBIMS) 부위코드 예측을 위한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [기술 스택](#기술-스택)
3. [프로젝트 구조](#프로젝트-구조)
4. [주요 기능](#주요-기능)
5. [설치 및 설정](#설치-및-설정)
6. [사용법](#사용법)
7. [Docker 배포](#docker-배포)
8. [트러블슈팅](#트러블슈팅)

## 프로젝트 개요

**SICT-BIMINI**는 BIM 객체를 한국 건축정보모델 표준(KBIMS)에 따라 자동으로 분류하고 부위코드를 예측하는 시스템입니다.

### 주요 특징

- **벡터 임베딩**: SentenceTransformer를 사용한 BIM 객체의 시맨틱 임베딩 생성
- **벡터 데이터베이스**: Milvus-lite를 활용한 대규모 유사도 검색
- **대규모 언어 모델**: Ollama 통합을 통한 지능형 예측 및 추론
- **데이터 파이프라인**: 다양한 입력 형식(Excel, JSON, CSV) 지원

### 핵심 기능

- BIM 객체의 KBIMS 부위코드 예측
- BIM 속성에 대한 시맨틱 유사도 검색
- 한국어 BIM 객체 분류 질의응답
- 다중 객체 배치 예측 처리
- 다양한 데이터 형식 지원 (XLSX, XLSM, JSON)

## 기술 스택

### 핵심 기술

| 컴포넌트 | 버전 | 용도 |
|----------|------|------|
| **Python** | 3.12+ | 메인 프로그래밍 언어 |
| **LangChain** | 1.0.5+ | LLM 오케스트레이션 및 프롬프트 관리 |
| **Ollama** | Latest | 로컬 LLM 추론 |
| **Milvus-lite** | 2.6.3+ | 임베딩용 벡터 데이터베이스 |
| **SentenceTransformers** | 5.1.2+ | 텍스트 임베딩 생성 |
| **PyTorch** | 2.9.1+ | 딥러닝 프레임워크 |
| **Pandas** | Latest | 데이터 처리 |
| **Docker** | Latest | 컨테이너화 |

### 주요 의존성

```
langchain-core>=1.0.5          # LLM 프레임워크
langchain-ollama>=1.0.0        # Ollama 통합
pymilvus[milvus-lite]>=2.6.3   # 벡터 데이터베이스
sentence-transformers>=5.1.2   # 임베딩
torch>=2.9.1                   # 딥러닝
pandas                         # 데이터 처리
openpyxl                       # Excel 처리
huggingface-hub               # 모델 허브 접근
```

## 프로젝트 구조

```
sict-bimini/
├── src/
│   ├── bim_vector_store.py       # 벡터 스토어 구현
│   ├── rag.py                    # RAG 시스템
│   └── converters/
│       ├── __init__.py
│       ├── json_to_csv.py        # JSON to CSV 변환
│       ├── xlsm2json.py          # XLSM to JSON 변환
│       └── xlsx2csv.py           # XLSX to CSV 변환
│
├── data/
│   ├── json/                     # JSON 소스 파일
│   ├── csv/                      # 처리된 CSV 파일
│   └── xlsx/                     # Excel 소스 파일
│
├── script/
│   └── ollama.sh                # Ollama 설정 스크립트
│
├── milvus_data/                 # 벡터 데이터베이스 저장소
├── pyproject.toml               # 프로젝트 메타데이터
├── uv.lock                      # 의존성 락 파일
├── Dockerfile                   # 컨테이너 이미지 정의
├── docker-compose.yaml          # 멀티 서비스 오케스트레이션
└── .exemple.env                 # 환경변수 템플릿
```

### 디렉토리 설명

- **src/**: 핵심 애플리케이션 코드
  - `bim_vector_store.py`: Milvus 데이터베이스 작업 및 벡터 임베딩 처리
  - `rag.py`: KBIMS 예측을 위한 RAG 시스템
  - `converters/`: 데이터 형식 변환 유틸리티

- **data/**: 데이터 저장소
  - `json/`: JSON 형식의 원본 BIM 데이터
  - `csv/`: 벡터 임베딩용 처리된 CSV 파일
  - `xlsx/`: BIM 속성이 포함된 Excel 소스 파일

- **milvus_data/**: 벡터 데이터베이스 영속 계층

## 주요 기능

### 1. BIM 벡터 스토어 (`bim_vector_store.py`)

벡터 스토어는 모든 시맨틱 임베딩과 유사도 검색 작업을 관리합니다.

**주요 기능:**
- SentenceTransformer를 사용한 자동 임베딩 생성 (768차원 벡터)
- COSINE 메트릭과 IVF_FLAT 인덱싱을 사용한 효율적인 유사도 검색
- 설정 가능한 크기의 배치 삽입
- 컬렉션 통계 및 관리
- 자동 중복 제거가 포함된 CSV 데이터 로딩

**지원 속성:**
- 카테고리
- 패밀리명
- KBIMS 부위코드
- 패밀리
- 타입
- 타입ID

### 2. RAG 시스템 (`rag.py`)

RAG 시스템은 벡터 검색과 LLM 추론을 결합하여 지능형 예측을 수행합니다.

**주요 기능:**
- **부위코드 예측**: 근거와 함께 KBIMS 코드 예측
- **채팅 모드**: 한국어로 BIM 객체에 대한 질문 응답
- **시맨틱 검색**: 유사한 BIM 객체 검색
- **배치 처리**: 여러 객체를 효율적으로 처리

**프롬프팅 전략:**
- 도메인 이해를 위해 최적화된 한국어 프롬프트
- 검색된 유사 객체를 사용한 컨텍스트 인식 생성
- 신뢰도 점수 (높음/중간/낮음)

### 3. 데이터 변환기 (`converters/`)

#### JSON to CSV 변환기 (`json_to_csv.py`)
- JSON 소스 파일에서 BIM 속성 추출
- 속성 조합의 자동 중복 제거
- 상세 추출 통계
- 한국어 텍스트를 위한 UTF-8 인코딩 지원

#### XLSM to JSON 변환기 (`xlsm2json.py`)
- Excel 속성 테이블을 JSON으로 변환
- 계층 구조 보존
- 속성 집합 및 메타데이터 처리

#### XLSX to CSV 변환기 (`xlsx2csv.py`)
- KBIMS Excel 시트의 배치 변환
- 컬럼 선택 및 커스텀 매핑
- 시트 탐색 및 처리
- 한국어 문자를 위한 UTF-8 인코딩

## 설치 및 설정

### 사전 요구사항

- **Python**: 3.12 이상
- **시스템 의존성**: GPU 지원을 위해 NVIDIA CUDA 툴킷 권장
- **디스크 공간**: 모델 및 데이터베이스용 ~10GB
- **RAM**: 최소 16GB

### 로컬 설치

#### 1. 저장소 클론

```bash
git clone <repository-url>
cd sict-bimini
```

#### 2. 가상환경 생성

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 또는
.venv\Scripts\activate  # Windows
```

#### 3. 의존성 설치

UV 사용 (권장):
```bash
uv sync
```

pip 사용:
```bash
pip install -e .
```

<!-- #### 4. 환경변수 설정

```bash
cp .exemple.env .env
# .env 파일을 편집하여 설정 추가
``` -->

### Docker 설치

#### 빠른 시작 (CPU)

```bash
docker compose up sict-bimini
```

#### GPU 지원

```bash
docker compose --profile gpu build sict-bimini-gpu
```

**GPU 요구사항:**
- CUDA Compute Capability 3.5+ NVIDIA GPU
- nvidia-docker 플러그인 설치

## 사용법

### 1. 데이터 준비

#### Excel을 CSV로 변환

```bash
python src/converters/xlsx2csv.py
# data/xlsx/KBIMS.xlsx의 모든 시트를 data/csv/로 변환
```

#### JSON을 CSV로 변환

```bash
python src/converters/json_to_csv.py \
  --json-dir ./data/json \
  --output-dir ./data/csv \
  --output-file bim_attributes.csv
```

### 2. 벡터 스토어 관리

#### 데이터 로드 및 스토어 초기화

```bash
python -m src.bim_vector_store --csv ./data/csv/bim_attributes.csv
```

#### 벡터 스토어 검색

```bash
python -m src.bim_vector_store --search "콘크리트 기둥" --limit 10
```

#### 컬렉션 통계 조회

```bash
python -m src.bim_vector_store --stats
```

#### 컬렉션 초기화 (모든 데이터 삭제)

```bash
python -m src.bim_vector_store --reset --csv ./data/csv/bim_attributes.csv
```

### 3. RAG 시스템 (Ollama 필요)

#### Ollama 서버 설정

```bash
# Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# Ollama 서비스 실행
ollama serve &

# 필요한 모델 다운로드
ollama pull gpt-oss:20b

# 또는 제공된 스크립트 사용
bash script/ollama.sh
```

#### KBIMS 코드 예측

```bash
python -m src.rag --predict "콘크리트 기둥 RC기둥-600x600" --top-k 5
```

예상 출력:
```
예측 부위코드: [예측된 코드]
근거: [유사 객체와의 비교를 통한 근거 설명]
신뢰도: [높음/중간/낮음]
```

#### 채팅 모드

```bash
python -m src.rag --chat "기초의 종류에는 무엇이 있나요?" --top-k 5
```

#### 시맨틱 검색

```bash
python -m src.rag --search "철근콘크리트 보" --top-k 5
```

#### 시스템 통계

```bash
python -m src.rag --stats
```