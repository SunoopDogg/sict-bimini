# SICT-BIMINI

**KBIMS 부위코드 자동 예측 시스템** — BIM 객체를 한국 건축정보모델 표준(KBIMS)에 맞게 분류하고 부위코드를 예측합니다.

## 핵심 구조

```
src/
├── api/                  # REST API (FastAPI)
│   ├── server.py         # API 서버
│   └── schemas.py        # 요청/응답 스키마
├── bim_vector_store.py   # 벡터 DB (Milvus) + 임베딩
├── rag.py                # RAG 예측 시스템 (Ollama LLM)
├── converters/           # 데이터 변환 (XLSX→JSON→CSV)
└── utils/                # 유틸리티 (프롬프트, 파싱, 포맷팅)

data/
├── xlsx/                 # 원본 BIM 속성 엑셀
├── json/                 # 변환된 JSON
└── csv/                  # 벡터 스토어용 CSV

prompts/
└── kbims_prediction.txt  # LLM 프롬프트 템플릿
```

## 기술 스택

| 구성요소 | 역할 |
|----------|------|
| **FastAPI** | REST API 서버 |
| **Milvus-lite** | 벡터 유사도 검색 |
| **SentenceTransformers** | 텍스트 임베딩 (300D) |
| **Ollama** | 로컬 LLM 추론 |
| **LangChain** | LLM 오케스트레이션 |

## 설치

```bash
# 의존성 설치 (UV 권장)
uv sync

# 또는 pip
pip install -e .
```

**요구사항**: Python 3.12+, RAM 16GB+

## 워크플로우

### 1. 데이터 변환

```bash
# XLSX → JSON
python src/converters/xlsx2json.py

# JSON → CSV (벡터 스토어용)
python src/converters/json_to_csv.py
```

### 2. 벡터 스토어 구축

```python
from src.bim_vector_store import BIMVectorStore

store = BIMVectorStore()
store.load_from_csv("data/csv/bim_attributes.csv")

# 유사 객체 검색
results = store.search("콘크리트 기둥", limit=5)
```

### 3. KBIMS 부위코드 예측

```bash
# Ollama 실행 (필수)
ollama serve &
ollama pull gpt-oss:20b
```

```python
from src.rag import BIMRAGSystem

rag = BIMRAGSystem()

# 단일 예측
result = rag.predict_part_code({
    "category": "구조 기둥",
    "family_name": "RC기둥",
    "family": "RC기둥-600x600",
    "type": "600x600"
})

# 배치 예측
results = rag.batch_predict(bim_objects_list)
```

**예측 결과 형식:**
```json
{
  "predicted_code": "25.21.10.01",
  "reasoning": "유사 객체 분석 기반 근거...",
  "confidence": 0.85
}
```

## REST API

### 서버 실행

```bash
# 개발 모드
uvicorn src.api.server:app --reload

# Docker
docker compose up -d sict-bimini
```

### Endpoints

| Endpoint | Method | 설명 |
|----------|--------|------|
| `/api/v1/health` | GET | 서버 상태 및 연결 확인 |
| `/api/v1/predict` | POST | 단일 BIM 객체 KBIMS 코드 예측 |
| `/api/v1/batch-predict` | POST | 배치 예측 (최대 100개) |
| `/api/v1/search` | GET | 유사 BIM 객체 검색 |

### 예측 API 예제

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "구조기둥",
    "family_name": "RC기둥",
    "family": "콘크리트-직사각형-기둥",
    "type": "400 x 600mm"
  }'
```

**응답:**
```json
{
  "success": true,
  "data": {
    "predicted_code": "25.21.10.01",
    "reasoning": "유사 객체 분석 기반 근거...",
    "confidence": 0.85
  }
}
```

### 검색 API 예제

```bash
curl "http://localhost:8000/api/v1/search?query=콘크리트%20기둥&top_k=5"
```

### API 문서

서버 실행 후 자동 생성:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Docker

```bash
# CPU (포트 8000 노출)
docker compose up -d sict-bimini

# GPU
docker compose --profile gpu up -d sict-bimini-gpu

# API 헬스 체크
curl http://localhost:8000/api/v1/health
```

