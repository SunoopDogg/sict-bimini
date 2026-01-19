# SICT-BIMINI

**KBIMS 부위코드 자동 예측 시스템** — BIM 객체를 한국 건축정보모델 표준(KBIMS)에 맞게 분류하고 부위코드를 예측합니다.

## 핵심 구조

```
src/
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
| **Milvus-lite** | 벡터 유사도 검색 |
| **SentenceTransformers** | 텍스트 임베딩 (768D) |
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
ollama pull gemma3:27b
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
  "confidence": "high",
  "rationale": "유사 객체 분석 기반 근거..."
}
```

## 주요 API

### BIMVectorStore

| 메서드 | 설명 |
|--------|------|
| `load_from_csv(path)` | CSV에서 BIM 속성 로드 및 임베딩 |
| `search(query, limit)` | 시맨틱 유사도 검색 |
| `reset()` | 컬렉션 초기화 |

### BIMRAGSystem

| 메서드 | 설명 |
|--------|------|
| `predict_part_code(bim_info)` | KBIMS 코드 예측 |
| `batch_predict(bim_list)` | 배치 예측 |
| `search(query, top_k)` | 유사 객체 검색 |

## Docker

```bash
# CPU
docker compose up sict-bimini

# GPU
docker compose --profile gpu build sict-bimini-gpu
```

## 파일 설명

| 파일 | 역할 |
|------|------|
| `bim_vector_store.py` | Milvus 컬렉션 관리, 임베딩 생성, 유사도 검색 |
| `rag.py` | LLM 기반 부위코드 예측, 컨텍스트 생성 |
| `converters/xlsx2json.py` | Excel 속성 테이블 → JSON 변환 |
| `converters/json_to_csv.py` | JSON → CSV 변환 (중복 제거) |
| `utils/prompt.py` | 프롬프트 템플릿 로더 |
| `utils/parsing.py` | LLM JSON 응답 파서 |
| `utils/formatters.py` | 결과 포맷팅 |
| `utils/file_selector.py` | 인터랙티브 파일 선택 |
