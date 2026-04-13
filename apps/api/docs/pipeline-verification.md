# 파이프라인 E2E 수동 검증 체크리스트

**언제 사용**: Task 1~9 구현 완료 후, 본 스펙(§13)의 성공 기준을 로컬에서 확인할 때.

**전제**:
- 외부 TEI 서비스가 기동 중이며 Qwen3-Embedding-8B를 서빙 (예: `http://localhost:8080`)
- 외부 Qdrant가 기동 중 (예: `http://localhost:6333`)
- `apps/api/data/xlsx/`에 최소 1개의 BIM xlsx 파일이 있음

## 1. 단위 테스트 통과 (성공 기준 #1)

```bash
bunx nx run api:test
```

**기대**: `tests/bim/*.py` + `tests/cli/*.py` 전부 PASS.

## 2. 파이프라인 1회 실행으로 3개 산출물 생성 (성공 기준 #2)

```bash
bunx nx run api:pipeline
```

**기대 출력**:
- `apps/api/data/json/raw/<name>.json` 생성됨
- `apps/api/data/json/normalized/<name>.json` 생성됨
- Qdrant 컬렉션 `bim__qwen8b_d2048`에 최소 1개 point 존재

**검증 커맨드**:

```bash
ls apps/api/data/json/raw/
ls apps/api/data/json/normalized/

curl -s http://localhost:6333/collections/bim__qwen8b_d2048 | jq '.result.points_count'
# 기대: 1 이상
```

## 3. Idempotency (성공 기준 #3)

```bash
# 2번째 실행
bunx nx run api:pipeline

curl -s http://localhost:6333/collections/bim__qwen8b_d2048 | jq '.result.points_count'
# 기대: 1번째 실행과 동일한 카운트 (stable_id upsert)
```

## 4. 실험 격리 (성공 기준 #4)

```bash
# experiment_id를 바꿔 재실행
BIM_EXPERIMENT_ID=qwen8b_d1024 BIM_EMBEDDING_DIM=1024 bunx nx run api:pipeline

# 원래 컬렉션 점수 불변 확인
curl -s http://localhost:6333/collections/bim__qwen8b_d2048 | jq '.result.points_count'

# 새 컬렉션 생성 확인
curl -s http://localhost:6333/collections/bim__qwen8b_d1024 | jq '.result.points_count'
# 기대: 둘 다 ≥1, 서로 독립
```

## 5. 워크스페이스 체인 (성공 기준 #5)

```bash
bunx nx run api:test
bunx nx run api:lint
bunx nx run api:build
```

**기대**: 모두 exit 0.

## 6. Predict-eval smoke (RAG 평가 CLI)

`api:pipeline`으로 라벨 있는 레코드가 Qdrant에 들어간 상태 + 외부 vLLM(`BIM_LLM_URL`) 기동 전제.

```bash
# vLLM 서빙 확인
bunx nx run api:llm-check

# 전량 평가 (kbims_code)
bunx nx run api:predict-eval -- --target kbims_code

# 필터 + 샘플 평가
bunx nx run api:predict-eval -- --target kbims_code --ifc-type IfcColumn --limit 20 --seed 42
```

**기대 출력**:
- stdout에 `=== predict-eval [kbims_code] ===` 헤더 + Top-1/Top-N accuracy + mode 분포 + latency p50/p95 + `Report: ...` 경로
- `data/reports/predict-eval/{UTC_ISO}_{target}/summary.json` 생성됨 (Pydantic AggregatedMetrics 직렬화)
- `data/reports/predict-eval/{UTC_ISO}_{target}/predictions.jsonl` 생성됨 (per-record `{stable_id, ground_truth, top1, top_k, mode, pool_size, latency_ms, error}` 1줄당 1레코드)

**검증 커맨드**:

```bash
# 최신 실행 디렉토리
LATEST=$(ls -td apps/api/data/reports/predict-eval/*/ | head -1)

# summary.json 구조 확인
jq 'keys' "$LATEST/summary.json"
# 기대: ["accuracy_by_ifc_type", "accuracy_by_mode", "errors_by_type", ...]

# top-1 accuracy
jq '.top1_accuracy' "$LATEST/summary.json"

# 틀린 예측만 필터링
jq -c 'select(.top1 != .ground_truth)' "$LATEST/predictions.jsonl" | head -5
```

**실패 경로 확인**:

```bash
# 매칭 0개 필터 → exit 1
bunx nx run api:predict-eval -- --target kbims_code --ifc-type IfcNonExistent
# 기대: stderr "predict-eval: No samples match filter ..." + exit 1

# 잘못된 target → exit 1
bunx nx run api:predict-eval -- --target bogus
# 기대: stderr "--target must be one of ..." + exit 1
```

## 차원 불일치 에러 경로 (보너스 확인)

```bash
# 기존 bim__qwen8b_d2048가 2048-D인 상태에서 1024로 시도
BIM_EXPERIMENT_ID=qwen8b_d2048 BIM_EMBEDDING_DIM=1024 bunx nx run api:upsert-qdrant
# 기대: DimensionMismatchError로 즉시 종료 (upsert 수행되지 않음)
```
