# apps/api — Python / uv / FastAPI

Python 3.13 (`.python-version`), uv 프로젝트, hatchling 빌드, src 레이아웃.

## 실행

- `bunx nx run api:install` — uv sync (`.venv` 설치)
- `bunx nx run api:serve` — `uv run fastapi dev src/api/main.py` (0.0.0.0:8000)
- `bunx nx run api:test` — pytest + coverage + html/junit 리포트
- `bunx nx run api:lint` / `api:format` — ruff check / format
- `bunx nx run api:build` — hatchling으로 wheel/sdist 생성 (`dist/`)
- 의존성 추가: `bunx nx run api:add --name=<pkg> [--extras=<x>] [--group=dev]`

## 레이아웃

- 엔트리: `src/api/main.py` — `app = FastAPI(...)` 정의, 라우터는 여기서 `include_router`
- 소스: `src/api/{core,routers}/*.py` (src/ 레이아웃, hatchling이 자동 감지)
- 테스트: `tests/` — `conftest.py`의 `client` fixture는 파일 간 자동 공유
- 설정: `core/config.py`의 pydantic-settings `Settings`, env prefix `API_`
  - 예: `API_DEBUG=true bunx nx run api:serve` (필드 이름은 대소문자 무시)
- 새 라우터: `routers/<name>.py`에 `router = APIRouter()` 선언 → `main.py`에서 `app.include_router(<name>.router)`

## 알려진 함정

- `fastapi dev`는 `fastapi[standard]` extras 필요 (plain `fastapi`는 CLI 없음)
- `fastapi dev` 종료 시 reloader 자식이 남을 수 있음 → `pkill -f 'fastapi dev'`
- `project.json` 수정 후 Nx 데몬 캐시로 target 미인식 → `bunx nx reset`
- `@nxlv/python:uv-project --srcDir=true` 생성 시 `sourceRoot`가 잘못 기록됨 → 생성 직후 `project.json`의 `sourceRoot`를 `apps/api/src/api`로 확인·수정
- `uvicorn[standard]`은 `fastapi[standard]`에 전이적으로 포함 (중복 명시 무해)

## 테스트 관례

- FastAPI `TestClient` 사용 (`httpx` dev dep 필요)
- fixture는 `conftest.py`에 정의 → pytest 자동 주입
- coverage threshold는 설정하지 않음 (초기 단계) — 필요 시 `pyproject.toml`의 `[tool.pytest.ini_options].addopts`에 `--cov-fail-under=N` 추가
