from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SICT Bimini API"
    debug: bool = False

    model_config = SettingsConfigDict(env_prefix="API_", case_sensitive=False)


class BIMSettings(BaseSettings):
    """BIM converter pipeline settings (env prefix: BIM_)."""

    model_config = SettingsConfigDict(env_prefix="BIM_", case_sensitive=False)

    # 실험 식별자 → Qdrant 컬렉션 네이밍
    experiment_id: str = "qwen4b_d2048"

    # 임베딩 (vLLM /v1/embeddings)
    embedding_url: str = "http://192.168.0.76:8080"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_dim: int = 2048  # MRL 절단 (native 2560 → 2048)

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # 데이터 경로 (apps/api 기준 상대 경로)
    data_root: Path = Path("data")

    # vLLM (외부 독립 운영 서버)
    llm_url: str = "http://localhost:8000"
    llm_model: str = "gemma-4"
    llm_timeout_seconds: float = 60.0

    # RAG 튜닝
    retrieval_k_min: int = 10
    retrieval_k_multiplier: int = 3
    mode_sim_threshold: float = 0.55

    # 코드 포맷 정규식 (Weak 모드 schema)
    # 실데이터(속성테이블 xlsx): KBIMS는 단일 'E' prefix + 숫자 (E77, E275 …)
    kbims_code_regex: str = r"^E\d+$"
    # 실데이터: 대문자+영숫자 세그먼트를 `+`로 결합 (AMB161A, AJG3, AND011+AGA3105C+…)
    pps_code_regex: str = r"^[A-Z][A-Z0-9]*(\+[A-Z][A-Z0-9]*)*$"

    @property
    def collection_name(self) -> str:
        return f"bim__{self.experiment_id}"


settings = Settings()
