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
    experiment_id: str = "qwen8b_d2048"

    # 임베딩 (TEI)
    tei_url: str = "http://localhost:8080"
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    embedding_dim: int = 2048  # MRL 절단

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # 데이터 경로 (apps/api 기준 상대 경로)
    data_root: Path = Path("data")

    # vLLM (외부 독립 운영 서버)
    llm_url: str = "http://localhost:8001"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_timeout_seconds: float = 60.0

    # RAG 튜닝
    retrieval_k_min: int = 10
    retrieval_k_multiplier: int = 3
    mode_sim_threshold: float = 0.55

    # 코드 포맷 정규식 (Weak 모드 schema)
    # TODO: finalize after KBIMS catalog audit
    kbims_code_regex: str = r"^[A-Z]{2}\d+$"
    # TODO: finalize after PPS catalog audit
    pps_code_regex: str = r"^[A-Z]-\d+(-\d+)*$"

    @property
    def collection_name(self) -> str:
        return f"bim__{self.experiment_id}"


settings = Settings()
