from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SICT Bimini API"
    debug: bool = False

    model_config = SettingsConfigDict(env_prefix="API_", case_sensitive=False)


settings = Settings()
