from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    anthropic_api_key: str = ""
    api_key: str = ""
    redis_url: str = "redis://redis:6379/0"

    class Config:
        env_file = ".env"

    @property
    def async_database_url(self) -> str:
        url = self.database_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://") and "+asyncpg" not in url:
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    @property
    def sync_database_url(self) -> str:
        # Celery tasks run sync SQLAlchemy (no event loop available in a worker task)
        url = self.database_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif url.startswith("postgresql://") and "+" not in url:
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url


settings = Settings()