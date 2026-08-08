from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    anthropic_api_key: str = ""
    api_key: str = ""  # set this to require auth; empty = auth disabled

    class Config:
        env_file = ".env"

    @property
    def async_database_url(self) -> str:
        # Render (and most managed Postgres providers) give plain
        # postgresql:// or postgres:// URLs; async SQLAlchemy needs
        # the asyncpg driver specified explicitly.
        url = self.database_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://") and "+asyncpg" not in url:
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url


settings = Settings()