from fastapi import Header, HTTPException

from app.core.config import settings


async def require_api_key(x_api_key: str = Header(default="")):
    if not settings.api_key:
        return  # auth disabled if no key is configured
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")