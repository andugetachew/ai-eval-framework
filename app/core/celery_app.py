from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "ai_eval_framework",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.core.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
)