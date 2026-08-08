import ssl

from celery import Celery

from app.core.config import settings

_is_secure_redis = settings.redis_url.startswith("rediss://")

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
    task_default_queue="ai_eval_framework_queue",
    result_backend_transport_options={"global_keyprefix": "ai_eval_framework:"},
)

if _is_secure_redis:
    ssl_opts = {"ssl_cert_reqs": ssl.CERT_NONE}
    celery_app.conf.broker_use_ssl = ssl_opts
    celery_app.conf.redis_backend_use_ssl = ssl_opts