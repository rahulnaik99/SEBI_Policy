import redis.asyncio as redis
from app.core.settings import settings

_redis_client = None

def get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDSI_PORT,
            decode_responses=True
        )
    return _redis_client