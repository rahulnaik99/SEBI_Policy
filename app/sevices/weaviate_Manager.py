import weaviate
from weaviate.classes.config import DataType, Property
from app.core.settings import settings
_client=None
def WEAVIATE_CLIENT():
    global _client
    if _client is None:
        _client = weaviate.connect_to_custom(
            http_host="weaviate",   # ← service name from docker-compose.yml
            http_port=8080,
            grpc_host="weaviate",
            grpc_port=50051,
        )
    return _client

def SET_SCHEMA():
    client = WEAVIATE_CLIENT()
    if not client.collections.exists(settings.COLLECTION):
        client.collections.create(
            name=settings.COLLECTION,
            properties=([
                Property(name="text",data_type=DataType.TEXT),
                Property(name="source",data_type=DataType.TEXT),
                Property(name="page_number",data_type=DataType.TEXT),
                Property(name="total_pages",data_type=DataType.TEXT),
                Property(name="chunk_number",data_type=DataType.TEXT),
                Property(name="total_chunks",data_type=DataType.TEXT),
                Property(name="hashing",data_type=DataType.TEXT),
                Property(name="is_active",data_type=DataType.BOOL),
                
            ])
        )
    client.close