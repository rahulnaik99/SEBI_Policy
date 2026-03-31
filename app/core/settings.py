from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class settings(BaseSettings):
    OPENAI_API_KEY : str
    CHAT_MODEL_OPENAI : str
    EMBEDDING_MODEL_OPENAI : str
    COLLECTION : str
    EMBEDDING_MODEL_ST : str
    CHAT_MODEL_GROQ : str
    GROQ_API_KEY:str
    CHAT_MODEL_OLLAMA : str
    EMBEDDING_MODEL_OLLAMA : str
    PDF_PATH :str
    OLLAMA_URL:str
    HUGGINGFACE_CROSS_ENCODER: str
    ENCODER: str
    Top_Retrieval : int
    Top_MMR : int
    Top_Encoder :int 
    REDIS_HOST: str
    REDSI_PORT: int
    REDIS_TTL: int
    model_config=ConfigDict(env_file=".env")
    

settings = settings()