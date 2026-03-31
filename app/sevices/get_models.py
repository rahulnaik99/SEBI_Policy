from sentence_transformers import SentenceTransformer
from app.core.settings import settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings, ChatOllama


def OPENAI_CHAT():
    return ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=settings.CHAT_MODEL_OPENAI)

def GROQ_CHAT():
    return ChatGroq(api_key=settings.GROQ_API_KEY,model=settings.CHAT_MODEL_GROQ)

def OLLAMA_CHAT():
    return ChatOllama(model=settings.CHAT_MODEL_OLLAMA)

_Ollama_Embedding = None
def OLLAMA_EMBEDDING():
    global _Ollama_Embedding
    if _Ollama_Embedding is None:
        _Ollama_Embedding = OllamaEmbeddings(model=settings.EMBEDDING_MODEL_OLLAMA,base_url=settings.OLLAMA_URL)
    return _Ollama_Embedding

_Openai_Embedding = None
def OPENAI_EMBEDDING():
    global _Openai_Embedding
    if _Openai_Embedding is None:
        return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL_OPENAI)

from sentence_transformers import SentenceTransformer

_Embedding_ST = None
def EMBEDDING_ST():
    global _Embedding_ST
    if _Embedding_ST is None:   # ✅ correct
        _Embedding_ST = SentenceTransformer(
            settings.EMBEDDING_MODEL_ST
        )
    return _Embedding_ST


