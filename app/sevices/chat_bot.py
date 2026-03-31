from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from app.sevices.weaviate_Manager import WEAVIATE_CLIENT
from app.sevices.get_models import OLLAMA_EMBEDDING, OPENAI_CHAT
from app.core.settings import settings
from ragas import SingleTurnSample
from app.sevices.ragas_evaluvation import evaluvate
from app.core.logger import get_logger
from app.sevices.redis_Manager import get_redis
import uuid
import numpy as np
import time
import hashlib
import json
logger = get_logger(__name__)

_session_id = None
def get_session_id():
    global _session_id
    if _session_id is None:
        _session_id=uuid.uuid4()
    return _session_id
    
_memory ={}
def get_memory(session_id):
    if session_id not in _memory:
        _memory[session_id] = InMemoryChatMessageHistory()
    return _memory[session_id]

async def invoke(question:str):
    start = time.time()
    redis = await get_redis()
    cache_key = f"invoke:{hashlib.md5(question.strip().lower().encode()).hexdigest()}"
    cached = await redis.get(cache_key)
    if cached:
        data = json.loads(cached)
        logger.info("Cache hit for thw question")
        return data
    else:
        logger.info("Cache not found!")
        
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""You are a Fiancial Examinar and adviser and answer the questions
                                                  from {context} and chat_history,
                                                  Rules:
                                                  - Don't Use external Knowledge
                                                  - Don't assume anything not in context
                                                  - If answer is not clearn present -> say I don't know
                                                  - Cite or stick closely to context
                                                  """),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("""Question : {question}""")

    ])
    logger.info(f"Prompt buliding is comppleted!")
    from langchain_core.runnables import RunnableLambda
    ragas_chain = (
        RunnablePassthrough.assign(
            docs = RunnableLambda(retrieval_wrapper)
        )
        | RunnablePassthrough.assign(
            context = lambda x: x["docs"]["context"],
            sources = lambda x: x["docs"]["sources"]
        )
        |
        RunnablePassthrough.assign(
            answer = prompt | OPENAI_CHAT() | StrOutputParser()
        )
    )
    logger.info(f"Rag chain built!")
    chain = (RunnableWithMessageHistory(
        ragas_chain,
        get_memory,
        input_messages_key="question",
        output_messages_key="answer",
        history_messages_key="chat_history"
        
    ))
    logger.info(f"Main  chain built & Invoking the Chain")
    response =await chain.ainvoke({"question":question},config={"configurable":{"session_id":get_session_id()}})
    logger.info(f"Response recieved")
    

    sample = SingleTurnSample(
    user_input=question,
    response=response["answer"],
    retrieved_contexts=[info["context"] for info in response["sources"]]
    )
    logger.info(f"Invoking RAGAS Evaluation")
    results = await evaluvate(sample)
    elaspsed_duration = round(time.time()-start,3)
    final_result = {
        "time_taken":elaspsed_duration,
        "question":question,
        "answer":response["answer"],
        "source":response["sources"],
        "session_id": str(get_session_id()),
        **results
    }
    await redis.setex(cache_key,settings.REDIS_TTL, json.dumps(final_result))
    
    return final_result