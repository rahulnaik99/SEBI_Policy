from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas import experiment, SingleTurnSample
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextEntityRecall
from openai import AsyncOpenAI
from app.core.settings import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
_evaluvation_llm = llm_factory( settings.CHAT_MODEL_OPENAI,client=_client,max_tokens=4096)
_evaluvation_embedding = embedding_factory(model=settings.EMBEDDING_MODEL_OPENAI, client=_client)
_Faithfulness_metrics= Faithfulness(llm=_evaluvation_llm)
_AnswerRelevancy_metrics = AnswerRelevancy(llm=_evaluvation_llm, embeddings=_evaluvation_embedding)
@experiment()
async def evaluvate(sample:SingleTurnSample):
    logger.info("Evaulvationg Faithfulness score")
    Faithfulness_score =await _Faithfulness_metrics.ascore(
        user_input=sample.user_input,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts
        
    )
    logger.info("Answer Relavancy score")
    AnswerRelevancy_score = await _AnswerRelevancy_metrics.ascore(
        user_input=sample.user_input,
        response=sample.response,
    )
    logger.info(f"RAGAS Scores Generated: {Faithfulness_score.value} - {AnswerRelevancy_score.value}")
    return {
        "Faithfullness":round((Faithfulness_score.value)*100,2),
        "Answer_releavency":round((AnswerRelevancy_score.value)*100,2)
    }

