from app.graph.state import RAGstate
from app.core.settings import settings
import numpy as np
from app.sevices.weaviate_Manager import WEAVIATE_CLIENT
from app.sevices.get_models import OLLAMA_EMBEDDING
from sentence_transformers import cross_encoder
from app.graph.state import RAGstate
def mmr(query_vec,doc_vec,lam=0.5):
    q_array = (np.array(query_vec))
    d_array = np.array(doc_vec)
    
    q_norm = q_array / np.linalg.norm(q_array)
    d_norm = d_array/np.linalg.norm(d_array, axis=1, keepdims=True)
    
    relavance = d_norm @ q_norm
    candidates = list(range(len(d_array)))
    selected= []
    for _ in range(settings.Top_MMR):
        if not selected:
            best = max(candidates, key=lambda x : relavance[x])
            
        else:
            sel_vec = d_norm[selected]
            best = max(candidates, key=lambda x : lam * relavance[x] - (1- lam) * np.max(d_norm[x] @ sel_vec.T))

        selected.append(best)
        candidates.remove(best)
    return selected

async def cross_encoder(question,mmr_docs):
    from  sentence_transformers import CrossEncoder
    _cross_encoder = CrossEncoder(settings.ENCODER)
    pair = [(question,doc.properties["text"]) for doc in mmr_docs]
    score = _cross_encoder.predict(pair)  
    rerank = sorted(zip(score,mmr_docs), key=lambda x: x[0], reverse=True)
    return [docs for _,docs in rerank[:settings.Top_Encoder]]

async def retrieval(state:RAGstate):
    question = state.get("question")
    client= WEAVIATE_CLIENT()
    embbeding_model = OLLAMA_EMBEDDING()
    query_vec = embbeding_model.embed_query(question)
    results = client.collections.get(settings.COLLECTION).query.hybrid(
        limit=settings.Top_Retrieval,
        query=question,
        vector=query_vec,
        return_properties=["text","source","title","chunk_number", "total_chunks","pages_number","total_pages"],
        include_vector=True
    )
    docs = results.objects
    doc_vectors = [result.vector["default"] for result in  results.objects]
    indexs = mmr(query_vec,doc_vectors)
    mmr_docs = [docs[i] for i in indexs]
    top_docs = await cross_encoder(question,mmr_docs)
    context = "\n\n".join([doc.properties["text"] for doc in top_docs])
    source = [{"context":doc.properties["text"],
               "source": doc.properties["source"]
               } for doc in top_docs]   
    client.close()
    return {
        "context":context,
        "sources":source
    }
    

