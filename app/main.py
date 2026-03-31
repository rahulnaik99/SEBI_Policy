from fastapi import FastAPI
from app.Ingestion.ingestion import load_pdf 
from app.sevices.chat_bot import invoke
from app.graph.graph import  graph
app =FastAPI(
    title="SEBI Chat Bot"
)

@app.get("/health")
def health():
    return {"health" : "ok"}

@app.post("/ingest")
def ingest():
    return {"Total Ingested files" : load_pdf()}

@app.post("/ask")
async def ask(question):
    return await graph.ainvoke({"question": question})