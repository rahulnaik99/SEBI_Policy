from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.sevices.weaviate_Manager import WEAVIATE_CLIENT
from app.core.settings import settings
from app.sevices.get_models import OLLAMA_EMBEDDING
from weaviate.classes.query import Filter
import glob
import hashlib
import re
from app.core.logger import get_logger
logger = get_logger(__name__)

def file_hasher(texts:str)->str:
    return hashlib.md5(texts.encode()).hexdigest()

def check_for_duplicate(hash)-> bool:
    results =client.collections.get(settings.COLLECTION).aggregate.over_all(
    group_by="hashing"
    )
    for group in results.groups:
        if group.grouped_by.value ==hash:
            return True
    
        
       
def clean_text(page_content:str):
    page_content = page_content.replace("\r","\n")
    page_content = re.sub(r"\n\s*\n+","\n",page_content)
    page_content = re.sub(r"Page\s+\d+\s+of\s+\d+", "", page_content)
    page_content = "\n".join([page.strip() for page in page_content.split("\n") if page.strip()])
    return page_content

def load_pdf():
    client = WEAVIATE_CLIENT()
    embedding_model = OLLAMA_EMBEDDING()
    total_documents=0
    path = "/Users/rahul/Projects/Learning_Projects/Data"
    pdf_files = glob.glob(path+"/*.pdf")

    for i, file in enumerate(pdf_files):
        logger.info(f"Ingesting {i+1} / {len(pdf_files)}")
        loader = PyMuPDFLoader(file)
        document = loader.load()
        hash = file_hasher(texts = "\n".join([page.page_content for page in document]))
        logger.info(f"hashing for file {i+1} is {hash}")
        if check_for_duplicate(hash):
            logger.info(f"File is already Ingested, Skipping Igestion for file {i+1}")
            continue
        for page_num, page in enumerate(document):
            page.metadata.update({
                "page_content":clean_text(page.page_content),
                "hash":hash,
                "total_pages": len(document),
                "page_number":int(page_num+1)
            })
        logger.info(f"chuking started for file {i+1}")
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150,separators=["\n\n", "\n", " ", ""]).split_documents(document)
        for chunk_number, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_number":chunk_number+1,
                "total_chunks":len(chunks)
            })
        logger.info(f"chuking completed for file {i+1}")    
        texts = [clean_text(chunk.page_content) for chunk in chunks]
        logger.info(f"Embedding Started for file {i+1}")   
        vectores = embedding_model.embed_documents(texts)
        logger.info(f"Embedding Completed for file {i+1} with total chunk {len(vectores)}")  
        logger.info(f"Igestion Started for {i+1}")  
        
        for chunk, text, vector in zip(chunks,texts,vectores):
            client.collections.get(settings.COLLECTION).data.insert(
                properties=({
                    "text":text,
                    "source":chunk.metadata.get("source","unknown"),
                    "title": chunk.metadata.get("title","Unknown"),
                    "pages_number":chunk.metadata.get("page_number","Unknown"),
                    "total_pages":chunk.metadata.get("total_pages","unknown"),
                    "chunk_number":chunk.metadata.get("chunk_number","unknown"),
                    "total_chunks":chunk.metadata.get("total_chunks","unknown"),
                    "hashing":chunk.metadata.get("hash","unknown"),
                    "is_active":chunk.metadata.get("is_active","unknown"),
                }),
                vector=vector
            )
        logger.info(f"Igestion Completed for {i+1}")  
        total_documents+=1
    client.close()
    del embedding_model
    return total_documents


            
