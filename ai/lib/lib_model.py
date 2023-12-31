import os
import dotenv
import logging
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
import pinecone
from langchain.vectorstores import Pinecone

dotenv.load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE"))

_vectordb = None
_embedding = None
_pinecone_index = None

def get_embedding_fn():
    global _embedding
    
    if _embedding:
        return _embedding
    
    _embedding = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'), timeout=30)
    return _embedding

def get_vectordb():
    global _vectordb
    global _pinecone_index

    if _vectordb:
        return _vectordb

    logging.info(f"Using Pinecone")
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
    pinecone_index = pinecone.Index(index_name=os.getenv("PINECONE_INDEX_NAME"))

    _vectordb = Pinecone(pinecone_index, get_embedding_fn(), "text")

    return _vectordb

def get_llm():
    return ChatOpenAI(model_name=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)