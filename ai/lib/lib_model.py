import logging
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
from langchain.vectorstores.pgvector import PGVector
import pinecone

_vectordb = None
_embedding = None
_pinecone_index = None
_llm = None
_embedding = None
_db = None


def init(model_name, api_key, db_connection_string, temp=0.5):
    global _llm
    global _embedding
    global _db

    logger = logging.getLogger(__name__)

    if _llm:
        logger.warning("LLM already initialized, skipping")
        return _llm

    _llm = ChatOpenAI(model_name=model_name, temperature=temp)
    _embedding = OpenAIEmbeddings(openai_api_key=api_key, timeout=30)
    _db = initialize_db(db_connection_string)

    return _llm


def get_embedding_fn():
    global _embedding

    logger = logging.getLogger(__name__)

    if not _embedding:
        logger.error("Embedding not initialized, call init() first")
        raise Exception("Embedding not initialized, call init() first")
    
    return _embedding

def initialize_db(db_connection_string, db_collection_name="docs"):
    global _db

    if _db:
        raise Exception("DB already initialized")

    _db = PGVector(
        embedding_function=get_embedding_fn(),
        collection_name=db_collection_name,
        connection_string=db_connection_string
    )

    return _db
    
def get_vectordb():
    return _db

def get_llm():
    global _llm

    logger = logging.getLogger(__name__)

    if not _llm:
        logger.error("LLM not initialized, call init() first")
        raise Exception("LLM not initialized, call init() first")

    return _llm