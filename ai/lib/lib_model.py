import logging
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.globals import set_llm_cache
from langchain.indexes import SQLRecordManager, index
from langchain.cache import SQLiteCache
from langchain.vectorstores.pgvector import PGVector
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
import pinecone

_vectordb = None
_embedding = None
_pinecone_index = None
_llm = None
_json_llm = None
_embedding = None
_db = None
_record_manager = None




def init(model_name, api_key, db_connection_string, record_manager_connection_string, temp=0.5):
    global _llm
    global _embedding
    global _db
    global _record_manager
    global _json_llm

    logger = logging.getLogger(__name__)

    if _llm:
        logger.warning("LLM already initialized, skipping")
        return _llm

    _llm = ChatOpenAI(model_name=model_name, temperature=temp)
    _embedding = OpenAIEmbeddings(openai_api_key=api_key, timeout=30)
    _db = initialize_db(db_connection_string, record_manager_connection_string)
    set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    _json_llm = ChatOpenAI(model_name=model_name, temperature=temp, timeout=30).bind(
        response_format= {
            "type": "json_object"
        }
    )


def get_embedding_fn():
    global _embedding

    logger = logging.getLogger(__name__)

    if not _embedding:
        logger.error("Embedding not initialized, call init() first")
        raise Exception("Embedding not initialized, call init() first")
    
    return _embedding

def initialize_db(db_connection_string, record_manager_connection_string, db_collection_name="docs"):
    global _db
    global _record_manager

    if _db:
        raise Exception("DB already initialized")

    _db = PGVector(
        embedding_function=get_embedding_fn(),
        collection_name=db_collection_name,
        connection_string=db_connection_string
    )

    namespace = f"pgvector/{db_collection_name}"
    _record_manager = SQLRecordManager(namespace, db_url=record_manager_connection_string)

    _record_manager.create_schema()

    return _db

def get_record_manager():
    global _record_manager

    logger = logging.getLogger(__name__)

    if not _record_manager:
        logger.error("Record manager not initialized, call initialize_db() first")
        raise Exception("Record manager not initialized, call initialize_db() first")

    return _record_manager
    
def get_vectordb():
    return _db

def get_llm():
    global _llm

    logger = logging.getLogger(__name__)

    if not _llm:
        logger.error("LLM not initialized, call init() first")
        raise Exception("LLM not initialized, call init() first")

    return _llm

def get_json_llm():
    global _json_llm

    logger = logging.getLogger(__name__)

    if not _json_llm:
        logger.error("JSON LLM not initialized, call init() first")
        raise Exception("JSON LLM not initialized, call init() first")

    return _json_llm
