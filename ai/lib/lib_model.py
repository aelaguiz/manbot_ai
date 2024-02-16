import logging
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.globals import set_llm_cache
from langchain.indexes import SQLRecordManager, index
from langchain.cache import SQLiteCache
from langchain.vectorstores.pgvector import PGVector
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.callbacks import OpenAICallbackHandler
import httpx
import dspy

_vectordb = None
_embedding = None
_pinecone_index = None
_fast_llm = None
_json_fast_llm = None
_smart_llm = None
_json_smart_llm = None
_embedding = None
_db = None
_record_manager = None
_oai = OpenAICallbackHandler()

turbo = gpt4 = None

def init(smart_model_name, fast_model_name, api_key, db_connection_string, record_manager_connection_string, temp=0.5):
    global _fast_llm
    global _json_fast_llm
    global _embedding
    global _db
    global _record_manager
    global _smart_llm
    global _json_smart_llm

    global turbo
    global gpt4

    turbo = dspy.OpenAI(fast_model_name, api_key=api_key, temperature=0.7, max_tokens=1000)
    gpt4 = dspy.OpenAI(smart_model_name, api_key=api_key, temperature=0.7, max_tokens=1000)

    dspy.settings.configure(lm=turbo, trace=[])
    import dsp
    dsp.settings.log_openai_usage = True


    logger = logging.getLogger(__name__)
    logger.debug(f"Initializing model")

    if _fast_llm:
        logger.warning("LLM already initialized, skipping")
        return _fast_llm

    _fast_llm = ChatOpenAI(model_name=fast_model_name, temperature=temp)
    _embedding = OpenAIEmbeddings(openai_api_key=api_key, timeout=30, model='text-embedding-3-small')
    _db = initialize_db(db_connection_string, record_manager_connection_string)
    # set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    _json_fast_llm = ChatOpenAI(model_name=fast_model_name, temperature=temp, timeout=httpx.Timeout(15.0, read=60.0, write=10.0, connect=3.0), max_retries=0).bind(
        response_format= {
            "type": "json_object"
        }
    )

    _smart_llm = ChatOpenAI(model_name=smart_model_name, temperature=temp)
    _json_smart_llm = ChatOpenAI(model_name=smart_model_name, temperature=temp, timeout=httpx.Timeout(15.0, read=60.0, write=10.0, connect=3.0), max_retries=0).bind(
        response_format= {
            "type": "json_object"
        }
    )

def get_oai():
    return _oai

def get_embedding_fn():
    global _embedding
    logger = logging.getLogger(__name__)
    if not _embedding:
        logger.error("Embedding not initialized, call init() first")
        raise Exception("Embedding not initialized, call init() first")
    
    return _embedding

def get_smart_llm():
    global _smart_llm

    logger = logging.getLogger(__name__)

    if not _smart_llm:
        logger.error("Smart LLM not initialized, call init() first")
        raise Exception("Smart LLM not initialized, call init() first")

    return _smart_llm

def get_json_smart_llm():
    global _json_smart_llm

    logger = logging.getLogger(__name__)

    if not _json_smart_llm:
        logger.error("JSON Smart LLM not initialized, call init() first")
        raise Exception("JSON Smart LLM not initialized, call init() first")

    return _json_smart_llm

def initialize_db(db_connection_string, record_manager_connection_string, db_collection_name="docs"):
    global _db
    global _record_manager

    if _db:
        raise Exception("DB already initialized")

    logger = logging.getLogger(__name__)
    logger.debug(f"Initializing database {db_connection_string}")

    _db = PGVector(
        embedding_function=get_embedding_fn(),
        collection_name=db_collection_name,
        connection_string=db_connection_string
    )

    namespace = f"pgvector/{db_collection_name}"
    _record_manager = SQLRecordManager(namespace, db_url=record_manager_connection_string)

    logger.debug(f'Creating schema for namespace {namespace}')
    _record_manager.create_schema()

    logger.debug(f"Done initializing database")
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

def get_fast_llm():
    global _fast_llm

    logger = logging.getLogger(__name__)

    if not _fast_llm:
        logger.error("LLM not initialized, call init() first")
        raise Exception("LLM not initialized, call init() first")

    return _fast_llm

def get_json_fast_llm():
    global _json_fast_llm

    logger = logging.getLogger(__name__)

    if not _json_fast_llm:
        logger.error("JSON LLM not initialized, call init() first")
        raise Exception("JSON LLM not initialized, call init() first")

    return _json_fast_llm
