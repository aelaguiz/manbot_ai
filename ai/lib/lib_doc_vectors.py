# lib_vectordb.py
import os
import logging
from dotenv import load_dotenv
# Assuming some document database abstraction is available as model_abstraction
from .lib_model import get_vectordb, get_record_manager
from langchain_core.documents import Document
from langchain.indexes import SQLRecordManager, index

def bulk_add_docs(docs: list):
    """
    Add multiple langchain Document objects to the vectordb in bulk.

    Parameters:
    docs (list of Document): A list of langchain Document objects to add.
    """
    logger = logging.getLogger(__name__)
    try:
        vectordb = get_vectordb()
        res = index(
            docs,
            get_record_manager(),
            vectordb,
            cleanup=None,
            source_id_key="source"
        )
        logger.info(f"Successfully added {len(docs)} documents in bulk: {res}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error adding documents in bulk: {e}")