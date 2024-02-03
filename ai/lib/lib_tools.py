from langchain.tools import BaseTool, StructuredTool, tool

import datetime

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever

from langchain.tools import Tool

import logging
logger = logging.getLogger(__name__)

class RetrieverInput(BaseModel):
    """Input to the retriever."""

    query: str = Field(description="query to look up in retriever")

class DocumentRetrieverInput(BaseModel):
    """Input to the retriever."""

    document: str = Field(description="history of communication with the user that can be used to find relevant docs")

def create_retriever_tool(
    retriever: BaseRetriever, name: str, description: str
) -> Tool:
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """

    def _get_relevant_documents(query: str) -> str:
        # print(f"Getting relevant documents for query: {query}")
        docs = retriever.get_relevant_documents(query)

        logger.info(f"Getting relevant documents for query {query}")

        for doc in docs:
            # created_at = doc.metadata['created_at']
            # # Convert current time to datetime for comparison
            # now = datetime.datetime.now(created_at.tzinfo)  # Preserving timezone of created_at if it has one
            # # Calculate the difference in hours
            # hours_ago = int((now - created_at).total_seconds() / 3600)

            created_at_timestamp = doc.metadata['created_at']
            created_at = datetime.datetime.fromtimestamp(created_at_timestamp)

            # Get the current time
            # If created_at is timezone-aware, use the same timezone for 'now'
            if created_at.tzinfo:
                now = datetime.datetime.now(created_at.tzinfo)
            else:
                now = datetime.datetime.now()

            # Calculate the difference
            time_difference = now - created_at
            hours_difference = int(time_difference.total_seconds() / 3600)
            doc.page_content = f"CREATED {hours_difference} HOURS AGO\n\n{doc.page_content}"
            logger.debug(f"DOC {doc.metadata['name']} {doc.metadata['created_at']} {doc.metadata['source']} {doc.metadata['type']} {doc.page_content[:100]}\n\n")
            # logger.debug(f"DOC {doc.metadata['name']} {doc.metadata['created_at']} {doc.metadata['last_accessed_at']} {doc.metadata['source']} {doc.metadata['type']} {doc.metadata['buffer_idx']} {doc.page_content[:100]}\n\n")

        return docs

    return Tool(
        name=name,
        description=description,
        func=_get_relevant_documents,
        coroutine=retriever.aget_relevant_documents,
        args_schema=DocumentRetrieverInput,
    )
