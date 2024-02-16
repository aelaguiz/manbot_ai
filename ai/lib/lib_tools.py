from langchain.tools import BaseTool, StructuredTool, tool

import datetime

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.tools import Tool

import logging

class RetrieverInput(BaseModel):
    """Input to the retriever."""

    query: str = Field(description="query to look up in retriever")

class DocumentRetrieverInput(BaseModel):
    """Input to the retriever."""

    chat_history: str = Field(description="Chat history with the user")

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
        logger = logging.getLogger(__name__)

        # print(f"Getting relevant documents for query: {query}")
        logger.info(f"TOOL: Getting relevant documents for query: {query}")
        docs = retriever.get_relevant_documents(query)
        for doc in docs:
            logger.debug(f"DOC {doc.metadata['source']} {doc.metadata['type']} {doc.page_content}\n\n")

        # logger.info(f"Getting relevant documents for query {query}")

        # for doc in docs:
        #     # created_at = doc.metadata['created_at']
        #     # # Convert current time to datetime for comparison
        #     # now = datetime.datetime.now(created_at.tzinfo)  # Preserving timezone of created_at if it has one
        #     # # Calculate the difference in hours
        #     # hours_ago = int((now - created_at).total_seconds() / 3600)

        #     created_at_timestamp = doc.metadata['created_at']
        #     created_at = datetime.datetime.fromtimestamp(created_at_timestamp)

        #     # Get the current time
        #     # If created_at is timezone-aware, use the same timezone for 'now'
        #     if created_at.tzinfo:
        #         now = datetime.datetime.now(created_at.tzinfo)
        #     else:
        #         now = datetime.datetime.now()

        #     # Calculate the difference
        #     time_difference = now - created_at
        #     hours_difference = int(time_difference.total_seconds() / 3600)
        #     doc.page_content = f"CREATED {hours_difference} HOURS AGO\n\n{doc.page_content}"
        #     logger.debug(f"DOC {doc.metadata['name']} {doc.metadata['created_at']} {doc.metadata['source']} {doc.metadata['type']} {doc.page_content[:100]}\n\n")
        #     # logger.debug(f"DOC {doc.metadata['name']} {doc.metadata['created_at']} {doc.metadata['last_accessed_at']} {doc.metadata['source']} {doc.metadata['type']} {doc.metadata['buffer_idx']} {doc.page_content[:100]}\n\n")

        return docs

    return Tool(
        name=name,
        description=description,
        func=_get_relevant_documents,
        coroutine=retriever.aget_relevant_documents,
        args_schema=DocumentRetrieverInput,
    )

class MimicRobbieInput(BaseModel):
    """Input to the retriever."""

    chat_history: str = Field(description="Chat history with the user, including the latest prompt")
    message: str = Field(description="The message to mimic Robbie's style")

def create_style_tool(
) -> StructuredTool:

    from . import lib_model, lib_retrievers, lc_logger, lib_formatters

    llm = lib_model.get_smart_llm()
    vectordb = lib_model.get_vectordb()
    retriever = lib_retrievers.get_retriever(vectordb, 2, type_filter="whatsapp_chat")
    oai = lib_model.get_oai()

    def make_retrieval_context(obj):
        logger = logging.getLogger(__name__)
        logger.debug(f"Making retrievel call with: {obj}")

        # return get_buffer_string(obj['message'])
        return obj['message']

    def _mimic_style(chat_history: str, message: str) -> str:
        logger = logging.getLogger(__name__)
        logger.debug(f"Mimic Robbie's style for message: {message}")

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""Your task is to reformat a chat message to match the style and tone of Robbie Kramer.

## Instructions 
1. Review the message and the chat history to understand the context of the conversation
2. Use the samples of relevant conversations involving Robbie Kramer to understand his style and tone
3. Reformat the message to match Robbie's style and tone
4. Strip all markdown or other formatting that would maket he message look like it was generated by a machine

## Conversation Samples
{conversation_samples}

## Chat History:
{chat_history}

**Message to rewrite to mimic Robbie's style:** {message}
"""),
        ])

        chain = (
            {
                "chat_history": lambda x: x["chat_history"],
                "conversation_samples": make_retrieval_context | retriever | lib_formatters.format_docs,
                "message": lambda x: x["message"]
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        res = chain.invoke({
            "chat_history": chat_history,
            "message": message 
        }, config={'callbacks': [lc_logger.LlmDebugHandler(), oai]})

        logger.debug(f"REWRITE\n: {message} -> {res}")
        logger.debug(oai)

        return res

    return StructuredTool(
        name="mimic_robbie",
        description="Mimic's Robbies style, turning the input into messages that mimic Robbie's style",
        func=_mimic_style,
        args_schema=MimicRobbieInput,
    )
