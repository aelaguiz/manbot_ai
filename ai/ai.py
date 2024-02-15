"""
This is the main interface to the library


The idea is that the web app will have a session id representing a single user, then a chat id representing an individual chat session

Each chat has a the chat history, anything necessary for the langchain memory we will just call chat chat state

Basically our AI library is going to have to inde on session and chat and its going to have to return the latest model reply as well as the chat state
The chat state has to be json serializable

it's also likely that at the beginning of a chat they there will be some context being passed through along with the initial user question

"""
from langchain.prompts import MessagesPlaceholder
from langchain.chains import LLMChain, ConversationChain
from operator import itemgetter

from langchain.callbacks.tracers import ConsoleCallbackHandler

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.agents import AgentExecutor

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)



from langchain.agents import OpenAIFunctionsAgent



from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import json
import logging

from .lib import lib_model, lc_logger, prompts, lib_tools, lib_retrievers, lib_formatters


def get_chat_reply(user_input, session_id, chat_id, chat_context=None, initial_messages=None):
    """
    Main interface for handling chat sessions with the AI.

    Args:
        session_id (str): Unique identifier for the user's session.
        chat_id (str): Unique identifier for the individual chat within the session.
        chat_context (dict, optional): The current chat context object. None if it's a new chat.

    Returns:
        tuple: A tuple containing:
            - reply (str): The AI model's reply to the latest message in the chat.
            - new_chat_context (dict): Updated chat context after processing the latest message.

    Raises:
        ChatError: An error occurred during chat processing.
    """



    return res, new_chat_context