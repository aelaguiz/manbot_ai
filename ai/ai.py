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

class ChatError(Exception):
    """
    Custom exception for errors encountered during chat processing.

    Attributes:
        message (str): Explanation of the error.
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(prompts.main_ai_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)


def debug_chain(obj):

    # obj['retrieval_context'] = get_buffer_string(obj['history']) + "\nHuman: " + obj['input']
    logger = logging.getLogger(__name__)
    # logger.info(f"Debug chain: {obj}")

    return obj

def make_retrieval_context(obj):
    return get_buffer_string(obj['history']) + "\nHuman: " + obj['input']

    # return obj

    
def simple_get_chat_reply(user_input):
    logger = logging.getLogger(__name__)
    logger.debug(f"AI: simple_get_chat_reply called with user_input: {user_input}")
    llm = lib_model.get_smart_llm()

    vectordb = lib_model.get_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""Your task is to list out each document in the relevant documents below and explain the conclusions/learnings contained in the document and how it is relevant to the user's query:\n\n{relevant_docs}"""),
        HumanMessagePromptTemplate.from_template("Human: {input}")
    ])

    chain = (
        {
            "input": RunnablePassthrough(),
            "relevant_docs": itemgetter("input") | retriever | lib_formatters.format_docs
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    res = chain.invoke({"input": user_input}, config={'callbacks': [lc_logger.LlmDebugHandler()]})

    return res

def get_memory(session_id, chat_id, chat_context=None, initial_messages=None):
    if chat_context:
        retrieve_from_db = json.loads(chat_context)
        retrieved_messages = messages_from_dict(retrieve_from_db)
        retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
        retrieved_memory = ConversationBufferWindowMemory(chat_memory=retrieved_chat_history, input_key="input", return_messages=True, k=10)

        memory = retrieved_memory
    else:
        if initial_messages:
            translated_messages = []

            for input_message in initial_messages:
                output_message = {
                    "type": input_message["type"].lower(),
                    "data": {
                        "content": input_message["text"],
                        "additional_kwargs": {},
                        "type": input_message["type"].lower(),
                        "example": False
                    }
                }

                translated_messages.append(output_message)


            retrieved_chat_history = ChatMessageHistory(messages=messages_from_dict(translated_messages))
            retrieved_memory = ConversationBufferWindowMemory(chat_memory=retrieved_chat_history, input_key="input", output_key="output", return_messages=True, k=10)

            memory = retrieved_memory
        else:
            memory = ConversationBufferWindowMemory(input_key="input", output_key="output", return_messages=True, k=10)

    return memory

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
    logger = logging.getLogger(__name__)
    logger.debug(f"AI: get_chat_reply called with user_input: {user_input}, session_id: {session_id}, chat_id: {chat_id}, chat_context: {chat_context}")
    # try:
    llm = lib_model.get_smart_llm()
    lmd = lc_logger.LlmDebugHandler()

    memory = get_memory(session_id, chat_id, chat_context, initial_messages)

    vectordb = lib_model.get_vectordb()
    whatsapp_retriever = lib_retrievers.get_retriever(vectordb, 3, type_filter="whatsapp_chat")
    discord_retriever = lib_retrievers.get_retriever(vectordb, 3, type_filter="discord")

    agent_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=prompts.agent_prompt)), 
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])


    book_tool = lib_tools.create_retriever_tool(
        lib_retrievers.get_retriever(vectordb, 5, type_filter="book"),
        "book_search",
        "Search books",
    )

    whatsapp_tool = lib_tools.create_retriever_tool(
        lib_retrievers.get_retriever(vectordb, 5, type_filter="whatsapp_chat"),
        "coaching_search",
        "Find coaching conversations with Robbie Kramer that are relevant to this user's challenges. Past in the chat history with the client including the latest prompt.",
    )

    style_tool = lib_tools.create_style_tool()

    tools = [book_tool, whatsapp_tool]
    llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    runnable_memory = RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )

    # "related_whatsapp_conversations": make_retrieval_context | whatsapp_retriever | format_docs,
    # "related_discord_conversations": make_retrieval_context | discord_retriever | format_docs,

    oai = lib_model.get_oai()
    chain = (
        runnable_memory
        | {
            "input": lambda x: x['input'],
            "chat_history": lambda x: x['history'],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
            )
        }
        | debug_chain
        | agent_prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=chain, tools=tools, callbacks=[oai], verbose=True)

    # main_agent = RunnableWithMessageHistory(
    #     agent_executor,
    #     # This is needed because in most real world scenarios, a session id is needed
    #     # It isn't really used here because we are using a simple in memory ChatMessageHistory
    #     lambda session_id: memory.chat_memory,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    # )

    logger.debug(f"Invoking agent with user_input: {user_input}")
    reply = agent_executor.invoke({"input": user_input}, config={'callbacks': [oai, lmd]})
    logger.debug(oai)

    # # extracted_messages = convo.memory.chat_memory.messages
    # # ingest_to_db = messages_to_dict(extracted_messages)
    # # new_chat_context = json.dumps(ingest_to_db)

    # # input key outpu tkey ChatMessageHistory
    # print(memory.outputs)
    # logger.debug(reply)


    res2 = style_tool.invoke({"chat_history": user_input, "message": reply['output']}, config={'callbacks': [lmd]})

    logger.debug(f"STYLE TOOL: {res2}")


    memory.save_context({"input": user_input}, {"output": res2})

    extracted_messages = memory.chat_memory.messages
    ingest_to_db = messages_to_dict(extracted_messages)
    new_chat_context = json.dumps(ingest_to_db)



    return res2, new_chat_context

    # except Exception as e:
    #     import traceback
    #     traceback.print_exc()
    #     raise ChatError(f"Failed to process chat: {str(e)}")