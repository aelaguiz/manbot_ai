import sys
import json
import os

import html
import html2text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatMessagePromptTemplate, ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

from collections.abc import Iterable
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import numpy as np

from operator import itemgetter

# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai.lib import ai_defaults, lib_model, lib_doc_vectors, prompts, lc_logger
from ai.lib.loaders import wp_loader
from ai.lib.agents import ExpertAgent
from langchain.agents import OpenAIFunctionsAgent

import logging
import logging.config
import dotenv
import os

dotenv.load_dotenv()


# Define the configuration file path based on the environment
config_path = os.getenv('LOGGING_CONF_PATH')

# Use the configuration file appropriate to the environment
logging.config.fileConfig(config_path)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.connection").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
logging.getLogger("openai._base_client").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field
from typing import Dict, List

oai = OpenAICallbackHandler()
lmd = lc_logger.LlmDebugHandler()

class ClientProfileResponseObject(BaseModel):
    beliefs: List[str]

class ClientGoalsResponseObject(BaseModel):
    specific_problems_or_goals: List[str]


# @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_chain(chain, input_obj):
    # logger.debug(f"Prompting with: \n{prompt.format(**input_obj)}")

    res = chain.invoke(input_obj, config={
        'callbacks': [oai]
    })

    logger.debug(oai)
    return res



def main():
    chat_path = sys.argv[1]
    client_name = sys.argv[2]

    logger.debug("Loading model")

    lib_model.init(os.getenv("SMART_OPENAI_MODEL"), os.getenv("FAST_OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), os.getenv("PGVECTOR_CONNECTION_STRING"), os.getenv("RECORDMANAGER_CONNECTION_STRING"), temp=os.getenv("OPENAI_TEMPERATURE"))

    logger.debug("Done loading")

    lmd = lc_logger.LlmDebugHandler()  
    # db = lib_docdb.get_docdb()
    llm = lib_model.get_fast_llm()


    # Load chat_path file into an array of strings, one per line
    with open(chat_path, 'r') as file:
        chat_lines = file.readlines()
    chat_lines = [line.strip() for line in chat_lines]

    profile_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=prompts.client_profile_prompt)), 
        # ChatPromptTemplate.from_template(f"**AI Conversation History**:\n"),
        # MessagesPlaceholder(variable_name='history'),
        HumanMessagePromptTemplate.from_template("\n**Last 10 client/coach messages**:\n{last_messages}"),
        HumanMessagePromptTemplate.from_template("\n**Existing beliefs**:\n{existing_beliefs}"),
        HumanMessagePromptTemplate.from_template("**New messages**: \"{input}\""),
    ])

    profile_chat_history = ChatMessageHistory()
    profile_memory = ConversationBufferMemory(chat_memory=profile_chat_history, input_key="input", output_key="output", return_messages=True)
    profile_loaded_memory = RunnablePassthrough.assign(
        history=RunnableLambda(profile_memory.load_memory_variables) | itemgetter("history"),
    )

    client_profile_agent = (
        # profile_loaded_memory
        # | 
        {
            "input": lambda x: x['input'],
            # "history": lambda x: x['history'],
            "client_name": lambda x: x['client_name'],
            "last_messages": lambda x: x['last_messages'],
            "existing_beliefs": lambda x: x['existing_beliefs']
        }
        | profile_prompt
        | llm
        | StrOutputParser()
    )

    goals_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=prompts.client_goals_prompt)), 
        # ChatPromptTemplate.from_template(f"**AI Conversation History**:\n"),
        # MessagesPlaceholder(variable_name='history'),
        HumanMessagePromptTemplate.from_template("\n**Last 10 client/coach messages**:\n{last_messages}"),
        HumanMessagePromptTemplate.from_template("\n**Existing goals or insights**:\n{existing_beliefs}"),
        HumanMessagePromptTemplate.from_template("**New messages**: \"{input}\""),
    ])

    goals_chat_history = ChatMessageHistory()
    goals_memory = ConversationBufferMemory(chat_memory=goals_chat_history, input_key="input", output_key="output", return_messages=True)
    goals_loaded_memory = RunnablePassthrough.assign(
        history=RunnableLambda(goals_memory.load_memory_variables) | itemgetter("history"),
    )
    client_goals_agent = (
        # goals_loaded_memory
        # | 
        {
            "input": lambda x: x['input'],
            # "history": lambda x: x['history'],
            "client_name": lambda x: x['client_name'],
            "last_messages": lambda x: x['last_messages'],
            "existing_beliefs": lambda x: x['existing_beliefs']
        }
        | goals_prompt
        | llm
        | StrOutputParser()
    )
    
    experts = {
        'client_profile': {'agent': client_profile_agent, 'beliefs': [], 'memory': profile_memory},
        # 'client_goals': {'agent': client_goals_agent, 'beliefs': [], 'memory': goals_memory}
    }

    summarize_notes_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=prompts.summarize_notes)), 
        HumanMessagePromptTemplate.from_template("\n**Existing coach notes**:\n{input}"),
    ])

    summarize_notes_chain = (
        # goals_loaded_memory
        # | 
        {
            "input": lambda x: x['input'],
            "client_name": lambda x: x['client_name'],
        }
        | summarize_notes_prompt
        | llm
        | StrOutputParser()
    )

    history_size = 10
    chunk_size = 6
    for idx in range(0, len(chat_lines), chunk_size):
        idx_bottom = idx-history_size if idx-history_size > 0 else 0
        last_10_lines = chat_lines[idx_bottom:idx]
        latest_messages = chat_lines[idx:idx+chunk_size]

        
        for expert_name, expert_data in experts.items():
            expert_agent = expert_data['agent']
            expert_beliefs = expert_data['beliefs']
            expert_memory = expert_data['memory']
            logger.debug(f"Invoking expert {expert_name}")
            reply = call_chain(expert_agent, {
                'client_name': client_name,
                'existing_beliefs': "\n\n".join(expert_beliefs),
                'last_messages': "\n".join(last_10_lines),
                'input': "\n".join(latest_messages)
            })
            logger.debug(f"Reply: {reply}")
            # expert_memory.save_context({
            #     'client_name': client_name,
            #     'existing_beliefs': json.dumps({expert_beliefs_key: expert_beliefs}, indent=4),
            #     'last_messages': "\n".join(last_10_lines),
            #     'input': latest_message
            # }, {"output": reply.model_dump_json()})

            expert_beliefs.append(reply)

            reply = call_chain(summarize_notes_chain, {
                'client_name': client_name,
                'input': "\n".join(expert_beliefs)
            })
            logger.debug(f"Summarized beliefs: {reply}")

            expert_beliefs = [reply]
            expert_data['beliefs'] = expert_beliefs



        # if line.startswith("Dan:"):
        #     chat_lines[idx] = line.replace("Dan:", "HUMAN:")
        # elif line.startswith("AI:"):
        #     chat_lines[idx] = line.replace("AI:", "AI:")




if __name__ == "__main__":
    main()