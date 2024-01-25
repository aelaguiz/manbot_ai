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


from collections.abc import Iterable
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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

class ResponseObject(BaseModel):
    beliefs: List[str]


# @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_chain(chain, input_obj):
    # logger.debug(f"Prompting with: \n{prompt.format(**input_obj)}")

    res = chain.invoke(input_obj, config={
        'callbacks': [oai, lmd]
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

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=prompts.profile_prompt)), 
        # MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template("Last 10 messages:\n{last_messages}"),
        HumanMessagePromptTemplate.from_template("Existing beliefs:\n{existing_beliefs}"),
        HumanMessagePromptTemplate.from_template("New message: \"{input}\""),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])
    tools = []
    a = (
        {
            "input": lambda x: x['input'],
            "client_name": lambda x: x['client_name'],
            "last_messages": lambda x: x['last_messages'],
            "existing_beliefs": lambda x: x['existing_beliefs'],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            )
        }
        | prompt
        | llm
        | OpenAIToolsAgentOutputParser()
    )
    
    print(a)
    experts = {
        'client_profile': AgentExecutor(agent=a, tools=tools, verbose=True)
    }

    for idx, line in enumerate(chat_lines):
        idx_bottom = idx-11 if idx-11 > 0 else 0
        idx_top = idx-1 if idx > 0 else 0
        latest_message = line
        last_10_lines = chat_lines[idx_bottom:idx_top]
        print(f"Chat_lines from {idx_bottom} to {idx_top}")

        
        for expert_name, expert_agent in experts.items():
            logger.debug(f"Invoking expert {expert_name}")
            call_chain(expert_agent, {
                'client_name': client_name,
                'existing_beliefs': json.dumps({"beliefs": []}, indent=4),
                'last_messages': "\n".join(last_10_lines),
                'input': latest_message
            })

        break


        # if line.startswith("Dan:"):
        #     chat_lines[idx] = line.replace("Dan:", "HUMAN:")
        # elif line.startswith("AI:"):
        #     chat_lines[idx] = line.replace("AI:", "AI:")




if __name__ == "__main__":
    main()