import sys
import json
import os

import html
import html2text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatMessagePromptTemplate, ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from collections.abc import Iterable
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.output_parsers import PydanticOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai.lib import ai_defaults, lib_model, lib_doc_vectors, prompts, lc_logger
from ai.lib.loaders import wp_loader

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


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_chain(chain, prompt, input_obj):
    # logger.debug(f"Prompting with: \n{prompt.format(**input_obj)}")

    res = chain.invoke(input_obj, config={
        'callbacks': [oai, lmd]
    })

    logger.debug(oai)
    return res


def client_profile_expert(client_name, expert_data, latest_message, last_10_lines):    
    llm = lib_model.get_json_fast_llm()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompts.profile_prompt),
        HumanMessagePromptTemplate.from_template("Last 10 messages:\n{last_messages}"),
        HumanMessagePromptTemplate.from_template("Existing beliefs:\n{existing_beliefs}"),
        HumanMessagePromptTemplate.from_template("New message: \"{new_message}\""),
    ])

    chain = (
        prompt
        | llm
        | PydanticOutputParser(pydantic_object=ResponseObject)
    )

    existing_beliefs = expert_data.setdefault("beliefs", [])

    res = call_chain(chain, prompt, {
        'client_name': client_name,
        'last_messages': "\n".join(last_10_lines),
        'existing_beliefs': json.dumps({"beliefs": existing_beliefs}, indent=4),
        'new_message': latest_message
    })

    logger.info(res)
    new_beliefs = res.beliefs
    if len(new_beliefs) == 1 and new_beliefs[0] == "No new beliefs":
        logger.debug(f"No new beliefs")
        return

    existing_beliefs.extend(new_beliefs)
    logger.info(f"BELIEFS: \n{existing_beliefs}")

def main():
    chat_path = sys.argv[1]
    client_name = sys.argv[2]

    logger.debug("Loading model")

    lib_model.init(os.getenv("SMART_OPENAI_MODEL"), os.getenv("FAST_OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), os.getenv("PGVECTOR_CONNECTION_STRING"), os.getenv("RECORDMANAGER_CONNECTION_STRING"), temp=os.getenv("OPENAI_TEMPERATURE"))

    logger.debug("Done loading")


    # Load chat_path file into an array of strings, one per line
    with open(chat_path, 'r') as file:
        chat_lines = file.readlines()
    chat_lines = [line.strip() for line in chat_lines]

    experts = {'client_profile': {'expert_fn': client_profile_expert, 'notes': []}}

    for idx, line in enumerate(chat_lines):
        idx_bottom = idx-11 if idx-11 > 0 else 0
        latest_message = line
        last_10_lines = chat_lines[idx_bottom:idx-1]

        
        for expert, expert_data in experts.items():
            logger.debug(f"Invoking expert {expert}")
            expert_fn = expert_data['expert_fn']
            expert_fn(client_name, expert_data, latest_message, last_10_lines)



        # if line.startswith("Dan:"):
        #     chat_lines[idx] = line.replace("Dan:", "HUMAN:")
        # elif line.startswith("AI:"):
        #     chat_lines[idx] = line.replace("AI:", "AI:")




if __name__ == "__main__":
    main()