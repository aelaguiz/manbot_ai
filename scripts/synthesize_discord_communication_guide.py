import sys
import json
import os
import time
import random

import html
import html2text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
from ai.lib.loaders import wp_loader, discord_loader

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

from pydantic import BaseModel, Field
from typing import List, Dict

class SampleChatExchange(BaseModel):
    question: str = Field(None, alias="Question")
    interaction: str = Field(None, alias="Interaction")
    response: str = Field(..., alias="Response")

class DiscordChatGuide(BaseModel):
    greetings_and_sign_offs: List[str] = Field(..., alias="Greetings and Sign-offs")
    common_responses: List[str] = Field(..., alias="Common Responses")
    emoji_usage: List[str] = Field(..., alias="Emoji Usage")
    interactive_engagement: List[str] = Field(..., alias="Interactive Engagement")
    sample_chat_exchanges: List[List[str]] = Field(..., alias="Sample Chat Exchanges")

    class Config:
        allow_population_by_field_name = True

def vectorize_texts(texts):
    """Convert a list of texts to TF-IDF vectors."""
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)

def calculate_similarity(tfidf_matrix):
    """Calculate cosine similarity from a TF-IDF matrix."""
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

def is_similar_with_tolerance(ai_response_text, example_text, threshold=0.5):
    """Check if AI response is similar to an example text with a given threshold."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([ai_response_text, example_text])
    
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity[0][0] >= threshold



def is_similar(dict1, dict2):
    # logger.debug(f"Comparing {dict1}\n\n and {dict2}")
    """Recursively compare two dictionaries to determine similarity."""
    if dict1.keys() != dict2.keys():
        # logger.debug(f"Keys don't match: {dict1.keys()} != {dict2.keys()}")
        return True

    for key in dict1:
        # logger.debug(f"Checking key {key}")
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            # logger.debug(f"Comparing two matching {key}...")
            if is_similar(dict1[key], dict2[key]):
                # logger.debug(f"Found similar {key} - {dict1[key]} and {dict2[key]}")
                return True
        elif isinstance(dict1[key], Iterable) and isinstance(dict2[key], Iterable):
            # logger.debug(f"Comparing two lists {key}...")
            for item1, item2 in zip(dict1[key], dict2[key]):
                if is_similar_with_tolerance(item1, item2):
                    # logger.debug(f"Found similar items {item1} and {item2}")
                    return True

        else:
            # logger.debug(f"Comparing records on {key}...")
            if is_similar_with_tolerance(dict1[key], dict2[key]):
                # logger.debug(f"Similar records on {key} - {dict1[key]} and {dict2[key]}")
                return True

    return False

def check_similarity_with_examples(ai_response, example):
    if is_similar(ai_response, example):
        return True
    return False

example_data = DiscordChatGuide.model_validate(ai_defaults.discord_comguide_example).model_dump()

def process_document(doc, chain, lmd):
    for i in range(3):
        try:
            logger.debug(f"Processing {doc.metadata['title']}")
            new_comguide = chain.invoke({
                "input": doc.page_content,
                "subject_name": "robbiekramer",
                "example": ai_defaults.discord_comguide_example_json,
            }, config={'callbacks': []})
            logger.debug(f"Returning {doc.metadata['title']}")
            comguide_dict = new_comguide.model_dump()

            logger.debug(f"Comguide: {json.dumps(comguide_dict, indent=4)}")

            return comguide_dict

            # if check_similarity_with_examples(comguide_dict, example_data):
            #     logger.debug(f"Found similar comguide for {doc.metadata['title']}")
            #     return None
            # else:
            #     logger.debug(f"Found new comguide for {doc.metadata['title']}")
            #     return comguide_dict
        except Exception as e:
            # Handle the exception here
            import traceback
            traceback.print_exc()
            logger.error(f"An error occurred: {e}")
            sleep_time = random.randint(1, 5)
            logging.info(f"Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)

def combine_json_objects(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    combined_dict = {}
    for item in data:
        for key, value in item.items():
            if key in combined_dict:
                if isinstance(value, list):
                    combined_dict[key].extend(value)
                else:
                    for subkey, subvalue in value.items():
                        if subkey in combined_dict[key]:
                            combined_dict[key][subkey].extend(subvalue)
                        else:
                            combined_dict[key][subkey] = subvalue
            else:
                combined_dict[key] = value

    return combined_dict


def main():
    lmd = lc_logger.LlmDebugHandler()

    comguide_path = sys.argv[1]

    print(f"Got comguide path {comguide_path}")
    comguide = combine_json_objects(comguide_path)
    print(comguide.keys())


    lib_model.init(os.getenv("SMART_OPENAI_MODEL"), os.getenv("FAST_OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), os.getenv("PGVECTOR_CONNECTION_STRING"), os.getenv("RECORDMANAGER_CONNECTION_STRING"), temp=os.getenv("OPENAI_TEMPERATURE"))


    llm = lib_model.get_smart_llm()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompts.synthesize_tone_section),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # new_comguide = {}
    # for key in comguide.keys():
    res = chain.invoke({
        "input": comguide,
        "subject_name": "robbiekramer"
    }, config={'callbacks': []})

    # new_comguide[key] = res
    print(res)

    # print(new_comguide)


#     # all_docs = all_docs[:1]
#     guidelog = open('guidelog.txt', 'w')
#     comguides = []
#     num_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4



if __name__ == "__main__":
    main()