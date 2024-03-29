import logging
import logging.config
import dotenv
import os
from rich.logging import RichHandler

dotenv.load_dotenv()


# Define the configuration file path based on the environment
config_path = os.getenv('LOGGING_CONF_PATH')

# Use the configuration file appropriate to the environment
logging.config.fileConfig(config_path)
rich_handler = RichHandler()
logging.getLogger().handlers[0] = rich_handler  # Replace the first handler, assuming it's the console handler

import sys
import os
import typing
from typing import Optional, Union, Literal, AbstractSet, Collection, Any, List
from langchain.text_splitter import split_text_on_tokens

import html
import html2text
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai.lib import lib_model, lib_doc_vectors
from ai.lib.loaders import wp_loader, discord_loader
from langchain.document_loaders import DirectoryLoader

import ai.lib.conversation_splitter

import glob
import random


def main():
    path = sys.argv[1]

    lib_model.init(os.getenv("SMART_OPENAI_MODEL"), os.getenv("FAST_OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), os.getenv("PGVECTOR_CONNECTION_STRING"), os.getenv("RECORDMANAGER_CONNECTION_STRING"), temp=os.getenv("OPENAI_TEMPERATURE"))

    txt_files = list(glob.glob(os.path.join(path, "*.txt")))

    random.shuffle(txt_files)
    total_files = len(txt_files)
    for i, file_path in enumerate(txt_files):
        print(f"Processing file {i+1}/{total_files}: {file_path}")
        loader = discord_loader.DiscordChatLoader(file_path)
        for doc in loader.load():
            text = doc.page_content
            metadata = doc.metadata
            print(f"\n\nAdding document title:{metadata['title']} author:{metadata['author']} guid:{metadata['guid']} {len(text)} first 50 chars: {text[:50]}")
            print(doc.page_content)
            lib_doc_vectors.bulk_add_docs([doc])
            print("Done loading")




if __name__ == "__main__":
    main()