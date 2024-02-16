import sys
import os

# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import os
import sys
import copy
import json
import random
import logging
import dotenv
from ai import ai, init
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
import logging.config
import logging

import time

dotenv.load_dotenv()

# Define the configuration file path based on the environment
config_path = os.getenv('LOGGING_CONF_PATH')

# Use the configuration file appropriate to the environment
logging.config.fileConfig(config_path)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.connection").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
logging.getLogger("openai._base_client").setLevel(logging.CRITICAL)

from ai.lib import lib_model, lc_logger, lib_conversation, lib_retrievers, lib_formatters
from prompt_toolkit.history import FileHistory


logger = logging.getLogger(__name__)

CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")
RECORDMANAGER_CONNECTION_STRING = os.getenv("RECORDMANAGER_CONNECTION_STRING")


init(os.getenv("SMART_OPENAI_MODEL"), os.getenv("SMART_OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), CONNECTION_STRING, RECORDMANAGER_CONNECTION_STRING, temp=os.getenv("OPENAI_TEMPERATURE"))

vectordb = lib_model.get_vectordb()
retriever = lib_retrievers.get_retriever(vectordb, 5, type_filter=sys.argv[1])


def process_command(user_input):
    res = lib_formatters.format_docs(retriever.get_relevant_documents(user_input))
    print(res)



def main():
    lib_conversation.init_conversation()
    bindings = KeyBindings()
    history = FileHistory('./gpt_prompt_history.txt')  # specify the path to your history file

    while True:
        multiline = False

        while True:
            try:
                if not multiline:
                    # Single-line input mode
                    line = prompt('Human: ', key_bindings=bindings, history=history)
                    if line.strip().lower() == 'quit':
                        return  # Exit the CLI
                    else:
                        chat_context = process_command(line)
                        break
            except EOFError:
                return


if __name__ == "__main__":
    main()
