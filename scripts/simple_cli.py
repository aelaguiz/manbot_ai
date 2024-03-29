import sys
import os

# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

from ai.lib import lib_model, lc_logger


logger = logging.getLogger(__name__)

CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")
RECORDMANAGER_CONNECTION_STRING = os.getenv("RECORDMANAGER_CONNECTION_STRING")


init(os.getenv("SMART_OPENAI_MODEL"), os.getenv("SMART_OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), CONNECTION_STRING, RECORDMANAGER_CONNECTION_STRING, temp=os.getenv("OPENAI_TEMPERATURE"))

def process_command(user_input):
    # print(f"Asking AI about: {user_input}")

    reply  = ai.simple_get_chat_reply(user_input)

    # res = convo.predict(input=user_input)
    print(f"\nai: {reply}\n")
    # print(new_context)


def main():
    bindings = KeyBindings()

    while True:
        multiline = False

        while True:
            try:
                if not multiline:
                    # Single-line input mode
                    line = prompt('Human: ', key_bindings=bindings)
                    if line.strip() == '"""':
                        multiline = True
                        continue
                    elif line.strip().lower() == 'quit':
                        return  # Exit the CLI
                    else:
                        process_command(line)
                        break
                else:
                    # Multiline input mode
                    line = prompt('... ', multiline=True, key_bindings=bindings)
                    process_command(line)
                    multiline = False
            except EOFError:
                return


if __name__ == "__main__":
    main()
