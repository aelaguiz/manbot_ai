import sys
import base64
from requests.auth import HTTPBasicAuth
import requests
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
# logging.getLogger("httpx").setLevel(logging.CRITICAL)
# logging.getLogger("httpcore.connection").setLevel(logging.CRITICAL)
# logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
# logging.getLogger("openai._base_client").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("httpcore.connection").setLevel(logging.DEBUG)
logging.getLogger("httpcore.http11").setLevel(logging.DEBUG)
logging.getLogger("openai._base_client").setLevel(logging.DEBUG)

from ai import ai, init
from ai.lib import lib_model, lc_logger, lib_conversation
from prompt_toolkit.history import FileHistory


logger = logging.getLogger(__name__)

CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")
RECORDMANAGER_CONNECTION_STRING = os.getenv("RECORDMANAGER_CONNECTION_STRING")

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import pytesseract
from pytesseract import Output
from PIL import Image
import io

def download_image(url, username, password):
    try:
        # Make a request to the URL with basic authentication
        response = requests.get(url, auth=HTTPBasicAuth(username, password))

        # Check if the request was successful
        if response.status_code == 200:
            image = response.content

            image_bytes = io.BytesIO(response.content)
            image = Image.open(image_bytes)

            return image

            return encoded_image.decode('utf-8')
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


init(os.getenv("IMAGE_OPENAI_MODEL"), os.getenv("SMART_OPENAI_MODEL"), os.getenv("FAST_OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), CONNECTION_STRING, RECORDMANAGER_CONNECTION_STRING, int(os.getenv("OPENAI_MAX_TOKENS")), int(os.getenv("OPENAI_IMAGE_MAX_TOKENS")), temp=os.getenv("OPENAI_TEMPERATURE"))

# image_url = "https://api.twilio.com/2010-04-01/Accounts/AC7278c4ac3f5c1ae283833f4b167b3f65/Messages/MMa87d68aed7ff1b094e7a7f0f966ab857/Media/ME1d2f0bc2d8c0f945f0ffe2029ad7d1e8"
# image = download_image(image_url, os.getenv("IMAGE_USERNAME"), os.getenv("IMAGE_PASSWORD"))

from ai.lib import lib_image_ai
from ai import ChatMessage
# image_data = base64.b64decode(base64_image)



# # https://api.twilio.com/2010-04-01/Accounts/AC7278c4ac3f5c1ae283833f4b167b3f65/Messages/MM9ee3dab948f1b43a754cbf8f00c1908e/Media/ME715240f2a6dacce533af4bbf2132b0dc
def process_image(image_url, chat_history, chat_context):
    image = download_image(image_url, os.getenv("IMAGE_USERNAME"), os.getenv("IMAGE_PASSWORD"))
    print(f"Downloaded image of length: {image.size}")

    image_description = ai.describe_image(image)

    # res = convo.predict(input=user_input)
    # print(f"\nai: {image_description}\n")
    # lib_conversation.save_message(image_description, "ai")

    msg = ChatMessage(sender="client", content=f"Uploaded image", msg_type="image", image_description=image_description)
    lib_conversation.save_message(msg.content, msg.sender)
    chat_history.append(msg)

def process_command(user_input, chat_history, chat_context):
    # print(f"Asking AI about: {user_input}")

    msg = ChatMessage(sender="client", content=user_input, msg_type="text")
    lib_conversation.save_message(msg.content, msg.sender)
    chat_history.append(msg)

    reply_list, new_context = ai.get_chat_reply(session_id="test", chat_id="test", chat_history=chat_history, chat_context=chat_context)

    chat_history.extend(reply_list)

    for msg in reply_list:
        # res = convo.predict(input=user_input)
        print(f"\n{msg.sender}: {msg.content}\n")
        lib_conversation.save_message(msg.content, msg.sender)

    return new_context



def main():
    lib_conversation.init_conversation()
    bindings = KeyBindings()
    history = FileHistory('./gpt_prompt_history.txt')  # specify the path to your history file

    chat_history = []
    chat_context = None

    while True:
        multiline = False

        while True:
            try:
                if not multiline:
                    # Single-line input mode
                    line = prompt('Human (type image to put an image url in): ', key_bindings=bindings, history=history)
                    if line.strip() == '"""':
                        multiline = True
                        continue
                    elif line.strip() == 'image':
                        # image_url = prompt('Image URL: ', key_bindings=bindings, history=history)

                        # Copyrighted
                        # image_url = "https://api.twilio.com/2010-04-01/Accounts/AC7278c4ac3f5c1ae283833f4b167b3f65/Messages/MM9ee3dab948f1b43a754cbf8f00c1908e/Media/ME715240f2a6dacce533af4bbf2132b0dc"
                        # Chat convo
                        image_url = "https://api.twilio.com/2010-04-01/Accounts/AC7278c4ac3f5c1ae283833f4b167b3f65/Messages/MMa87d68aed7ff1b094e7a7f0f966ab857/Media/ME1d2f0bc2d8c0f945f0ffe2029ad7d1e8"
                        process_image(image_url, chat_history, chat_context)
                        break
                    elif line.strip().lower() == 'quit':
                        return  # Exit the CLI
                    else:
                        chat_context = process_command(line, chat_history, chat_context)
                else:
                    # Multiline input mode
                    line = prompt('... ', multiline=True, key_bindings=bindings, history=history)
                    chat_context = process_command(line, chat_history, chat_context)
                    multiline = False
            except EOFError:
                return


if __name__ == "__main__":
    main()