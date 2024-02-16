import logging
import logging.config
import dotenv
import os
from rich.logging import RichHandler

# Load environment variables
dotenv.load_dotenv()

# Define the configuration file path based on the environment
# config_path = os.getenv('LOGGING_CONF_PATH')

# # Use the configuration file appropriate to the environment
# logging.config.fileConfig(config_path)
# rich_handler = RichHandler()
# # Replace the first handler, assuming it's the console handler
# logging.getLogger().handlers[0] = rich_handler

import sys
import glob
import random
import dspy

# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the WhatsAppChatLoader class from where it is defined
from ai.lib.loaders.whatsapp_loader import WhatsAppChatLoader 
from ai.lib.loaders import wp_loader, discord_loader
from ai.lib import lib_model, lib_doc_vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from sklearn.model_selection import train_test_split

# logging.getLogger("httpx").setLevel(logging.CRITICAL)
# logging.getLogger("httpcore.connection").setLevel(logging.CRITICAL)
# logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
# logging.getLogger("openai._base_client").setLevel(logging.CRITICAL)


import dspy

class GenerateRobbieReplyQuery(dspy.Signature):
    chats = dspy.InputField(desc="The client chat history to generate a reply for.")
    answer = dspy.OutputField(desc="Robbie Kramer's reply")

class RobbieReply(dspy.Module):
    def __init__(self, num_chats=3):
        # self.retrieve = dspy.Retrieve(k=num_chats)
        self.generate_answer = dspy.ChainOfThought(GenerateRobbieReplyQuery)

    def forward(self, chats):
        # context = self.retrieve(chats).passages
        answer = self.generate_answer(chats=chats)
        return answer

turbo = dspy.OpenAI(model=os.getenv("FAST_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
gpt4 = dspy.OpenAI(model=os.getenv("SMART_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))

dspy.settings.configure(lm=gpt4)


model = RobbieReply(num_chats=3)
model.load("robbie_reply_model.json")
        

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit import prompt

history = []
def process_command(user_input):
    history.append(user_input)

    res = model(chats="\n".join(history))
    # print(f"Asking AI about: {user_input}")
    print(f"AI: {res.answer}")

    history.append(res.answer)



    # res = convo.predict(input=user_input)
    # print(f"\nai: {reply}\n")
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
