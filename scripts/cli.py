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
from ai import ai
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

import time

dotenv.load_dotenv()


def process_command(user_input, chat_context, initial_messages):
    print(f"Asking AI about: {user_input}")

    reply, new_context = ai.get_chat_reply(user_input, session_id="test", chat_id="test", chat_context=chat_context, initial_messages=initial_messages)

    # res = convo.predict(input=user_input)
    print(reply)
    print(new_context)

    return new_context



def main():
    bindings = KeyBindings()

    initial_messages = [{
        "type": "ai",
        "text": "Hey! Let's start with the basics. Are you looking for help with a specific girl or are you looking for more general advice?"
    }]

    chat_context = None

    for msg in initial_messages:
        print(f"{msg['type']}: {msg['text']}")

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
                        chat_context = process_command(line, chat_context, initial_messages)
                        break
                else:
                    # Multiline input mode
                    line = prompt('... ', multiline=True, key_bindings=bindings)
                    chat_context = process_command(line, chat_context, initial_messages)
                    multiline = False
            except EOFError:
                return


if __name__ == "__main__":
    main()
