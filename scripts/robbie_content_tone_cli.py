import logging
import logging.config
import dotenv
import os
from rich.logging import RichHandler

# Load environment variables
dotenv.load_dotenv()

# Define the configuration file path based on the environment
config_path = os.getenv('LOGGING_CONF_PATH')

# Use the configuration file appropriate to the environment
logging.config.fileConfig(config_path)
rich_handler = RichHandler()
# Replace the first handler, assuming it's the console handler
logging.getLogger().handlers[0] = rich_handler

import sys
import glob
import random
import dspy

# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.connection").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
logging.getLogger("openai._base_client").setLevel(logging.CRITICAL)

import dspy
turbo = dspy.OpenAI(model=os.getenv("FAST_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
gpt4 = dspy.OpenAI(model=os.getenv("SMART_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))

dspy.settings.configure(lm=gpt4)

class TrainingExample(dspy.Example):
    pass

class GenerateRobbieContentQuery(dspy.Signature):
    # books = dspy.InputField(desc="The books to use for generating content.")
    chats = dspy.InputField(desc="A collection of the last X messages exchanged with the client, providing raw conversational data without prior analysis.")
    intuition = dspy.OutputField(desc="A synthesized insight reflecting Robbie Kramer's understanding of the client's current issues and needs, informed by patterns observed in historical coaching interactions.")

class GenerateRobbieReplyQuery(dspy.Signature):
    chats = dspy.InputField(desc="The most recent X messages exchanged with the client, serving as the immediate conversational context for generating a reply.")
    inuition = dspy.InputField(desc="An analytical summary capturing Robbie Kramer's intuitive grasp on the client's situation, drawn from similar past coaching experiences.")
    answer = dspy.OutputField(desc="Robbie Kramer's reply")

class RobbieReply(dspy.Module):
    def __init__(self, num_chats=3):
        # self.retrieve = dspy.Retrieve(k=num_chats)
        self.generate_content = dspy.Predict(GenerateRobbieContentQuery)
        self.generate_answer = dspy.Predict(GenerateRobbieReplyQuery)

    def forward(self, chats):
        # context = self.retrieve(chats).passages
        intuition = self.generate_content(chats=chats)
        answer = self.generate_answer(chats=chats, intuition=intuition)
        return answer

class AssessResponse(dspy.Signature):
    chats = dspy.InputField(desc="A snapshot of recent exchanges between the client and Robbie, used as a basis for generating and evaluating responses.")
    generated_answer = dspy.InputField(desc="The AI-generated response intended to emulate Robbie Kramer's style and advice.")
    actual_answer = dspy.InputField(desc="A reference response from Robbie Kramer, used as a benchmark for evaluating the AI-generated answer's authenticity.")
    assessment_question = dspy.InputField(desc="A targeted question guiding the evaluation of the generated answer against specific criteria of authenticity or relevance.")
    assessment_answer = dspy.OutputField(desc="The evaluator's judgement ('Yes' or 'No') regarding the generated answer's alignment with the specified assessment criteria.")

model_path = "robbie_content_reply_model.json"
model = RobbieReply(num_chats=3)
model.load(model_path)
        

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit import prompt

history = ["Hey! What's up? Do you need help with a specific girl or more general advice?"]
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
    print(history[0])

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
