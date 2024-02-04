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

# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the WhatsAppChatLoader class from where it is defined
from ai.lib.loaders.whatsapp_loader import WhatsAppChatLoader 
from ai.lib.loaders import wp_loader, discord_loader
from ai.lib import lib_model, lib_doc_vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from datetime import datetime, timedelta, timezone
from typing import List, Dict

def ensure_aware(datetime_obj):
    """Ensure a datetime object is offset-aware, assuming UTC if it's naive."""
    if datetime_obj.tzinfo is None or datetime_obj.tzinfo.utcoffset(datetime_obj) is None:
        return datetime_obj.replace(tzinfo=timezone.utc)
    return datetime_obj


def get_all_messages(whatsapp_path, discord_path):
    txt_files = list(glob.glob(os.path.join(whatsapp_path, "*.txt")))

    random.shuffle(txt_files)
    total_files = len(txt_files)
    docs = []

    all_messages = []

    for i, file_path in enumerate(txt_files):
        print(f"Processing file {i+1}/{total_files}: {file_path}")
        loader = WhatsAppChatLoader(file_path)
        messages = loader.load_messages()
        all_messages.extend(messages)

    txt_files = list(glob.glob(os.path.join(discord_path, "*.txt")))

    random.shuffle(txt_files)
    total_files = len(txt_files)
    for i, file_path in enumerate(txt_files):
        print(f"Processing file {i+1}/{total_files}: {file_path}")
        loader = discord_loader.DiscordChatLoader(file_path)
        messages = loader.load_messages()
        all_messages.extend(messages)

    print(f"Loaded {len(all_messages)} messages")

    return all_messages


def split_into_conversations(messages: List[Dict], time_threshold_minutes: int = 30) -> List[List[Dict]]:
    # First, sort messages by their timestamp to ensure they're in chronological order
    for msg in messages:
        msg["timestamp"] = ensure_aware(msg["timestamp"])

    # Now, sort messages by their timestamp to ensure they're in chronological order
    messages.sort(key=lambda msg: msg["timestamp"])

    print("Messages sorted by timestamp.")

    conversations = []
    current_conversation = []

    for i, message in enumerate(messages):
        # For the first message, just add it to the current conversation
        if i == 0:
            current_conversation.append(message)
            continue

        # Calculate the time difference between the current message and the previous one
        time_difference = message["timestamp"] - messages[i - 1]["timestamp"]

        # If the time difference exceeds the threshold, start a new conversation
        if time_difference > timedelta(minutes=time_threshold_minutes):
            # Save the current conversation and start a new one
            conversations.append(current_conversation)
            print(f"New conversation started due to a gap of {time_difference}.")
            current_conversation = [message]
        else:
            current_conversation.append(message)

    # Don't forget to add the last conversation if it exists
    if current_conversation:
        conversations.append(current_conversation)
        print("Final conversation added.")

    return conversations

def filter_conversations_for_robbie(conversations):
    return [conv for conv in conversations if any(msg['user'] == 'Robbie Kramer' or msg['user'] == 'robbiekramer' for msg in conv)]

def generate_training_samples(conversations):
    training_samples = []

    for conversation in conversations:
        context = []  # Initialize context for accumulating messages
        current_sample = {'X': [], 'y': {'number_of_replies': 0, 'replies': []}}
        robbie_replied = False

        for message in conversation:
            # Check if message is from Robbie
            if message['user'] == 'Robbie Kramer' or message['user'] == 'robbiekramer':
                robbie_replied = True
                current_sample['y']['replies'].append(message['message'])
                current_sample['y']['number_of_replies'] += 1
            else:
                # If Robbie has already replied, finalize the current sample and prepare for the next
                if robbie_replied:
                    # Ensure Robbie is not the first speaker in the sample
                    if current_sample['X']:
                        training_samples.append(current_sample)
                    # Reset for the next potential sample
                    context = context[-10:]  # Keep only up to the last 10 messages for context if needed
                    current_sample = {'X': list(context), 'y': {'number_of_replies': 0, 'replies': []}}
                    robbie_replied = False
                
                # Accumulate context
                context.append(message['message'])
                current_sample['X'].append(message['message'])

        # Check and add the last sample if it ends with Robbie's replies
        if robbie_replied and current_sample['X']:
            training_samples.append(current_sample)

    return format_training_samples(training_samples)

class TrainingExample:
    def __init__(self, question, answer):
        self.question = "\n-".join(question)  # Combine context messages into a single string
        self.answer = "\n-".join(answer['replies'])  # Combine Robbie's replies into a single string

    def with_inputs(self, input_field):
        if input_field == 'question':
            return self
        else:
            raise ValueError(f"Unknown input field: {input_field}")

# Convert the generated training samples into the desired format
def format_training_samples(training_samples):
    formatted_samples = [TrainingExample(sample['X'], sample['y']) for sample in training_samples]
    return formatted_samples



def main():
    whatsapp_path = sys.argv[1]
    discord_path = sys.argv[2]

    # Initialize the library with environment variables
    lib_model.init(os.getenv("SMART_OPENAI_MODEL"), os.getenv("FAST_OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), os.getenv("PGVECTOR_CONNECTION_STRING"), os.getenv("RECORDMANAGER_CONNECTION_STRING"), temp=os.getenv("OPENAI_TEMPERATURE"))

    all_messages = get_all_messages(whatsapp_path, discord_path)

    conversations = split_into_conversations(all_messages, 60)
    print(f"Split into {len(conversations)} conversations.")

    conversations = filter_conversations_for_robbie(conversations)
    print(f"Got {len(conversations)} conversations with Robbie.")

    
    training_samples = generate_training_samples(conversations)
    for sample in training_samples:
        print(f"Question: {sample.question}\nAnswer: {sample.answer}\n\n")

    # for i, convo in enumerate(conversations):
    #     print(f"\n\nConversation {i+1} has {len(convo)} messages.")
    #     for message in convo:
    #         print(f"{message['timestamp']} {message['user']}: {message['message']}")


if __name__ == "__main__":
    main()