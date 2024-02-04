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

# Import the WhatsAppChatLoader class from where it is defined
from ai.lib.loaders.whatsapp_loader import WhatsAppChatLoader 
from ai.lib.loaders import wp_loader, discord_loader
from ai.lib import lib_model, lib_doc_vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from sklearn.model_selection import train_test_split


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
        logging.debug(f"Processing file {i+1}/{total_files}: {file_path}")
        loader = WhatsAppChatLoader(file_path)
        messages = loader.load_messages()
        all_messages.extend(messages)

    # txt_files = list(glob.glob(os.path.join(discord_path, "*.txt")))

    # random.shuffle(txt_files)
    # total_files = len(txt_files)
    # for i, file_path in enumerate(txt_files):
    #     logging.debug(f"Processing file {i+1}/{total_files}: {file_path}")
    #     loader = discord_loader.DiscordChatLoader(file_path)
    #     messages = loader.load_messages()
    #     all_messages.extend(messages)

    # logging.debug(f"Loaded {len(all_messages)} messages")

    return all_messages


def split_into_conversations(messages: List[Dict], time_threshold_minutes: int = 30) -> List[List[Dict]]:
    # First, sort messages by their timestamp to ensure they're in chronological order
    for msg in messages:
        msg["timestamp"] = ensure_aware(msg["timestamp"])

    # # Now, sort messages by their timestamp to ensure they're in chronological order
    messages.sort(key=lambda msg: msg["timestamp"])

    logging.debug("Messages sorted by timestamp.")

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
            # logging.debug(f"New conversation started due to a gap of {time_difference}.")
            current_conversation = [message]
        else:
            current_conversation.append(message)

    # Don't forget to add the last conversation if it exists
    if current_conversation:
        conversations.append(current_conversation)
        # logging.debug("Final conversation added.")

    return conversations

def filter_conversations_for_robbie(conversations):
    convos = [conv for conv in conversations if any(msg['user'] == 'Robbie Kramer' or msg['user'] == 'robbiekramer' for msg in conv)]
    convos = [conv for conv in convos if all("image omitted" not in msg['message'] for msg in conv)]

    return convos


def generate_training_samples(conversations):
    training_samples = []

    for idx, conversation in enumerate(conversations):
        speakers = set()  # Track unique speakers
        context = []  # Initialize context for accumulating messages
        current_sample = {'X': [], 'y': {'number_of_replies': 0, 'replies': []}}
        robbie_replied = False
        last_speaker = None  # Keep track of the last speaker

        # logging.debug(f"Starting conversation {idx+1} with {len(conversation)} messages")

        for message_idx, message in enumerate(conversation):
            # Normalize Robbie's user name
            if message['user'] in ['Robbie Kramer', 'robbiekramer']:
                normalized_user = 'Robbie'
            else:
                normalized_user = f"Client [{message['user']}]"
            message['user'] = normalized_user  # Apply normalization

            # logging.debug(f"Message {message_idx+1}: '{message['message']}' from {normalized_user}")

            speakers.add(normalized_user)

            if normalized_user == 'Robbie':
                robbie_replied = True
                current_sample['y']['replies'].append(message)
                current_sample['y']['number_of_replies'] += 1
                # logging.debug(f"Robbie replied. Total replies: {current_sample['y']['number_of_replies']}")
            else:  # Message from a client
                if robbie_replied and (last_speaker == 'Robbie' or len(speakers) > 2):
                    # logging.debug("Transition detected. Finalizing current sample and starting a new one.")
                    training_samples.append(current_sample) if current_sample['X'] else None
                    # logging.debug(f"Sample finalized with {len(current_sample['X'])} messages and {current_sample['y']['number_of_replies']} replies.")
                    # Reset for a new conversation sample
                    context = [message]  # Start with the current message as context
                    current_sample = {'X': list(context), 'y': {'number_of_replies': 0, 'replies': []}}
                    robbie_replied = False
                    speakers = {normalized_user}  # Reset speakers for the new sample
                else:
                    # Accumulate context for ongoing conversation
                    context.append(message)
                    current_sample['X'].append(message)
                # logging.debug(f"Context updated with {len(current_sample['X'])} messages.")

            last_speaker = normalized_user  # Update the last speaker

        # Check and add the last sample if it ends with Robbie's replies
        if current_sample['X'] and (robbie_replied and len(speakers) <= 2):
            training_samples.append(current_sample)
            # logging.debug(f"Final sample added from conversation {idx+1} with {len(current_sample['X'])} messages and {current_sample['y']['number_of_replies']} replies.")

    # logging.debug(f"Total training samples generated: {len(training_samples)}")
    return training_samples


import dspy

def split_dataset(data, train_size, validate_size, test_size):
    """
    Splits the dataset into training, validation, and test sets based on the specified percentages.
    
    Parameters:
    - data: The complete dataset as a list of TrainingExamples.
    - train_size: The percentage of the dataset to allocate to the training set.
    - validate_size: The percentage of the dataset to allocate to the validation set.
    - test_size: The percentage of the dataset to allocate to the test set.
    
    Returns:
    - A tuple containing the training set, validation set, and test set.
    """
    
    # Ensure that the percentages add up to 1 (or 100%)
    if (train_size + validate_size + test_size) != 1:
        raise ValueError("The sum of train, validate, and test sizes must equal 1")
    
    # First split to separate out the training set
    initial_train_size = train_size + validate_size
    train_val_set, test_set = train_test_split(data, test_size=test_size, shuffle=True)
    
    # Adjust validate_size for the second split
    validate_size_adjusted = validate_size / initial_train_size
    
    # Second split to separate out the validation set from the training set
    train_set, validate_set = train_test_split(train_val_set, test_size=validate_size_adjusted, shuffle=True)
    
    return train_set, validate_set, test_set


class TrainingExample(dspy.Example):
    pass

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

# Convert the generated training samples into the desired format
def format_training_samples(training_samples):
    for sample in training_samples:
        qs = []
        for q in sample['X']:
            qs.append(f"{q['user']}: {q['message']}")

        answers = []
        for a in sample['y']['replies']:
            answers.append(f"{a['message']}")

        yield TrainingExample(chats="\n".join(qs), answer="\n".join(answers)).with_inputs("chats")


class AssessResponse(dspy.Signature):
    chats = dspy.InputField(desc="The client chat history to generate a reply for.")
    generated_answer = dspy.InputField(desc="The generated answer from the model")
    actual_answer = dspy.InputField(desc="The actual answer from Robbie Kramer")
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

turbo = dspy.OpenAI(model=os.getenv("FAST_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
gpt4 = dspy.OpenAI(model=os.getenv("SMART_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))

dspy.settings.configure(lm=turbo)

def metric(example, pred, trace=None):
    tone = "Does the answer sound like Robbie in terms of tone? In particular, does the message start the way he would start a message?"
    format = "Is the answer formatted like Robbie's?"
    content = "Is the content of the answer similar to Robbie's?"
    length = "Is the length of the answer similar to Robbie's?"

    
    with dspy.context(lm=turbo):
        t_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=tone)
        f_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=format)
        c_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=content)
        l_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=length)

        
    t_score, f_score, c_score, l_score = [m.assessment_answer.split()[0].lower() == 'yes' for m in [t_res, f_res, c_res, l_res]]
    score = (t_score + f_score + c_score + l_score)
    

    logging.debug(f"Chats: {example.chats}")
    logging.debug(f"Actual result: {example.answer}")
    logging.debug(f"Model result: {pred.answer}")
    logging.debug(f"Assessment results: tone = {t_res.assessment_answer}, format = {f_res.assessment_answer}, content = {c_res.assessment_answer}, length = {l_res.assessment_answer}")
    logging.debug(f"Score = {score}")

    return score
    

def main():
    whatsapp_path = sys.argv[1]
    discord_path = sys.argv[2]

    # Initialize the library with environment variables
    # lib_model.init(os.getenv("SMART_OPENAI_MODEL"), os.getenv("FAST_OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), os.getenv("PGVECTOR_CONNECTION_STRING"), os.getenv("RECORDMANAGER_CONNECTION_STRING"), temp=os.getenv("OPENAI_TEMPERATURE"))

    all_messages = get_all_messages(whatsapp_path, discord_path)

    conversations = split_into_conversations(all_messages, 120)
    logging.debug(f"Split into {len(conversations)} conversations.")

    conversations = filter_conversations_for_robbie(conversations)
    logging.debug(f"Got {len(conversations)} conversations with Robbie.")

    
    training_samples = list(generate_training_samples(conversations))
    random.shuffle(training_samples)
    training_samples = training_samples[:50]
    logging.debug(f"Got {len(training_samples)} training samples.")

    # for sample in training_samples:
    #     logging.debug("Question:")
    #     for q in sample['X']:
    #         logging.debug(f"\t{q['timestamp']} {q['user']}: {q['message']}")

    #     logging.debug("Answer:")
    #     for a in sample['y']['replies']:
    #         logging.debug(f"\t{a['timestamp']} {a['user']}: {a['message']}")

    #     logging.debug("\n\n")

    dspy_samples = list(format_training_samples(training_samples))

    train_set, validate_set, test_set = split_dataset(dspy_samples, 0.6, 0.2, 0.2)
    
    model = RobbieReply(num_chats=3)

    from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFewShot
    
    optimizer = BootstrapFewShotWithRandomSearch(metric=metric, num_threads=4, num_candidate_programs=2, max_bootstrapped_demos=2, teacher_settings=dict(lm=gpt4))
    compiled_model = optimizer.compile(model, trainset=train_set, valset=validate_set)
    compiled_model.save("robbie_reply_model.json")


    for t in train_set[:3]:
        res = compiled_model(chats=t.chats)
        # logging.debug(f"Client Chat: {t.chats}")
        # logging.debug("\n\n")
        # logging.debug(f"Robbie's Actual reply: {train_set[1].answer}")
        # logging.debug(f"Model reply: {res.answer}")
        # logging.debug(f"Model rationale: {res.rationale}")
        # turbo.inspect_history(n=1)

        score = metric(t, res)
        logging.debug(f"Score: {score}")


    # import dspy.teleprompt
    # tp = dspy.teleprompt.BootstrapFewShot(
    #     metric=validate
    # )


    # for i, convo in enumerate(conversations):
    #     logging.debug(f"\n\nConversation {i+1} has {len(convo)} messages.")
    #     for message in convo:
    #         logging.debug(f"{message['timestamp']} {message['user']}: {message['message']}")

        


if __name__ == "__main__":
    main()