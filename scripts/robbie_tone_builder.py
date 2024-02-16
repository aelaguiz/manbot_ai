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

logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.connection").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
logging.getLogger("openai._base_client").setLevel(logging.CRITICAL)

import dspy

class TrainingExample(dspy.Example):
    pass

class GenerateRobbieReplyQuery(dspy.Signature):
    chats = dspy.InputField(desc="The most recent X messages exchanged with the client, serving as the immediate conversational context for generating a reply.")
    answer = dspy.OutputField(desc="Robbie Kramer's reply")

class RobbieReply(dspy.Module):
    def __init__(self, num_chats=3):
        # self.retrieve = dspy.Retrieve(k=num_chats)
        self.generate_answer = dspy.Predict(GenerateRobbieReplyQuery)

    def forward(self, chats):
        # context = self.retrieve(chats).passages
        answer = self.generate_answer(chats=chats)
        return answer

class AssessResponse(dspy.Signature):
    chats = dspy.InputField(desc="A snapshot of recent exchanges between the client and Robbie, used as a basis for generating and evaluating responses.")
    generated_answer = dspy.InputField(desc="The AI-generated response intended to emulate Robbie Kramer's style and advice.")
    actual_answer = dspy.InputField(desc="A reference response from Robbie Kramer, used as a benchmark for evaluating the AI-generated answer's authenticity.")
    assessment_question = dspy.InputField(desc="A targeted question guiding the evaluation of the generated answer against specific criteria of authenticity or relevance.")
    assessment_answer = dspy.OutputField(desc="The evaluator's judgement ('Yes' or 'No') regarding the generated answer's alignment with the specified assessment criteria.")

turbo = dspy.OpenAI(model=os.getenv("FAST_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
gpt4 = dspy.OpenAI(model=os.getenv("SMART_OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))

dspy.settings.configure(lm=gpt4)

def robbie_style_score(example, pred, trace=None):
    tone = "Does the answer sound like Robbie in terms of tone? In particular, does the message start the way he would start a message?"
    format = "Is the answer formatted like Robbie's, including punctuation, emojis, and capitalization?"
    diction = "Does the answer use language, slang, and phrases that Robbie would typically use?"
    personalization = "Does the answer include personal touches or specific references that Robbie would likely include?"
    length = "Does the length of the answer match what is typical for Robbie, considering both brevity and detail?"

    with dspy.context(lm=turbo):
        t_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=tone)
        f_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=format)
        d_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=diction)
        p_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=personalization)
        l_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=length)

    # Evaluating scores based on 'yes' responses indicating alignment with Robbie's style
    t_score, f_score, d_score, p_score, l_score = [m.assessment_answer.split()[0].lower() == 'yes' for m in [t_res, f_res, d_res, p_res, l_res]]
    score = (t_score + f_score + d_score + p_score + (l_score * 2))
    score /= 6.0

    logging.debug(f"Chats: {example.chats}")
    logging.debug(f"Actual result: {example.answer}")
    logging.debug(f"Model result: {pred.answer}")
    logging.debug(f"Assessment results: tone = {t_res.assessment_answer}, format = {f_res.assessment_answer}, diction = {d_res.assessment_answer}, personalization = {p_res.assessment_answer}, length = {l_res.assessment_answer}")
    logging.debug(f"Score = {score}")

    return score
 
def main():
    dataset_path = sys.argv[1]
    import pickle
    with open(dataset_path, 'rb') as f:
        (train_set, validate_set, test_set) = pickle.load(f)
    
    train_set = train_set[:50]
    test_set = test_set[:10]
    validate_set = validate_set[:10]

    model = RobbieReply(num_chats=3)
    # model.load("robbie_reply_model.json")

    from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFewShot
    from dspy.evaluate import Evaluate

    evaluator = Evaluate(devset=test_set, num_threads=4, display_progress=True, display_table=False, metric=robbie_style_score)

    avg_score = evaluator(model)
    logging.info(f"BEFORE OPTIMIZATION EVALUATION: {avg_score}")
    
    # compiled_model = model
    optimizer = BootstrapFewShotWithRandomSearch(metric=robbie_style_score, num_threads=8, num_candidate_programs=3, max_bootstrapped_demos=3, teacher_settings=dict(lm=gpt4))
    compiled_model = optimizer.compile(model, trainset=train_set, valset=validate_set)
    compiled_model.save("robbie_reply_model.json")

    avg_score = evaluator(compiled_model)
    logging.info(f"AFTER OPTIMIZATION EVALUATION: {avg_score}")

    logging.debug("TESTING THE MODEL")

    for t in test_set:
        res = compiled_model(chats=t.chats)
        score = robbie_style_score(t, res)
        gpt4.inspect_history(n=1)


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