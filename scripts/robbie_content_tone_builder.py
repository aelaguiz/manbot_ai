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

def robbie_style_score(example, pred, trace=None):
    comprehensive_relevance_and_insight = "Does the answer not only directly address the client's issue with clear relevance but also demonstrate an intuitive understanding of their underlying needs and challenges, mirroring Robbie's ability to provide deeply insightful and contextually appropriate guidance?"
    strategic_depth_with_actionable_solutions = "Does the advice offer a strategic perspective that considers both the immediate and long-term implications of the client's situation, providing practical, actionable steps in a manner consistent with Robbie's approach to delivering solution-focused and strategically sound advice?"
    style_and_expression = "Does the answer embody Robbie's distinctive communication style, from the tone and personal touches to the specific use of language, including slang and phrases, and is it presented with Robbie's typical formatting preferences, such as punctuation, emojis, and capitalization?"
    content_and_form = "Does the content of the answer reflect Robbie's way of personalizing responses with relevant references or anecdotes, and does its length mirror the typical depth and detail Robbie provides, balancing brevity with comprehensive insight?"

    with dspy.context(lm=turbo):
        cr_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=comprehensive_relevance_and_insight)
        sd_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=strategic_depth_with_actionable_solutions)
        se_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=style_and_expression)
        cf_res = dspy.Predict(AssessResponse)(chats=example.chats, generated_answer=pred.answer, actual_answer=example.answer, assessment_question=content_and_form)

    # Evaluating scores based on 'yes' responses indicating alignment with Robbie's style
    scores = [m.assessment_answer.split()[0].lower() == 'yes' for m in [cr_res, sd_res, se_res, cf_res]]
    score = sum(scores) / 4.0  # Normalizing score based on the number of questions

    logging.debug(f"Chats: {example.chats}")
    logging.debug(f"Actual result: {example.answer}")
    logging.debug(f"Model result: {pred.answer}")
    logging.debug(f"Assessment results: Comprehensive relevance and insight = {cr_res.assessment_answer}, Strategic depth with actionable solutions = {sd_res.assessment_answer}, Style and expression = {se_res.assessment_answer}, Content and form = {cf_res.assessment_answer}")
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

    model_path = "robbie_content_reply_model.json"
    model = RobbieReply(num_chats=3)
    # # model.load(model_path)

    from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFewShot
    from dspy.evaluate import Evaluate

    evaluator = Evaluate(devset=test_set, num_threads=4, display_progress=True, display_table=False, metric=robbie_style_score)

    avg_score = evaluator(model)
    logging.info(f"BEFORE OPTIMIZATION EVALUATION: {avg_score}")
    
    optimizer = BootstrapFewShotWithRandomSearch(metric=robbie_style_score, num_threads=8, num_candidate_programs=3, max_bootstrapped_demos=3, teacher_settings=dict(lm=gpt4))
    compiled_model = optimizer.compile(model, trainset=train_set, valset=validate_set)
    compiled_model.save(model_path)

    # compiled_model = model
    avg_score = evaluator(compiled_model)
    logging.info(f"AFTER OPTIMIZATION EVALUATION: {avg_score}")

    logging.debug("TESTING THE MODEL")

    for t in test_set:
        res = compiled_model(chats=t.chats)
        score = robbie_style_score(t, res)
        gpt4.inspect_history(n=1)

        


if __name__ == "__main__":
    main()