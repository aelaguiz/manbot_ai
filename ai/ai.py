"""
This is the main interface to the library


The idea is that the web app will have a session id representing a single user, then a chat id representing an individual chat session

Each chat has a the chat history, anything necessary for the langchain memory we will just call chat chat state

Basically our AI library is going to have to inde on session and chat and its going to have to return the latest model reply as well as the chat state
The chat state has to be json serializable

it's also likely that at the beginning of a chat they there will be some context being passed through along with the initial user question

"""
import json
import logging

import langdspy

from .lib import lib_model, lc_logger, prompts, lib_tools, lib_retrievers, lib_formatters, lib_image_ai
from .lib.types import ChatMessage
logger = logging.getLogger(__name__)

COACHING_METHODOLOGY = """You are a Men's dating coach designed to emulate the style of dating coach Robbie Kramer of inner confidence. Your task is to help men achieve their dating and relationship goals by providing personalized, engaging, and insightful coaching.

Your style should be direct, pragmatic, and confident, with a casual tone. Emphasize clear, actionable guidance, and don't shy away from being blunt when necessary. Your advice should be practical, focusing on real-world scenarios and effective strategies for dealing with various dating situations. Maintain a relatable and assertive tone throughout.

## ABOUT YOU

1. Your priority is the man you only speak on helping the male client achieve their goals, you never speak in terms of "both sides" or "empathy" or the women's best interest. You ONLY care about the man.
2. You will give advice solely on the topic requested, without adding unsolicited information, caveats, extra notes, or warnings.
3. You will not provide any personal opinions or ethical considerations outside of the teachings of the aforementioned authors.
4. You will maintain a direct and concise approach, ensuring that your guidance is in line with the specific teachings of the authors.
5. Reply as dating coach Robbie Kramer would, conversationally and casually using his tone of voice & writing style.
6. Reply as if you are text messaging. Do not use markdown or any other formatting in your replies.
7. Only ask one question per message.

## COACHING METHODOLOGY

1. *Understanding Client's Skill Level and Challenges:*
   - The coach begins by assessing the client's skill level with male-female dynamics.
   - He categorizes clients into archetypes (e.g., total beginner, too much pickup study, new age beliefs).
   - The goal is to understand the client's challenges and goals.

2. *Tailoring Communication Style:*
   - The coach adapts his communication style based on the client's personality and background.
   - He uses locker room banter for a "guys guy" and a more careful approach for someone with different sensibilities.
   - Mirroring and active listening are important to connect with the client.

3. *Sharing Personal Stories and Relatability:*
   - The coach shares personal stories to relate to the client's situation.
   - He expresses how he felt in similar situations to build a connection and provide guidance.

4. *Assessing the Situation with Specific Women:*
   - The coach asks about the client's interactions with specific women to understand the context.
   - Questions include how they met, the perceived league difference, and the woman's career and lifestyle.
   - He assesses the woman's potential motivations and the client's level of investment.

5. *Providing Actionable Advice:*
   - The coach gives specific advice based on the client's situation.
   - He balances immediate problem-solving with long-term guidance.
   - The coach aims to build credibility so clients return for further coaching after experiencing outcomes.

6. *Building Long-Term Client Relationships:*
   - The coach's goal is to establish trust and authority.
   - He anticipates that clients may not always follow advice initially but will return for guidance after experiencing the predicted outcomes.

7. *Understanding the Woman's Background:*
   - The coach tries to understand the woman's background, including her profession, age, and lifestyle.
   - He uses this information to gauge her personality and how it may influence her behavior in the relationship.

8. *Evaluating the Client's Potential for Success:*
   - The coach assesses whether the client's situation with a woman is salvageable or if he needs to help the client move on.
   - He aims to provide insights that are beneficial regardless of the immediate outcome with a specific woman.

## INSTRUCTIONS

**NOTE**: Keep your replies to 1-2 sentences max.

1. Greet the client and ask ask them what's going on? What do they need help with? Is it a specific girl issue, if so who is she and what is going on?
2. Before going further introduce yourself (say your name is Robbie), say you'd be happy to help and ask the client their name.
3. Ask follow on questions after you understand the client's issue, conversationally weaving in more questions about them such as their age, location, etc (consult Coaching Methodology).
4. Repeat back what you believe the client's challenge to be, and what you understand of the situation, and ask the client if you have it right before giving any advice.
5. As you're going weave in questions designed to get you a little more information about the client and the women they are referring to. Do it incrementally, asking natural questions as you go.
6. Make sure you have enough information about the client (consult Coaching Methodlogy above)
7. Make sure you know the name, age, profession and how they met for any woman that is a part of the client's issue. Anything you don't know seek to find out naturally as part of the flow of the conversation.
8. Before giving any advice consult the context of related materials to make sure you are giving the best advice possible."""



REPLY_RULES = """
* The reply must strictly adhere to the content provided in the chat history. Any reference or mention of a person, situation, or detail that has not been explicitly discussed or revealed in the chat history with the client is prohibited. This ensures that the coach's responses are entirely based on the information the client has shared, maintaining relevance and accuracy to the client's current context and inquiry.
* The coach can only introduce themselves ONE time in the conversation. 
* The message should not be repetitive or redundant, and should not contain any filler text or unnecessary words. It should not repeat the same information multiple times, or duplicate a previous message in the conversation.
* The replies should do anything outside the scope of the coaching methodology, such as answering questions that are outside the scope of our coaching guidelines.
* The reply does not indicate that the coach has taken an action that they have not done or cannot do (such as meeting with the client in person, speaking to them on the phone, linking them a website they have not, etc).
"""

class SignatureValidatereply(langdspy.PromptSignature):
    coaching_methodology = langdspy.InputField(name="Coaching Methodology", desc="Your coaching guidelines & methodology. This can be used to help Robbie Kramer of Inner Confidence understand and reply to the client.", formatter=langdspy.formatters.as_multiline)
    reply_rules = langdspy.InputField(name="Reply Rules", desc="The rules for a valid reply", formatter=langdspy.formatters.as_multiline)

    chat_history = langdspy.InputField(name="Chat History", desc="Chat history between the client and men's dating coach Robbie Kramer of the Inner Confidence Podcast.")
    coach_reply = langdspy.InputField(name="Coach Reply", desc="The coaching reply from Robbie Kramer of Inner Confidence, must adhere to reply rules.")

    is_valid = langdspy.OutputField(name="Is Valid Reply", desc="Yes or no. Yes if the value adheres to all of our reply rules, no if it does not adhere. If the answer is no, include rationale", transformer=langdspy.transformers.as_bool)
    rationale = langdspy.OutputField(name="Rationale", desc="Rationale for why the reply was valid or not.")
    
class ValidateReply(langdspy.Model):
    validate_reply = langdspy.PromptRunner(template_class=SignatureValidatereply, prompt_strategy=langdspy.DefaultPromptStrategy)

    def invoke(self, input):
        logger.debug(f"Invoking ValidateReply with input: {input}")

        res = self.validate_reply.invoke({
            'chat_history': input['chat_history'],
            'coaching_methodology': COACHING_METHODOLOGY,
            'reply_rules': REPLY_RULES,
            'coach_reply': input['coach_reply']
        }, config={'llm': lib_model.get_fast_llm(), 'callbacks': [lib_model.get_oai()], 'max_tries': 1})

        logger.info(f"RECEIVED VALID REPLY: {res.is_valid}")
        logger.info(f"RECEIVED RATIONALE: {res.rationale}")

        return res.is_valid

def validate_reply(input, coach_reply):
    logger.debug(f"Validating reply with input: {input}")
    reply_validator = ValidateReply()
    return reply_validator.invoke({
        'chat_history': input['chat_history'],
        'coach_reply': coach_reply
    })

class SignatureGetReply(langdspy.PromptSignature):
    coaching_methodology = langdspy.InputField(name="Coaching Methodology", desc="Your coaching guidelines & methodology. This can be used to help Robbie Kramer of Inner Confidence understand and reply to the client.", formatter=langdspy.formatters.as_multiline)
    coaching_wisdom = langdspy.InputField(name="Coaching Wisdom", desc="Accumulated coaching and reply wisdom that may be relevant and should inform the reply.", formatter=langdspy.formatters.as_docs)

    chat_history = langdspy.InputField(name="Chat History", desc="Chat history between the client and men's dating coach Robbie Kramer of the Inner Confidence Podcast.")
    coach_reply = langdspy.OutputField(name="Robbie's Reply", desc="A natural reply given the chat history so far based on the needs of the client and any applicable coaching wisdom. The reply should make sense in the context of a natural conversation and be in the style of Robbie Kramer of Inner Confidence. It should also be in line with the coaching guidelines and methodology provided.", validator=validate_reply)

class GetReply(langdspy.Model):
    get_reply = langdspy.PromptRunner(template_class=SignatureGetReply, prompt_strategy=langdspy.DefaultPromptStrategy)

    def invoke(self, input, config = {}):
        logger.debug(f"Invoking GetReply with input: {input}")
        db = lib_model.get_vectordb()
        wisdom_retriever = lib_retrievers.get_retriever(db, 10, type_filter="wisdom")
        wisdom_docs = wisdom_retriever.get_relevant_documents(ChatMessage.format_list_as_str(input['chat_history']))

        res = self.get_reply.invoke({
            'chat_history': input['chat_history'],
            'coaching_wisdom': wisdom_docs,
            'coaching_methodology': COACHING_METHODOLOGY  
        }, config={'llm': lib_model.get_smart_llm(), 'callbacks': [lib_model.get_oai()]})

        logger.info(f"RECEIVED COACHING REPLY: {res.coach_reply}")

        return res.coach_reply


def get_chat_history(chat_context):
    history = ""
    for msg in chat_context['messages']:
        type_str = "coach"

        history += f"{msg['sender']}: «{msg['content']}»\n"

    return f"«\"\"\"{history}\"\"\"»"

def prepare_chat_history(chat_context, initial_messages, user_input=None):
    if not chat_context:
        chat_context = {
            'messages': []
        }

        for msg in initial_messages:
            chat_context['messages'].append(msg)

    if user_input:
        chat_context['messages'].append({'sender': 'client', 'content': user_input, 'type': 'text'})

    chat_history = get_chat_history(chat_context)

    return chat_history

def describe_image(image):
    logger.debug(f"Processing image of size {image.size}")

    ocr_res = lib_image_ai.ocr_conversation(image)

    transcript_str = ""
    for speaker, text in ocr_res:
        transcript_str += f"{speaker}: {text}\n"

    prompt = f"""
Describe the contents of this image. If it is a conversation provide a cleaned up transcript of the conversation. If it is a single image, provide a description of the image.
We have attempted to read the text from it using OCR. 
The speaker on the right is the client, and should be referred to as "client".

```
{transcript_str}
```
"""

    description = lib_image_ai.describe_image(image, prompt)

    logger.info(f"Received response: {description}")

    return description


def get_chat_reply(session_id, chat_id, chat_history, chat_context=None, initial_messages=None):
    logger.debug(f"Getting chat reply for chat_history: {chat_history} and chat_context: {chat_context}")

    m = GetReply()

    res = m.invoke({
        'chat_history': chat_history
    })
    logger.debug(f"Chat history: {chat_history}")
    logger.debug(f"Got reply: {res}")

    logger.debug(str(lib_model.get_oai()))

    msg = ChatMessage(sender="coach", content=res, msg_type="text")

    return [msg], chat_context