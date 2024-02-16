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

from dsp.modules import cache_utils
cache_utils.cache_turn_on = False

import dspy

from .lib import lib_model, lc_logger, prompts, lib_tools, lib_retrievers, lib_formatters
logger = logging.getLogger(__name__)

conversation_stages = """
You are a Men's dating coach designed to emulate the style of dating coach Robbie Kramer of inner confidence. Your task is to help men achieve their dating and relationship goals by providing personalized, engaging, and insightful coaching.

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

1. Greet the client and ask ask them what's going on? What do they need help with? Is it a specific girl issue, if so who is she and what is going on?
2. Before going further introduce yourself (say your name is Robbie), say you'd be happy to help and ask the client their name.
3. Ask follow on questions after you understand the client's issue, conversationally weaving in more questions about them such as their age, location, etc (consult Coaching Methodology).
4. Repeat back what you believe the client's challenge to be, and what you understand of the situation, and ask the client if you have it right before giving any advice.
5. As you're going weave in questions designed to get you a little more information about the client and the women they are referring to. Do it incrementally, asking natural questions as you go.
6. Make sure you have enough information about the client (consult Coaching Methodlogy above)
7. Make sure you know the name, age, profession and how they met for any woman that is a part of the client's issue. Anything you don't know seek to find out naturally as part of the flow of the conversation.
8. Before giving any advice consult the context of related materials to make sure you are giving the best advice possible.
"""

class SignatureGetReply(dspy.Signature):
    chat_history = dspy.InputField(desc="Chat history between the client and men's dating coach Robbie Kramer of the Inner Confidence Podcast.")

    about_you = dspy.InputField(desc="Your coaching guidelines & methodology. This can be used to help Robbie Kramer of Inner Confidence understand and reply to the client.")

    # context = dspy.InputField(desc="Books from men's dating coaches and experts on dating and relationships. This can be used to help Robbie Kramer of Inner Confidence understand and reply to the client.")

    next_message = dspy.OutputField(desc="A natural reply given the chat history so far. The reply should make sense in the context of a natural conversation and be in the style of Robbie Kramer of Inner Confidence. It should also be in line with the coaching guidelines and methodology provided.")

class SignatureRobbieTone(dspy.Signature):
    chat_history = dspy.InputField(desc="Chat history between the client and men's dating coach Robbie Kramer of the Inner Confidence Podcast.")

    raw_message = dspy.InputField(desc="The suggested reply from Robbie Kramer, raw and unprocessed for tone")

    context = dspy.InputField(desc="Examples of how Robbie speaks to clients")

    adjusted_message = dspy.OutputField(desc="The suggested reply from Robbie Kramer, adjusted to sound more like Robbie speaks based on the reference context provided.")

class GetReply(dspy.Module):
    def __init__(self):
        self.get_reply = dspy.Predict(SignatureGetReply)
        self.adjust_tone = dspy.Predict(SignatureRobbieTone)

    def forward(self, chat_history):
        db = lib_model.get_vectordb()
        book_retriever = lib_retrievers.get_retriever(db, 3, type_filter="book")
        book_docs = book_retriever.get_relevant_documents(chat_history)
        book_formatted_passages = [lib_formatters._format_doc(d) for d in book_docs]

        res = self.get_reply(chat_history=chat_history, about_you=conversation_stages, lm=lib_model.gpt4)
        # res = self.get_reply(chat_history=chat_history, about_you=conversation_stages, context=book_formatted_passages, lm=lib_model.gpt4)
        raw_message = res.next_message

        whatsapp_retriever = lib_retrievers.get_retriever(db, 3, type_filter="discord")
        whatsapp_docs = whatsapp_retriever.get_relevant_documents(chat_history)
        whatsapp_formatted_passages = [lib_formatters._format_doc(d) for d in whatsapp_docs]

        # for p in whatsapp_formatted_passages:
        #     print(p)
        
        res = self.adjust_tone(chat_history=chat_history, raw_message=raw_message, context=whatsapp_formatted_passages, lm=lib_model.turbo)
        adjusted_message = res.adjusted_message
        
        logger.info(f"RAW MESSAGGE: {raw_message}")
        logger.info(f"ADJUSTED MESSAGE: {adjusted_message}")
        return adjusted_message

def get_chat_history(chat_context):
    history = ""
    for msg in chat_context['messages']:
        type_str = "coach"

        history += f"{msg['sender']}: {msg['content']}\n\n"

    return history

def get_chat_reply(user_input, session_id, chat_id, chat_context=None, initial_messages=None):
    logger.debug(f"Getting chat reply for user input: {user_input}")

    if not chat_context:
        chat_context = {
            'messages': []
        }

        for msg in initial_messages:
            chat_context['messages'].append(msg)

    chat_context['messages'].append({'sender': 'client', 'content': user_input, 'type': 'text'})

    chat_history = get_chat_history(chat_context)

    m = GetReply()
    dspy.assert_transform_module(m)

    res = m(chat_history)
    logger.debug(f"Chat history: {chat_history}")
    logger.debug(f"Got reply: {res}")

    chat_context['messages'].append({'sender': 'coach', 'content': res, 'type': 'text'})

    logger.info(lib_model.turbo)
    logger.info(lib_model.gpt4)

    return res, chat_context

    # from ai.lib.dspy.dspy_pgvector_retriever import PGVectorRM



    # return res, new_chat_context