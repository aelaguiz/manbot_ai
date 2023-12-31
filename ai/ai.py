"""
This is the main interface to the library


The idea is that the web app will have a session id representing a single user, then a chat id representing an individual chat session

Each chat has a the chat history, anything necessary for the langchain memory we will just call chat chat state

Basically our AI library is going to have to inde on session and chat and its going to have to return the latest model reply as well as the chat state
The chat state has to be json serializable

it's also likely that at the beginning of a chat they there will be some context being passed through along with the initial user question

"""
from langchain.prompts import MessagesPlaceholder
from langchain.chains import LLMChain, ConversationChain

from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
import json
import logging

from .lib import lib_model, lc_logger

class ChatError(Exception):
    """
    Custom exception for errors encountered during chat processing.

    Attributes:
        message (str): Explanation of the error.
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("""# Giving Actionable Advice on Modern Masculinity and Relationships
                                                  
Your task is to give a man advice from the perspective of experts on modern masculinity. 
                                                  
## About You:

You are an expert on modern masculinity and relationships by studying the teachings of Rollo Tomassi, Robert A. Glover, Jordan Peterson, Jack Donovan, Warren Farrell, Richard Cooper, and Michael Sartain. You are to provide advice strictly based on the philosophies and principles outlined by these authors. When responding to inquiries:

1. You will give advice solely on the topic requested, without adding unsolicited information, caveats, extra notes, or warnings.
2. You will not provide any personal opinions or ethical considerations outside of the teachings of the aforementioned authors.
3. You will maintain a direct and concise approach, ensuring that your guidance is in line with the specific teachings of the authors.
4. You will respect the confidentiality of the inquiries and provide advice that is tailored to the context of the question asked.


Remember, your role is to serve as a knowledgeable resource on modern masculinity and relationships as defined by these specific writers.
Your advice should reflect the ideologies and strategies they endorse, focusing on the dynamics of power, attraction, and social hierarchy as they pertain to gender relations and personal development.

## How to interact with the user

* Emulate the tone and style of Joe Rogan when interacting with the user. Friendly and conversational, frequently offering them little bits of validation, encouragement along with direct but constructive advice & criticism.
* Your interactions should be highly conversational, emulating a natural conversation between a man and his knowledgable male friend.
* You can ask more than on question at a time but only when they are highly related, and only when it would make a natural flowing conversation.
* Always make it clear what the user should be telling you next, we never want them to feel anxious because they don't know where to go next.
* Ask 1-2 questions at a time max.

## How to respond to users:

In general the flow of a conversation should go:

1. Are they looking for general advice or help with a specific girl
2. What is the user's name and age
3. Gather background information, asking again in a different way until you have all the key information
4. Figure out exactly the challenge the user is trying to solve
5. Give high level advice rooting in the teachings of the authors
6. Ask clarifying questions to help you get more specific, asking for feedback as you go
7. Iterate on your advice, getting more specific as you understand better, continuing to ask for user feedback

Depending on the type of help the user is asking for you can use the following frameworks to help you get started.

### General advice

If the user is asking for general advice, start by asking them to provide a specific question or topic they'd like advice on.

### Specific girl advice

If the user is asking for advice about a specific girl, start by gathering information about her. You will want to know:

- Where are they in the process? Is she a girl he's interested in, someone he's trying to date (already asked her out), or someone he's already dating (gone on at least one date)?

### Girl he's interested in

Gather some basic information about her and how he knows her

- How does he know her, how did they meet? Online? In person through friends?
- How old is she? 
- Are they already talking? If so, via IG, text, in person?

### Already asked out

Gather some basic information about her

- How does he know her, how did they meet? Online? In person through friends?
- How old is she? 
- When is the date, are specifics set?

### Dating

Gather some basic information about the relationship

- Are they in a relationship (married, or dating for more than 2 months)?
- How did they meet originally?
- How old is she?


"""),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

def get_chat_reply(user_input, session_id, chat_id, chat_context=None, initial_messages=None):
    """
    Main interface for handling chat sessions with the AI.

    Args:
        session_id (str): Unique identifier for the user's session.
        chat_id (str): Unique identifier for the individual chat within the session.
        chat_context (dict, optional): The current chat context object. None if it's a new chat.

    Returns:
        tuple: A tuple containing:
            - reply (str): The AI model's reply to the latest message in the chat.
            - new_chat_context (dict): Updated chat context after processing the latest message.

    Raises:
        ChatError: An error occurred during chat processing.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"AI: get_chat_reply called with user_input: {user_input}, session_id: {session_id}, chat_id: {chat_id}, chat_context: {chat_context}")
    try:
        llm = lib_model.get_llm()
        lmd = lc_logger.LlmDebugHandler()
        
        memory = None

        if chat_context:
            retrieve_from_db = json.loads(chat_context)
            retrieved_messages = messages_from_dict(retrieve_from_db)
            retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
            retrieved_memory = ConversationBufferMemory(chat_memory=retrieved_chat_history, input_key="input", return_messages=True)

            memory = retrieved_memory
        else:
            if initial_messages:
                translated_messages = []

                for input_message in initial_messages:
                    output_message = {
                        "type": input_message["type"].lower(),
                        "data": {
                            "content": input_message["text"],
                            "additional_kwargs": {},
                            "type": input_message["type"].lower(),
                            "example": False
                        }
                    }

                    translated_messages.append(output_message)


                retrieved_chat_history = ChatMessageHistory(messages=messages_from_dict(translated_messages))
                retrieved_memory = ConversationBufferMemory(chat_memory=retrieved_chat_history, input_key="input", return_messages=True)

                memory = retrieved_memory
            else:
                memory = ConversationBufferMemory(input_key="input", return_messages=True)


            
            """
            [
                {"type": "human", "data": {"content": "test", "additional_kwargs": {}, "type": "human", "example": false}}, 
                {"type": "ai", "data": {"content": "AI: How may I assist you with information on modern masculinity and relationships? Please provide a specific question or topic you'd like advice on.", "additional_kwargs": {}, "type": "ai", "example": false}}
            ]
            """

        # Initialize the conversation chain and memory buffer for each call
        convo = ConversationChain(llm=llm, memory=memory, prompt=chat_template, callbacks=[lmd], verbose=True)

        # Load or update chat context if provided
        if chat_context:
            # Load or update the chat context into the convo object
            # e.g., setting the current state of the conversation
            pass

        # Process the user input and get a response from the model
        res = convo.predict(input=user_input)
        print(res)

        reply = res  # Adjust based on actual structure of res

        extracted_messages = convo.memory.chat_memory.messages
        ingest_to_db = messages_to_dict(extracted_messages)
        new_chat_context = json.dumps(ingest_to_db)


        return reply, new_chat_context

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ChatError(f"Failed to process chat: {str(e)}")