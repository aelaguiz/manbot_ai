import sys
import os
import typing
from typing import Optional, Union, Literal, AbstractSet, Collection, Any, List
from langchain.text_splitter import split_text_on_tokens

import html
import html2text
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai.lib import lib_model, lib_doc_vectors
from ai.lib.loaders import wp_loader, discord_loader


import logging
import logging.config
import dotenv
import os

dotenv.load_dotenv()


# Define the configuration file path based on the environment
config_path = os.getenv('LOGGING_CONF_PATH')

# Use the configuration file appropriate to the environment
logging.config.fileConfig(config_path)

"""
Be specific and clear: The more specific and clear your prompt is, the better the response will be. Make sure to provide all the necessary information and context for GPT-4 to understand your request.
Use examples: Providing examples can help GPT-4 understand the format and style of the desired output.
Use natural language: Use natural, conversational language when prompting GPT-4. This will help it understand your request better and provide a more natural response.
Be concise: Keep your prompts concise and to the point. Long, rambling prompts can confuse GPT-4 and lead to less accurate responses.
Use context: If you are using GPT-4 to generate text for a specific purpose, make sure to provide the context for that purpose. This will help GPT-4 generate more relevant and useful text.
Use feedback: If you receive a response from GPT-4 that is not quite what you were looking for, provide feedback and try again. This will help GPT-4 learn and improve its responses over time.
"""

classify_conversations_prompt = """
Prompt to LLM:

"# Discord Conversation Classification

Your task is to analyze a set of new Discord messages, each with their timestamp, and classify them into conversations

## Conversation Buffer

These are messages that have been reviewed but do not yet belong to a conversation:
{conversation_buffer}

## Existing Conversations

These are the conversations that are currently open:
{existing_conversations}

## Conversation Guidelines

When determining what messages belong to a conversation, follow these guidelines:

1. Topic Consistency: Group messages into a conversation if they discuss the same main topic.
2. Response Chain: Include a message in a conversation if it directly responds to or follows up on an earlier message.
3. Extended Time Window: Use a 2-hour window to determine if messages belong to the same conversation.
4. Natural Conversation Flow: Identify the start and end of conversations based on the natural flow of messages.
5. Multiple Conversations: Recognize and categorize simultaneous conversations based on topic relevance.

## Instructions

For each message:
1. Determine if the message fits into an existing open conversation, if so mark the message with the existing conversation topic
2. If a message does not fit into an existing open conversation, add it to the c

For each open conversation, provide the following details:
- Conversation Topic
- Initial Message of the Conversation (who said it and what was said)
- List of Participants in the Conversation
- Last Message of the Conversation (who said it and what was said)

   - Status: Closed (No new messages in the last 2 hours)

After analyzing the new messages, provide a summary including:
- Any new conversations that have started, with their topic, initial message, participants, and the last message.
- Existing conversations that are still ongoing, with their current status and the last message.
- Conversations that have been marked as closed, with the last message before closure.

Classify each new message accordingly and update the status of existing conversations based on these guidelines."
"""

close_conversations_prompt = """

Your task is to review open conversations and determine if they should be closed.
"""


"""
1. Split messages by time into chunks (long delays between messages)
2. Classify messages into an existing open conversation OR into the conversation buffer
3. Review the conversation buffer and create new conversations if any jump out, removing them from conversation buffer
4. Close topics
4. Evaluate orphan messages
"""

"""
Example Input 1:
    {"user": "UserD", "timestamp": "01/01/2024, 06:30 PM", "message": "We need to talk about the project deadline, team."},
    {"user": "UserA", "timestamp": "01/01/2024, 08:15 PM", "message": "Hey everyone, how about a game night this Saturday?"},
    {"user": "UserB", "timestamp": "01/01/2024, 08:20 PM", "message": "Sounds fun, I'm in!"},
    {"user": "UserC", "timestamp": "01/01/2024, 08:45 PM", "message": "Saturday works for me!"}
    {"user": "UserE", "timestamp": "01/01/2024, 07:00 PM", "message": "@UserD Agreed, it's too tight."},
    {"user": "UserG", "timestamp": "01/01/2024, 05:00 PM", "message": "Anyone have good movie suggestions?"},
    {"user": "UserF", "timestamp": "01/01/2024, 07:50 PM", "message": "I think we can extend it by two days."}
    {"user": "UserH", "timestamp": "01/01/2024, 05:05 PM", "message": "@UserG Check out 'Space Odyssey'!"},
    {"user": "UserG", "timestamp": "01/01/2024, 05:15 PM", "message": "Thanks for the suggestions, everyone!"}

Example Output 1:
[
    [
        {"user": "UserD", "timestamp": "01/01/2024, 06:30 PM", "message": "We need to talk about the project deadline, team."},
        {"user": "UserE", "timestamp": "01/01/2024, 07:00 PM", "message": "@UserD Agreed, it's too tight."},
        {"user": "UserF", "timestamp": "01/01/2024, 07:50 PM", "message": "I think we can extend it by two days."}
    ], [
        {"user": "UserA", "timestamp": "01/01/2024, 08:15 PM", "message": "Hey everyone, how about a game night this Saturday?"},
        {"user": "UserB", "timestamp": "01/01/2024, 08:20 PM", "message": "Sounds fun, I'm in!"},
        {"user": "UserC", "timestamp": "01/01/2024, 08:45 PM", "message": "Saturday works for me!"}
    [
        {"user": "UserG", "timestamp": "01/01/2024, 05:00 PM", "message": "Anyone have good movie suggestions?"},
        {"user": "UserH", "timestamp": "01/01/2024, 05:05 PM", "message": "@UserG Check out 'Space Odyssey'!"},
        {"user": "UserG", "timestamp": "01/01/2024, 05:15 PM", "message": "Thanks for the suggestions, everyone!"}
    ]
]
"""
    
"""
1. New Conversation:
{
  "Topic": "Game Night Planning",
  "Messages": ,
  "Participants": ["UserA", "UserB", "UserC"],
  "Status": "Open",
  "LastMessage": {"User": "UserC", "timestamp": "01/01/2024, 08:45 PM", "message": "Saturday works for me!"}
}

2. Existing Open Conversation:
{
  "Topic": "Project Deadline Discussion",
  "Messages": [
    {"User": "UserD", "timestamp": "01/01/2024, 06:30 PM", "message": "We need to talk about the project deadline, team."},
    {"User": "UserE", "timestamp": "01/01/2024, 07:00 PM", "message": "Agreed, it's too tight."},
    {"User": "UserF", "timestamp": "01/01/2024, 07:50 PM", "message": "I think we can extend it by two days."}
  ],
  "Participants": ["UserD", "UserE", "UserF"],
  "Status": "Open",
  "LastMessage": {"User": "UserF", "timestamp": "01/01/2024, 07:50 PM", "message": "I think we can extend it by two days."}
}

3. Closed Conversation:
{
  "Topic": "Movie Recommendations",
  "Messages": [
  ],
  "Participants": ["UserG", "UserH"],
  "Status": "Closed",
  "LastMessage": {"User": "UserG", "timestamp": "01/01/2024, 05:15 PM", "Content": "Thanks for the suggestions, everyone!"}
}
"""




class ConversationTextSplitter(TextSplitter):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        return []

def main():


    # lib_model.init(os.getenv("OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), os.getenv("PGVECTOR_CONNECTION_STRING"), os.getenv("RECORDMANAGER_CONNECTION_STRING"), temp=os.getenv("OPENAI_TEMPERATURE"))

    # vectordb = lib_doc_vectors.get_vectordb()
    # print(vectordb)

    loader = discord_loader.DiscordChatLoader('documents/misc-2024-01-02-15-50-45.txt')
    docs = loader.load()
    print(len(docs))

    splitter = ConversationTextSplitter()
    all_docs = splitter.split_documents(docs)
    print(f"Split into {len(all_docs)} documents")



    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(2000), chunk_overlap=200, add_start_index=True)
    # all_docs = text_splitter.split_documents(docs)

    # for doc in all_docs:
    #     text = doc.page_content
    #     metadata = doc.metadata
    #     print(f"Adding document title:{metadata['title']} author:{metadata['author']} guid:{metadata['guid']} link:{metadata['url']}  start_index:{metadata['start_index']} {len(text)} first 50 chars: {text[:50]}")
    #     print(doc.page_content)

    #     # lib_doc_vectors.add_doc(doc, metadata['guid'])
    # print(f"Adding {len(all_docs)} documents...")
    # lib_doc_vectors.bulk_add_docs(all_docs)
    # print("Done")



if __name__ == "__main__":
    main()