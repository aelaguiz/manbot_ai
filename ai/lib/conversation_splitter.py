import sys
import os
import typing
from typing import Optional, Union, Literal, AbstractSet, Collection, Any, List
from langchain.text_splitter import TextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from . import lib_model, lc_logger


import logging
import logging.config
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.connection").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
logging.getLogger("openai._base_client").setLevel(logging.CRITICAL)


"""
I think the steps should be:

1. Review the messages and extract all unique conversations, parciipants and when and who started the conversation
2. Classify messages one by one into the conversation they most likely fit into including
"""

topic_prompt_template = """# Discord message conversation classification

Your task is to analyze a list of Discord messages and identify unique conversations. For each conversation, extract the following information:

1. Conversation Topic: Determine the main topic or subject of the conversation.
2. Participants: List all users who have participated in the conversation.
3. Initial Message: Identify the first message that started the conversation, including who said it and when.

Consider the content of the messages, the participants involved, and the flow of the conversation to distinguish between different topics. Ignore time gaps between messages, as conversations can be ongoing over extended periods.

Analyze the messages and output the extracted conversations with the required details.

## Guidelines for conversation topics

1. Topic Consistency: Group messages into a conversation if they discuss the same main topic.
2. Response Chain: Include a message in a conversation if it directly responds to or follows up on an earlier message.
3. Extended Time Window: In addition to the contextual markets you can also use a 2-hour window to determine if messages belong to the same conversation when other methods fail.
4. Natural Conversation Flow: Identify the start and end of conversations based on the natural flow of messages.
5. Multiple Conversations: Recognize and categorize simultaneous conversations based on topic relevance.


## Examples

### Example 1:
Example Input:
```json
[
    {{"user": "UserA", "timestamp": "01/01/2024, 08:15 PM", "message": "Hey everyone, how about a game night this Saturday?"}},
    {{"user": "UserB", "timestamp": "01/01/2024, 08:20 PM", "message": "Sounds fun, I'm in!"}},
    {{"user": "UserC", "timestamp": "01/01/2024, 08:45 PM", "message": "Saturday works for me!"}}
]
```

Expected Output:
```json
[
    {{
        "topic": "Game Night Planning",
        "participants": ["UserA", "UserB", "UserC"],
        "first_message": {{"user": "UserA", "timestamp": "01/01/2024, 08:15 PM", "message": "Hey everyone, how about a game night this Saturday?"}}
    }}
]
```

### Example 2:
Example Input:
```json
[
    {{"user": "UserX", "timestamp": "01/01/2024, 10:00 AM", "message": "Has anyone seen the latest space documentary?"}},
    {{"user": "UserY", "timestamp": "01/01/2024, 10:05 AM", "message": "Yes, watched it last night. It's mind-blowing!"}},
    {{"user": "UserZ", "timestamp": "01/01/2024, 10:15 AM", "message": "I think our project needs a new approach."}},
    {{"user": "UserX", "timestamp": "01/01/2024, 10:20 AM", "message": "Totally agree on the documentary. What did you think of the Mars segment, UserY?"}},
    {{"user": "UserA", "timestamp": "01/01/2024, 10:30 AM", "message": "@UserZ, I'm open to suggestions. What are you thinking?"}},
    {{"user": "UserY", "timestamp": "01/01/2024, 10:35 AM", "message": "Mars segment was the best. Also, @UserZ, are you proposing a complete overhaul?"}},
    {{"user": "UserZ", "timestamp": "01/01/2024, 10:40 AM", "message": "Not a complete overhaul, but significant changes. Let's discuss this afternoon."}},
    {{"user": "UserB", "timestamp": "01/01/2024, 10:45 AM", "message": "I missed the documentary. Can anyone summarize it?"}},
    {{"user": "UserA", "timestamp": "01/01/2024, 10:50 AM", "message": "Looking forward to the meeting, @UserZ. We definitely need fresh ideas."}}
]
```

Expected Output:
```json
[
    {{
        "topic": "Space Documentary Discussion",
        "participants": ["UserX", "UserY", "UserB"],
        "first_message": {{"user": "UserX", "timestamp": "01/01/2024, 10:00 AM", "message": "Has anyone seen the latest space documentary?"}}
    }},
    {{
        "topic": "Project Strategy Discussion",
        "participants": ["UserZ", "UserA", "UserY"],
        "first_message": {{"user": "UserZ", "timestamp": "01/01/2024, 10:15 AM", "message": "I think our project needs a new approach."}}
    }}
]```

# INPUT MESSAGES:
{input_messages}
"""


message_classify_prompt_template = """# Discord message classification

Given a list of new Discord messages and the identified conversation topics from Step 1, your task now is to classify each of these new messages into the most relevant existing conversation. Use the following criteria for your classification:

1. Message Content: Compare the content of each new message to the topics of existing conversations.
2. Participants: Consider if the sender of the new message is already a participant in an existing conversation.
3. Mentions and Context: Look for direct mentions (@username) and contextual hints in the message that might link it to an existing conversation.

## Guidelines for Classification

1. Topic Relevance: Assign a message to a conversation where its content closely aligns with the identified topic.
2. Existing Participants: If a message is from a user already participating in a conversation, it's likely to belong to that conversation.
3. Conversational Flow: Use any indications of ongoing dialogue, like direct responses or follow-up questions, to classify messages.

## Examples

### Example 1

Messages:
```json
[
    {"user": "UserX", "timestamp": "01/01/2024, 10:00 AM", "message": "Has anyone seen the latest space documentary?"},
    {"user": "UserZ", "timestamp": "01/01/2024, 10:15 AM", "message": "I think our project needs a new approach."},
    {"user": "UserY", "timestamp": "01/01/2024, 10:05 AM", "message": "Yes, watched it last night. It's mind-blowing!"},
    {"user": "UserA", "timestamp": "01/01/2024, 08:15 PM", "message": "Hey everyone, how about a game night this Saturday?"},
    {"user": "UserB", "timestamp": "01/01/2024, 08:20 PM", "message": "Sounds fun, I'm in!"},
    {"user": "UserC", "timestamp": "01/01/2024, 08:45 PM", "message": "Saturday works for me!"},
    {"user": "UserD", "timestamp": "01/01/2024, 09:00 PM", "message": "What time are we starting the game night?"}
]
```

Conversations:
```json
[
    {
        "topic": "Space Documentary Discussion",
        "participants": ["UserX", "UserY"],
        "first_message": {"user": "UserX", "timestamp": "01/01/2024, 10:00 AM", "message": "Has anyone seen the latest space documentary?"}
    },
    {
        "topic": "Game Night Planning",
        "participants": ["UserA", "UserB", "UserC"],
        "first_message": {"user": "UserA", "timestamp": "01/01/2024, 08:15 PM", "message": "Hey everyone, how about a game night this Saturday?"}
    },
    {
        "topic": "Project Strategy Discussion",
        "participants": ["UserZ"],
        "first_message": {"user": "UserZ", "timestamp": "01/01/2024, 10:15 AM", "message": "I think our project needs a new approach."}
    }
]
```

Expected Output:
```json
[
    {
        "topic": "Space Documentary Discussion",
        "messages": [
            {"user": "UserX", "timestamp": "01/01/2024, 10:00 AM", "message": "Has anyone seen the latest space documentary?"},
            {"user": "UserY", "timestamp": "01/01/2024, 10:05 AM", "message": "Yes, watched it last night. It's mind-blowing!"}
        ]
    },
    {
        "topic": "Game Night Planning",
        "messages": [
            {"user": "UserA", "timestamp": "01/01/2024, 08:15 PM", "message": "Hey everyone, how about a game night this Saturday?"},
            {"user": "UserB", "timestamp": "01/01/2024, 08:20 PM", "message": "Sounds fun, I'm in!"},
            {"user": "UserC", "timestamp": "01/01/2024, 08:45 PM", "message": "Saturday works for me!"},
            {"user": "UserD", "timestamp": "01/01/2024, 09:00 PM", "message": "What time are we starting the game night?"}
        ]
    },
    {
        "topic": "Project Strategy Discussion",
        "messages": [
            {"user": "UserZ", "timestamp": "01/01/2024, 10:15 AM", "message": "I think our project needs a new approach."},
        ]
    }
]
```

# INPUT MESSAGES:
{input_messages}

# EXISTING CONVERSATIONS:
{identified_conversations}
"""


from datetime import timedelta

def split_into_time_chunks(messages, interval_minutes=60):
    """Splits messages into chunks based on time intervals.

    Args:
        messages (list): List of message dictionaries with 'timestamp' as datetime objects.
        interval_minutes (int): Time interval in minutes for each chunk.

    Returns:
        list: List of message chunks, each chunk being a list of messages.
    """

    if not messages:
        return []

    # Sort messages by timestamp
    messages = sorted(messages, key=lambda x: x['timestamp'])

    chunks = []
    current_chunk = []
    chunk_start_time = messages[0]['timestamp']

    for message in messages:
        if message['timestamp'] < chunk_start_time + timedelta(minutes=interval_minutes):
            current_chunk.append(message)
        else:
            chunks.append(current_chunk)
            current_chunk = [message]
            chunk_start_time = message['timestamp']

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def format_message_for_json(message):
    """Formats a message dictionary for JSON serialization, converting datetime to string."""
    formatted_message = message.copy()
    formatted_message['timestamp'] = message['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    return formatted_message


def split_conversations(messages: List[str]) -> List[str]:
    logger = logging.getLogger(__name__)
    logger.debug(f"Splitting conversation on {len(messages)} messages")
    time_chunks = split_into_time_chunks(messages, interval_minutes=60*24*2)
    logger.debug(f"Split into {len(time_chunks)} time chunks")

    total_msgs = 0
    for idx, chunk in enumerate(time_chunks):
        logger.debug(f"Chunk {idx+1}: {len(chunk)} messages")
        logger.debug(f"\tFirst message: {chunk[0]}")
        logger.debug(f"\tLast message: {chunk[-1]}")
        total_msgs += len(chunk)

    logger.debug(f"Counted {total_msgs} messages in {len(time_chunks)} chunks")

    llm = lib_model.get_llm()
    lmd = lc_logger.LlmDebugHandler()

    topic_prompt = PromptTemplate.from_template(topic_prompt_template)
    message_classify_prompt = PromptTemplate.from_template(message_classify_prompt_template)

    topic_chain = (
        topic_prompt
        | llm
        | StrOutputParser()
    )
    
    message_classify_chain = (
        message_classify_prompt
        | llm
        | StrOutputParser()
    )

    im_string = "\n".join(json.dumps(format_message_for_json(m)) for m in chunk)
    res = topic_chain.batch([{"input_messages": im_string} for chunk in time_chunks], config={'callbacks': [lmd]})
    logger.debug(res)

    ## TODO: Need to have the topic_chain return json that goes in to the message_classify_chain as identified_conversations

    res2 = message_classify_chain.batch(
        {
            "input_messages": im_string,
            "identified_conversations": json.dumps(res)
        }
    )
    return []