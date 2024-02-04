import re
import json
from typing import List, Dict
from datetime import datetime
from langchain.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document

import logging
import hashlib

class WhatsAppChatLoader(UnstructuredFileLoader):
    def __init__(self, file_path):
        super().__init__(file_path)

    def load_messages(self) -> List[Dict]:
        logger = logging.getLogger(__name__)
        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()

        logger.info(f"Loading file {self.file_path}")

        # Define a regular expression to match the WhatsApp chat structure
        pattern = re.compile(r'\[(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}:\d{2}\s[AP]M)\]\s(.*?):\s(.*)')

        messages = []
        message_number = 1
        current_msg = None

        for line in content:
            match = pattern.match(line)
            if match:
                if current_msg:
                    # If there's an ongoing message, finalize it
                    messages.append(current_msg)
                    message_number += 1

                timestamp_str, user, message = match.groups()
                # WhatsApp uses a different timestamp format, adjust parsing accordingly
                parsed_timestamp = datetime.strptime(timestamp_str, '%m/%d/%y, %I:%M:%S %p')
                current_msg = {
                    "number": message_number,
                    'message_type': 'whatsapp',
                    "timestamp": parsed_timestamp,
                    "user": user,
                    "message": message
                }
            elif current_msg:
                # If the line doesn't match and there's a current message, append the line to the current message
                current_msg["message"] += "\n" + line.strip()
            else:
                # Log unmatched lines only if they are not just empty lines or separators
                if line.strip():
                    logger.warning(f"Failed to match line: {line}")

        # Add the last message if there is one
        if current_msg:
            messages.append(current_msg)

        logger.debug(f"Loaded {len(messages)} messages")
        return messages

    def load_messages_as_docs(self) -> List[Document]:
        messages = self.load_messages()
        doc = Document(page_content="\n".join(f"{c['timestamp'].strftime('%d/%m/%Y, %I:%M %p')} {c['user']}: {c['message']}" for c in messages), metadata={
            "type": "whatsapp_chat",
            "title": f"WhatsApp Chat {self.file_path}",
            "source": f"WhatsApp Chat Export {self.file_path}",
            "filename": self.file_path
        })

        content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
        metadata_hash = hashlib.sha256(json.dumps(doc.metadata).encode()).hexdigest()
        guid = f"{content_hash}-{metadata_hash}"
        doc.metadata['guid'] = guid
        return [doc]

    def load(self) -> List[Document]:
        logger = logging.getLogger(__name__)
        doc = self.load_messages_as_docs()[0]
        yield doc
