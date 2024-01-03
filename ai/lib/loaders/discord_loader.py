import re
from typing import List, Dict
from datetime import datetime
from langchain.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document

class DiscordChatLoader(UnstructuredFileLoader):
    def __init__(self, file_path):
        super().__init__(file_path)

    def load_messages(self) -> List[Dict]:
        # Open the file
        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()

        # Define a regular expression to match the Discord chat structure
        pattern = re.compile(r'\[(.*?)\] (.*?): (.*)')

        messages = []
        for line in content:
            match = pattern.match(line)
            if match:
                timestamp_str, user, message = match.groups()
                parsed_timestamp = datetime.fromisoformat(timestamp_str)

                print(f"{parsed_timestamp} {user} {message}")
                messages.append({
                    "timestamp": parsed_timestamp,
                    "user": user,
                    "message": message
                })

        return messages


    def load(self) -> List[Document]:
        # Open the file
        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()

        # Define a regular expression to match the Discord chat structure
        pattern = re.compile(r'\[(.*?)\] (.*?): (.*)')

        lines = []
        for line in content:
            match = pattern.match(line)
            if match:
                timestamp, user, message = match.groups()
                print(f"{timestamp} {user} {message}")
                lines.append(line)

        doc = Document(page_content="\n".join(lines), metadata={
            "type": "discord",
            "source": f"Discord Chat Export {self.file_path}",
            "filename": self.file_path
        })

        return [doc]

        # "title": title if title else 'Unknown Title',
        # "author": "Robbie Kramer",
        # 'type': 'wordpress',
        # "filename": filename,
        # "url": link,
        # "guid": guid,
        # "source": filename