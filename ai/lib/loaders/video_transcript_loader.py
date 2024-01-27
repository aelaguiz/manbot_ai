import json
import os
from typing import List
from langchain.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document

class VideoTranscriptLoader(UnstructuredFileLoader):

    def load(self) -> List[Document]:
        # Read the JSON file
        with open(self.file_path, 'r') as file:
            data = json.load(file)

        # Create metadata dictionary
        metadata = {
            "type": "video",
            "title": data["title"],
            "source": f"Video {data['video_id']}",
            "video_id": data["video_id"],
            "video_url": data["video_url"],
            "description": data["description"],
            "published_at": data["published_at"],
            "channel_id": data["channel_id"],
            "channel_title": data["channel_title"],
            "tags": data["tags"],
            "category_id": data["category_id"],
            "duration": data["duration"],
            "view_count": data["view_count"],
            "like_count": data["like_count"],
            "comment_count": data["comment_count"],
            "channel_info": data["channel_info"]
        }

        try:
            transcript = json.loads(data['transcription'])
            transcript = transcript['text']
        except:
            transcript = data['transcription']

        # Create a single document for the file
        doc = Document(page_content=transcript, metadata=metadata)

        return [doc]
