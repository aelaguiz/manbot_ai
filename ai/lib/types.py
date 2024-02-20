from datetime import datetime
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator, Extra, HttpUrl
from langchain_core.pydantic_v1 import validator
from langchain_core.language_models import BaseLLM
from typing import List, Optional

def current_timestamp():
    return datetime.now()

class ChatMessage(BaseModel):
    sender: str = Field(default="text", regex="^(client|coach)$")
    content: str
    msg_type: str = Field(default="text", regex="^(text|image)$")
    image_url: HttpUrl = None
    image_user: str = None
    image_password: str = None
    image_description: str = None
    timestamp: datetime = Field(default_factory=current_timestamp)
    
    @validator('image_url', always=True)
    def validate_image_fields(cls, v, values, **kwargs):
        if v and (not values.get('image_user') or not values.get('image_password')):
            raise ValueError('image_user and image_password must be provided with an image_url')
        return v

        
    def __str__(self):
        return f"{self.sender}: {self.content}"

    # Custom JSON encoder for datetime
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


    @classmethod
    def format_list_as_str(cls, chat_history: List['ChatMessage']):
        return "\n".join([f"{msg.sender}: {msg.content}" for msg in chat_history])