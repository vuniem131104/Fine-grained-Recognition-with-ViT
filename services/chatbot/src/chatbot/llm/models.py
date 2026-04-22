from __future__ import annotations 

from typing import Optional 
from pydantic import BaseModel
from enum import Enum

class Role(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

class CompletionMessage(BaseModel):
    role: Role
    content: Optional[str] = None
    image_url: Optional[str] = None
    file_url: Optional[str] = None


Messages = list[CompletionMessage]

class LiteLLMInput(BaseModel):
    messages: Messages
    model: Optional[str] = None 
    response_format: type[BaseModel] | None = None
    temperature: Optional[float] = None 
    top_p: Optional[float] = None
    n: Optional[int] = None
    frequency_penalty: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None 
    
class LiteLLMOutput(BaseModel):
    response: BaseModel | str
    completion_tokens: int
    
class LiteLLMEmbeddingInput(BaseModel):
    text: str

class LiteLLMEmbeddingOutput(BaseModel):
    embedding: list[float]
    
