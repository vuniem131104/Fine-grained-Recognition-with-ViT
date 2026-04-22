import uuid
from typing import TypedDict, Optional


class ChatbotState(TypedDict):
    """State for the chatbot workflow"""
    original_query: str
    img_b64: Optional[str]
    enriched_queries: Optional[list[dict]]
    tool_results: Optional[dict]
    final_response: Optional[str]
    user_id: Optional[int]
    conversation_id: Optional[uuid.UUID]
    conversation_history: Optional[list[dict]]
    rephrased_original_query: Optional[str]