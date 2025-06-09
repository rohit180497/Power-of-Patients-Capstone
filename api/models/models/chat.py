from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class ChatMessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "message": "I've been having headaches recently. What should I do?",
                "session_id": "d290f1ee-6c54-4b01-90e6-d701748f0851"
            }
        }

class ChatMessageResponse(BaseModel):
    id: int
    message: str
    is_patient: bool
    session_id: str
    created_at: str
    
    class Config:
        schema_extra = {
            "example": {
                "id": 42,
                "message": "I'm sorry to hear about your headaches. Can you tell me more about their frequency and intensity?",
                "is_patient": False,
                "session_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
                "created_at": "2025-05-15T14:30:45.123Z"
            }
        }

class ChatMessage(BaseModel):
    id: int
    message: str
    is_patient: bool
    created_at: str
    
    class Config:
        schema_extra = {
            "example": {
                "id": 42,
                "message": "I've been having headaches recently. What should I do?",
                "is_patient": True,
                "created_at": "2025-05-15T14:30:00.123Z"
            }
        }

class ChatSession(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
                "created_at": "2025-05-15T14:30:00.123Z",
                "updated_at": "2025-05-15T14:45:00.123Z",
                "message_count": 10
            }
        }

class ChatSessionList(BaseModel):
    sessions: List[ChatSession]
    
    class Config:
        schema_extra = {
            "example": {
                "sessions": [
                    {
                        "session_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
                        "created_at": "2025-05-15T14:30:00.123Z",
                        "updated_at": "2025-05-15T14:45:00.123Z",
                        "message_count": 10
                    },
                    {
                        "session_id": "e290f1ee-6c54-4b01-90e6-d701748f0852",
                        "created_at": "2025-05-14T10:30:00.123Z",
                        "updated_at": "2025-05-14T11:45:00.123Z",
                        "message_count": 15
                    }
                ]
            }
        }

class ChatHistory(BaseModel):
    messages: List[ChatMessage]
    
    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {
                        "id": 41,
                        "message": "I've been having headaches recently. What should I do?",
                        "is_patient": True,
                        "created_at": "2025-05-15T14:30:00.123Z"
                    },
                    {
                        "id": 42,
                        "message": "I'm sorry to hear about your headaches. Can you tell me more about their frequency and intensity?",
                        "is_patient": False,
                        "created_at": "2025-05-15T14:30:45.123Z"
                    }
                ]
            }
        }

class NewChatSessionResponse(BaseModel):
    session_id: str
    created_at: str
    message: str
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
                "created_at": "2025-05-15T14:30:00.123Z",
                "message": "Chat session created successfully"
            }
        }