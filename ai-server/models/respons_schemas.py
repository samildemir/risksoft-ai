from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ChatResponse(BaseModel):

    success: bool


class RiskAssessmentResponse(BaseModel):
    success: bool
    message: str
    data: Union[
        str, Dict[str, Any], None
    ]  # Flexible data type for different response types


class VectorStoreResponse(BaseModel):
    id: int
    account_id: int
    created_at: datetime
    document_count: int
    total_chunks: int
    embedding_model: str
    status: str
    metadata: Optional[dict] = None

    class Config:
        from_attributes = True


class VectorStoreOperationResponse(BaseModel):
    success: bool
    message: str


class ConversationResponse(BaseModel):
    content: str
    role: str


class AIResponse(BaseModel):
    """Schema for agent responses including usage metrics"""

    content: str

    prompt_tokens: Optional[int] = Field(
        None, description="Number of tokens in the prompt"
    )

    completion_tokens: Optional[int] = Field(
        None, description="Number of tokens in the completion"
    )

    input_tokens: Optional[int] = Field(
        None, description="Total input tokens processed"
    )

    output_tokens: Optional[int] = Field(
        None, description="Total output tokens generated"
    )

    cost: Optional[float] = Field(None, description="Total cost of the API call")

    model: Optional[str] = Field(None, description="Model used for generation")

    response_time_ms: Optional[int] = Field(
        None, description="Response time in milliseconds"
    )

    class Config:

        json_schema_extra = {
            "example": {
                "content": "İşte analiz sonuçlarınız...",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "input_tokens": 100,
                "output_tokens": 50,
                "cost": 0.002,
                "model": "gpt-4-test",
                "response_time_ms": 1500,
            }
        }


class SupportChatResponse(BaseModel):
    """Schema for support chat responses"""

    response: str = Field(..., description="AI assistant's response to the user")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    needsHumanSupport: bool = Field(
        ..., description="Whether the request should be escalated to human support"
    )
    intent: str = Field(..., description="Detected intent (general, escalate, error)")
    suggestions: List[str] = Field(
        default_factory=list, description="List of suggested actions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Size yardımcı olmaktan mutluluk duyarım...",
                "confidence": 0.8,
                "needsHumanSupport": False,
                "intent": "general",
                "suggestions": [],
            }
        }
