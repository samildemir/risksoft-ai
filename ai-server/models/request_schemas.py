from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from models.enum import ChatMode
from models.respons_schemas import ConversationResponse
from models.schemas import DocumentSource


class RiskAssessmentRequest(BaseModel):
    account_id: int
    files: List[str]
    method: str


class SupportMetadata(BaseModel):
    user_id: Optional[int] = Field(None, description="Original user id")
    account_id: Optional[int] = Field(None, description="Support account id override")
    user: Optional[Dict[str, Any]] = Field(
        default=None, description="Snapshot of the user for support-specific context"
    )


class ChatRequest(BaseModel):
    content: str
    context: List[ConversationResponse] = Field(
        default_factory=list, description="Conversation history for context"
    )
    account_id: Optional[int] = Field(None, description="Account id for data access")
    siteMap: str = Field(default="", description="Optional sitemap text")
    mode: ChatMode = Field(default=ChatMode.STANDARD, description="Chat interaction mode")
    supportMetadata: Optional[SupportMetadata] = Field(
        default=None, description="Support-specific metadata"
    )


class SupportChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    context: Optional[str] = Field(default="support", description="Chat context")
    user_id: Optional[int] = Field(None, description="User ID")
    account_id: Optional[int] = Field(None, description="Account ID")


class GenerateConversationTitleRequest(BaseModel):
    messages: List[ConversationResponse]


class RiskAssessmentGenerationRequest(BaseModel):
    question_id: int
    question: str
    control_list_name: str
    keywords: Optional[str] = None
    uploaded_documents: Optional[List[Dict]] = None
    language: Optional[str] = "tr"


class DocumentEmbeddingRequest(BaseModel):
    documents: List[DocumentSource]
    account_id: int
    bucket_id: str


class RiskAssessmentQuestionGenerationRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Title cannot be empty")
    description: str = Field(
        ..., min_length=1, description="Description cannot be empty"
    )


class AIHelpAnalysisRequest(BaseModel):
    account_id: int
    files: List[str] = Field(
        ..., min_items=1, description="At least one image file is required"
    )
    additional_context: str = Field(
        ...,
        min_length=10,
        description="Additional context must be at least 10 characters",
    )
    question_context: Optional[Dict] = Field(
        None, description="Optional context about the current question"
    )
    language: str = Field(default="tr", description="Response language")
