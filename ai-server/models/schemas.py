from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal, Type
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from core.database import Base
from models.enum import LogType, Status, MessageRole
from typing_extensions import TypedDict


# Vector Store and Document Processing Models
class DocumentSource(BaseModel):
    path: str
    name: str
    id: int
    description: Optional[str] = None


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


class ModelUsage(BaseModel):
    """Individual model usage details"""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    response_time_ms: int = 0


class ChatbotUsageLog(BaseModel):
    conversation_id: Optional[int] = None
    message_id: Optional[int] = None

    # List of model usages
    model_usages: List[ModelUsage] = Field(default_factory=list)

    # Aggregated totals
    total_tokens: int = 0
    total_cost: float = 0.0
    total_response_time_ms: int = 0

    # Log details
    log_type: LogType = LogType.INFO
    status: Status = Status.SUCCESS
    message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def add_usage(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost: float = 0.0,
        response_time_ms: int = 0,
    ):
        """Add usage for a specific model"""
        usage = ModelUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            response_time_ms=response_time_ms,
        )
        self.model_usages.append(usage)

        # Update totals
        self.total_tokens += total_tokens
        self.total_cost += cost
        self.total_response_time_ms += response_time_ms

    @classmethod
    def create_error_log(cls, message: str) -> "ChatbotUsageLog":
        """Create an error log instance"""
        return cls(log_type=LogType.ERROR, status=Status.ERROR, message=message)


class ChatHistory(BaseModel):
    """Schema for chat history entries"""

    model_config = ConfigDict(from_attributes=True)

    role: MessageRole
    content: str

    def __init__(self, role: str, content: str):
        super().__init__(role=role, content=content)

    def model_dump(self, **kwargs):
        return (self.role, self.content)


class ChatbotSqlTemplate(Base):
    """Model for storing SQL query templates for the chatbot."""

    __tablename__ = "chatbot_sql_templates"
    id = Column(Integer, primary_key=True)
    input_text = Column(String, nullable=False)
    query = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @staticmethod
    def format_templates(templates: List["ChatbotSqlTemplate"]) -> List[Dict[str, str]]:
        """
        Format SQL templates for few-shot learning.

        Args:
            templates: List of ChatbotSqlTemplate instances

        Returns:
            List of dictionaries with input and query fields
        """
        if not templates:
            return []

        return [
            {"input": template.input_text, "query": template.query}
            for template in templates
        ]


class AgentResponse(BaseModel):
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
                "model": "gpt-4",
                "response_time_ms": 1500,
            }
        }


class DetermineAnswerSourceResult(BaseModel):
    """Validated response payload for determine_answer_source."""

    sources: List[Literal["database", "document", "casual"]] = Field(
        default_factory=lambda: ["casual"],
        min_length=1,
        max_length=3,
        description=(
            "Knowledge sources the assistant should query "
            "(database for analytics, document for procedures, casual for small talk)."
        ),
    )
    improved_question: str = Field(
        ...,
        description=(
            "Latest user question after gently fixing obvious typos; "
            "should match the original text when no fixes are necessary."
        ),
    )
    casual_response: Optional[str] = Field(
        default=None,
        description=(
            "If the assistant decides only the 'casual' source is needed, "
            "this field should contain the final user-facing response."
        ),
    )

    def prioritized_sources(self) -> List[str]:
        """Remove casual when higher-fidelity sources exist."""
        prioritized = list(self.sources) or ["casual"]
        if len(prioritized) > 1 and "casual" in prioritized:
            prioritized = [s for s in prioritized if s != "casual"] or ["casual"]
        return prioritized


class DocumentQAResponse(BaseModel):
    """Structured answer for document QA responses."""

    answer: str = Field(..., description="Comprehensive answer synthesized from documents.")
    key_points: List[str] = Field(
        default_factory=list,
        description="Key bullet points extracted from the context.",
    )


class DocumentAnalysisResponse(BaseModel):
    """Structured answer for advanced document analysis responses."""

    answer: str = Field(..., description="Detailed Turkish response grounded in documents.")
    analysis_notes: List[str] = Field(
        default_factory=list,
        description="Short insights or caveats derived from the context.",
    )


class SQLQueryResponse(BaseModel):
    """Model for structured SQL generation responses."""

    sql_query: str = Field(
        ...,
        description="Final PostgreSQL query without trailing semicolons.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Short explanation of how the query answers the question.",
    )

DETERMINE_ANSWER_SOURCE_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "determine_answer_source",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "sources": {
                    "type": "array",
                    "description": (
                        "Unique list of knowledge sources the assistant must query "
                        "(database for analytics, document for procedures, casual for small talk)."
                    ),
                    "items": {
                        "type": "string",
                        "enum": ["database", "document", "casual"],
                    },
                    "minItems": 1,
                    "maxItems": 3,
                    "uniqueItems": True,
                },
                "improved_question": {
                    "type": "string",
                    "description": (
                        "The user's latest question after the AI gently fixes obvious typos "
                        "while keeping acronyms/intent intact; return the original text if no change is needed."
                    ),
                },
                "casual_response": {
                    "type": "string",
                    "description": (
                        "When only the casual source is selected, provide the final assistant response here."
                    ),
                },
            },
            "required": ["sources", "improved_question"],
        },
    },
}


def build_openrouter_schema(name: str, model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Helper to convert Pydantic models into OpenRouter response_format payloads."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": model_cls.model_json_schema(),
        },
    }


DOCUMENT_QA_RESPONSE_SCHEMA = build_openrouter_schema(
    "document_qa_response", DocumentQAResponse
)
DOCUMENT_ANALYSIS_RESPONSE_SCHEMA = build_openrouter_schema(
    "document_analysis_response", DocumentAnalysisResponse
)
SQL_QUERY_RESPONSE_SCHEMA = build_openrouter_schema(
    "sql_query_response", SQLQueryResponse
)


class RiskAssessmentModel(BaseModel):
    legal_basis: str = Field(description="Riskin yasal dayanaklarını belirtir.")
    affected_people: List[str] = Field(
        description="Riskten etkilenebilecek kişi veya grupları belirtir."
    )
    risks: str = Field(description="Riskin olası sonuçlarını veya etkilerini belirtir.")
    cautions: str = Field(
        description="Riski önlemek veya azaltmak için alınması gereken önlemleri belirtir."
    )
    current_cautions: str = Field(
        description="Mevcut durumda riskleri kontrol etmek için alınan önlemleri belirtir."
    )
    possibility: float = Field(
        description="Olasılık değeri (1-5 arasında veya 0.1-12 arasında)."
    )
    intensity: float = Field(
        description="Şiddet değeri (1-5 arasında veya 1-100 arasında)."
    )
    frequency: Optional[float] = Field(
        description="Sıklık değeri (0.5-10 arasında, sadece FINE_KINNEY için)."
    )
