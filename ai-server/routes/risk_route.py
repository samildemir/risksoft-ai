from fastapi import APIRouter, status

from business.risk_service import RiskService
from models.request_schemas import (
    RiskAssessmentGenerationRequest,
    RiskAssessmentQuestionGenerationRequest,
)
from models.respons_schemas import RiskAssessmentResponse

router = APIRouter(prefix="/risk", tags=["risk"])


def get_risk_service() -> RiskService:
    return RiskService()


@router.post(
    "/analysis",
    response_model=RiskAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def analyze_risk_factors(
    request: RiskAssessmentGenerationRequest,
):
    """
    Generate AI help analysis for a specific risk assessment question.
    The process involves:
    1. Question analysis with control list context
    2. AI-powered response generation
    3. Markdown formatted response with recommendations
    """
    service = get_risk_service()
    print("request", request.uploaded_documents)
    return await service.generate_ai_help_analysis(
        question_id=request.question_id,
        question=request.question,
        control_list_name=request.control_list_name,
        keywords=request.keywords,
        uploaded_documents=request.uploaded_documents,
        language=request.language or "tr",
    )


@router.post(
    "/questions",
    response_model=RiskAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def generate_risk_assessment_question(
    request: RiskAssessmentQuestionGenerationRequest,
):
    try:
        # Debug: Log the request data
        print(
            f"Received request: title='{request.title}', description='{request.description}'"
        )
        print(
            f"Title length: {len(request.title)}, Description length: {len(request.description)}"
        )

        service = get_risk_service()
        return await service.generate_risk_assessment_question(request)
    except Exception as e:
        print(f"Error in generate_risk_assessment_question: {str(e)}")
        raise e
