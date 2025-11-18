import json
import logging
from typing import Dict, List, Optional

from core.exceptions import ChatbotException
from models.schemas import ChatbotUsageLog
from models.request_schemas import RiskAssessmentQuestionGenerationRequest
from prompts.risk_assesmet_prompts import RiskAssessmentPrompts
from services.open_router_service import OpenRouterService

logger = logging.getLogger(__name__)


class RiskService:
    def __init__(self):
        """Initialize risk analysis service dependencies."""
        try:
            self.prompts = RiskAssessmentPrompts()
            self.llm_service = OpenRouterService()
            logger.info("Risk service initialized successfully")
        except Exception as exc:
            logger.error(f"Error initializing risk service: {exc}")
            raise ChatbotException("Failed to initialize risk service") from exc

    async def generate_ai_help_analysis(
        self,
        question_id: int,
        question: str,
        control_list_name: str,
        keywords: Optional[str] = None,
        uploaded_documents: Optional[List] = None,
        language: str = "tr",
    ) -> Dict:
        """Generate AI help analysis for a specific risk assessment question."""
        try:
            usage_log = ChatbotUsageLog()

            prompt_payload = self.prompts.build_ai_help_prompt_payload(
                question=question,
                control_list_name=control_list_name,
                keywords=keywords,
                uploaded_documents=uploaded_documents,
                language=language,
                question_id=question_id,
            )
            prompt = prompt_payload["prompt"]
            image_urls = prompt_payload.get("image_urls") or []

            if image_urls:
                # Send prompt + actual images per https://openrouter.ai/docs/features/multimodal/images
                response = self.llm_service.multimodal_completion(
                    text=prompt,
                    image_urls=image_urls,
                    usage_log=usage_log,
                    response_format={"type": "json_object"},
                )
            else:
                response = self.llm_service.generate_text(
                    prompt,
                    usage_log=usage_log,
                    response_format={"type": "json_object"},
                )

            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            response_text = (
                response_text.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )

            try:
                parsed_response = json.loads(response_text)
                if isinstance(parsed_response, list) and len(parsed_response) > 0:
                    parsed_response = parsed_response[0]

                return {
                    "success": True,
                    "message": "AI help analysis generated successfully",
                    "data": parsed_response,
                    "usage_log": usage_log,
                }
            except json.JSONDecodeError as exc:
                logger.error(f"JSON parse error: {exc}, Response: {response_text}")
                return {
                    "success": False,
                    "message": f"Failed to parse AI response: {exc}",
                    "data": None,
                }
        except Exception as exc:
            logger.error(f"Error generating AI help analysis: {exc}")
            return {"success": False, "message": str(exc), "data": None}

    async def generate_risk_assessment_question(
        self, request: RiskAssessmentQuestionGenerationRequest
    ) -> Dict:
        """Generate risk assessment question from documents."""
        try:
            usage_log = ChatbotUsageLog()
            prompt = self.prompts.generate_risk_assessment_questions(
                request.title, request.description
            )
            response = self.llm_service.generate_text(
                prompt,
                usage_log=usage_log,
            )

            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            return {
                "success": True,
                "message": "Risk assessment question generated successfully",
                "data": response_text,
            }
        except Exception as exc:
            logger.error(f"Error generating risk assessment question: {exc}")
            raise
