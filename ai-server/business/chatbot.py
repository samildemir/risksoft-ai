import logging
import asyncio
from typing import Any, Dict, Optional, List, Tuple, Awaitable, Set

from pydantic import ValidationError

from services.open_router_service import OpenRouterService
from services.semantic_search_service import SemanticSearchService
from services.sql_query_agent_service import SQLQueryAgentService
from models.schemas import (
    ChatbotUsageLog,
    DETERMINE_ANSWER_SOURCE_SCHEMA,
    DetermineAnswerSourceResult,
)
from models.respons_schemas import SupportChatResponse
from core.exceptions import ChatbotException
from models.enum import ChatMode
from prompts.chatbot_prompts import (
    build_conversation_title_prompt,
    build_routing_prompt,
    build_service_response_prompt,
)
from models.request_schemas import (
    ChatRequest,
    ConversationResponse,
    SupportChatRequest,
    SupportMetadata,
)

logger = logging.getLogger(__name__)
database_keywords = [
    # database keywords tr-TR
    "dfi",
    "olay bildirimi",
    "Emniyet Turu",
    "Operasyonal Denetim Raporu",
    "ODR",
    "secg",
    "secg iç denetim raporu",
    "Denetim Raporu",
    "Risk Analizi",
    "istatistik",
    # database keywords en-US
    "dfi",
    "incident report",
    "safety report",
    "operational audit report",
    "odr",
    "secg internal audit report",
    "audit report",
    "risk assessment",
    "statistics",
    "numbers",
]
document_keywords = [
    "document",
    "policy",
    "procedure",
    "general knowledge",
]


class Chatbot:
    def __init__(self):
        """Initialize chatbot with required services."""
        try:
            self.semantic_search_service = SemanticSearchService()
            self.sql_query_agent_service = SQLQueryAgentService()

            # Initialize OpenRouter service
            self.openrouter_service = OpenRouterService()

            logger.info("Chatbot services initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing chatbot services: {str(e)}")

            raise ChatbotException("Failed to initialize chatbot services")

    async def generate_conversation_title(
        self, messages: List[ConversationResponse]
    ) -> Dict[str, Any]:
        """Generate a title for a conversation."""
        try:
            # Create a short conversation summary for title generation
            conversation_text = self._serialize_context(messages, limit=3)

            prompt = build_conversation_title_prompt(conversation_text)

            usage_log = ChatbotUsageLog()

            response = self.openrouter_service.generate_text(
                prompt=prompt,
                model="google/gemini-2.5-flash-lite-preview-06-17",
                temperature=0.3,
                usage_log=usage_log,
            )

            result = response.content.strip()

            return {"success": True, "title": result, "usage_log": usage_log}

        except Exception as e:
            logger.error(f"Error generating conversation title: {str(e)}")
            return {
                "success": False,
                "title": "Conversation Title",
                "usage_log": ChatbotUsageLog.create_error_log(str(e)),
            }

    async def plan_answer_route(
        self,
        question: str,
        context: List[ConversationResponse],
        site_map: Optional[str],
        mode: ChatMode = ChatMode.STANDARD,
    ) -> tuple[List[str], str, ChatbotUsageLog, Optional[str]]:
        """Determine best answer route and optionally provide immediate casual response."""
        try:
            usage_log = ChatbotUsageLog()
            immediate_response: Optional[str] = None

            sanitized_history = self._serialize_context(context, limit=5)
            prompt = build_routing_prompt(
                question=question,
                conversation_history=sanitized_history,
                database_keywords=database_keywords,
                document_keywords=document_keywords,
                conversation_mode=mode.value if mode else ChatMode.STANDARD.value,
                site_map=site_map or "",
            )

            response = self.openrouter_service.generate_text(
                prompt=prompt,
                model="meta-llama/llama-4-scout",
                temperature=0.1,
                usage_log=usage_log,
                response_format=DETERMINE_ANSWER_SOURCE_SCHEMA,
            )

            sources = ["casual"]
            improved_question = question

            casual_answer: Optional[str] = None
            try:
                parsed_response = DetermineAnswerSourceResult.model_validate_json(
                    response.content.strip()
                )
                normalized_sources = parsed_response.prioritized_sources()
                if mode == ChatMode.SUPPORT and "casual" in (normalized_sources or []):
                    normalized_sources = ["casual"]
                if normalized_sources:
                    sources = normalized_sources

                cleaned_question = parsed_response.improved_question.strip()
                if cleaned_question:
                    improved_question = cleaned_question

                text_response = (parsed_response.casual_response or "").strip()
                if text_response:
                    casual_answer = text_response

            except (ValidationError, ValueError) as parse_error:
                logger.warning(
                    "Structured routing response could not be parsed: %s", parse_error
                )

            requires_structured = any(
                src in {"database", "document"} for src in sources or []
            )
            if not requires_structured:
                improved_question = question
                immediate_response = casual_answer or (
                    "Şu anda isteğinizi yerine getiremedim. Lütfen tekrar dener misiniz?"
                )

            return sources, improved_question, usage_log, immediate_response

        except Exception as e:
            logger.error(f"Error determining answer source: {str(e)}")
            # Default to document search on error, return as list
            return (
                ["document"],
                question,
                ChatbotUsageLog.create_error_log(str(e)),
                None,
            )

    def _update_usage_log(
        self, target: ChatbotUsageLog, source: Optional[ChatbotUsageLog]
    ) -> None:
        """Utility to merge model usage entries."""
        if not target or not source:
            return
        for usage in source.model_usages:
            target.add_usage(
                model=usage.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cost=usage.cost,
                response_time_ms=usage.response_time_ms,
            )

    def _format_sql_templates(self) -> Dict[str, Any]:
        templates = getattr(self.sql_query_agent_service, "chatbot_sql_templates", [])
        if not templates:
            return {"text": "", "list": []}

        lines = []
        serialized = []
        for template in templates:
            # template is tuple (input_text, query, description)
            input_text, query, description = template
            lines.append(
                f"Input: {input_text}\nQuery: {query}\nDescription: {description or '—'}"
            )
            serialized.append(
                {
                    "input": input_text,
                    "query": query,
                    "description": description,
                }
            )
        return {
            "text": "\n\n".join(lines),
            "list": serialized,
        }

    @staticmethod
    def _serialize_context(
        context: Optional[List[ConversationResponse]], limit: Optional[int] = None
    ) -> str:
        """Convert a list of conversation entries to a printable string."""
        if not context:
            return ""
        selected = context[-limit:] if limit else context
        return "\n".join(f"{msg.role}: {msg.content}" for msg in selected if msg)

    async def _execute_answer_sources(
        self,
        answer_sources: List[str],
        question: str,
        context: List[ConversationResponse],
        account_id: int,
    ) -> Tuple[str, str, ChatbotUsageLog, List[str], Dict[str, Any]]:
        """Execute the necessary services based on determined sources."""
        deduped_sources: List[str] = []
        seen_sources: Set[str] = set()
        for source in answer_sources or ["casual"]:
            normalized = (source or "casual").strip().lower()
            if normalized not in {"database", "document", "casual"}:
                continue
            if normalized in seen_sources:
                continue
            deduped_sources.append(normalized)
            seen_sources.add(normalized)

        context = context or []

        # Limit heavy services (SQL/document) to the first prioritized source
        heavy_seen = False
        prioritized_sources: List[str] = []
        for source in deduped_sources:
            if source in {"database", "document"}:
                if heavy_seen:
                    continue
                heavy_seen = True
            prioritized_sources.append(source)
        deduped_sources = prioritized_sources

        if not account_id:
            raise ChatbotException("Structured sources require a valid account_id.")

        if not deduped_sources:
            deduped_sources = ["casual"]

        logger.debug(
            "Active answer sources resolved to %s for account_id=%s",
            deduped_sources,
            account_id,
        )

        combined_usage = ChatbotUsageLog()
        extras: Dict[str, Any] = {}

        async_calls: List[Tuple[str, Awaitable[Tuple[str, ChatbotUsageLog]]]] = []

        if "database" in deduped_sources:
            async_calls.append(
                (
                    "database",
                    self.sql_query_agent_service.advanced_database_chat(
                        question=question, account_id=account_id
                    ),
                )
            )
        if "document" in deduped_sources:
            async_calls.append(
                (
                    "document",
                    self.semantic_search_service.advanced_document_chat(
                        question=question, account_id=account_id
                    ),
                )
            )

        gathered = await asyncio.gather(*[coro for _, coro in async_calls])
        section_texts: Dict[str, str] = {}
        active_sources: List[str] = []

        for (name, _), outcome in zip(async_calls, gathered):
            content, usage = outcome
            self._update_usage_log(combined_usage, usage)
            active_sources.append(name)
            section_texts[name] = (content or "").strip()

        label_map = {
            "database": "Database Result",
            "document": "Document Result",
        }
        if len(active_sources) == 1:
            result = section_texts.get(active_sources[0], "")
        else:
            sections = [
                f"{label_map.get(name, name.title())}:\n{section_texts.get(name, '')}"
                for name in active_sources
            ]
            result = "\n\n".join(sections).strip()

        if "database" in active_sources:
            template_payload = self._format_sql_templates()
            if template_payload["text"]:
                suffix = (
                    f"Kullanılabilir SQL Şablonları:\n{template_payload['text']}"
                ).strip()
                result = f"{result}\n\n{suffix}".strip()
            extras["sql_templates"] = template_payload["list"]
            extras["sql_templates_text"] = template_payload["text"]

        processed_result = result
        try:
            history = self._serialize_context(context)
            prompt = build_service_response_prompt(
                question=question,
                source_types=active_sources,
                raw_result=result,
                conversation_history=history,
            )
            processing_usage = ChatbotUsageLog()
            response = self.openrouter_service.generate_text(
                prompt=prompt,
                model="meta-llama/llama-4-scout",
                temperature=0.7,
                usage_log=processing_usage,
            )
            processed_result = response.content.strip()
            self._update_usage_log(combined_usage, processing_usage)
        except Exception as e:
            logger.error(f"Error processing service result: {str(e)}")
            self._update_usage_log(
                combined_usage, ChatbotUsageLog.create_error_log(str(e))
            )

        return processed_result, result, combined_usage, active_sources, extras

    async def _run_chat_pipeline(
        self,
        *,
        message: str,
        context: Optional[List[ConversationResponse]],
        account_id: Optional[int],
        site_map: Optional[str],
        mode: ChatMode = ChatMode.STANDARD,
        raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        """Build the full chat response without bouncing through multiple helpers."""
        context = context or []
        site_map = site_map or ""

        try:
            aggregated_usage = ChatbotUsageLog()

            # Determine answer route
            (
                answer_sources,
                refined_question,
                source_usage,
                immediate_response,
            ) = await self.plan_answer_route(message, context, site_map, mode=mode)
            self._update_usage_log(aggregated_usage, source_usage)

            if immediate_response is not None:
                return {
                    "answer": immediate_response,
                    "raw_result": immediate_response,
                    "usage_log": aggregated_usage,
                    "sources": answer_sources or ["casual"],
                    "improved_question": message,
                    "extras": {},
                }

            active_question = refined_question

            processed_result, raw_result, service_usage, active_sources, extras = (
                await self._execute_answer_sources(
                    answer_sources, active_question, context, account_id
                )
            )
            self._update_usage_log(aggregated_usage, service_usage)

            return {
                "answer": processed_result,
                "raw_result": raw_result,
                "usage_log": aggregated_usage,
                "sources": active_sources,
                "improved_question": active_question,
                "extras": extras,
            }
        except Exception as exc:
            if raise_on_error:
                raise
            logger.warning(f"Chat pipeline fallback: {exc}")
            usage_log = ChatbotUsageLog.create_error_log(str(exc))
            return {
                "answer": "",
                "raw_result": "",
                "usage_log": usage_log,
                "sources": ["casual"],
                "improved_question": message,
                "extras": {},
            }

    async def interact_with_agent(self, request: ChatRequest):
        """Process user message and generate AI response using relevant services."""
        try:
            pipeline = await self._run_chat_pipeline(
                message=request.content,
                context=request.context,
                account_id=request.account_id,
                site_map=request.siteMap,
                mode=request.mode,
                raise_on_error=request.mode != ChatMode.SUPPORT,
            )

            response_payload = {
                "success": True,
                "response": pipeline["answer"],
                "usage_logs": pipeline["usage_log"],
                "conversation_type": ", ".join(pipeline["sources"]),
                "improved_question": pipeline["improved_question"],
                "mode": request.mode.value,
            }

            return response_payload

        except Exception as e:
            logger.error(f"Error in interact_with_agent: {str(e)}")
            return {
                "success": False,
                "response": "I encountered an error processing your request. Please try again.",
                "conversation_type": "error",
                "usage_logs": ChatbotUsageLog.create_error_log(str(e)),
            }

    async def handle_support_chat(
        self, request: SupportChatRequest
    ) -> SupportChatResponse:
        """
        Handle support chat requests with AI
        Returns response with confidence score and escalation recommendation
        """
        try:
            message = request.message

            chat_request = ChatRequest(
                content=message,
                context=[],
                account_id=request.account_id,
                siteMap="",
                mode=ChatMode.SUPPORT,
                supportMetadata=SupportMetadata(
                    user_id=request.user_id,
                    account_id=request.account_id,
                ),
            )

            agent_response = await self.interact_with_agent(chat_request)

            logger.info(f"Agent response: {agent_response}")
            base_response = agent_response.get("response", "").strip()
            if not base_response:
                logger.warning("Empty support response, escalating to human support")
                return self._build_support_error_response()

            return SupportChatResponse(
                response=base_response,
                confidence=0.7,
                needsHumanSupport=False,
                intent="general",
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Error in handle_support_chat: {str(e)}")
            return self._build_support_error_response()

    @staticmethod
    def _build_support_error_response() -> SupportChatResponse:
        """Return a consistent fallback response when support handling fails."""
        return SupportChatResponse(
            response="Üzgünüm, bir hata oluştu. Lütfen canlı destek talep edin.",
            confidence=0.0,
            needsHumanSupport=True,
            intent="error",
            suggestions=["Canlı destek talep et"],
        )
