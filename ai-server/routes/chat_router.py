from fastapi import APIRouter, status
from models.request_schemas import (
    ChatRequest,
    GenerateConversationTitleRequest,
    SupportChatRequest,
)
from business.chatbot import Chatbot


router = APIRouter(prefix="/chat", tags=["chat"])


def get_chatbot_service() -> Chatbot:
    return Chatbot()


@router.post("/agent", status_code=status.HTTP_200_OK)
async def interact_with_agent(
    request: ChatRequest,
):
    """Interact with agent"""
    service = get_chatbot_service()
    return await service.interact_with_agent(request)


@router.post("/agent/title", status_code=status.HTTP_200_OK)
async def generate_conversation_title(
    request: GenerateConversationTitleRequest,
):
    """Generate conversation title"""
    service = get_chatbot_service()
    return await service.generate_conversation_title(request.messages)


@router.post("/support", status_code=status.HTTP_200_OK)
async def support_chat(
    request: SupportChatRequest,
):
    """
    Handle support chat requests
    Returns AI response with confidence score and escalation flag
    """
    service = get_chatbot_service()
    return await service.handle_support_chat(request)
