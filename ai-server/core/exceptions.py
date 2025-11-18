class ChatbotException(Exception):
    """Base exception for chatbot related errors"""
    pass

class ConversationNotFoundError(ChatbotException):
    """Raised when a conversation is not found"""
    pass

class UnauthorizedAccessError(ChatbotException):
    """Raised when a user tries to access unauthorized resources"""
    pass

class DatabaseOperationError(ChatbotException):
    """Raised when database operations fail"""
    pass

class AIServiceError(ChatbotException):
    """Raised when AI service operations fail"""
    pass 