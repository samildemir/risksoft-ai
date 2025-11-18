from enum import Enum


class LogType(str, Enum):
    TOKEN_USAGE = "token_usage"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SECURITY = "security"
    PERFORMANCE = "performance"


class Status(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class MessageRole(str, Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    USER = "user"


class ChatMode(str, Enum):
    STANDARD = "standard"
    SUPPORT = "support"
