# src/core/exceptions.py

class EduMantriException(Exception):
    """Base exception for all EduMantri application errors."""
    pass

class AgentExecutionError(EduMantriException):
    """Raised when an agent fails to execute its run step."""
    pass

class IngestionError(EduMantriException):
    """Raised when document ingestion or database building fails."""
    pass

class StateRoutingError(EduMantriException):
    """Raised when the orchestrator fails to route states between agents."""
    pass

class GuardrailTriggeredException(EduMantriException):
    """Raised when input or output violates guardrails."""
    pass
