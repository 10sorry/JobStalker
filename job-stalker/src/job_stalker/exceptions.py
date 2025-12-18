"""
Custom Exceptions for JobStalker

Provides specific exceptions for different error types with context logging.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger("exceptions")


class JobStalkerError(Exception):
    """Base exception for all JobStalker errors"""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause
        self._log_error()

    def _log_error(self) -> None:
        """Log error with context"""
        log.error(
            f"{self.__class__.__name__}: {self.message}",
            extra={"context": self.context},
            exc_info=self.cause
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "context": self.context
        }


# ============== CONFIGURATION ERRORS ==============

class ConfigurationError(JobStalkerError):
    """Configuration-related errors"""
    pass


class MissingConfigError(ConfigurationError):
    """Required configuration is missing"""

    def __init__(self, param: str, hint: str = ""):
        message = f"Missing required configuration: {param}"
        if hint:
            message += f". {hint}"
        super().__init__(message, context={"param": param})


class InvalidConfigError(ConfigurationError):
    """Configuration value is invalid"""

    def __init__(self, param: str, value: Any, expected: str):
        super().__init__(
            f"Invalid configuration for {param}: {value!r}. Expected: {expected}",
            context={"param": param, "value": value, "expected": expected}
        )


# ============== AUTHENTICATION ERRORS ==============

class AuthenticationError(JobStalkerError):
    """Authentication-related errors"""
    pass


class NotAuthorizedError(AuthenticationError):
    """User is not authorized"""

    def __init__(self, message: str = "Telegram authorization required"):
        super().__init__(message)


class InvalidCodeError(AuthenticationError):
    """Verification code is invalid"""

    def __init__(self):
        super().__init__("Invalid verification code")


class CodeExpiredError(AuthenticationError):
    """Verification code has expired"""

    def __init__(self):
        super().__init__("Verification code expired. Please request a new one")


class InvalidPasswordError(AuthenticationError):
    """2FA password is invalid"""

    def __init__(self):
        super().__init__("Invalid 2FA password")


class SessionExpiredError(AuthenticationError):
    """Session has expired"""

    def __init__(self):
        super().__init__("Session expired. Please re-authenticate")


# ============== TELEGRAM ERRORS ==============

class TelegramError(JobStalkerError):
    """Telegram API-related errors"""
    pass


class ChannelNotFoundError(TelegramError):
    """Channel was not found"""

    def __init__(self, channel: str):
        super().__init__(
            f"Channel not found: {channel}",
            context={"channel": channel}
        )


class ChannelAccessDeniedError(TelegramError):
    """Access to channel is denied"""

    def __init__(self, channel: str):
        super().__init__(
            f"Access denied to channel: {channel}. Make sure you're a member.",
            context={"channel": channel}
        )


class RateLimitError(TelegramError):
    """Rate limit exceeded"""

    def __init__(self, retry_after: Optional[int] = None):
        message = "Rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, context={"retry_after": retry_after})
        self.retry_after = retry_after


class ConnectionError(TelegramError):
    """Connection to Telegram failed"""

    def __init__(self, cause: Optional[Exception] = None):
        super().__init__(
            "Failed to connect to Telegram",
            cause=cause
        )


# ============== AI/LLM ERRORS ==============

class AIError(JobStalkerError):
    """AI/LLM-related errors"""
    pass


class AIConnectionError(AIError):
    """Failed to connect to AI service"""

    def __init__(self, service: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to connect to {service}",
            context={"service": service},
            cause=cause
        )


class AITimeoutError(AIError):
    """AI request timed out"""

    def __init__(self, service: str, timeout: float):
        super().__init__(
            f"{service} request timed out after {timeout}s",
            context={"service": service, "timeout": timeout}
        )


class AIResponseError(AIError):
    """AI returned invalid response"""

    def __init__(self, service: str, reason: str, response: Optional[str] = None):
        super().__init__(
            f"{service} returned invalid response: {reason}",
            context={"service": service, "reason": reason, "response": response[:200] if response else None}
        )


class ModelNotFoundError(AIError):
    """Requested model not found"""

    def __init__(self, model: str):
        super().__init__(
            f"Model not found: {model}",
            context={"model": model}
        )


class JSONParseError(AIError):
    """Failed to parse JSON from AI response"""

    def __init__(self, response: str, cause: Optional[Exception] = None):
        super().__init__(
            "Failed to parse JSON from AI response",
            context={"response_preview": response[:500] if response else None},
            cause=cause
        )


# ============== STORAGE ERRORS ==============

class StorageError(JobStalkerError):
    """Storage-related errors"""
    pass


class DatabaseError(StorageError):
    """Database operation failed"""

    def __init__(self, operation: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Database operation failed: {operation}",
            context={"operation": operation},
            cause=cause
        )


class FileStorageError(StorageError):
    """File storage operation failed"""

    def __init__(self, operation: str, path: str, cause: Optional[Exception] = None):
        super().__init__(
            f"File storage operation failed: {operation} on {path}",
            context={"operation": operation, "path": path},
            cause=cause
        )


class VacancyNotFoundError(StorageError):
    """Vacancy not found in storage"""

    def __init__(self, vacancy_id: str):
        super().__init__(
            f"Vacancy not found: {vacancy_id}",
            context={"vacancy_id": vacancy_id}
        )


# ============== RESUME ERRORS ==============

class ResumeError(JobStalkerError):
    """Resume-related errors"""
    pass


class ResumeNotLoadedError(ResumeError):
    """Resume is not loaded"""

    def __init__(self):
        super().__init__("Resume not loaded. Please upload a resume first.")


class ResumeParseError(ResumeError):
    """Failed to parse resume file"""

    def __init__(self, file_type: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to parse resume file ({file_type})",
            context={"file_type": file_type},
            cause=cause
        )


class UnsupportedFileTypeError(ResumeError):
    """File type is not supported"""

    def __init__(self, file_type: str):
        super().__init__(
            f"Unsupported file type: {file_type}. Supported: PDF, DOCX, TXT, HTML",
            context={"file_type": file_type}
        )


# ============== VALIDATION ERRORS ==============

class ValidationError(JobStalkerError):
    """Data validation errors"""
    pass


class InvalidInputError(ValidationError):
    """Input data is invalid"""

    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Invalid {field}: {reason}",
            context={"field": field, "value": str(value)[:100], "reason": reason}
        )


class MissingFieldError(ValidationError):
    """Required field is missing"""

    def __init__(self, field: str):
        super().__init__(
            f"Missing required field: {field}",
            context={"field": field}
        )


# ============== TASK ERRORS ==============

class TaskError(JobStalkerError):
    """Background task errors"""
    pass


class TaskNotFoundError(TaskError):
    """Task not found"""

    def __init__(self, task_id: str):
        super().__init__(
            f"Task not found: {task_id}",
            context={"task_id": task_id}
        )


class TaskCancelledError(TaskError):
    """Task was cancelled"""

    def __init__(self, task_id: str):
        super().__init__(
            f"Task was cancelled: {task_id}",
            context={"task_id": task_id}
        )


class MonitoringNotActiveError(TaskError):
    """Monitoring is not active"""

    def __init__(self):
        super().__init__("Monitoring is not active")


class MonitoringAlreadyActiveError(TaskError):
    """Monitoring is already active"""

    def __init__(self):
        super().__init__("Monitoring is already active")


# ============== HELPER FUNCTIONS ==============

def handle_exception(e: Exception, default_message: str = "An error occurred") -> JobStalkerError:
    """Convert any exception to JobStalkerError"""
    if isinstance(e, JobStalkerError):
        return e
    return JobStalkerError(default_message, cause=e)


def log_and_raise(
    error_class: type,
    *args,
    level: str = "error",
    **kwargs
) -> None:
    """Log and raise an exception"""
    error = error_class(*args, **kwargs)
    getattr(log, level)(str(error), exc_info=error.cause)
    raise error
