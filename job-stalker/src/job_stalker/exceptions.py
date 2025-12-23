from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger("exceptions")


class JobStalkerError(Exception):
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause
        self._log_error()

    def _log_error(self) -> None:
        log.error(
            f"{self.__class__.__name__}: {self.message}",
            extra={"context": self.context},
            exc_info=self.cause,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"error": self.__class__.__name__, "message": self.message, "context": self.context}


class ConfigurationError(JobStalkerError):
    pass


class MissingConfigError(ConfigurationError):
    def __init__(self, param: str, hint: str = ""):
        message = f"Missing required configuration: {param}"
        if hint:
            message += f". {hint}"
        super().__init__(message, context={"param": param})


class InvalidConfigError(ConfigurationError):
    def __init__(self, param: str, value: Any, expected: str):
        super().__init__(
            f"Invalid configuration for {param}: {value!r}. Expected: {expected}",
            context={"param": param, "value": value, "expected": expected},
        )


class AuthenticationError(JobStalkerError):
    pass


class NotAuthorizedError(AuthenticationError):
    def __init__(self, message: str = "Telegram authorization required"):
        super().__init__(message)


class InvalidCodeError(AuthenticationError):
    def __init__(self):
        super().__init__("Invalid verification code")


class CodeExpiredError(AuthenticationError):
    def __init__(self):
        super().__init__("Verification code expired. Please request a new one")


class InvalidPasswordError(AuthenticationError):
    def __init__(self):
        super().__init__("Invalid 2FA password")


class SessionExpiredError(AuthenticationError):
    def __init__(self):
        super().__init__("Session expired. Please re-authenticate")


class TelegramError(JobStalkerError):
    pass


class ChannelNotFoundError(TelegramError):
    def __init__(self, channel: str):
        super().__init__(f"Channel not found: {channel}", context={"channel": channel})


class ChannelAccessDeniedError(TelegramError):
    def __init__(self, channel: str):
        super().__init__(
            f"Access denied to channel: {channel}. Make sure you're a member.",
            context={"channel": channel},
        )


class RateLimitError(TelegramError):
    def __init__(self, retry_after: Optional[int] = None):
        message = "Rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, context={"retry_after": retry_after})
        self.retry_after = retry_after


class ConnectionError(TelegramError):
    def __init__(self, cause: Optional[Exception] = None):
        super().__init__("Failed to connect to Telegram", cause=cause)


class AIError(JobStalkerError):
    pass


class AIConnectionError(AIError):
    def __init__(self, service: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to connect to {service}", context={"service": service}, cause=cause
        )


class AITimeoutError(AIError):
    def __init__(self, service: str, timeout: float):
        super().__init__(
            f"{service} request timed out after {timeout}s",
            context={"service": service, "timeout": timeout},
        )


class AIResponseError(AIError):
    def __init__(self, service: str, reason: str, response: Optional[str] = None):
        super().__init__(
            f"{service} returned invalid response: {reason}",
            context={
                "service": service,
                "reason": reason,
                "response": response[:200] if response else None,
            },
        )


class ModelNotFoundError(AIError):
    def __init__(self, model: str):
        super().__init__(f"Model not found: {model}", context={"model": model})


class JSONParseError(AIError):
    def __init__(self, response: str, cause: Optional[Exception] = None):
        super().__init__(
            "Failed to parse JSON from AI response",
            context={"response_preview": response[:500] if response else None},
            cause=cause,
        )


class StorageError(JobStalkerError):
    pass


class DatabaseError(StorageError):
    def __init__(self, operation: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Database operation failed: {operation}", context={"operation": operation}, cause=cause
        )


class FileStorageError(StorageError):
    def __init__(self, operation: str, path: str, cause: Optional[Exception] = None):
        super().__init__(
            f"File storage operation failed: {operation} on {path}",
            context={"operation": operation, "path": path},
            cause=cause,
        )


class VacancyNotFoundError(StorageError):
    def __init__(self, vacancy_id: str):
        super().__init__(f"Vacancy not found: {vacancy_id}", context={"vacancy_id": vacancy_id})


class ResumeError(JobStalkerError):
    pass


class ResumeNotLoadedError(ResumeError):
    def __init__(self):
        super().__init__("Resume not loaded. Please upload a resume first.")


class ResumeParseError(ResumeError):
    def __init__(self, file_type: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to parse resume file ({file_type})",
            context={"file_type": file_type},
            cause=cause,
        )


class UnsupportedFileTypeError(ResumeError):
    def __init__(self, file_type: str):
        super().__init__(
            f"Unsupported file type: {file_type}. Supported: PDF, DOCX, TXT, HTML",
            context={"file_type": file_type},
        )


class ValidationError(JobStalkerError):
    pass


class InvalidInputError(ValidationError):
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Invalid {field}: {reason}",
            context={"field": field, "value": str(value)[:100], "reason": reason},
        )


class MissingFieldError(ValidationError):
    def __init__(self, field: str):
        super().__init__(f"Missing required field: {field}", context={"field": field})


class TaskError(JobStalkerError):
    pass


class TaskNotFoundError(TaskError):
    def __init__(self, task_id: str):
        super().__init__(f"Task not found: {task_id}", context={"task_id": task_id})


class TaskCancelledError(TaskError):
    def __init__(self, task_id: str):
        super().__init__(f"Task was cancelled: {task_id}", context={"task_id": task_id})


class MonitoringNotActiveError(TaskError):
    def __init__(self):
        super().__init__("Monitoring is not active")


class MonitoringAlreadyActiveError(TaskError):
    def __init__(self):
        super().__init__("Monitoring is already active")


def handle_exception(e: Exception, default_message: str = "An error occurred") -> JobStalkerError:
    if isinstance(e, JobStalkerError):
        return e
    return JobStalkerError(default_message, cause=e)


def log_and_raise(error_class: type, *args, level: str = "error", **kwargs) -> None:
    error = error_class(*args, **kwargs)
    getattr(log, level)(str(error), exc_info=error.cause)
    raise error
