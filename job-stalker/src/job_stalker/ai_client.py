"""
AI Client Module - Unified interface for LLM providers

Provides:
- Base class for AI clients with retry logic
- Ollama client for local inference
- Gemini client for Google's API
- Shared HTTP session management
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, AsyncGenerator, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager

import aiohttp

log = logging.getLogger("ai_client")

DEFAULT_CONNECT_TIMEOUT = 10
DEFAULT_READ_TIMEOUT = 300
DEFAULT_TOTAL_TIMEOUT = 600

MAX_RETRIES = 3
RETRY_DELAY_BASE = 1.0
RETRY_DELAY_MAX = 30.0


@dataclass
class AIResponse:
    """Standardized AI response"""

    text: str
    success: bool
    error: Optional[str] = None
    tokens_used: int = 0
    model: str = ""


class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        base_delay: float = RETRY_DELAY_BASE,
        max_delay: float = RETRY_DELAY_MAX,
        retryable_exceptions: tuple = (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ConnectionError,
        ),
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff"""
        delay: float = self.base_delay * (2**attempt)
        return min(delay, self.max_delay)


class HTTPSessionManager:
    """
    Manages shared aiohttp ClientSession for all AI clients.
    Prevents creating new sessions for each request.
    """

    _instance: Optional["HTTPSessionManager"] = None
    _session: Optional[aiohttp.ClientSession] = None
    _lock: asyncio.Lock

    def __new__(cls) -> "HTTPSessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._lock = asyncio.Lock()
        return cls._instance

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create shared HTTP session"""
        async with self._lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(
                    connect=DEFAULT_CONNECT_TIMEOUT,
                    sock_read=DEFAULT_READ_TIMEOUT,
                    total=DEFAULT_TOTAL_TIMEOUT,
                )
                self._session = aiohttp.ClientSession(timeout=timeout)
                log.debug("Created new HTTP session")
            return self._session

    async def close(self):
        """Close the shared session"""
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
                log.debug("Closed HTTP session")

    @asynccontextmanager
    async def request(self, method: str, url: str, **kwargs):
        """Context manager for making requests with the shared session"""
        session = await self.get_session()
        async with session.request(method, url, **kwargs) as response:
            yield response


_session_manager: Optional[HTTPSessionManager] = None


def get_session_manager() -> HTTPSessionManager:
    """Get or create session manager singleton"""
    global _session_manager
    if _session_manager is None:
        _session_manager = HTTPSessionManager()
    return _session_manager


async def close_session():
    """Close the global HTTP session"""
    if _session_manager is not None:
        await _session_manager.close()


class BaseAIClient(ABC):
    """
    Abstract base class for AI clients.

    Provides:
    - Retry logic with exponential backoff
    - Timeout handling
    - Unified response format
    - Stream callback support
    """

    def __init__(
        self, retry_config: Optional[RetryConfig] = None, stream_callback: Optional[Callable] = None
    ):
        self.retry_config = retry_config or RetryConfig()
        self._stream_callback = stream_callback

    def set_stream_callback(self, callback: Optional[Callable]):
        """Set callback for streaming responses"""
        self._stream_callback = callback

    async def _notify_stream(self, chunk: str, stream_type: str = "analysis"):
        """Send chunk to stream callback"""
        if self._stream_callback:
            try:
                await self._stream_callback(
                    {"type": "stream", "stream_type": stream_type, "chunk": chunk}
                )
            except Exception as e:
                log.warning(f"Stream callback error: {e}")

    async def generate_with_retry(
        self, prompt: str, stream_type: Optional[str] = None, **kwargs
    ) -> AIResponse:
        """Generate with automatic retry on failure"""
        last_error = None

        for attempt in range(self.retry_config.max_retries):
            try:
                return await self.generate(prompt, stream_type=stream_type, **kwargs)
            except self.retry_config.retryable_exceptions as e:
                last_error = e
                if attempt < self.retry_config.max_retries - 1:
                    delay = self.retry_config.get_delay(attempt)
                    log.warning(
                        f"AI request failed (attempt {attempt + 1}/{self.retry_config.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error(
                        f"AI request failed after {self.retry_config.max_retries} attempts: {e}"
                    )

        return AIResponse(text="", success=False, error=str(last_error))

    @abstractmethod
    async def generate(
        self, prompt: str, stream_type: Optional[str] = None, **kwargs: Any
    ) -> AIResponse:
        """Generate response from the AI model"""
        pass

    @abstractmethod
    async def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        yield ""

    @abstractmethod
    async def is_available(self) -> bool:
        pass


class OllamaClient(BaseAIClient):
    DEFAULT_URL = "http://localhost:11434"

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        default_model: str = "mistral7",
        retry_config: Optional[RetryConfig] = None,
        stream_callback: Optional[Callable] = None,
    ):
        super().__init__(retry_config, stream_callback)
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream_type: Optional[str] = None,
        num_predict: int = 2048,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AIResponse:
        model = model or self.default_model
        full_response = ""

        if stream_type:
            await self._notify_stream("[START]", stream_type)

        try:
            async for chunk in self.stream(
                prompt, model=model, num_predict=num_predict, temperature=temperature
            ):
                if chunk.startswith("[ERROR:"):
                    return AIResponse(text="", success=False, error=chunk, model=model)
                full_response += chunk
                if stream_type:
                    await self._notify_stream(chunk, stream_type)
                    await asyncio.sleep(0.005)

            if stream_type:
                await self._notify_stream("[END]", stream_type)

            return AIResponse(text=full_response, success=True, model=model)

        except Exception as e:
            log.error(f"Ollama generate error: {e}")
            return AIResponse(text="", success=False, error=str(e), model=model)

    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        num_predict: int = 2048,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream response from Ollama"""
        model = model or self.default_model
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": num_predict, "temperature": temperature},
        }

        session_manager = get_session_manager()

        try:
            session = await session_manager.get_session()
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    log.error(f"Ollama API error: {error}")
                    yield f"[ERROR: {response.status}]"
                    return

                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            log.error(f"Ollama connection error: {e}")
            yield f"[ERROR: {e}]"
        except Exception as e:
            log.error(f"Ollama stream error: {e}")
            yield f"[ERROR: {e}]"

    async def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            session_manager = get_session_manager()
            session = await session_manager.get_session()
            async with session.get(
                f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            session_manager = get_session_manager()
            session = await session_manager.get_session()
            async with session.get(
                f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    for model_info in data.get("models", []):
                        models.append(
                            {
                                "name": model_info.get("name", ""),
                                "display_name": model_info.get("name", ""),
                                "size": model_info.get("size", 0),
                            }
                        )
                    log.info(f"Found {len(models)} Ollama models")
                    return models
        except Exception as e:
            log.warning(f"Could not get Ollama models: {e}")

        return [
            {"name": "mistral7", "display_name": "mistral7", "size": 0},
            {"name": "llama3.2:3b", "display_name": "llama3.2:3b", "size": 0},
        ]


class GeminiClient(BaseAIClient):
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        retry_config: Optional[RetryConfig] = None,
        stream_callback: Optional[Callable] = None,
    ):
        super().__init__(retry_config, stream_callback)
        self.api_key = api_key
        self.model = model

    async def generate(
        self,
        prompt: str,
        stream_type: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs,
    ) -> AIResponse:
        """Generate response from Gemini"""
        url = f"{self.BASE_URL}/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }

        session_manager = get_session_manager()

        try:
            session = await session_manager.get_session()
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    log.error(f"Gemini API error: {response.status} - {error}")
                    return AIResponse(
                        text="",
                        success=False,
                        error=f"Gemini API error {response.status}",
                        model=self.model,
                    )

                data = await response.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

                return AIResponse(text=text, success=True, model=self.model)

        except Exception as e:
            log.error(f"Gemini generate error: {e}")
            return AIResponse(text="", success=False, error=str(e), model=self.model)

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Gemini doesn't support streaming in this implementation"""
        response = await self.generate(prompt, **kwargs)
        if response.success:
            yield response.text
        else:
            yield f"[ERROR: {response.error}]"

    async def is_available(self) -> bool:
        """Check if Gemini API is accessible"""
        if not self.api_key:
            return False

        try:
            url = f"{self.BASE_URL}/models?key={self.api_key}"
            session_manager = get_session_manager()
            session = await session_manager.get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                return response.status == 200
        except Exception:
            return False


class GroqClient(BaseAIClient):
    # For GROQ
    BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llama-3.1-70b-versatile"

    # Available Groq models
    AVAILABLE_MODELS = [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "llama-3.2-90b-text-preview",
        "llama-3.2-11b-text-preview",
        "llama-3.2-3b-text-preview",
    ]

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        retry_config: Optional[RetryConfig] = None,
        stream_callback: Optional[Callable] = None,
    ):
        super().__init__(retry_config, stream_callback)
        self.api_key = api_key
        self.model = model

    async def generate(
        self,
        prompt: str,
        stream_type: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> AIResponse:
        """Generate response from Groq"""
        if stream_type:
            full_response = ""
            await self._notify_stream("[START]", stream_type)

            try:
                async for chunk in self.stream(
                    prompt, temperature=temperature, max_tokens=max_tokens
                ):
                    if chunk.startswith("[ERROR:"):
                        await self._notify_stream("[END]", stream_type)
                        return AIResponse(text="", success=False, error=chunk, model=self.model)
                    full_response += chunk
                    await self._notify_stream(chunk, stream_type)
                    await asyncio.sleep(0.005)  # Small delay for UI updates

                await self._notify_stream("[END]", stream_type)
                return AIResponse(text=full_response, success=True, model=self.model)
            except Exception as e:
                log.error(f"Groq generate error: {e}")
                await self._notify_stream("[END]", stream_type)
                return AIResponse(text="", success=False, error=str(e), model=self.model)

        url = f"{self.BASE_URL}/chat/completions"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        session_manager = get_session_manager()

        try:
            session = await session_manager.get_session()
            async with session.post(
                url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    log.error(f"Groq API error: {response.status} - {error}")
                    return AIResponse(
                        text="",
                        success=False,
                        error=f"Groq API error {response.status}",
                        model=self.model,
                    )

                data = await response.json()
                text = data["choices"][0]["message"]["content"].strip()

                tokens_used = data.get("usage", {}).get("total_tokens", 0)

                return AIResponse(
                    text=text, success=True, tokens_used=tokens_used, model=self.model
                )

        except Exception as e:
            log.error(f"Groq generate error: {e}")
            return AIResponse(text="", success=False, error=str(e), model=self.model)

    async def stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Stream response from Groq"""
        url = f"{self.BASE_URL}/chat/completions"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        session_manager = get_session_manager()

        try:
            session = await session_manager.get_session()
            async with session.post(
                url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    log.error(f"Groq API error: {error}")
                    yield f"[ERROR: {response.status}]"
                    return

                async for line in response.content:
                    if line:
                        try:
                            line_text = line.decode("utf-8").strip()
                            if line_text.startswith("data: "):
                                data_str = line_text[6:]
                                if data_str == "[DONE]":
                                    break
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    chunk = delta.get("content", "")
                                    if chunk:
                                        yield chunk
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            log.warning(f"Groq stream parsing error: {e}")
                            continue

        except aiohttp.ClientError as e:
            log.error(f"Groq connection error: {e}")
            yield f"[ERROR: {e}]"
        except Exception as e:
            log.error(f"Groq stream error: {e}")
            yield f"[ERROR: {e}]"

    async def is_available(self) -> bool:
        """Check if Groq API is accessible"""
        if not self.api_key:
            return False

        try:
            url = f"{self.BASE_URL}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            session_manager = get_session_manager()
            session = await session_manager.get_session()
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available Groq models"""
        models = []
        for model_name in self.AVAILABLE_MODELS:
            models.append(
                {
                    "name": model_name,
                    "display_name": f"Groq {model_name}",
                    "size": 0,
                    "provider": "groq",
                }
            )
        return models


class AIClientFactory:
    # Factory for AI-clients(LLM)
    _ollama_client: Optional[OllamaClient] = None
    _gemini_client: Optional[GeminiClient] = None
    _groq_client: Optional[GroqClient] = None

    @classmethod
    def get_ollama_client(
        cls, model: str = "mistral7", stream_callback: Optional[Callable] = None
    ) -> OllamaClient:
        if cls._ollama_client is None:
            cls._ollama_client = OllamaClient(default_model=model)
        if stream_callback:
            cls._ollama_client.set_stream_callback(stream_callback)
        return cls._ollama_client

    @classmethod
    def get_gemini_client(
        cls, api_key: str, stream_callback: Optional[Callable] = None
    ) -> Optional[GeminiClient]:
        if not api_key:
            return None
        if cls._gemini_client is None:
            cls._gemini_client = GeminiClient(api_key)
        if stream_callback:
            cls._gemini_client.set_stream_callback(stream_callback)
        return cls._gemini_client

    @classmethod
    def get_groq_client(
        cls,
        api_key: str,
        model: str = "llama-3.1-70b-versatile",
        stream_callback: Optional[Callable] = None,
    ) -> Optional[GroqClient]:
        if not api_key:
            return None
        if cls._groq_client is None:
            cls._groq_client = GroqClient(api_key, model=model)
        elif cls._groq_client.model != model:
            cls._groq_client.model = model
        if stream_callback:
            cls._groq_client.set_stream_callback(stream_callback)
        return cls._groq_client

    @classmethod
    def get_client(
        cls,
        model_type: str,
        api_key: Optional[str] = None,
        stream_callback: Optional[Callable] = None,
    ) -> Optional[BaseAIClient]:
        if model_type == "gemini" and api_key:
            return cls.get_gemini_client(api_key, stream_callback)

        is_groq_model = (
            model_type == "groq"
            or model_type.startswith("groq:")
            or model_type in GroqClient.AVAILABLE_MODELS
            or model_type.startswith("llama-3.")
            or model_type.startswith("llama-3.1")
            or model_type.startswith("llama-3.2")
            or model_type.startswith("gemma2")
            or model_type == "mixtral-8x7b-32768"
        )

        if is_groq_model:
            if api_key:
                if model_type == "mixtral-8x7b-32768":
                    log.warning(
                        "Model mixtral-8x7b-32768 has been decommissioned. Using default model instead."
                    )
                    model = GroqClient.DEFAULT_MODEL
                elif model_type.startswith("groq:"):
                    model = model_type[5:]  # Remove "groq:" prefix
                elif model_type in GroqClient.AVAILABLE_MODELS:
                    model = model_type
                elif model_type == "groq":
                    model = GroqClient.DEFAULT_MODEL
                else:
                    model = model_type
                return cls.get_groq_client(api_key, model=model, stream_callback=stream_callback)
            else:
                log.warning("Groq API key not provided, falling back to Ollama")
                model = "mistral7" if model_type == "mistral" else "llama3.2:3b"
                return cls.get_ollama_client(model, stream_callback)
        else:
            model = "mistral7" if model_type == "mistral" else "llama3.2:3b"
            return cls.get_ollama_client(model, stream_callback)

    @classmethod
    async def cleanup(cls):
        """Cleanup all clients and sessions"""
        cls._ollama_client = None
        cls._gemini_client = None
        cls._groq_client = None
        await close_session()
