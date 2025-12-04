"""
Local LLM Provider implementation (Ollama, LLama.cpp, vLLM, etc.)
"""
import logging
import json
from typing import AsyncGenerator, List, Dict, Any
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from src.models.schemas import GenerationRequest, GenerationResponse, StreamChunk, GenerationParameters
from src.services.base_provider import BaseLLMProvider
from src.services.prompts import PromptTemplates

logger = logging.getLogger(__name__)


class LocalLLMProvider(BaseLLMProvider):
    """
    Local LLM provider supporting:
    - Ollama (llama3, mistral, phi, etc.)
    - vLLM inference server
    - LLama.cpp server
    - Any OpenAI-compatible local API

    Follows Single Responsibility Principle - handles only local model interactions
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3",
        api_type: str = "ollama",
        timeout: int = 300
    ):
        """
        Initialize Local LLM Provider

        Args:
            base_url: Base URL of the local inference server
            default_model: Default model to use (e.g., 'llama3', 'mistral', 'phi')
            api_type: Type of API ('ollama', 'vllm', 'llamacpp', 'openai-compatible')
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.api_type = api_type
        self.timeout = timeout
        self.prompt_templates = PromptTemplates()

        logger.info(f"LocalLLMProvider initialized with {api_type} at {base_url}, model: {default_model}")

    def _get_client(self) -> httpx.AsyncClient:
        """Create a new HTTP client for each request to avoid connection caching issues."""
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

    def get_available_models(self) -> List[str]:
        """Get list of available local models"""
        return [
            "llama3",
            "llama3:70b",
            "llama2",
            "mistral",
            "mixtral",
            "phi3",
            "codellama",
            "neural-chat"
        ]

    def count_tokens(self, text: str, model: str) -> int:
        """
        Estimate token count for local models

        Args:
            text: Text to count tokens for
            model: Model name

        Returns:
            Token count (estimated)
        """
        # Rough estimate for local models: 1 token ~= 4 characters
        # This is less accurate than tiktoken but works for most models
        return len(text) // 4

    def _prepare_prompt(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """
        Prepare prompt for local models

        Args:
            query: User query
            context_documents: Context documents for RAG

        Returns:
            Complete prompt string
        """
        # For local models, we combine system and user prompts
        system_prompt = self.prompt_templates.SYSTEM_PROMPT
        user_prompt = self.prompt_templates.build_rag_prompt(query, context_documents)

        # Format depends on model type
        if "llama" in self.default_model.lower():
            # Llama format
            return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        elif "mistral" in self.default_model.lower():
            # Mistral format
            return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        else:
            # Generic format
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    def _build_ollama_request(
        self,
        prompt: str,
        params: GenerationParameters
    ) -> Dict[str, Any]:
        """Build request payload for Ollama API"""
        # Always use self.default_model for local Ollama provider
        # params.model defaults to "gpt-3.5-turbo" which Ollama doesn't support
        return {
            "model": self.default_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.temperature,
                "num_predict": params.max_tokens,
                "top_p": params.top_p,
            }
        }

    def _build_vllm_request(
        self,
        prompt: str,
        params: GenerationParameters
    ) -> Dict[str, Any]:
        """Build request payload for vLLM API"""
        # Always use self.default_model for local vLLM provider
        return {
            "prompt": prompt,
            "model": self.default_model,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "top_p": params.top_p,
        }

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate response using local LLM with retry logic

        Args:
            request: Generation request

        Returns:
            Generation response

        Raises:
            httpx.HTTPError: If generation fails after retries
        """
        try:
            params = request.parameters or GenerationParameters()
            # Always use self.default_model for local provider
            model = self.default_model

            context_docs = [doc.dict() for doc in request.context_documents]
            prompt = self._prepare_prompt(request.query, context_docs)

            logger.info(f"Generating response for query: {request.query[:100]}... using local model {model}")

            # Build request based on API type
            if self.api_type == "ollama":
                endpoint = "/api/generate"
                payload = self._build_ollama_request(prompt, params)
            elif self.api_type == "vllm":
                endpoint = "/v1/completions"
                payload = self._build_vllm_request(prompt, params)
            else:
                endpoint = "/v1/completions"
                payload = self._build_vllm_request(prompt, params)

            logger.info(f"Sending to {self.base_url}{endpoint} with model: {payload.get('model')}")

            # Make request to local server with fresh client
            async with self._get_client() as client:
                response = await client.post(endpoint, json=payload)
                logger.info(f"Response status: {response.status_code}")
                response.raise_for_status()
                result = response.json()

            # Parse response based on API type
            if self.api_type == "ollama":
                generated_text = result.get("response", "")
                tokens_used = result.get("eval_count", 0) + result.get("prompt_eval_count", 0)
                finish_reason = "stop" if result.get("done", False) else "length"
            else:  # vLLM or OpenAI-compatible
                choice = result.get("choices", [{}])[0]
                generated_text = choice.get("text", "")
                tokens_used = result.get("usage", {}).get("total_tokens", 0)
                finish_reason = choice.get("finish_reason", "stop")

            sources = self.prompt_templates.extract_sources(context_docs)

            logger.info(f"Generated response with {tokens_used} tokens, finish_reason: {finish_reason}")

            return GenerationResponse(
                generated_text=generated_text.strip(),
                query=request.query,
                tokens_used=tokens_used,
                model_used=model,
                finish_reason=finish_reason,
                sources_used=sources,
                session_id=request.session_id,
                metadata={
                    "context_documents_count": len(request.context_documents),
                    "user_id": request.user_id,
                    "provider": "local",
                    "api_type": self.api_type
                }
            )

        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling local LLM: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generation: {e}", exc_info=True)
            raise Exception(f"Local LLM generation failed: {str(e)}")

    async def generate_stream(
        self,
        request: GenerationRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate streaming response using local LLM

        Args:
            request: Generation request

        Yields:
            Stream chunks

        Raises:
            httpx.HTTPError: If generation fails
        """
        try:
            params = request.parameters or GenerationParameters()
            # Always use self.default_model for local provider
            model = self.default_model

            context_docs = [doc.dict() for doc in request.context_documents]
            prompt = self._prepare_prompt(request.query, context_docs)

            logger.info(f"Starting streaming generation for query: {request.query[:100]}... using local model {model}")

            # Build request based on API type
            if self.api_type == "ollama":
                endpoint = "/api/generate"
                payload = self._build_ollama_request(prompt, params)
                payload["stream"] = True
            elif self.api_type == "vllm":
                endpoint = "/v1/completions"
                payload = self._build_vllm_request(prompt, params)
                payload["stream"] = True
            else:
                endpoint = "/v1/completions"
                payload = self._build_vllm_request(prompt, params)
                payload["stream"] = True

            # Make streaming request with fresh client
            async with self._get_client() as client:
                async with client.stream("POST", endpoint, json=payload) as response:
                    response.raise_for_status()

                    tokens_used = 0
                    full_content = ""

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        try:
                            # Remove "data: " prefix if present
                            if line.startswith("data: "):
                                line = line[6:]

                            if line == "[DONE]":
                                break

                            data = json.loads(line)

                            # Parse based on API type
                            if self.api_type == "ollama":
                                content = data.get("response", "")
                                done = data.get("done", False)

                                if content:
                                    full_content += content
                                    tokens_used += 1

                                    yield StreamChunk(
                                        content=content,
                                        finish_reason=None,
                                        tokens_used=None
                                    )

                                if done:
                                    finish_reason = "stop"
                                    tokens_used = data.get("eval_count", tokens_used)

                                    yield StreamChunk(
                                        content="",
                                        finish_reason=finish_reason,
                                        tokens_used=tokens_used
                                    )
                            else:  # vLLM or OpenAI-compatible
                                choice = data.get("choices", [{}])[0]
                                content = choice.get("text", "")
                                finish_reason = choice.get("finish_reason")

                                if content:
                                    full_content += content
                                    tokens_used += 1

                                    yield StreamChunk(
                                        content=content,
                                        finish_reason=None,
                                        tokens_used=None
                                    )

                                if finish_reason:
                                    yield StreamChunk(
                                        content="",
                                        finish_reason=finish_reason,
                                        tokens_used=tokens_used
                                    )

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse streaming line: {line}")
                            continue

                    logger.info(f"Streaming completed with estimated {tokens_used} tokens")

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}", exc_info=True)
            raise Exception(f"Local LLM streaming generation failed: {str(e)}")

    async def health_check(self) -> bool:
        """
        Check if local LLM server is accessible

        Returns:
            True if healthy, False otherwise
        """
        try:
            async with self._get_client() as client:
                if self.api_type == "ollama":
                    response = await client.get("/api/tags")
                else:
                    response = await client.get("/health")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client (no-op since we create fresh clients per request)"""
        logger.info("LocalLLMProvider closed")
