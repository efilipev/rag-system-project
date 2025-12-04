"""
LLM Generation Service with OpenAI integration
"""
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional
import tiktoken
from openai import AsyncOpenAI, OpenAIError, RateLimitError, APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from src.models.schemas import (
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
    GenerationParameters
)
from src.services.prompts import PromptTemplates

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    LLM Generator with OpenAI API integration
    Handles text generation, streaming, token counting, and retry logic
    """

    def __init__(self, api_key: str, default_model: str = "gpt-3.5-turbo"):
        """
        Initialize LLM Generator

        Args:
            api_key: OpenAI API key
            default_model: Default model to use
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model
        self.prompt_templates = PromptTemplates()

        # Token encoding for counting
        self.encodings: Dict[str, Any] = {}

        logger.info(f"LLMGenerator initialized with model: {default_model}")

    def _get_encoding(self, model: str):
        """
        Get or create tiktoken encoding for a model

        Args:
            model: Model name

        Returns:
            Tiktoken encoding
        """
        if model not in self.encodings:
            try:
                self.encodings[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Use cl100k_base encoding as fallback
                logger.warning(f"Model {model} not found, using cl100k_base encoding")
                self.encodings[model] = tiktoken.get_encoding("cl100k_base")

        return self.encodings[model]

    def count_tokens(self, text: str, model: str) -> int:
        """
        Count tokens in text for a specific model

        Args:
            text: Text to count tokens for
            model: Model name

        Returns:
            Token count
        """
        try:
            encoding = self._get_encoding(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Rough estimate: 1 token ~= 4 characters
            return len(text) // 4

    def _prepare_messages(
        self,
        query: str,
        context_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for OpenAI API

        Args:
            query: User query
            context_documents: Context documents for RAG

        Returns:
            List of message dictionaries
        """
        # Build the user prompt with context
        user_prompt = self.prompt_templates.build_rag_prompt(query, context_documents)

        messages = [
            {"role": "system", "content": self.prompt_templates.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        return messages

    def _validate_token_count(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int
    ) -> None:
        """
        Validate that total tokens don't exceed model limits

        Args:
            messages: Messages to send
            model: Model name
            max_tokens: Max tokens for completion

        Raises:
            ValueError: If token count exceeds limits
        """
        # Calculate input tokens
        input_text = " ".join([msg["content"] for msg in messages])
        input_tokens = self.count_tokens(input_text, model)

        # Model context limits
        context_limits = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
        }

        # Find matching context limit
        context_limit = context_limits.get(model, 4096)
        for key in context_limits:
            if model.startswith(key):
                context_limit = context_limits[key]
                break

        total_tokens = input_tokens + max_tokens

        if total_tokens > context_limit:
            raise ValueError(
                f"Token count ({total_tokens}) exceeds model limit ({context_limit}). "
                f"Input tokens: {input_tokens}, Max output tokens: {max_tokens}"
            )

        logger.debug(f"Token validation passed: {input_tokens} input + {max_tokens} max = {total_tokens}/{context_limit}")

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """
        Generate response using OpenAI API with retry logic

        Args:
            request: Generation request

        Returns:
            Generation response

        Raises:
            OpenAIError: If generation fails after retries
            ValueError: If token count exceeds limits
        """
        try:
            # Prepare parameters
            params = request.parameters or GenerationParameters()
            model = params.model or self.default_model

            # Convert documents to dict format
            context_docs = [doc.dict() for doc in request.context_documents]

            # Prepare messages
            messages = self._prepare_messages(request.query, context_docs)

            # Validate token count
            self._validate_token_count(messages, model, params.max_tokens)

            logger.info(f"Generating response for query: {request.query[:100]}... using {model}")

            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                frequency_penalty=params.frequency_penalty,
                presence_penalty=params.presence_penalty,
            )

            # Extract response
            generated_text = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            tokens_used = response.usage.total_tokens

            # Extract sources
            sources = self.prompt_templates.extract_sources(context_docs)

            logger.info(f"Generated response with {tokens_used} tokens, finish_reason: {finish_reason}")

            return GenerationResponse(
                generated_text=generated_text,
                query=request.query,
                tokens_used=tokens_used,
                model_used=model,
                finish_reason=finish_reason,
                sources_used=sources,
                session_id=request.session_id,
                metadata={
                    "context_documents_count": len(request.context_documents),
                    "user_id": request.user_id
                }
            )

        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generation: {e}", exc_info=True)
            raise OpenAIError(f"Generation failed: {str(e)}")

    async def generate_stream(
        self,
        request: GenerationRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate streaming response using OpenAI API

        Args:
            request: Generation request

        Yields:
            Stream chunks

        Raises:
            OpenAIError: If generation fails
        """
        try:
            # Prepare parameters
            params = request.parameters or GenerationParameters()
            model = params.model or self.default_model

            # Convert documents to dict format
            context_docs = [doc.dict() for doc in request.context_documents]

            # Prepare messages
            messages = self._prepare_messages(request.query, context_docs)

            # Validate token count
            self._validate_token_count(messages, model, params.max_tokens)

            logger.info(f"Starting streaming generation for query: {request.query[:100]}... using {model}")

            # Call OpenAI API with streaming
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                frequency_penalty=params.frequency_penalty,
                presence_penalty=params.presence_penalty,
                stream=True
            )

            tokens_used = 0
            full_content = ""

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    tokens_used += 1  # Rough estimate

                    yield StreamChunk(
                        content=content,
                        finish_reason=None,
                        tokens_used=None
                    )

                # Check for finish
                if chunk.choices[0].finish_reason is not None:
                    finish_reason = chunk.choices[0].finish_reason
                    # Estimate tokens more accurately
                    tokens_used = self.count_tokens(full_content, model)

                    yield StreamChunk(
                        content="",
                        finish_reason=finish_reason,
                        tokens_used=tokens_used
                    )

            logger.info(f"Streaming completed with estimated {tokens_used} tokens")

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}", exc_info=True)
            raise OpenAIError(f"Streaming generation failed: {str(e)}")

    async def close(self):
        """Close the OpenAI client"""
        await self.client.close()
        logger.info("LLMGenerator closed")
