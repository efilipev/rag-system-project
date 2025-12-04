"""
Service-specific HTTP clients for RAG system microservices
"""
import logging
from typing import List, Dict, Any, Optional
from shared.clients.base_client import BaseHTTPClient

logger = logging.getLogger(__name__)


class QueryAnalysisClient(BaseHTTPClient):
    """Client for Query Analysis Service"""

    def __init__(self, base_url: str = "http://localhost:8101", **kwargs):
        super().__init__(
            base_url=base_url,
            service_name="query-analysis",
            **kwargs
        )

    async def analyze_query(
        self,
        query: str,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a query

        Args:
            query: User query to analyze
            correlation_id: Request correlation ID

        Returns:
            Analysis result with keywords, entities, intent
        """
        return await self.post(
            "/api/v1/analyze",
            json={"query": query},
            correlation_id=correlation_id
        )


class DocumentRetrievalClient(BaseHTTPClient):
    """Client for Document Retrieval Service"""

    def __init__(self, base_url: str = "http://localhost:8102", **kwargs):
        super().__init__(
            base_url=base_url,
            service_name="document-retrieval",
            **kwargs
        )

    async def search_documents(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant documents

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            correlation_id: Request correlation ID

        Returns:
            Search results with documents and scores
        """
        payload = {
            "query": query,
            "top_k": top_k
        }
        if filters:
            payload["filters"] = filters

        return await self.post(
            "/api/v1/search",
            json=payload,
            correlation_id=correlation_id
        )


class DocumentRankingClient(BaseHTTPClient):
    """Client for Document Ranking Service"""

    def __init__(self, base_url: str = "http://localhost:8103", **kwargs):
        super().__init__(
            base_url=base_url,
            service_name="document-ranking",
            **kwargs
        )

    async def rank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rank documents based on relevance

        Args:
            query: Search query
            documents: List of documents to rank
            top_k: Number of top documents to return
            correlation_id: Request correlation ID

        Returns:
            Ranked documents with scores
        """
        payload = {
            "query": query,
            "documents": documents
        }
        if top_k:
            payload["top_k"] = top_k

        return await self.post(
            "/api/v1/rank",
            json=payload,
            correlation_id=correlation_id
        )


class LatexParserClient(BaseHTTPClient):
    """Client for LaTeX Parser Service"""

    def __init__(self, base_url: str = "http://localhost:8104", **kwargs):
        super().__init__(
            base_url=base_url,
            service_name="latex-parser",
            **kwargs
        )

    async def parse_latex(
        self,
        latex_string: str,
        output_format: str = "mathml",
        simplify: bool = False,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse LaTeX formula

        Args:
            latex_string: LaTeX formula to parse
            output_format: Desired output format (mathml, text, unicode, etc.)
            simplify: Whether to simplify expressions
            correlation_id: Request correlation ID

        Returns:
            Parsed LaTeX in requested format
        """
        return await self.post(
            "/api/v1/parse",
            json={
                "latex_string": latex_string,
                "output_format": output_format,
                "simplify": simplify
            },
            correlation_id=correlation_id
        )

    async def validate_latex(
        self,
        latex_string: str,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate LaTeX syntax

        Args:
            latex_string: LaTeX formula to validate
            correlation_id: Request correlation ID

        Returns:
            Validation result
        """
        return await self.post(
            "/api/v1/validate",
            json=latex_string,
            correlation_id=correlation_id
        )


class LLMGenerationClient(BaseHTTPClient):
    """Client for LLM Generation Service"""

    def __init__(self, base_url: str = "http://localhost:8105", **kwargs):
        super().__init__(
            base_url=base_url,
            service_name="llm-generation",
            timeout=120.0,  # Longer timeout for LLM generation
            **kwargs
        )

    async def generate_response(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response using LLM

        Args:
            query: User query
            context_documents: Context documents for RAG
            parameters: Generation parameters (temperature, max_tokens, etc.)
            correlation_id: Request correlation ID

        Returns:
            Generated response with metadata
        """
        payload = {
            "query": query,
            "context_documents": context_documents
        }
        if parameters:
            payload["parameters"] = parameters

        return await self.post(
            "/api/v1/generate",
            json=payload,
            correlation_id=correlation_id
        )


class ResponseFormatterClient(BaseHTTPClient):
    """Client for Response Formatter Service"""

    def __init__(self, base_url: str = "http://localhost:8106", **kwargs):
        super().__init__(
            base_url=base_url,
            service_name="response-formatter",
            **kwargs
        )

    async def format_response(
        self,
        content: str,
        query: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        output_format: str = "markdown",
        include_citations: bool = True,
        include_metadata: bool = False,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format response with sources and citations

        Args:
            content: Content to format
            query: Original user query
            sources: Source citations
            output_format: Desired output format (markdown, html, json, etc.)
            include_citations: Whether to include citations
            include_metadata: Whether to include metadata
            correlation_id: Request correlation ID

        Returns:
            Formatted response
        """
        payload = {
            "content": content,
            "output_format": output_format,
            "include_citations": include_citations,
            "include_metadata": include_metadata
        }

        if query:
            payload["query"] = query
        if sources:
            payload["sources"] = sources

        return await self.post(
            "/api/v1/format",
            json=payload,
            correlation_id=correlation_id
        )


# Factory function for creating clients
def create_client(service_name: str, base_url: Optional[str] = None, **kwargs):
    """
    Factory function to create service clients

    Args:
        service_name: Name of the service
        base_url: Optional base URL (uses default if not provided)
        **kwargs: Additional arguments for client

    Returns:
        Service client instance

    Raises:
        ValueError: If service name is not recognized
    """
    clients = {
        "query-analysis": QueryAnalysisClient,
        "document-retrieval": DocumentRetrievalClient,
        "document-ranking": DocumentRankingClient,
        "latex-parser": LatexParserClient,
        "llm-generation": LLMGenerationClient,
        "response-formatter": ResponseFormatterClient,
    }

    if service_name not in clients:
        raise ValueError(
            f"Unknown service: {service_name}. "
            f"Available services: {list(clients.keys())}"
        )

    client_class = clients[service_name]

    if base_url:
        return client_class(base_url=base_url, **kwargs)
    else:
        return client_class(**kwargs)
