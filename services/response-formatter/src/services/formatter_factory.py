"""
Formatter Factory - Factory Pattern with Dependency Injection
"""
import logging
from typing import Optional, Any, Dict, List

from src.services.base_formatter import BaseFormatter
from src.services.jinja_formatter import JinjaFormatter

logger = logging.getLogger(__name__)


class FormatterFactory:
    """
    Factory for creating response formatters (Factory Pattern)
    """

    _instance: Optional['FormatterFactory'] = None
    _formatter: Optional[BaseFormatter] = None

    def __new__(cls):
        """Singleton pattern to ensure single factory instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def create_formatter(
        cls,
        formatter_type: str = "jinja2",
        **kwargs
    ) -> BaseFormatter:
        """
        Create a formatter based on configuration.

        Args:
            formatter_type: Type of formatter ('jinja2', 'default').
            **kwargs: Formatter-specific configuration.

        Returns:
            Formatter instance.

        Raises:
            ValueError: If formatter type is not supported.
        """
        formatter_type = formatter_type.lower()

        try:
            if formatter_type in ["default", "jinja2", "jinja"]:
                logger.info("Creating JinjaFormatter")
                return JinjaFormatter()
            else:
                raise ValueError(
                    f"Unsupported formatter type: {formatter_type}. "
                    f"Supported types: default, jinja2, jinja"
                )
        except Exception as e:
            logger.error(f"Failed to create formatter {formatter_type}: {e}")
            raise

    @classmethod
    def get_formatter(cls) -> Optional[BaseFormatter]:
        """Get the current formatter instance."""
        return cls._formatter

    @classmethod
    def set_formatter(cls, formatter: BaseFormatter):
        """Set the formatter instance (Dependency Injection)."""
        cls._formatter = formatter
        logger.info(f"Formatter set to: {formatter.__class__.__name__}")

    @classmethod
    async def close_formatter(cls):
        """Close the current formatter and cleanup resources."""
        if cls._formatter:
            await cls._formatter.close()
            cls._formatter = None
            logger.info("Formatter closed and cleaned up")


class FormattingService:
    """
    High-level service class for formatting operations.
    """

    def __init__(self, formatter: BaseFormatter):
        """
        Initialize service with a formatter.

        Args:
            formatter: Formatter instance (injected dependency).
        """
        self.formatter = formatter
        logger.info(f"FormattingService initialized with {formatter.__class__.__name__}")

    async def format(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        output_format: str = "markdown",
        query: Optional[str] = None,
        include_citations: bool = True,
        include_metadata: bool = False,
        custom_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format content with sources and citations.

        Args:
            content: The main content to format.
            sources: List of source documents.
            output_format: Output format (markdown, html, json, plain_text).
            query: Original user query.
            include_citations: Whether to include citations.
            include_metadata: Whether to include metadata.
            custom_template: Optional custom Jinja2 template.

        Returns:
            Formatted result dictionary.
        """
        sources = sources or []

        # Build context for template
        context = {
            "content": content,
            "sources": sources,
            "query": query,
            "include_citations": include_citations,
            "include_metadata": include_metadata,
        }

        # Use custom template or default based on output format
        if custom_template:
            template = custom_template
        else:
            template = self._get_default_template(output_format)

        # Format using the formatter
        formatted_content = await self.formatter.format(template, context)

        return {
            "formatted_content": formatted_content,
            "output_format": output_format,
            "original_query": query,
            "num_sources": len(sources),
            "metadata": {"include_citations": include_citations} if include_metadata else {}
        }

    def _get_default_template(self, output_format: str) -> str:
        """Get default template for output format."""
        templates = {
            "markdown": """{{ content }}
{% if include_citations and sources %}

---
**Sources:**
{% for source in sources %}
{{ loop.index }}. {{ source.title|default('Unknown') }} - {{ source.source|default('Unknown source') }}
{% endfor %}
{% endif %}""",
            "html": """<div class="response">{{ content }}</div>
{% if include_citations and sources %}
<hr>
<div class="sources">
<strong>Sources:</strong>
<ol>
{% for source in sources %}
<li>{{ source.title|default('Unknown') }} - {{ source.source|default('Unknown source') }}</li>
{% endfor %}
</ol>
</div>
{% endif %}""",
            "plain_text": """{{ content }}
{% if include_citations and sources %}

Sources:
{% for source in sources %}
{{ loop.index }}. {{ source.title|default('Unknown') }} - {{ source.source|default('Unknown source') }}
{% endfor %}
{% endif %}""",
            "json": """{{ content }}"""
        }
        return templates.get(output_format, templates["markdown"])

    async def format_batch(
        self, template: str, contexts: List[Dict[str, Any]]
    ) -> List[str]:
        """Delegate to formatter."""
        return await self.formatter.format_batch(template, contexts)

    async def validate_template(self, template: str) -> bool:
        """Delegate to formatter."""
        return await self.formatter.validate_template(template)

    def get_formatter_name(self) -> str:
        """Delegate to formatter."""
        return self.formatter.get_formatter_name()

    def get_available_variables(self) -> List[str]:
        """Delegate to formatter."""
        return self.formatter.get_available_variables()

    async def health_check(self) -> bool:
        """Delegate to formatter."""
        return await self.formatter.health_check()
