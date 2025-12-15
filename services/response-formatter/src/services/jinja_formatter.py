"""
Jinja2-based Response Formatter.
"""
import logging
from typing import Any, Dict, List

from jinja2 import Environment, BaseLoader, TemplateSyntaxError

from src.services.base_formatter import BaseFormatter

logger = logging.getLogger(__name__)


class JinjaFormatter(BaseFormatter):
    """
    Response formatter using Jinja2 templating engine.
    """

    def __init__(self):
        """Initialize Jinja2 environment."""
        self.env = Environment(loader=BaseLoader())
        logger.info("JinjaFormatter initialized")

    async def format(self, template: str, context: Dict[str, Any]) -> str:
        """
        Format a response using Jinja2 template.

        Args:
            template: Jinja2 template string.
            context: Dictionary of variables.

        Returns:
            Formatted string.
        """
        try:
            jinja_template = self.env.from_string(template)
            return jinja_template.render(**context)
        except Exception as e:
            logger.error(f"Formatting error: {e}")
            raise

    async def format_batch(
        self, template: str, contexts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Format multiple responses using the same template.

        Args:
            template: Jinja2 template string.
            contexts: List of context dictionaries.

        Returns:
            List of formatted strings.
        """
        results = []
        jinja_template = self.env.from_string(template)
        for context in contexts:
            try:
                results.append(jinja_template.render(**context))
            except Exception as e:
                logger.error(f"Batch formatting error: {e}")
                results.append(f"Error: {str(e)}")
        return results

    async def validate_template(self, template: str) -> bool:
        """
        Validate a Jinja2 template string.

        Args:
            template: Template string to validate.

        Returns:
            True if template is valid.
        """
        try:
            self.env.from_string(template)
            return True
        except TemplateSyntaxError:
            return False

    def get_formatter_name(self) -> str:
        """Get formatter name."""
        return "jinja2"

    def get_available_variables(self) -> List[str]:
        """Get commonly available template variables."""
        return [
            "query",
            "answer",
            "sources",
            "metadata",
            "timestamp",
            "confidence",
        ]

    async def health_check(self) -> bool:
        """Check if formatter is healthy."""
        try:
            test_template = "Hello {{ name }}!"
            result = await self.format(test_template, {"name": "World"})
            return result == "Hello World!"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
