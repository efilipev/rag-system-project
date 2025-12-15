"""
Base Formatter - Abstract base class for formatters.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseFormatter(ABC):
    """
    Abstract base class for response formatters.
    """

    @abstractmethod
    async def format(self, template: str, context: Dict[str, Any]) -> str:
        """
        Format a response using the given template and context.

        Args:
            template: Template string or template name.
            context: Dictionary of variables to use in formatting.

        Returns:
            Formatted string.
        """
        pass

    @abstractmethod
    async def format_batch(
        self, template: str, contexts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Format multiple responses using the same template.

        Args:
            template: Template string or template name.
            contexts: List of context dictionaries.

        Returns:
            List of formatted strings.
        """
        pass

    @abstractmethod
    async def validate_template(self, template: str) -> bool:
        """
        Validate a template string.

        Args:
            template: Template string to validate.

        Returns:
            True if template is valid, False otherwise.
        """
        pass

    @abstractmethod
    def get_formatter_name(self) -> str:
        """
        Get the name of the formatter.

        Returns:
            Formatter name.
        """
        pass

    @abstractmethod
    def get_available_variables(self) -> List[str]:
        """
        Get list of commonly available template variables.

        Returns:
            List of variable names.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the formatter is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        pass

    async def close(self):
        """Cleanup resources."""
        pass
