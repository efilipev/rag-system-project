"""
Abstract base class for LaTeX parsers following SOLID principles
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.models.schemas import OutputFormat, ValidationResult


class BaseParser(ABC):
    """
    Abstract base class for LaTeX parsers (Dependency Inversion Principle)

    This interface defines the contract that all parsers must implement,
    allowing the system to work with any LaTeX parsing backend.
    """

    @abstractmethod
    async def parse(
        self,
        latex_string: str,
        output_format: OutputFormat,
        simplify: bool = False
    ) -> Dict[str, Any]:
        """
        Parse LaTeX string to desired format

        Args:
            latex_string: LaTeX formula to parse
            output_format: Desired output format
            simplify: Whether to simplify expressions

        Returns:
            Dict with parsed output and metadata

        Raises:
            Exception: If parsing fails
        """
        pass

    @abstractmethod
    async def validate(self, latex_string: str) -> ValidationResult:
        """
        Validate LaTeX syntax

        Args:
            latex_string: LaTeX formula to validate

        Returns:
            Validation result

        Raises:
            Exception: If validation fails
        """
        pass

    @abstractmethod
    async def parse_batch(
        self,
        latex_strings: List[str],
        output_format: OutputFormat,
        simplify: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Parse multiple LaTeX strings in batch

        Args:
            latex_strings: List of LaTeX formulas
            output_format: Desired output format
            simplify: Whether to simplify expressions

        Returns:
            List of parse results

        Raises:
            Exception: If parsing fails
        """
        pass

    @abstractmethod
    def get_parser_name(self) -> str:
        """
        Get the name of the parser

        Returns:
            Parser name string
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the parser is healthy and ready

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    async def close(self):
        """Clean up resources"""
        pass
