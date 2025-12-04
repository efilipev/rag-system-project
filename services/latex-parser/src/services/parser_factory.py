"""
Parser Factory - Factory Pattern with Dependency Injection
"""
import logging
from typing import Optional

from src.services.base_parser import BaseParser
from src.services.latex_parser import LatexParser

logger = logging.getLogger(__name__)


class ParserFactory:
    """
    Factory for creating LaTeX parsers (Factory Pattern)

    This class follows the Open/Closed Principle - open for extension
    (can add new parsers) but closed for modification.
    """

    _instance: Optional['ParserFactory'] = None
    _parser: Optional[BaseParser] = None

    def __new__(cls):
        """Singleton pattern to ensure single factory instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def create_parser(
        cls,
        parser_type: str = "default",
        **kwargs
    ) -> BaseParser:
        """
        Create a parser based on configuration

        Args:
            parser_type: Type of parser ('default', 'sympy')
            **kwargs: Parser-specific configuration

        Returns:
            Parser instance

        Raises:
            ValueError: If parser type is not supported
        """
        parser_type = parser_type.lower()

        try:
            if parser_type in ["default", "sympy", "latex"]:
                logger.info(f"Creating LatexParser")
                return LatexParser()

            else:
                raise ValueError(
                    f"Unsupported parser type: {parser_type}. "
                    f"Supported types: default, sympy, latex"
                )

        except Exception as e:
            logger.error(f"Failed to create parser {parser_type}: {e}")
            raise

    @classmethod
    def get_parser(cls) -> Optional[BaseParser]:
        """
        Get the current parser instance

        Returns:
            Current parser or None if not initialized
        """
        return cls._parser

    @classmethod
    def set_parser(cls, parser: BaseParser):
        """
        Set the parser instance (Dependency Injection)

        Args:
            parser: Parser instance to set
        """
        cls._parser = parser
        logger.info(f"Parser set to: {parser.__class__.__name__}")

    @classmethod
    async def close_parser(cls):
        """Close the current parser and cleanup resources"""
        if cls._parser:
            await cls._parser.close()
            cls._parser = None
            logger.info("Parser closed and cleaned up")


class ParsingService:
    """
    High-level service class that uses dependency injection
    to work with any parser (Dependency Inversion Principle)

    This class depends on the BaseParser abstraction, not on
    concrete implementations, allowing easy swapping of parsers.
    """

    def __init__(self, parser: BaseParser):
        """
        Initialize service with a parser

        Args:
            parser: Parser instance (injected dependency)
        """
        self.parser = parser
        logger.info(f"ParsingService initialized with {parser.__class__.__name__}")

    async def parse(self, latex_string: str, output_format, simplify: bool = False):
        """Delegate to parser"""
        return await self.parser.parse(latex_string, output_format, simplify)

    async def validate(self, latex_string: str):
        """Delegate to parser"""
        return await self.parser.validate(latex_string)

    async def parse_batch(self, latex_strings: list, output_format, simplify: bool = False):
        """Delegate to parser"""
        return await self.parser.parse_batch(latex_strings, output_format, simplify)

    def get_parser_name(self) -> str:
        """Delegate to parser"""
        return self.parser.get_parser_name()

    async def health_check(self) -> bool:
        """Delegate to parser"""
        return await self.parser.health_check()
