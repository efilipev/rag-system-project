"""
LaTeX Parser implementation using sympy, latex2mathml, and pylatexenc
"""
import logging
import asyncio
import re
from typing import Dict, Any, List
import sympy
from sympy.parsing.latex import parse_latex
from latex2mathml.converter import convert as latex_to_mathml
from pylatexenc.latex2text import LatexNodes2Text

from src.models.schemas import OutputFormat, ValidationResult
from src.services.base_parser import BaseParser

logger = logging.getLogger(__name__)


class LatexParser(BaseParser):
    """
    LaTeX parser using multiple backends
    Follows Single Responsibility Principle - handles LaTeX parsing
    """

    # Dangerous LaTeX commands that should be blocked
    DANGEROUS_COMMANDS = [
        r'\\write', r'\\input', r'\\include', r'\\openin',
        r'\\openout', r'\\read', r'\\csname', r'\\expandafter',
        r'\\catcode', r'\\def', r'\\let', r'\\futurelet'
    ]

    def __init__(self):
        """Initialize LaTeX parser"""
        self.parser_name = "SymPy+Latex2MathML+PyLatexEnc"
        self.latex2text_converter = LatexNodes2Text()
        logger.info(f"LatexParser initialized")

    def _sanitize_latex(self, latex_string: str) -> str:
        """
        Sanitize LaTeX input by removing dangerous commands

        Args:
            latex_string: LaTeX string to sanitize

        Returns:
            Sanitized LaTeX string

        Raises:
            ValueError: If dangerous commands are detected
        """
        # Check for dangerous commands
        for dangerous_cmd in self.DANGEROUS_COMMANDS:
            if re.search(dangerous_cmd, latex_string, re.IGNORECASE):
                raise ValueError(f"Dangerous LaTeX command detected: {dangerous_cmd}")

        # Remove comments
        latex_string = re.sub(r'%.*$', '', latex_string, flags=re.MULTILINE)

        return latex_string.strip()

    def _extract_commands(self, latex_string: str) -> List[str]:
        """
        Extract LaTeX commands from string

        Args:
            latex_string: LaTeX string

        Returns:
            List of command names
        """
        commands = re.findall(r'\\([a-zA-Z]+)', latex_string)
        return list(set(commands))

    async def validate(self, latex_string: str) -> ValidationResult:
        """
        Validate LaTeX syntax

        Args:
            latex_string: LaTeX formula to validate

        Returns:
            Validation result
        """
        try:
            # Sanitize first
            sanitized = self._sanitize_latex(latex_string)

            # Extract commands
            commands = self._extract_commands(sanitized)

            warnings = []

            # Try parsing with sympy
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, parse_latex, sanitized)
                is_valid = True
                error_message = None

            except Exception as e:
                # Sympy failed, but it might still be valid LaTeX
                # Try with latex2mathml as fallback validation
                try:
                    await loop.run_in_executor(None, latex_to_mathml, sanitized)
                    is_valid = True
                    error_message = None
                    warnings.append(f"Sympy parsing failed but MathML conversion succeeded: {str(e)}")

                except Exception as e2:
                    is_valid = False
                    error_message = f"Both Sympy and MathML conversion failed: {str(e2)}"

            return ValidationResult(
                is_valid=is_valid,
                latex_string=latex_string,
                error_message=error_message,
                warnings=warnings,
                detected_commands=commands
            )

        except ValueError as e:
            # Dangerous command detected
            return ValidationResult(
                is_valid=False,
                latex_string=latex_string,
                error_message=str(e),
                warnings=[],
                detected_commands=[]
            )

        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return ValidationResult(
                is_valid=False,
                latex_string=latex_string,
                error_message=str(e),
                warnings=[],
                detected_commands=[]
            )

    def _parse_to_mathml(self, latex_string: str) -> str:
        """Parse LaTeX to MathML"""
        try:
            mathml = latex_to_mathml(latex_string)
            return mathml
        except Exception as e:
            logger.error(f"MathML conversion error: {e}")
            raise

    def _parse_to_text(self, latex_string: str) -> str:
        """Parse LaTeX to plain text"""
        try:
            text = self.latex2text_converter.latex_to_text(latex_string)
            return text
        except Exception as e:
            logger.error(f"Text conversion error: {e}")
            raise

    def _parse_to_unicode(self, latex_string: str) -> str:
        """Parse LaTeX to Unicode representation"""
        try:
            # Use sympy to parse and then convert to Unicode
            expr = parse_latex(latex_string)
            unicode_str = sympy.pretty(expr, use_unicode=True)
            return unicode_str
        except Exception as e:
            logger.error(f"Unicode conversion error: {e}")
            # Fallback to text conversion
            return self._parse_to_text(latex_string)

    def _simplify_expression(self, latex_string: str) -> str:
        """Simplify mathematical expression"""
        try:
            expr = parse_latex(latex_string)
            simplified = sympy.simplify(expr)
            simplified_latex = sympy.latex(simplified)
            return simplified_latex
        except Exception as e:
            logger.warning(f"Simplification failed: {e}")
            return latex_string

    def _clean_latex(self, latex_string: str) -> str:
        """Clean and normalize LaTeX"""
        try:
            # Parse and re-export to get clean LaTeX
            expr = parse_latex(latex_string)
            clean_latex = sympy.latex(expr)
            return clean_latex
        except Exception as e:
            logger.warning(f"LaTeX cleaning failed: {e}")
            return latex_string

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
        """
        try:
            # Validate first
            validation = await self.validate(latex_string)
            if not validation.is_valid:
                raise ValueError(f"Invalid LaTeX: {validation.error_message}")

            # Sanitize
            sanitized = self._sanitize_latex(latex_string)

            # Run parsing in executor to avoid blocking
            loop = asyncio.get_event_loop()

            # Get simplified form if requested
            simplified_form = None
            if simplify:
                simplified_form = await loop.run_in_executor(
                    None,
                    self._simplify_expression,
                    sanitized
                )

            # Parse to requested format
            if output_format == OutputFormat.MATHML:
                parsed_output = await loop.run_in_executor(
                    None,
                    self._parse_to_mathml,
                    sanitized
                )
            elif output_format == OutputFormat.TEXT:
                parsed_output = await loop.run_in_executor(
                    None,
                    self._parse_to_text,
                    sanitized
                )
            elif output_format == OutputFormat.UNICODE:
                parsed_output = await loop.run_in_executor(
                    None,
                    self._parse_to_unicode,
                    sanitized
                )
            elif output_format == OutputFormat.SIMPLIFIED:
                if simplified_form:
                    parsed_output = simplified_form
                else:
                    parsed_output = await loop.run_in_executor(
                        None,
                        self._simplify_expression,
                        sanitized
                    )
            elif output_format == OutputFormat.LATEX:
                parsed_output = await loop.run_in_executor(
                    None,
                    self._clean_latex,
                    sanitized
                )
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(f"Successfully parsed LaTeX to {output_format}")

            return {
                "original_latex": latex_string,
                "parsed_output": parsed_output,
                "output_format": output_format,
                "is_valid": True,
                "simplified_form": simplified_form,
                "metadata": {
                    "commands": validation.detected_commands,
                    "warnings": validation.warnings
                }
            }

        except Exception as e:
            logger.error(f"Parsing error: {e}", exc_info=True)
            raise

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
        """
        try:
            logger.info(f"Batch parsing {len(latex_strings)} LaTeX formulas to {output_format}")

            # Parse each formula
            results = []
            for latex_string in latex_strings:
                try:
                    result = await self.parse(latex_string, output_format, simplify)
                    results.append(result)

                except Exception as e:
                    # Add error result for failed parses
                    logger.warning(f"Failed to parse LaTeX: {e}")
                    results.append({
                        "original_latex": latex_string,
                        "parsed_output": "",
                        "output_format": output_format,
                        "is_valid": False,
                        "simplified_form": None,
                        "metadata": {
                            "error": str(e)
                        }
                    })

            logger.info(f"Batch parsing completed. Success: {sum(1 for r in results if r['is_valid'])}/{len(results)}")

            return results

        except Exception as e:
            logger.error(f"Batch parsing error: {e}", exc_info=True)
            raise

    def get_parser_name(self) -> str:
        """Get the name of the parser"""
        return self.parser_name

    async def health_check(self) -> bool:
        """
        Check if the parser is healthy

        Returns:
            True if healthy
        """
        try:
            # Try a simple parse
            test_latex = r"\frac{1}{2}"
            result = await self.parse(test_latex, OutputFormat.MATHML)
            return result["is_valid"]

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Clean up resources"""
        # No cleanup needed for this parser
        logger.info("LatexParser closed")
