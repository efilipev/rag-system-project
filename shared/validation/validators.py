"""
Input Validation and Sanitization Utilities
Provides enhanced validation beyond Pydantic for security
"""
import re
import logging
from typing import Optional, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Enhanced input validation for security
    """

    # Patterns for dangerous inputs
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|;|\/\*|\*\/)",
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bOR\b.*=.*)",
        r"(xp_cmdshell|sp_executesql)"
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers like onclick, onload
        r"<iframe",
        r"<embed",
        r"<object"
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$()]",
        r"\b(cat|ls|pwd|chmod|rm|mv|cp|curl|wget|nc|bash|sh)\b"
    ]

    @staticmethod
    def sanitize_string(
        text: str,
        max_length: Optional[int] = None,
        allow_html: bool = False,
        remove_control_chars: bool = True
    ) -> str:
        """
        Sanitize string input

        Args:
            text: Input text
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags
            remove_control_chars: Remove control characters

        Returns:
            Sanitized string
        """
        if not isinstance(text, str):
            text = str(text)

        # Remove control characters (except newlines and tabs if needed)
        if remove_control_chars:
            text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')

        # Remove HTML if not allowed
        if not allow_html:
            text = re.sub(r'<[^>]+>', '', text)

        # Trim to max length
        if max_length and len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Input truncated to {max_length} characters")

        return text.strip()

    @classmethod
    def check_sql_injection(cls, text: str) -> bool:
        """
        Check if text contains SQL injection patterns

        Args:
            text: Text to check

        Returns:
            True if suspicious patterns found
        """
        text_upper = text.upper()

        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_upper, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {pattern}")
                return True

        return False

    @classmethod
    def check_xss(cls, text: str) -> bool:
        """
        Check if text contains XSS patterns

        Args:
            text: Text to check

        Returns:
            True if suspicious patterns found
        """
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potential XSS detected: {pattern}")
                return True

        return False

    @classmethod
    def check_command_injection(cls, text: str) -> bool:
        """
        Check if text contains command injection patterns

        Args:
            text: Text to check

        Returns:
            True if suspicious patterns found
        """
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potential command injection detected: {pattern}")
                return True

        return False

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format

        Args:
            email: Email address

        Returns:
            True if valid email format
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> bool:
        """
        Validate URL format and scheme

        Args:
            url: URL to validate
            allowed_schemes: List of allowed schemes (e.g., ['http', 'https'])

        Returns:
            True if valid and safe URL
        """
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']

        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in allowed_schemes:
                logger.warning(f"Invalid URL scheme: {parsed.scheme}")
                return False

            # Check for localhost/private IPs (prevent SSRF)
            if parsed.hostname:
                hostname = parsed.hostname.lower()
                dangerous_hosts = [
                    'localhost', '127.0.0.1', '0.0.0.0',
                    '::1', '169.254.169.254',  # AWS metadata
                    'metadata.google.internal'  # GCP metadata
                ]

                if hostname in dangerous_hosts:
                    logger.warning(f"Dangerous hostname detected: {hostname}")
                    return False

                # Check for private IP ranges
                if hostname.startswith('10.') or hostname.startswith('192.168.'):
                    logger.warning(f"Private IP detected: {hostname}")
                    return False

            return True

        except Exception as e:
            logger.error(f"URL validation failed: {e}")
            return False

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = filename.replace('\\', '/').split('/')[-1]

        # Remove dangerous characters
        filename = re.sub(r'[^\w\s.-]', '', filename)

        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')

        return filename or 'file'

    @staticmethod
    def validate_json_depth(obj: any, max_depth: int = 10, current_depth: int = 0) -> bool:
        """
        Validate JSON object depth to prevent DoS

        Args:
            obj: JSON object to validate
            max_depth: Maximum allowed nesting depth
            current_depth: Current depth (internal)

        Returns:
            True if depth is acceptable
        """
        if current_depth > max_depth:
            logger.warning(f"JSON depth exceeds maximum: {current_depth} > {max_depth}")
            return False

        if isinstance(obj, dict):
            return all(
                InputValidator.validate_json_depth(v, max_depth, current_depth + 1)
                for v in obj.values()
            )
        elif isinstance(obj, list):
            return all(
                InputValidator.validate_json_depth(item, max_depth, current_depth + 1)
                for item in obj
            )

        return True

    @classmethod
    def validate_query(cls, query: str, max_length: int = 2000) -> tuple[bool, Optional[str]]:
        """
        Comprehensive query validation

        Args:
            query: User query
            max_length: Maximum query length

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Length check
        if len(query) > max_length:
            return False, f"Query exceeds maximum length of {max_length} characters"

        # Check for malicious patterns
        if cls.check_sql_injection(query):
            return False, "Query contains suspicious SQL patterns"

        if cls.check_xss(query):
            return False, "Query contains suspicious XSS patterns"

        if cls.check_command_injection(query):
            return False, "Query contains suspicious command injection patterns"

        return True, None
