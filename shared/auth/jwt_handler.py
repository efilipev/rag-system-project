"""
JWT Token Handler for Authentication
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)


class JWTHandler:
    """
    JWT token handler for authentication
    Handles token generation, verification, and user authentication
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7
    ):
        """
        Initialize JWT handler

        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm (default: HS256)
            access_token_expire_minutes: Access token expiration time
            refresh_token_expire_days: Refresh token expiration time
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days

        # Password hashing context
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        logger.info("JWT Handler initialized")

    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password

        Returns:
            True if password matches
        """
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token

        Args:
            data: Data to encode in token (typically user_id, roles, etc.)
            expires_delta: Optional custom expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """
        Create JWT refresh token

        Args:
            data: Data to encode in token

        Returns:
            Encoded JWT refresh token
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode and verify JWT token

        Args:
            token: JWT token to decode

        Returns:
            Decoded token payload or None if invalid

        Raises:
            jwt.ExpiredSignatureError: If token is expired
            jwt.InvalidTokenError: If token is invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise

    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify access token and return payload

        Args:
            token: Access token

        Returns:
            Token payload or None if invalid
        """
        try:
            payload = self.decode_token(token)

            if payload.get("type") != "access":
                logger.warning("Token is not an access token")
                return None

            return payload

        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None

    def verify_refresh_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify refresh token and return payload

        Args:
            token: Refresh token

        Returns:
            Token payload or None if invalid
        """
        try:
            payload = self.decode_token(token)

            if payload.get("type") != "refresh":
                logger.warning("Token is not a refresh token")
                return None

            return payload

        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None


class APIKeyHandler:
    """
    API Key handler for service-to-service authentication
    """

    def __init__(self, secret_key: str):
        """
        Initialize API key handler

        Args:
            secret_key: Secret key for hashing
        """
        self.secret_key = secret_key
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def generate_api_key(self) -> str:
        """
        Generate a new API key

        Returns:
            API key string (to be hashed before storage)
        """
        import secrets
        return f"rag_{secrets.token_urlsafe(32)}"

    def hash_api_key(self, api_key: str) -> str:
        """
        Hash an API key for storage

        Args:
            api_key: Plain API key

        Returns:
            Hashed API key
        """
        return self.pwd_context.hash(api_key)

    def verify_api_key(self, plain_key: str, hashed_key: str) -> bool:
        """
        Verify API key against its hash

        Args:
            plain_key: Plain API key
            hashed_key: Hashed API key

        Returns:
            True if key matches
        """
        return self.pwd_context.verify(plain_key, hashed_key)
