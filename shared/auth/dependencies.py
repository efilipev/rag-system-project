"""
FastAPI Dependencies for Authentication
Provides reusable dependencies for JWT and API key authentication
"""
import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

from shared.auth.jwt_handler import JWTHandler

logger = logging.getLogger(__name__)

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthDependency:
    """
    Authentication dependency provider
    Manages JWT and API key authentication
    """

    def __init__(
        self,
        jwt_handler: JWTHandler,
        api_key_hashes: Optional[Dict[str, str]] = None
    ):
        """
        Initialize authentication dependency

        Args:
            jwt_handler: JWT handler instance
            api_key_hashes: Dictionary of service names to hashed API keys
        """
        self.jwt_handler = jwt_handler
        self.api_key_hashes = api_key_hashes or {}

    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
    ) -> Dict[str, Any]:
        """
        Verify JWT token and return user payload

        Args:
            credentials: Bearer token credentials

        Returns:
            User payload from token

        Raises:
            HTTPException: If token is invalid or expired
        """
        token = credentials.credentials

        try:
            payload = self.jwt_handler.verify_access_token(token)

            if payload is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return payload

        except Exception as e:
            logger.warning(f"Token verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def get_current_user_optional(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
    ) -> Optional[Dict[str, Any]]:
        """
        Optionally verify JWT token (allows anonymous access)

        Args:
            credentials: Optional bearer token credentials

        Returns:
            User payload if token provided and valid, None otherwise
        """
        if credentials is None:
            return None

        try:
            return await self.get_current_user(credentials)
        except HTTPException:
            return None

    async def verify_api_key(
        self,
        api_key: Optional[str] = Security(api_key_header)
    ) -> Dict[str, Any]:
        """
        Verify API key for service-to-service authentication

        Args:
            api_key: API key from X-API-Key header

        Returns:
            Service information

        Raises:
            HTTPException: If API key is invalid
        """
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Check API key against stored hashes
        for service_name, hashed_key in self.api_key_hashes.items():
            if self.jwt_handler.verify_api_key(api_key, hashed_key):
                logger.info(f"API key verified for service: {service_name}")
                return {
                    "service_name": service_name,
                    "auth_type": "api_key"
                }

        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    async def get_current_user_or_service(
        self,
        user_credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
        api_key: Optional[str] = Security(api_key_header)
    ) -> Dict[str, Any]:
        """
        Verify either JWT token or API key (flexible authentication)

        Args:
            user_credentials: Optional JWT bearer token
            api_key: Optional API key

        Returns:
            User or service payload

        Raises:
            HTTPException: If neither authentication method is valid
        """
        # Try JWT first
        if user_credentials:
            try:
                return await self.get_current_user(user_credentials)
            except HTTPException:
                pass

        # Try API key
        if api_key:
            try:
                return await self.verify_api_key(api_key)
            except HTTPException:
                pass

        # Neither authentication method succeeded
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid JWT token or API key required",
            headers={"WWW-Authenticate": "Bearer, ApiKey"},
        )


def require_role(required_role: str):
    """
    Dependency factory for role-based access control

    Args:
        required_role: Required role name

    Returns:
        Dependency function that checks user role
    """
    async def role_checker(current_user: Dict[str, Any] = Depends(lambda: None)) -> Dict[str, Any]:
        """
        Check if user has required role

        Args:
            current_user: Current user from JWT token

        Returns:
            User payload if role matches

        Raises:
            HTTPException: If user doesn't have required role
        """
        if current_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )

        user_roles = current_user.get("roles", [])

        if required_role not in user_roles:
            logger.warning(
                f"User {current_user.get('user_id')} attempted to access "
                f"resource requiring role '{required_role}'"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )

        return current_user

    return role_checker


def require_any_role(*required_roles: str):
    """
    Dependency factory for role-based access control (any of multiple roles)

    Args:
        required_roles: List of acceptable roles

    Returns:
        Dependency function that checks user has at least one role
    """
    async def role_checker(current_user: Dict[str, Any] = Depends(lambda: None)) -> Dict[str, Any]:
        """
        Check if user has any of the required roles

        Args:
            current_user: Current user from JWT token

        Returns:
            User payload if has any required role

        Raises:
            HTTPException: If user doesn't have any required role
        """
        if current_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )

        user_roles = current_user.get("roles", [])

        if not any(role in user_roles for role in required_roles):
            logger.warning(
                f"User {current_user.get('user_id')} attempted to access "
                f"resource requiring one of roles: {required_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {required_roles} required"
            )

        return current_user

    return role_checker
