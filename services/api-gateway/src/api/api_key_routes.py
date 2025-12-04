"""
API Key Management Routes
Admin endpoints for managing service-to-service API keys
"""
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, Header
from pydantic import BaseModel, Field

from src.core.config import settings
from shared.auth.api_key_manager import APIKeyManager, APIKey

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/api-keys", tags=["API Key Management"])

# Initialize API Key Manager (singleton)
api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get or create API key manager instance"""
    global api_key_manager
    if api_key_manager is None:
        api_key_manager = APIKeyManager(secret_key=settings.JWT_SECRET_KEY)
    return api_key_manager


# Request/Response Models
class GenerateAPIKeyRequest(BaseModel):
    """Request to generate new API key"""
    service_name: str = Field(..., min_length=3, max_length=50)
    description: Optional[str] = None


class GenerateAPIKeyResponse(BaseModel):
    """Response with new API key"""
    service_name: str
    api_key: str
    created_at: datetime
    description: Optional[str] = None
    warning: str = "Save this API key - it will not be shown again!"


class APIKeyInfo(BaseModel):
    """API key information (without the actual key)"""
    service_name: str
    created_at: datetime
    is_active: bool
    description: Optional[str] = None
    last_used: Optional[datetime] = None
    usage_count: int


class ListAPIKeysResponse(BaseModel):
    """Response with list of API keys"""
    api_keys: List[APIKeyInfo]
    total: int


class BootstrapResponse(BaseModel):
    """Response from bootstrapping service keys"""
    services: List[str]
    api_keys: dict  # service_name -> api_key
    warning: str = "Save these API keys - they will not be shown again!"


# Simple admin authentication (in production, use proper RBAC)
async def verify_admin_token(x_admin_token: Optional[str] = Header(None)):
    """
    Verify admin token for API key management endpoints

    In production, replace with proper role-based access control
    """
    # For now, use a simple admin token from settings
    # In production, verify JWT token with admin role
    admin_token = getattr(settings, "ADMIN_TOKEN", "admin-secret-token")

    if x_admin_token != admin_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )


@router.post(
    "/generate",
    response_model=GenerateAPIKeyResponse,
    dependencies=[Depends(verify_admin_token)]
)
async def generate_api_key(
    request: GenerateAPIKeyRequest,
    manager: APIKeyManager = Depends(get_api_key_manager)
) -> GenerateAPIKeyResponse:
    """
    Generate a new API key for a service

    Requires admin authentication (X-Admin-Token header)

    Args:
        request: Service name and description

    Returns:
        New API key (only shown once!)
    """
    try:
        plain_key, api_key = manager.generate_api_key(
            service_name=request.service_name,
            description=request.description
        )

        logger.info(f"Admin generated API key for service: {request.service_name}")

        return GenerateAPIKeyResponse(
            service_name=api_key.service_name,
            api_key=plain_key,
            created_at=api_key.created_at,
            description=api_key.description
        )

    except Exception as e:
        logger.error(f"Failed to generate API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate API key: {str(e)}"
        )


@router.get(
    "/list",
    response_model=ListAPIKeysResponse,
    dependencies=[Depends(verify_admin_token)]
)
async def list_api_keys(
    include_inactive: bool = False,
    manager: APIKeyManager = Depends(get_api_key_manager)
) -> ListAPIKeysResponse:
    """
    List all API keys

    Requires admin authentication (X-Admin-Token header)

    Args:
        include_inactive: Include inactive/revoked keys

    Returns:
        List of API key information (without actual keys)
    """
    api_keys = manager.list_api_keys(include_inactive=include_inactive)

    key_infos = [
        APIKeyInfo(
            service_name=k.service_name,
            created_at=k.created_at,
            is_active=k.is_active,
            description=k.description,
            last_used=k.last_used,
            usage_count=k.usage_count
        )
        for k in api_keys
    ]

    return ListAPIKeysResponse(
        api_keys=key_infos,
        total=len(key_infos)
    )


@router.get(
    "/{service_name}",
    response_model=APIKeyInfo,
    dependencies=[Depends(verify_admin_token)]
)
async def get_api_key_info(
    service_name: str,
    manager: APIKeyManager = Depends(get_api_key_manager)
) -> APIKeyInfo:
    """
    Get API key information for a service

    Requires admin authentication (X-Admin-Token header)

    Args:
        service_name: Name of the service

    Returns:
        API key information (without actual key)
    """
    api_key = manager.get_api_key_info(service_name)

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No API key found for service: {service_name}"
        )

    return APIKeyInfo(
        service_name=api_key.service_name,
        created_at=api_key.created_at,
        is_active=api_key.is_active,
        description=api_key.description,
        last_used=api_key.last_used,
        usage_count=api_key.usage_count
    )


@router.delete(
    "/{service_name}/revoke",
    dependencies=[Depends(verify_admin_token)]
)
async def revoke_api_key(
    service_name: str,
    manager: APIKeyManager = Depends(get_api_key_manager)
) -> dict:
    """
    Revoke (deactivate) an API key

    Requires admin authentication (X-Admin-Token header)

    Args:
        service_name: Name of the service

    Returns:
        Success message
    """
    success = manager.revoke_api_key(service_name)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No API key found for service: {service_name}"
        )

    logger.info(f"Admin revoked API key for service: {service_name}")

    return {
        "message": f"API key revoked for service: {service_name}",
        "service_name": service_name,
        "is_active": False
    }


@router.delete(
    "/{service_name}",
    dependencies=[Depends(verify_admin_token)]
)
async def delete_api_key(
    service_name: str,
    manager: APIKeyManager = Depends(get_api_key_manager)
) -> dict:
    """
    Permanently delete an API key

    Requires admin authentication (X-Admin-Token header)

    Args:
        service_name: Name of the service

    Returns:
        Success message
    """
    success = manager.delete_api_key(service_name)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No API key found for service: {service_name}"
        )

    logger.info(f"Admin deleted API key for service: {service_name}")

    return {
        "message": f"API key permanently deleted for service: {service_name}",
        "service_name": service_name
    }


@router.post(
    "/bootstrap",
    response_model=BootstrapResponse,
    dependencies=[Depends(verify_admin_token)]
)
async def bootstrap_service_keys(
    manager: APIKeyManager = Depends(get_api_key_manager)
) -> BootstrapResponse:
    """
    Bootstrap API keys for all microservices

    Generates API keys for all core services in the RAG system.
    Use this for initial setup only!

    Requires admin authentication (X-Admin-Token header)

    Returns:
        API keys for all services (only shown once!)
    """
    services = [
        "query-analysis",
        "document-retrieval",
        "document-ranking",
        "latex-parser",
        "llm-generation",
        "response-formatter"
    ]

    api_keys = manager.bootstrap_service_keys(services)

    logger.info(f"Admin bootstrapped API keys for {len(services)} services")

    return BootstrapResponse(
        services=services,
        api_keys=api_keys
    )
