"""
API Key Manager
Manages API keys for service-to-service authentication
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from shared.auth.jwt_handler import APIKeyHandler

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """API Key metadata"""
    service_name: str
    api_key_hash: str
    created_at: datetime
    is_active: bool = True
    description: Optional[str] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0


class APIKeyManager:
    """
    Manages API keys for service-to-service authentication

    In production, this should be backed by a database.
    For now, uses in-memory storage.
    """

    def __init__(self, secret_key: str):
        """
        Initialize API key manager

        Args:
            secret_key: Secret key for hashing
        """
        self.handler = APIKeyHandler(secret_key)
        self.api_keys: Dict[str, APIKey] = {}  # service_name -> APIKey

    def generate_api_key(
        self,
        service_name: str,
        description: Optional[str] = None
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key for a service

        Args:
            service_name: Name of the service
            description: Optional description

        Returns:
            Tuple of (plain_api_key, APIKey object)
        """
        # Generate plain API key
        plain_key = self.handler.generate_api_key()

        # Hash the key for storage
        hashed_key = self.handler.hash_api_key(plain_key)

        # Create API key metadata
        api_key = APIKey(
            service_name=service_name,
            api_key_hash=hashed_key,
            created_at=datetime.utcnow(),
            description=description,
            is_active=True
        )

        # Store (overwrites if service already has a key)
        self.api_keys[service_name] = api_key

        logger.info(f"Generated API key for service: {service_name}")

        # Return plain key (only time it's visible!)
        return plain_key, api_key

    def verify_api_key(
        self,
        plain_key: str,
        update_usage: bool = True
    ) -> Optional[APIKey]:
        """
        Verify API key and return service information

        Args:
            plain_key: Plain API key to verify
            update_usage: Whether to update usage stats

        Returns:
            APIKey object if valid, None otherwise
        """
        # Check against all stored keys
        for service_name, api_key in self.api_keys.items():
            if not api_key.is_active:
                continue

            if self.handler.verify_api_key(plain_key, api_key.api_key_hash):
                # Update usage stats
                if update_usage:
                    api_key.last_used = datetime.utcnow()
                    api_key.usage_count += 1

                logger.debug(f"API key verified for service: {service_name}")
                return api_key

        logger.warning("Invalid API key provided")
        return None

    def revoke_api_key(self, service_name: str) -> bool:
        """
        Revoke (deactivate) an API key

        Args:
            service_name: Name of the service

        Returns:
            True if revoked, False if not found
        """
        api_key = self.api_keys.get(service_name)

        if api_key is None:
            return False

        api_key.is_active = False
        logger.info(f"Revoked API key for service: {service_name}")
        return True

    def delete_api_key(self, service_name: str) -> bool:
        """
        Permanently delete an API key

        Args:
            service_name: Name of the service

        Returns:
            True if deleted, False if not found
        """
        if service_name in self.api_keys:
            del self.api_keys[service_name]
            logger.info(f"Deleted API key for service: {service_name}")
            return True

        return False

    def list_api_keys(self, include_inactive: bool = False) -> List[APIKey]:
        """
        List all API keys

        Args:
            include_inactive: Whether to include inactive keys

        Returns:
            List of APIKey objects
        """
        if include_inactive:
            return list(self.api_keys.values())
        else:
            return [k for k in self.api_keys.values() if k.is_active]

    def get_api_key_info(self, service_name: str) -> Optional[APIKey]:
        """
        Get API key information for a service

        Args:
            service_name: Name of the service

        Returns:
            APIKey object if found, None otherwise
        """
        return self.api_keys.get(service_name)

    def bootstrap_service_keys(
        self,
        services: List[str]
    ) -> Dict[str, str]:
        """
        Generate API keys for multiple services at once
        Useful for initial setup

        Args:
            services: List of service names

        Returns:
            Dict mapping service names to plain API keys
        """
        keys = {}

        for service in services:
            plain_key, _ = self.generate_api_key(
                service_name=service,
                description=f"Auto-generated key for {service}"
            )
            keys[service] = plain_key

        logger.info(f"Bootstrapped API keys for {len(services)} services")
        return keys
