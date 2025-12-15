"""
FastAPI dependencies for Document Retrieval Service.
"""
from fastapi import Request, HTTPException, status

from src.services.vector_store import VectorStoreService


def get_vector_store(request: Request) -> VectorStoreService:
    """
    Dependency to get the vector store service instance.

    Args:
        request: FastAPI request object.

    Returns:
        VectorStoreService instance.

    Raises:
        HTTPException: If the service is not initialized.
    """
    service = getattr(request.app.state, "vector_store", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store service not initialized"
        )
    return service
