"""
FastAPI dependencies for API Gateway.
"""
from fastapi import Request, HTTPException, status

from src.services.orchestrator import RAGOrchestrator


def get_orchestrator(request: Request) -> RAGOrchestrator:
    """
    Dependency to get the RAG orchestrator instance.

    Args:
        request: FastAPI request object.

    Returns:
        RAGOrchestrator instance.

    Raises:
        HTTPException: If the orchestrator is not initialized.
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Orchestrator not initialized"
        )
    return orchestrator
