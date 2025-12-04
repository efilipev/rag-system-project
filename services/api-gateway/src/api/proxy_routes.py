"""
Proxy routes for API Gateway - forwards requests to downstream services.
This maintains the architectural pattern where frontend only communicates with api-gateway.
"""
import logging
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.core.config import settings
from src.api.dependencies import get_orchestrator
from src.services.orchestrator import RAGOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter()


class StreamQueryRequest(BaseModel):
    """Request model for streaming RAG query"""
    query: str
    collection: str = "wikipedia"
    top_k: int = 5
    score_threshold: float = 0.3
    use_hyde_colbert: bool = False
    hyde_colbert_options: Optional[dict] = None
    model: str = "llama3.2:1b"
    enable_query_analysis: bool = True  # Enabled by default for LaTeX detection
    enable_ranking: bool = True  # Enabled by default for better relevance

# Timeout configurations
RETRIEVAL_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=180.0,  # Long read timeout for streaming
    write=30.0,
    pool=5.0
)


@router.get("/collections")
async def list_collections() -> dict[str, Any]:
    """
    Proxy to document-retrieval service to list available collections.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.DOCUMENT_RETRIEVAL_URL}/api/v1/collections"
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Collections request failed: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
) -> dict[str, Any]:
    """
    Proxy document upload to document-retrieval service.
    """
    try:
        # Read file content
        file_content = await file.read()

        # Prepare multipart form data
        files = {"file": (file.filename, file_content, file.content_type)}
        data = {}
        if collection_name:
            data["collection_name"] = collection_name
        if session_id:
            data["session_id"] = session_id

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.DOCUMENT_RETRIEVAL_URL}/api/v1/documents/upload",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Document upload failed: {e}")
        error_detail = e.response.text if e.response else str(e)
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    collection_name: str,
) -> dict[str, Any]:
    """
    Proxy document deletion to document-retrieval service.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{settings.DOCUMENT_RETRIEVAL_URL}/api/v1/documents/{document_id}",
                params={"collection_name": collection_name}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str) -> dict[str, Any]:
    """
    Proxy collection deletion to document-retrieval service.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{settings.DOCUMENT_RETRIEVAL_URL}/api/v1/collections/{collection_name}"
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Collection deletion failed: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_stream(
    request: StreamQueryRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
) -> StreamingResponse:
    """
    Execute streaming RAG query using the full orchestrator pipeline.

    This endpoint uses the orchestrator to:
    1. Query Analysis (optional) - expands and analyzes the query
    2. Document Retrieval - finds relevant documents
    3. Document Ranking (optional) - re-ranks for better relevance
    4. LLM Generation (streaming) - generates response token by token

    The response is streamed as Server-Sent Events (SSE).
    """
    logger.info(f"Streaming query: {request.query[:50]}... (collection={request.collection})")

    return StreamingResponse(
        orchestrator.execute_rag_pipeline_stream(
            query=request.query,
            collection=request.collection,
            retrieval_top_k=request.top_k,
            ranking_top_k=min(request.top_k, 5),  # Rank top 5 at most
            score_threshold=request.score_threshold,
            use_hyde_colbert=request.use_hyde_colbert,
            hyde_colbert_options=request.hyde_colbert_options,
            enable_query_analysis=request.enable_query_analysis,
            enable_ranking=request.enable_ranking,
            model=request.model
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/retrieve")
async def retrieve_documents(request: Request) -> dict[str, Any]:
    """
    Proxy document retrieval to document-retrieval service.
    """
    try:
        body = await request.json()

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.DOCUMENT_RETRIEVAL_URL}/api/v1/retrieve",
                json=body
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Retrieval request failed: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/colbert/indexes")
async def list_colbert_indexes() -> dict[str, Any]:
    """
    Proxy ColBERT index listing to document-retrieval service.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.DOCUMENT_RETRIEVAL_URL}/api/v1/colbert/indexes"
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"ColBERT indexes request failed: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching ColBERT indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/colbert/index")
async def create_colbert_index(
    collection_name: str = "documents",
    batch_size: int = 32,
    limit: Optional[int] = None,
    domain: str = "general",
) -> dict[str, Any]:
    """
    Proxy ColBERT index creation to document-retrieval service.
    """
    try:
        params = {
            "collection_name": collection_name,
            "batch_size": batch_size,
            "domain": domain,
        }
        if limit is not None:
            params["limit"] = limit

        async with httpx.AsyncClient(timeout=600.0) as client:  # Long timeout for indexing
            response = await client.post(
                f"{settings.DOCUMENT_RETRIEVAL_URL}/api/v1/colbert/index",
                params=params
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"ColBERT index creation failed: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating ColBERT index: {e}")
        raise HTTPException(status_code=500, detail=str(e))
