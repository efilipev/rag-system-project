"""
API routes for Document Retrieval Service.
"""
import time
import httpx
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse

from src.core.logging import logger
from src.models.schemas import (
    Document,
    RetrievalRequest,
    RetrievalResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from src.services.upload_service import UploadService, UploadResult
from src.services.vector_store import VectorStoreService
from src.api.dependencies import get_vector_store

router = APIRouter()

LLM_SERVICE_URL = "http://llm-generation:8000/api/v1"
DEFAULT_COLLECTION = "wikipedia"


async def resolve_collection(vector_store, collection_name: str | None) -> str:
    """
    Resolve collection name with fallback to default.

    :param vector_store: Vector store service instance.
    :param collection_name: Requested collection name.
    :return: Resolved collection name.
    """
    # Use default if empty or None
    if not collection_name or collection_name.strip() == "":
        collection_name = DEFAULT_COLLECTION

    # Verify collection exists
    try:
        collections = await vector_store.list_collections()
        collection_names = [c["name"] for c in collections]

        if collection_name not in collection_names:
            # If requested collection doesn't exist, fall back to default
            if collection_name != DEFAULT_COLLECTION and DEFAULT_COLLECTION in collection_names:
                logger.warning(f"Collection '{collection_name}' not found, using '{DEFAULT_COLLECTION}'")
                return DEFAULT_COLLECTION
            logger.warning(f"Collection '{collection_name}' not found")
    except Exception as e:
        logger.warning(f"Could not verify collection: {e}")

    return collection_name


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(
    retrieval_request: RetrievalRequest,
    vector_store: VectorStoreService = Depends(get_vector_store)
) -> RetrievalResponse:
    """
    Retrieve relevant documents based on query.

    Supports HyDE-ColBERT retrieval when use_hyde_colbert=True.

    :param retrieval_request: Retrieval request with query and options.
    :param vector_store: Injected VectorStoreService instance.
    :return: Retrieval response with documents.
    """
    start_time = time.time()

    try:
        logger.info(f"Retrieving documents for query: {retrieval_request.query[:100]}")

        # Perform hybrid search (with optional HyDE-ColBERT)
        results = await vector_store.hybrid_search_with_hyde(
            query=retrieval_request.query,
            top_k=retrieval_request.top_k,
            filter_dict=retrieval_request.filters,
            score_threshold=retrieval_request.score_threshold,
            use_qdrant=retrieval_request.use_qdrant,
            use_chroma=retrieval_request.use_chroma,
            use_hyde_colbert=retrieval_request.use_hyde_colbert,
            hyde_colbert_options=retrieval_request.hyde_colbert_options,
            collection_name=retrieval_request.collection,
        )

        # Format documents
        documents = [
            Document(
                content=text,
                score=score,
                metadata=metadata,
                source=metadata.get("source", "unknown"),
            )
            for text, score, metadata in results
        ]

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        return RetrievalResponse(
            success=True,
            documents=documents,
            total_found=len(documents),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        processing_time = (time.time() - start_time) * 1000
        return RetrievalResponse(
            success=False,
            documents=[],
            total_found=0,
            error=str(e),
            processing_time_ms=processing_time,
        )


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    :return: Health status dictionary.
    """
    return {"status": "healthy", "service": "document-retrieval"}


@router.get("/collections")
async def list_collections(
    vector_store: VectorStoreService = Depends(get_vector_store)
) -> dict[str, Any]:
    """
    List all available collections from Qdrant.

    :param vector_store: Injected VectorStoreService instance.
    :return: Collections list with names and document counts.
    :raises HTTPException: If listing fails.
    """
    try:
        collections = await vector_store.list_collections()
        return {"success": True, "collections": collections}
    except Exception as e:
        logger.error(f"Error listing collections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/colbert/index")
async def create_colbert_index(
    vector_store: VectorStoreService = Depends(get_vector_store),
    collection_name: str = "documents",
    batch_size: int = 32,
    limit: int | None = None,
    domain: str = "general",
) -> dict[str, Any]:
    """
    Create a ColBERT index from a Qdrant collection.

    :param vector_store: Injected VectorStoreService instance.
    :param collection_name: Name of collection to index.
    :param batch_size: Batch size for processing.
    :param limit: Maximum number of documents to index.
    :param domain: Domain for the index.
    :return: Index creation statistics.
    :raises HTTPException: If index creation fails.
    """
    try:
        stats = await vector_store.create_colbert_index(
            collection_name=collection_name,
            batch_size=batch_size,
            limit=limit,
            domain=domain,
        )
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Error creating ColBERT index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/colbert/indexes")
async def list_colbert_indexes(
    vector_store: VectorStoreService = Depends(get_vector_store)
) -> dict[str, Any]:
    """
    List all available ColBERT indexes.

    :param vector_store: Injected VectorStoreService instance.
    :return: List of available ColBERT indexes.
    :raises HTTPException: If listing fails.
    """
    try:
        indexes = await vector_store.list_colbert_indexes()
        return {"success": True, "indexes": indexes}
    except Exception as e:
        logger.error(f"Error listing ColBERT indexes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse, deprecated=True)
async def query_rag(
    query_request: QueryRequest,
    vector_store: VectorStoreService = Depends(get_vector_store)
) -> QueryResponse:
    """
    [DEPRECATED] Full RAG query: retrieval + LLM generation.

    NOTE: This endpoint is deprecated. Use the API Gateway endpoint instead:
    POST http://api-gateway:8000/api/v1/query

    The API Gateway provides:
    - Query Analysis with LaTeX detection
    - Document Ranking
    - Response Formatting
    - Better error handling and monitoring

    This endpoint will be removed in a future version.

    :param query_request: Query request with query text and options.
    :param vector_store: Injected VectorStoreService instance.
    :return: Query response with generated answer and sources.
    """
    start_time = time.time()
    retrieval_time = 0.0
    generation_time = 0.0

    logger.warning("DEPRECATED: /query endpoint called directly. Use API Gateway /api/v1/query instead.")

    try:
        # Resolve collection (with fallback to default)
        collection = await resolve_collection(vector_store, query_request.collection)
        logger.info(f"RAG query using collection '{collection}': {query_request.query[:100]}")

        # Step 1: Retrieve documents
        retrieval_start = time.time()

        results = await vector_store.hybrid_search_with_hyde(
            query=query_request.query,
            top_k=query_request.top_k,
            filter_dict=None,
            score_threshold=query_request.score_threshold,
            use_qdrant=True,
            use_chroma=False,
            use_hyde_colbert=query_request.use_hyde_colbert,
            hyde_colbert_options=query_request.hyde_colbert_options,
            collection_name=collection,
        )
        retrieval_time = (time.time() - retrieval_start) * 1000

        if not results:
            return QueryResponse(
                success=False,
                response="No relevant documents found for your query. Try rephrasing your question.",
                sources=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                retrieval_time_ms=retrieval_time,
                error="No documents found",
            )

        # Format sources
        sources = [
            SourceDocument(
                title=metadata.get("title", f"Source {idx + 1}"),
                content=text[:500] + ("..." if len(text) > 500 else ""),
                score=score,
                metadata=metadata,
            )
            for idx, (text, score, metadata) in enumerate(results)
        ]

        # Step 2: Call LLM generation service
        generation_start = time.time()
        context_documents = [
            {
                "content": text,
                "title": metadata.get("title"),
                "source": metadata.get("source", "unknown"),
                "score": score,
                "metadata": metadata,
            }
            for text, score, metadata in results
        ]

        async with httpx.AsyncClient(timeout=180.0) as client:
            llm_response = await client.post(
                f"{LLM_SERVICE_URL}/generate",
                json={
                    "query": query_request.query,
                    "context_documents": context_documents,
                    "parameters": {
                        "model": query_request.model,
                        "temperature": query_request.temperature,
                        "max_tokens": query_request.max_tokens,
                    },
                    "session_id": query_request.session_id,
                },
            )

            if llm_response.status_code != 200:
                logger.error(f"LLM service error: {llm_response.text}")
                raise HTTPException(
                    status_code=llm_response.status_code,
                    detail=f"LLM generation failed: {llm_response.text}",
                )

            llm_result = llm_response.json()
            generation_time = (time.time() - generation_start) * 1000

        total_time = (time.time() - start_time) * 1000

        return QueryResponse(
            success=True,
            response=llm_result.get("generated_text", ""),
            sources=sources,
            processing_time_ms=total_time,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            tokens_used=llm_result.get("tokens_used"),
            model_used=llm_result.get("model_used"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG query error: {e}", exc_info=True)
        total_time = (time.time() - start_time) * 1000
        return QueryResponse(
            success=False,
            response="",
            sources=[],
            processing_time_ms=total_time,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            error=str(e),
        )


@router.post("/query/stream", deprecated=True)
async def query_rag_stream(
    query_request: QueryRequest,
    vector_store: VectorStoreService = Depends(get_vector_store)
) -> StreamingResponse:
    """
    [DEPRECATED] Full RAG query with streaming response.

    NOTE: This endpoint is deprecated. Use the API Gateway endpoint instead:
    POST http://api-gateway:8000/api/v1/query/stream

    The API Gateway provides:
    - Query Analysis with LaTeX detection
    - Document Ranking
    - Better error handling and monitoring

    This endpoint will be removed in a future version.

    :param query_request: Query request with query text and options.
    :param vector_store: Injected VectorStoreService instance.
    :return: Streaming response with SSE events.
    """
    import json

    logger.warning("DEPRECATED: /query/stream endpoint called directly. Use API Gateway /api/v1/query/stream instead.")

    async def event_generator():
        start_time = time.time()

        try:
            # Step 1: Resolve collection (with fallback to default)
            collection = await resolve_collection(vector_store, query_request.collection)
            logger.info(f"Using collection: {collection} for query: {query_request.query[:50]}...")

            # Step 2: Retrieve documents
            results = await vector_store.hybrid_search_with_hyde(
                query=query_request.query,
                top_k=query_request.top_k,
                filter_dict=None,
                score_threshold=query_request.score_threshold,
                use_qdrant=True,
                use_chroma=False,
                use_hyde_colbert=query_request.use_hyde_colbert,
                hyde_colbert_options=query_request.hyde_colbert_options,
                collection_name=collection,
            )

            if not results:
                yield f"data: {json.dumps({'type': 'error', 'error': 'No documents found'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Send sources first
            sources = [
                {
                    "title": metadata.get("title", f"Source {idx + 1}"),
                    "content": text[:500] + ("..." if len(text) > 500 else ""),
                    "score": score,
                    "metadata": metadata,
                }
                for idx, (text, score, metadata) in enumerate(results)
            ]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Step 2: Stream LLM response
            context_documents = [
                {
                    "content": text,
                    "title": metadata.get("title"),
                    "source": metadata.get("source", "unknown"),
                    "score": score,
                    "metadata": metadata,
                }
                for text, score, metadata in results
            ]

            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{LLM_SERVICE_URL}/generate/stream",
                    json={
                        "query": query_request.query,
                        "context_documents": context_documents,
                        "parameters": {
                            "model": query_request.model,
                            "temperature": query_request.temperature,
                            "max_tokens": query_request.max_tokens,
                        },
                        "session_id": query_request.session_id,
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                yield "data: [DONE]\n\n"
                                return
                            try:
                                chunk = json.loads(data)
                                if "content" in chunk:
                                    yield f"data: {json.dumps({'type': 'token', 'content': chunk['content']})}\n\n"
                            except json.JSONDecodeError:
                                pass

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Document Upload Endpoints
# =============================================================================


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> dict[str, Any]:
    """
    Upload a document (PDF, TXT, DOCX) and create embeddings.

    The document will be parsed, chunked, embedded, and stored in Qdrant.

    :param file: The document file to upload.
    :param collection_name: Target collection name (auto-generated if not provided).
    :param session_id: Optional session ID for tracking.
    :param vector_store: Injected VectorStoreService instance.
    :return: Upload result with document ID and collection name.
    :raises HTTPException: If upload fails.
    """
    try:
        logger.info(f"Uploading document: {file.filename}")

        # Read file content
        file_content = await file.read()

        # Create upload service
        upload_service = UploadService(vector_store)

        # Process upload
        result: UploadResult = await upload_service.upload_document(
            file_content=file_content,
            filename=file.filename or "unknown",
            content_type=file.content_type or "application/octet-stream",
            collection_name=collection_name,
            session_id=session_id,
        )

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)

        return {
            "success": True,
            "id": result.document_id,
            "collection": result.collection_name,
            "filename": result.filename,
            "chunks_created": result.chunks_created,
            "processing_time_ms": result.processing_time_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    collection_name: str,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> dict[str, Any]:
    """
    Delete a document from a collection.

    :param document_id: The document ID to delete.
    :param collection_name: The collection containing the document.
    :param vector_store: Injected VectorStoreService instance.
    :return: Success status.
    :raises HTTPException: If document not found or deletion fails.
    """
    try:
        upload_service = UploadService(vector_store)

        success = await upload_service.delete_document(document_id, collection_name)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found or could not be deleted")

        return {"success": True, "message": f"Document {document_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> dict[str, Any]:
    """
    Delete an entire collection.

    :param collection_name: The collection to delete.
    :param vector_store: Injected VectorStoreService instance.
    :return: Success status.
    :raises HTTPException: If collection not found or deletion fails.
    """
    try:
        upload_service = UploadService(vector_store)

        success = await upload_service.delete_collection(collection_name)

        if not success:
            raise HTTPException(status_code=404, detail="Collection not found or could not be deleted")

        return {"success": True, "message": f"Collection {collection_name} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
