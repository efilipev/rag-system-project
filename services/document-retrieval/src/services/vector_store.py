"""
Vector Store Service with Qdrant, Chroma, and HyDE-ColBERT integration
"""
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.core.config import settings
from src.core.logging import logger
from src.models.schemas import HyDEColBERTOptions


class VectorStoreService:
    """
    Service for managing vector stores (Qdrant, Chroma, and HyDE-ColBERT) with LangChain
    """

    def __init__(self):
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.chroma_client: Optional[chromadb.Client] = None
        self.qdrant_store: Optional[Qdrant] = None
        self.chroma_store: Optional[Chroma] = None

        # HyDE-ColBERT components (lazy loaded)
        self._hyde_colbert_retrieval = None
        self._hyde_colbert_initialized = False

    async def initialize(self) -> None:
        """
        Initialize vector stores and embeddings
        """
        try:
            logger.info("Initializing Vector Store Service")

            # Initialize embeddings
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # Initialize Qdrant
            if settings.USE_QDRANT:
                await self._initialize_qdrant()

            # Initialize Chroma
            if settings.USE_CHROMA:
                await self._initialize_chroma()

            logger.info("Vector Store Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Vector Store Service: {e}")
            raise

    async def _initialize_qdrant(self) -> None:
        """
        Initialize Qdrant client and collection
        """
        try:
            logger.info(f"Connecting to Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

            self.qdrant_client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=30,
            )

            # Create collection if it doesn't exist
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]

            if settings.QDRANT_COLLECTION not in collection_names:
                logger.info(f"Creating Qdrant collection: {settings.QDRANT_COLLECTION}")
                self.qdrant_client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=settings.VECTOR_DIMENSION,
                        distance=models.Distance.COSINE,
                    ),
                )

            # Initialize LangChain Qdrant vector store
            # Use "text" as content key to match Wikipedia ingestion format
            self.qdrant_store = Qdrant(
                client=self.qdrant_client,
                collection_name=settings.QDRANT_COLLECTION,
                embeddings=self.embeddings,
                content_payload_key="text",
            )

            logger.info("Qdrant initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise

    async def _initialize_chroma(self) -> None:
        """
        Initialize Chroma client and collection
        """
        try:
            logger.info(f"Connecting to Chroma: {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")

            # Initialize Chroma client
            self.chroma_client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                ),
            )

            # Initialize LangChain Chroma vector store
            self.chroma_store = Chroma(
                client=self.chroma_client,
                collection_name=settings.CHROMA_COLLECTION,
                embedding_function=self.embeddings,
            )

            logger.info("Chroma initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {e}")
            raise

    async def search_qdrant(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search documents in Qdrant

        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            score_threshold: Minimum similarity score
            collection_name: Collection to search in (defaults to settings.QDRANT_COLLECTION)

        Returns:
            List of tuples (document_text, score, metadata)
        """
        if not self.qdrant_client:
            raise RuntimeError("Qdrant not initialized")

        # Use provided collection or default
        target_collection = collection_name or settings.QDRANT_COLLECTION

        try:
            logger.info(f"Searching Qdrant collection '{target_collection}' for: {query[:100]}")

            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search using Qdrant client directly to support dynamic collection
            search_results = self.qdrant_client.search(
                collection_name=target_collection,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

            # Format results
            formatted_results = []
            for result in search_results:
                payload = result.payload or {}
                # Get text from payload - check common keys
                text = payload.get("text") or payload.get("page_content") or payload.get("content") or ""
                score = result.score

                # Create metadata dict (all payload except text field)
                metadata = {k: v for k, v in payload.items() if k not in ("text", "page_content", "content")}

                formatted_results.append((text, score, metadata))

            logger.info(f"Found {len(formatted_results)} documents in Qdrant collection '{target_collection}'")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching Qdrant collection '{target_collection}': {e}")
            raise

    async def search_chroma(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search documents in Chroma

        Returns:
            List of tuples (document_text, score, metadata)
        """
        if not self.chroma_store:
            raise RuntimeError("Chroma not initialized")

        try:
            logger.info(f"Searching Chroma for: {query[:100]}")

            # Perform similarity search with scores
            results = self.chroma_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter_dict,
            )

            # Format results
            formatted_results = []
            for doc, score in results:
                # Chroma returns distance, convert to similarity
                similarity_score = 1 - score
                if score_threshold is None or similarity_score >= score_threshold:
                    formatted_results.append((doc.page_content, similarity_score, doc.metadata))

            logger.info(f"Found {len(formatted_results)} documents in Chroma")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching Chroma: {e}")
            raise

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
        score_threshold: float = 0.7,
        use_qdrant: bool = True,
        use_chroma: bool = True,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Perform hybrid search across both vector stores

        Returns:
            Merged and ranked list of tuples (document_text, score, metadata)
        """
        results = []

        # Search Qdrant
        if use_qdrant and self.qdrant_store:
            try:
                qdrant_results = await self.search_qdrant(
                    query=query,
                    top_k=top_k,
                    filter_dict=filter_dict,
                    score_threshold=score_threshold,
                )
                results.extend(qdrant_results)
            except Exception as e:
                logger.warning(f"Qdrant search failed: {e}")

        # Search Chroma
        if use_chroma and self.chroma_store:
            try:
                chroma_results = await self.search_chroma(
                    query=query,
                    top_k=top_k,
                    filter_dict=filter_dict,
                    score_threshold=score_threshold,
                )
                results.extend(chroma_results)
            except Exception as e:
                logger.warning(f"Chroma search failed: {e}")

        # Remove duplicates and sort by score
        unique_results = {}
        for text, score, metadata in results:
            if text not in unique_results or unique_results[text][1] < score:
                unique_results[text] = (text, score, metadata)

        # Sort by score descending
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top_k results
        return sorted_results[:top_k]

    async def add_documents_qdrant(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Add documents to Qdrant

        Returns:
            List of document IDs
        """
        if not self.qdrant_store:
            raise RuntimeError("Qdrant not initialized")

        try:
            logger.info(f"Adding {len(texts)} documents to Qdrant")

            ids = self.qdrant_store.add_texts(
                texts=texts,
                metadatas=metadatas,
            )

            logger.info(f"Added {len(ids)} documents to Qdrant")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents to Qdrant: {e}")
            raise

    async def add_documents_chroma(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Add documents to Chroma

        Returns:
            List of document IDs
        """
        if not self.chroma_store:
            raise RuntimeError("Chroma not initialized")

        try:
            logger.info(f"Adding {len(texts)} documents to Chroma")

            ids = self.chroma_store.add_texts(
                texts=texts,
                metadatas=metadatas,
            )

            logger.info(f"Added {len(ids)} documents to Chroma")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}")
            raise

    async def _initialize_hyde_colbert(self) -> None:
        """
        Initialize HyDE-ColBERT retrieval service (lazy loaded)
        """
        if self._hyde_colbert_initialized:
            return

        try:
            logger.info("Initializing HyDE-ColBERT retrieval service...")
            from src.services.hyde_colbert_retrieval import get_hyde_colbert_retrieval
            self._hyde_colbert_retrieval = await get_hyde_colbert_retrieval()
            self._hyde_colbert_initialized = True
            logger.info("HyDE-ColBERT retrieval service initialized")
        except ImportError as e:
            logger.warning(f"HyDE-ColBERT not available: {e}")
            self._hyde_colbert_retrieval = None
        except Exception as e:
            logger.error(f"Failed to initialize HyDE-ColBERT: {e}")
            self._hyde_colbert_retrieval = None

    async def search_hyde_colbert(
        self,
        query: str,
        collection_name: str = "wikipedia",
        top_k: int = 10,
        options: Optional[HyDEColBERTOptions] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search documents using HyDE-ColBERT retrieval.

        Args:
            query: Search query
            collection_name: Name of the ColBERT indexed collection
            top_k: Number of documents to retrieve
            options: HyDE-ColBERT options

        Returns:
            List of tuples (document_text, score, metadata)
        """
        # Ensure HyDE-ColBERT is initialized
        await self._initialize_hyde_colbert()

        if self._hyde_colbert_retrieval is None:
            raise RuntimeError("HyDE-ColBERT not available")

        try:
            logger.info(f"Searching with HyDE-ColBERT: {query[:100]}")

            # Set default options if not provided
            if options is None:
                options = HyDEColBERTOptions()

            await self._ensure_colbert_index(collection_name, options.domain.value)

            results = await self._hyde_colbert_retrieval.retrieve(
                query=query,
                collection_name=collection_name,
                top_k=top_k,
                domain=options.domain.value,
                n_hypotheticals=options.n_hypotheticals,
                fusion_strategy=options.fusion_strategy.value,
                fusion_weight=options.fusion_weight,
                return_scores=True,
            )

            # Format results to match existing API
            formatted_results = []
            for result in results:
                formatted_results.append((
                    result.get("content", ""),
                    result.get("score", 0.0),
                    result.get("metadata", {}),
                ))

            logger.info(f"Found {len(formatted_results)} documents with HyDE-ColBERT")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching with HyDE-ColBERT: {e}")
            raise

    async def _ensure_colbert_index(
        self,
        collection_name: str,
        domain: str = "general",
    ) -> None:
        """
        Ensure ColBERT index exists for the collection, create if not.

        Args:
            collection_name: Name of the collection to index
            domain: Domain for encoder configuration
        """
        from src.services.colbert_index import get_colbert_index_manager

        index_manager = get_colbert_index_manager()
        index = index_manager.get_index(collection_name, auto_load=True)

        if index.is_loaded():
            logger.info(f"ColBERT index for '{collection_name}' already loaded")
            return

        # Index doesn't exist or isn't loaded, need to create it
        if self.qdrant_client is None:
            raise RuntimeError(
                f"Cannot create ColBERT index for '{collection_name}': Qdrant not initialized"
            )

        # Check if the collection exists in Qdrant
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            if collection_info.points_count == 0:
                raise RuntimeError(
                    f"Cannot create ColBERT index: Qdrant collection '{collection_name}' is empty"
                )
        except Exception as e:
            if "not found" in str(e).lower():
                raise RuntimeError(
                    f"Cannot create ColBERT index: Qdrant collection '{collection_name}' does not exist"
                )
            raise

        logger.info(
            f"ColBERT index for '{collection_name}' not found. "
            f"Creating index from Qdrant collection ({collection_info.points_count} documents)..."
        )

        # Create the index
        stats = await index_manager.create_index(
            collection_name=collection_name,
            qdrant_client=self.qdrant_client,
            batch_size=32,
            limit=None,  # Index all documents
            domain=domain,
            save=True,  # Persist to disk for future use
        )

        logger.info(
            f"ColBERT index created for '{collection_name}': "
            f"{stats['stats']['indexed']} documents indexed"
        )

    async def create_colbert_index(
        self,
        collection_name: str,
        batch_size: int = 32,
        limit: Optional[int] = None,
        domain: str = "general",
    ) -> Dict[str, Any]:
        """
        Create a ColBERT index from a Qdrant collection.

        Args:
            collection_name: Name of the Qdrant collection to index
            batch_size: Batch size for encoding
            limit: Maximum documents to index (None for all)
            domain: Domain for encoder configuration

        Returns:
            Index creation statistics
        """
        await self._initialize_hyde_colbert()

        if self._hyde_colbert_retrieval is None:
            raise RuntimeError("HyDE-ColBERT not available")

        if self.qdrant_client is None:
            raise RuntimeError("Qdrant not initialized")

        try:
            logger.info(f"Creating ColBERT index for collection: {collection_name}")

            from src.services.colbert_index import get_colbert_index_manager
            index_manager = get_colbert_index_manager()

            stats = await index_manager.create_index(
                collection_name=collection_name,
                qdrant_client=self.qdrant_client,
                batch_size=batch_size,
                limit=limit,
                domain=domain,
                save=True,
            )

            logger.info(f"ColBERT index created: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error creating ColBERT index: {e}")
            raise

    async def list_colbert_indexes(self) -> List[Dict[str, Any]]:
        """
        List all available ColBERT indexes.

        Returns:
            List of index information
        """
        try:
            from src.services.colbert_index import get_colbert_index_manager
            index_manager = get_colbert_index_manager()
            return index_manager.list_indexes()
        except Exception as e:
            logger.error(f"Error listing ColBERT indexes: {e}")
            return []

    async def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all available collections from Qdrant.

        Returns:
            List of collection information with name and document count
        """
        if self.qdrant_client is None:
            logger.warning("Qdrant not initialized, cannot list collections")
            return []

        try:
            collections = self.qdrant_client.get_collections().collections
            result = []

            for col in collections:
                try:
                    # Get collection info with point count
                    col_info = self.qdrant_client.get_collection(col.name)
                    result.append({
                        "name": col.name,
                        "documentCount": col_info.points_count or 0,
                        "description": f"Vector collection with {col_info.points_count or 0} documents",
                    })
                except Exception as e:
                    logger.warning(f"Error getting collection info for {col.name}: {e}")
                    result.append({
                        "name": col.name,
                        "documentCount": 0,
                        "description": "Unable to get collection info",
                    })

            logger.info(f"Found {len(result)} collections in Qdrant")
            return result

        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    async def hybrid_search_with_hyde(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
        score_threshold: float = 0.7,
        use_qdrant: bool = True,
        use_chroma: bool = True,
        use_hyde_colbert: bool = False,
        hyde_colbert_options: Optional[HyDEColBERTOptions] = None,
        collection_name: str = "wikipedia",
    ) -> List[Tuple[str, float, Dict]]:
        """
        Perform hybrid search across all vector stores including HyDE-ColBERT.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_dict: Metadata filters
            score_threshold: Minimum similarity score
            use_qdrant: Search in Qdrant
            use_chroma: Search in Chroma
            use_hyde_colbert: Use HyDE-ColBERT retrieval
            hyde_colbert_options: HyDE-ColBERT options
            collection_name: Collection name to search in

        Returns:
            Merged and ranked list of tuples (document_text, score, metadata)
        """
        results = []

        # Search Qdrant
        if use_qdrant and self.qdrant_client:
            try:
                qdrant_results = await self.search_qdrant(
                    query=query,
                    top_k=top_k,
                    filter_dict=filter_dict,
                    score_threshold=score_threshold,
                    collection_name=collection_name,
                )
                results.extend(qdrant_results)
            except Exception as e:
                logger.warning(f"Qdrant search failed: {e}")

        # Search Chroma
        if use_chroma and self.chroma_store:
            try:
                chroma_results = await self.search_chroma(
                    query=query,
                    top_k=top_k,
                    filter_dict=filter_dict,
                    score_threshold=score_threshold,
                )
                results.extend(chroma_results)
            except Exception as e:
                logger.warning(f"Chroma search failed: {e}")

        # Search HyDE-ColBERT
        if use_hyde_colbert:
            try:
                hyde_results = await self.search_hyde_colbert(
                    query=query,
                    collection_name=collection_name,
                    top_k=top_k,
                    options=hyde_colbert_options,
                )
                results.extend(hyde_results)
            except Exception as e:
                logger.warning(f"HyDE-ColBERT search failed: {e}")

        # Remove duplicates and sort by score
        unique_results = {}
        for text, score, metadata in results:
            if text not in unique_results or unique_results[text][1] < score:
                unique_results[text] = (text, score, metadata)

        # Sort by score descending
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top_k results
        return sorted_results[:top_k]

    async def close(self) -> None:
        """
        Close connections to vector stores
        """
        try:
            if self.qdrant_client:
                self.qdrant_client.close()
                logger.info("Closed Qdrant connection")

            # Chroma HTTP client doesn't need explicit closing

            logger.info("Vector store connections closed")

        except Exception as e:
            logger.error(f"Error closing vector stores: {e}")
