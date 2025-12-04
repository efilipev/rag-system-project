"""
ColBERT Index Manager

Manages ColBERT document indexes for efficient retrieval:
- Index documents from Qdrant collections
- Persist indexes to disk
- Load and manage indexes at runtime
- Support multiple collections
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

import torch
from qdrant_client import QdrantClient

from src.core.config import settings
from src.services.colbert_encoder import ColBERTEncoder, get_colbert_encoder

logger = logging.getLogger(__name__)


class ColBERTIndex:
    """
    Manages a ColBERT index for a document collection.

    Features:
    - Index documents from various sources
    - Persist to disk for reuse
    - Support incremental updates
    - Memory-efficient loading
    """

    def __init__(
        self,
        collection_name: str,
        index_path: Optional[str] = None,
        encoder: Optional[ColBERTEncoder] = None,
    ):
        """
        Initialize ColBERT index.

        Args:
            collection_name: Name of the document collection
            index_path: Path to store/load index files
            encoder: ColBERT encoder instance
        """
        self.collection_name = collection_name
        self.index_path = Path(index_path or settings.COLBERT_INDEX_PATH) / collection_name
        self.encoder = encoder

        # Index data
        self.doc_embeddings: List[torch.Tensor] = []
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []

        # Index stats
        self.stats = {
            "total_documents": 0,
            "total_tokens": 0,
            "index_size_mb": 0.0,
            "is_loaded": False,
        }

    def _ensure_encoder(self) -> ColBERTEncoder:
        """Ensure encoder is available."""
        if self.encoder is None:
            self.encoder = get_colbert_encoder()
        return self.encoder

    async def index_from_qdrant(
        self,
        qdrant_client: QdrantClient,
        collection_name: Optional[str] = None,
        batch_size: int = 32,
        limit: Optional[int] = None,
        domain: str = "general",
    ) -> Dict[str, int]:
        """
        Index documents from a Qdrant collection.

        Args:
            qdrant_client: Qdrant client instance
            collection_name: Qdrant collection name (defaults to self.collection_name)
            batch_size: Batch size for encoding
            limit: Maximum documents to index (None for all)
            domain: Domain for encoder configuration

        Returns:
            Statistics about indexed documents
        """
        collection_name = collection_name or self.collection_name
        encoder = self._ensure_encoder()

        # Apply domain configuration
        encoder.apply_domain_config(domain)

        logger.info(f"Indexing documents from Qdrant collection: {collection_name}")

        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        total_points = collection_info.points_count
        logger.info(f"Collection has {total_points} points")

        if limit:
            total_points = min(total_points, limit)
            logger.info(f"Limiting to {total_points} documents")

        # Scroll through all documents
        documents = []
        doc_ids = []
        doc_metadata = []

        offset = None
        while True:
            result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            points, next_offset = result

            if not points:
                break

            for point in points:
                # Extract text content
                text = point.payload.get("text", "")
                if not text:
                    text = point.payload.get("content", "")
                if not text:
                    text = point.payload.get("page_content", "")

                if text:
                    documents.append(text)
                    doc_ids.append(str(point.id))
                    doc_metadata.append(point.payload)

                if limit and len(documents) >= limit:
                    break

            if next_offset is None or (limit and len(documents) >= limit):
                break

            offset = next_offset
            logger.info(f"Fetched {len(documents)} documents...")

        if not documents:
            logger.warning(f"No documents found in collection {collection_name}")
            return {"indexed": 0, "total_tokens": 0}

        logger.info(f"Encoding {len(documents)} documents...")

        # Encode documents
        self.doc_embeddings = encoder.encode_documents(
            documents,
            batch_size=batch_size,
            adaptive_pooling=settings.COLBERT_USE_ADAPTIVE_POOLING,
        )
        self.doc_ids = doc_ids
        self.doc_texts = documents
        self.doc_metadata = doc_metadata

        # Update stats
        total_tokens = sum(emb.size(0) for emb in self.doc_embeddings)
        self.stats = {
            "total_documents": len(documents),
            "total_tokens": total_tokens,
            "index_size_mb": self._calculate_index_size(),
            "is_loaded": True,
        }

        logger.info(
            f"Indexed {len(documents)} documents with {total_tokens} tokens "
            f"({self.stats['index_size_mb']:.2f} MB)"
        )

        return {
            "indexed": len(documents),
            "total_tokens": total_tokens,
        }

    def index_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32,
        domain: str = "general",
    ) -> Dict[str, int]:
        """
        Index a list of documents.

        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs
            metadata: Optional list of metadata dicts
            batch_size: Batch size for encoding
            domain: Domain for encoder configuration

        Returns:
            Statistics about indexed documents
        """
        encoder = self._ensure_encoder()
        encoder.apply_domain_config(domain)

        logger.info(f"Indexing {len(documents)} documents...")

        # Generate IDs if not provided
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]

        if metadata is None:
            metadata = [{} for _ in documents]

        # Encode documents
        self.doc_embeddings = encoder.encode_documents(
            documents,
            batch_size=batch_size,
            adaptive_pooling=settings.COLBERT_USE_ADAPTIVE_POOLING,
        )
        self.doc_ids = doc_ids
        self.doc_texts = documents
        self.doc_metadata = metadata

        # Update stats
        total_tokens = sum(emb.size(0) for emb in self.doc_embeddings)
        self.stats = {
            "total_documents": len(documents),
            "total_tokens": total_tokens,
            "index_size_mb": self._calculate_index_size(),
            "is_loaded": True,
        }

        return {
            "indexed": len(documents),
            "total_tokens": total_tokens,
        }

    def _calculate_index_size(self) -> float:
        """Calculate index size in MB."""
        if not self.doc_embeddings:
            return 0.0

        total_bytes = 0
        for emb in self.doc_embeddings:
            total_bytes += emb.numel() * 4  # 4 bytes per float32

        return total_bytes / (1024 * 1024)

    def save(self, path: Optional[str] = None) -> str:
        """
        Save index to disk.

        Args:
            path: Optional path override

        Returns:
            Path where index was saved
        """
        save_path = Path(path) if path else self.index_path
        save_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        embeddings_path = save_path / "embeddings.pt"
        torch.save(self.doc_embeddings, embeddings_path)

        # Save metadata
        metadata_path = save_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'doc_ids': self.doc_ids,
                'doc_texts': self.doc_texts,
                'doc_metadata': self.doc_metadata,
                'stats': self.stats,
            }, f)

        logger.info(f"Saved ColBERT index to {save_path}")
        return str(save_path)

    def load(self, path: Optional[str] = None) -> bool:
        """
        Load index from disk.

        Args:
            path: Optional path override

        Returns:
            True if loaded successfully
        """
        load_path = Path(path) if path else self.index_path

        embeddings_path = load_path / "embeddings.pt"
        metadata_path = load_path / "metadata.pkl"

        if not embeddings_path.exists() or not metadata_path.exists():
            logger.warning(f"Index not found at {load_path}")
            return False

        # Load embeddings
        self.doc_embeddings = torch.load(embeddings_path, map_location='cpu')

        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.doc_ids = data['doc_ids']
            self.doc_texts = data['doc_texts']
            self.doc_metadata = data['doc_metadata']
            self.stats = data['stats']

        self.stats['is_loaded'] = True

        logger.info(
            f"Loaded ColBERT index from {load_path}: "
            f"{self.stats['total_documents']} documents"
        )
        return True

    def exists(self, path: Optional[str] = None) -> bool:
        """Check if index exists on disk."""
        check_path = Path(path) if path else self.index_path
        return (check_path / "embeddings.pt").exists()

    def clear(self) -> None:
        """Clear the index from memory."""
        self.doc_embeddings = []
        self.doc_ids = []
        self.doc_texts = []
        self.doc_metadata = []
        self.stats = {
            "total_documents": 0,
            "total_tokens": 0,
            "index_size_mb": 0.0,
            "is_loaded": False,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return dict(self.stats)

    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self.stats.get('is_loaded', False) and len(self.doc_embeddings) > 0


class ColBERTIndexManager:
    """
    Manages multiple ColBERT indexes for different collections.

    Provides centralized access to indexes and handles lazy loading.
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize index manager.

        Args:
            base_path: Base path for storing indexes
        """
        self.base_path = Path(base_path or settings.COLBERT_INDEX_PATH)
        self.indexes: Dict[str, ColBERTIndex] = {}
        self.encoder: Optional[ColBERTEncoder] = None

        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_encoder(self) -> ColBERTEncoder:
        """Get or create shared encoder."""
        if self.encoder is None:
            self.encoder = get_colbert_encoder()
        return self.encoder

    def get_index(
        self,
        collection_name: str,
        auto_load: bool = True,
    ) -> ColBERTIndex:
        """
        Get or create an index for a collection.

        Args:
            collection_name: Name of the collection
            auto_load: Whether to auto-load from disk if available

        Returns:
            ColBERT index for the collection
        """
        if collection_name not in self.indexes:
            index = ColBERTIndex(
                collection_name=collection_name,
                index_path=str(self.base_path),
                encoder=self._get_encoder(),
            )

            if auto_load and index.exists():
                index.load()

            self.indexes[collection_name] = index

        return self.indexes[collection_name]

    async def create_index(
        self,
        collection_name: str,
        qdrant_client: QdrantClient,
        batch_size: int = 32,
        limit: Optional[int] = None,
        domain: str = "general",
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a new index from a Qdrant collection.

        Args:
            collection_name: Name of the collection
            qdrant_client: Qdrant client instance
            batch_size: Batch size for encoding
            limit: Maximum documents to index
            domain: Domain for encoder configuration
            save: Whether to save to disk

        Returns:
            Index creation statistics
        """
        index = self.get_index(collection_name, auto_load=False)

        stats = await index.index_from_qdrant(
            qdrant_client=qdrant_client,
            batch_size=batch_size,
            limit=limit,
            domain=domain,
        )

        if save:
            index.save()

        return {
            "collection": collection_name,
            "stats": stats,
            "path": str(index.index_path),
        }

    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all available indexes."""
        indexes = []

        # Check disk
        for path in self.base_path.iterdir():
            if path.is_dir() and (path / "embeddings.pt").exists():
                collection_name = path.name
                index = self.get_index(collection_name, auto_load=True)
                indexes.append({
                    "collection": collection_name,
                    "loaded": index.is_loaded(),
                    "stats": index.get_stats(),
                })

        return indexes

    def delete_index(self, collection_name: str) -> bool:
        """
        Delete an index.

        Args:
            collection_name: Name of the collection

        Returns:
            True if deleted successfully
        """
        index_path = self.base_path / collection_name

        if collection_name in self.indexes:
            self.indexes[collection_name].clear()
            del self.indexes[collection_name]

        if index_path.exists():
            import shutil
            shutil.rmtree(index_path)
            logger.info(f"Deleted ColBERT index: {collection_name}")
            return True

        return False


# Singleton instance
_index_manager: Optional[ColBERTIndexManager] = None


def get_colbert_index_manager() -> ColBERTIndexManager:
    """Get or create the ColBERT index manager singleton."""
    global _index_manager
    if _index_manager is None:
        _index_manager = ColBERTIndexManager()
    return _index_manager
