"""
ColBERT Encoder with MaxSim Late Interaction

This module implements ColBERT encoding with:
- Token-level embeddings for queries and documents
- Adaptive token pooling for storage efficiency (50-75% reduction)
- MaxSim late interaction for retrieval scoring

Based on the ColBERT paper and HyDE-ColBERT implementation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.core.config import settings

logger = logging.getLogger(__name__)

# Check ColBERT availability
try:
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.infra import ColBERTConfig
    COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False
    logger.warning(
        "ColBERT library not available. Install with: pip install colbert-ai"
    )


class AdaptiveTokenPooling:
    """
    Adaptive token pooling for ColBERT embeddings.

    Reduces storage by 50-75% by pooling adjacent token embeddings based on
    document complexity.
    """

    def __init__(
        self,
        default_pool_factor: int = 2,
        adaptive: bool = True,
        complexity_threshold_high: float = 0.5,
        complexity_threshold_medium: float = 0.3,
        complexity_threshold_low: float = 0.15,
    ):
        """
        Initialize adaptive pooling.

        Args:
            default_pool_factor: Default pooling factor (1-4)
            adaptive: Whether to use adaptive pooling based on complexity
            complexity_threshold_high: Threshold for pool_factor=1
            complexity_threshold_medium: Threshold for pool_factor=2
            complexity_threshold_low: Threshold for pool_factor=3
        """
        self.default_pool_factor = default_pool_factor
        self.adaptive = adaptive
        self.thresholds = {
            'high': complexity_threshold_high,
            'medium': complexity_threshold_medium,
            'low': complexity_threshold_low,
        }

    def estimate_complexity(self, embeddings: torch.Tensor) -> float:
        """
        Estimate document complexity from token embeddings.

        Args:
            embeddings: Token embeddings [num_tokens, embedding_dim]

        Returns:
            Complexity score (higher = more complex)
        """
        variance = torch.var(embeddings, dim=0).mean().item()
        norms = torch.norm(embeddings, dim=-1)
        norm_std = norms.std().item()
        return (variance + norm_std) / 2

    def determine_pool_factor(self, complexity_score: float) -> int:
        """
        Determine pooling factor based on complexity score.

        Args:
            complexity_score: Document complexity score

        Returns:
            Pool factor (1-4)
        """
        if complexity_score > self.thresholds['high']:
            return 1
        elif complexity_score > self.thresholds['medium']:
            return 2
        elif complexity_score > self.thresholds['low']:
            return 3
        else:
            return 4

    def pool_embeddings(
        self,
        embeddings: torch.Tensor,
        pool_factor: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool token embeddings using average pooling.

        Args:
            embeddings: Token embeddings [num_tokens, embedding_dim]
            pool_factor: Pooling factor (if None, uses adaptive estimation)
            mask: Token mask [num_tokens]

        Returns:
            pooled_embeddings: Pooled embeddings
            pooled_mask: Pooled mask
        """
        if pool_factor is None:
            if self.adaptive:
                complexity = self.estimate_complexity(embeddings)
                pool_factor = self.determine_pool_factor(complexity)
            else:
                pool_factor = self.default_pool_factor

        if pool_factor == 1:
            return embeddings, mask if mask is not None else torch.ones(embeddings.size(0))

        num_tokens = embeddings.size(0)
        embed_dim = embeddings.size(1)
        num_pooled = (num_tokens + pool_factor - 1) // pool_factor

        # Pad embeddings if needed
        pad_size = num_pooled * pool_factor - num_tokens
        if pad_size > 0:
            embeddings = torch.cat([
                embeddings,
                torch.zeros(pad_size, embed_dim, device=embeddings.device)
            ], dim=0)

            if mask is not None:
                mask = torch.cat([
                    mask,
                    torch.zeros(pad_size, device=mask.device)
                ], dim=0)

        embeddings_reshaped = embeddings.view(num_pooled, pool_factor, embed_dim)

        if mask is not None:
            mask_reshaped = mask.view(num_pooled, pool_factor)
            mask_expanded = mask_reshaped.unsqueeze(-1)
            masked_embeddings = embeddings_reshaped * mask_expanded
            pooled_embeddings = masked_embeddings.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-10)
            pooled_mask = (mask_reshaped.sum(dim=1) > 0).float()
        else:
            pooled_embeddings = embeddings_reshaped.mean(dim=1)
            pooled_mask = torch.ones(num_pooled, device=embeddings.device)

        return pooled_embeddings, pooled_mask


class ColBERTEncoder:
    """
    ColBERT encoder with MaxSim late interaction scoring.

    Features:
    - Token-level embeddings for fine-grained matching
    - Adaptive token pooling for storage efficiency
    - Batched MaxSim computation for performance
    """

    # Domain-specific configurations
    DOMAIN_CONFIGS = {
        'biomedical': {'doc_maxlen': 350, 'nbits': 2},
        'scientific': {'doc_maxlen': 400, 'nbits': 2},
        'financial': {'doc_maxlen': 280, 'nbits': 2},
        'argumentative': {'doc_maxlen': 250, 'nbits': 2},
        'general': {'doc_maxlen': 300, 'nbits': 2},
    }

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        use_adaptive_pooling: bool = True,
        default_pool_factor: int = 2,
        doc_maxlen: int = 300,
        query_maxlen: int = 32,
        nbits: int = 2,
        device: Optional[str] = None,
    ):
        """
        Initialize ColBERT encoder.

        Args:
            checkpoint: ColBERT model checkpoint name or path
            use_adaptive_pooling: Whether to use adaptive token pooling
            default_pool_factor: Default pooling factor (1-4)
            doc_maxlen: Maximum document length in tokens
            query_maxlen: Maximum query length in tokens
            nbits: Number of bits for residual compression
            device: Device to use ('cuda' or 'cpu')
        """
        if not COLBERT_AVAILABLE:
            raise ImportError(
                "ColBERT library required. Install with: pip install colbert-ai"
            )

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.checkpoint_name = checkpoint or settings.COLBERT_CHECKPOINT

        # Initialize ColBERT config
        self.config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            kmeans_niters=4,
        )

        # Load ColBERT checkpoint
        logger.info(f"Loading ColBERT checkpoint: {self.checkpoint_name}")
        self.checkpoint = Checkpoint(self.checkpoint_name, colbert_config=self.config)

        if torch.cuda.is_available():
            self.checkpoint = self.checkpoint.cuda()

        # Initialize pooling
        if use_adaptive_pooling:
            self.pooling = AdaptiveTokenPooling(default_pool_factor=default_pool_factor)
            logger.info("Using adaptive token pooling")
        else:
            self.pooling = AdaptiveTokenPooling(
                default_pool_factor=default_pool_factor,
                adaptive=False,
            )
            logger.info(f"Using fixed pooling with factor={default_pool_factor}")

        logger.info(f"ColBERT encoder initialized on {self.device}")

    def apply_domain_config(self, domain: str) -> None:
        """
        Apply domain-specific ColBERT configuration.

        Args:
            domain: Domain name
        """
        domain = domain.lower()
        if domain not in self.DOMAIN_CONFIGS:
            domain = 'general'

        config = self.DOMAIN_CONFIGS[domain]
        self.doc_maxlen = config['doc_maxlen']
        self.config.doc_maxlen = config['doc_maxlen']
        self.config.nbits = config['nbits']

        logger.info(f"Applied domain config for '{domain}': doc_maxlen={config['doc_maxlen']}")

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Encode queries into token-level embeddings.

        Args:
            queries: Single query string or list of queries
            batch_size: Batch size for encoding

        Returns:
            Query embeddings [num_queries, max_tokens, embedding_dim]
        """
        if isinstance(queries, str):
            queries = [queries]

        all_embeddings = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]

            with torch.no_grad():
                result = self.checkpoint.queryFromText(batch, bsize=len(batch))
                if isinstance(result, tuple):
                    Q = result[0]
                else:
                    Q = result
                all_embeddings.append(Q.cpu())

        query_embeddings = torch.cat(all_embeddings, dim=0)
        logger.debug(f"Encoded {len(queries)} queries -> shape {query_embeddings.shape}")

        return query_embeddings

    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        adaptive_pooling: bool = True,
        pool_factor: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Encode documents into token-level embeddings with optional pooling.

        Args:
            documents: List of document strings
            batch_size: Batch size for encoding
            adaptive_pooling: Whether to apply token pooling
            pool_factor: Optional fixed pool factor

        Returns:
            List of document embeddings, each [num_tokens, embedding_dim]
        """
        all_doc_embeddings = []

        for i in tqdm(range(0, len(documents), batch_size), desc="Encoding documents"):
            batch = documents[i:i + batch_size]

            with torch.no_grad():
                result = self.checkpoint.docFromText(batch, bsize=len(batch))

                if isinstance(result, tuple):
                    D = result[0]
                    masks = result[1] if len(result) > 1 else None
                else:
                    D = result
                    masks = None

                batch_size_actual = D.size(0)

                for j in range(batch_size_actual):
                    doc_emb = D[j]

                    # Remove padding tokens
                    if masks is not None:
                        doc_mask = masks[j]
                        doc_emb = doc_emb[doc_mask.bool()]
                        token_mask = doc_mask.bool()
                    else:
                        nonzero_mask = doc_emb.norm(dim=-1) > 1e-6
                        doc_emb = doc_emb[nonzero_mask]
                        token_mask = nonzero_mask

                    # Apply pooling
                    if adaptive_pooling and len(doc_emb) > 0:
                        doc_emb, _ = self.pooling.pool_embeddings(
                            doc_emb.cpu(),
                            pool_factor=pool_factor,
                            mask=token_mask.cpu() if masks is not None else None,
                        )
                    else:
                        doc_emb = doc_emb.cpu()

                    all_doc_embeddings.append(doc_emb)

        logger.info(f"Encoded {len(documents)} documents")
        return all_doc_embeddings

    def maxsim_score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> float:
        """
        Compute MaxSim late interaction score.

        For each query token, find max similarity with any document token,
        then average across all query tokens.

        Args:
            query_embeddings: [num_query_tokens, embedding_dim]
            doc_embeddings: [num_doc_tokens, embedding_dim]

        Returns:
            MaxSim score
        """
        scores = torch.mm(query_embeddings, doc_embeddings.t())
        max_scores = scores.max(dim=1).values
        return max_scores.mean().item()

    def batch_maxsim_score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings_list: List[torch.Tensor],
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        Compute MaxSim scores for a query against multiple documents.

        Args:
            query_embeddings: [num_query_tokens, embedding_dim]
            doc_embeddings_list: List of document embeddings
            batch_size: Batch size for scoring

        Returns:
            Scores array [num_docs]
        """
        scores = np.zeros(len(doc_embeddings_list))
        query_embeddings = query_embeddings.to(self.device)

        for i in range(0, len(doc_embeddings_list), batch_size):
            batch_docs = doc_embeddings_list[i:i + batch_size]
            batch_docs_device = [doc.to(self.device) for doc in batch_docs]
            batch_scores = self._batched_maxsim(query_embeddings, batch_docs_device)
            scores[i:i + len(batch_docs)] = batch_scores

        return scores

    def _batched_maxsim(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings_list: List[torch.Tensor],
    ) -> np.ndarray:
        """
        Efficiently compute MaxSim for a batch of documents with varying lengths.

        Args:
            query_embeddings: [num_query_tokens, embedding_dim]
            doc_embeddings_list: List of [num_doc_tokens_i, embedding_dim]

        Returns:
            Batch scores [batch_size]
        """
        max_doc_len = max(doc.size(0) for doc in doc_embeddings_list)
        embed_dim = query_embeddings.size(1)
        batch_size = len(doc_embeddings_list)

        padded_docs = torch.zeros(
            batch_size, max_doc_len, embed_dim,
            device=self.device, dtype=query_embeddings.dtype
        )
        doc_masks = torch.zeros(batch_size, max_doc_len, device=self.device, dtype=torch.bool)

        for idx, doc_emb in enumerate(doc_embeddings_list):
            doc_len = doc_emb.size(0)
            padded_docs[idx, :doc_len] = doc_emb.to(self.device)
            doc_masks[idx, :doc_len] = True

        # Batched computation: Q @ D^T for all docs
        scores = torch.einsum('qe,bde->qbd', query_embeddings, padded_docs)
        scores = scores.masked_fill(~doc_masks.unsqueeze(0), float('-inf'))
        max_scores = scores.max(dim=2).values
        total_scores = max_scores.mean(dim=0)

        return total_scores.cpu().numpy()

    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using min-max scaling.

        Args:
            scores: Raw scores array

        Returns:
            Normalized scores in [0, 1]
        """
        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score < 1e-8:
            return np.ones_like(scores) * 0.5

        return (scores - min_score) / (max_score - min_score)

    @staticmethod
    def calibrate_scores(scores: np.ndarray) -> np.ndarray:
        """
        Calibrate scores using z-score normalization.

        Args:
            scores: Raw MaxSim scores

        Returns:
            Calibrated scores with mean=0, std=1
        """
        mean = np.mean(scores)
        std = np.std(scores)

        if std < 1e-8:
            return scores - mean

        return (scores - mean) / std

    def get_storage_stats(self, doc_embeddings: List[torch.Tensor]) -> Dict[str, float]:
        """
        Get storage statistics for indexed documents.

        Args:
            doc_embeddings: List of document embeddings

        Returns:
            Dict with storage metrics
        """
        if not doc_embeddings:
            return {}

        total_tokens = sum(emb.size(0) for emb in doc_embeddings)
        total_docs = len(doc_embeddings)
        avg_tokens = total_tokens / total_docs
        embedding_dim = doc_embeddings[0].size(1)
        storage_bytes = total_tokens * embedding_dim * 4
        storage_mb = storage_bytes / (1024 ** 2)

        return {
            'total_documents': total_docs,
            'total_tokens': total_tokens,
            'avg_tokens_per_doc': avg_tokens,
            'embedding_dim': embedding_dim,
            'storage_mb': storage_mb,
        }


# Singleton instance
_colbert_encoder: Optional[ColBERTEncoder] = None


def get_colbert_encoder() -> ColBERTEncoder:
    """Get or create the ColBERT encoder singleton."""
    global _colbert_encoder
    if _colbert_encoder is None:
        _colbert_encoder = ColBERTEncoder()
    return _colbert_encoder
