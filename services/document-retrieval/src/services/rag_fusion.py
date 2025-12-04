"""
RAG-Fusion: Multi-Query Retrieval with Reciprocal Rank Fusion
Generates multiple query variations and fuses results
"""
import logging
from typing import List, Dict, Any, Optional
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryVariation:
    """A variation of the original query"""
    text: str
    variation_type: str  # "original", "paraphrase", "broader", "specific"
    confidence: float


@dataclass
class FusedResult:
    """Result from fusion"""
    document_id: str
    content: str
    fused_score: float
    source_scores: Dict[str, float]  # scores from each query variation
    rank_positions: Dict[str, int]  # rank in each result list
    metadata: Dict[str, Any]


class QueryGenerator:
    """
    Generate multiple variations of a query
    Implements: Paraphrasing, Decomposition, Step-back, HyDE
    """

    def __init__(self, llm_client=None):
        """
        Initialize query generator

        Args:
            llm_client: LLM client for generating variations
        """
        self.llm_client = llm_client

    async def generate_paraphrases(
        self,
        query: str,
        num_variations: int = 3
    ) -> List[QueryVariation]:
        """
        Generate paraphrased versions of the query

        Args:
            query: Original query
            num_variations: Number of paraphrases to generate

        Returns:
            List of query paraphrases
        """
        prompt = f"""Generate {num_variations} different paraphrases of this question that preserve the same meaning:

Original question: {query}

Generate {num_variations} paraphrases (one per line):
1."""

        # Simulate LLM response (would use actual LLM client)
        paraphrases = await self._generate_variations(prompt, num_variations)

        return [
            QueryVariation(
                text=p,
                variation_type="paraphrase",
                confidence=0.9
            )
            for p in paraphrases
        ]

    async def generate_decomposition(self, query: str) -> List[QueryVariation]:
        """
        Decompose complex query into sub-queries

        Example:
        "How does photosynthesis work and what are its stages?"
        →
        - "What is photosynthesis?"
        - "What are the stages of photosynthesis?"
        - "How does each stage of photosynthesis work?"

        Args:
            query: Complex query

        Returns:
            List of sub-queries
        """
        prompt = f"""Break down this complex question into simpler sub-questions:

Complex question: {query}

Sub-questions:
1."""

        sub_queries = await self._generate_variations(prompt, 3)

        return [
            QueryVariation(
                text=sq,
                variation_type="decomposition",
                confidence=0.85
            )
            for sq in sub_queries
        ]

    async def generate_step_back(self, query: str) -> QueryVariation:
        """
        Generate a broader, more general version of the query

        Example:
        "How to implement quicksort in Python?"
        →
        "What are sorting algorithms in computer science?"

        Args:
            query: Specific query

        Returns:
            Broader query
        """
        prompt = f"""Generate a broader, more general version of this specific question:

Specific question: {query}

Broader question:"""

        broader = await self._generate_single_variation(prompt)

        return QueryVariation(
            text=broader,
            variation_type="step_back",
            confidence=0.8
        )

    async def generate_hyde(self, query: str) -> QueryVariation:
        """
        HyDE: Hypothetical Document Embeddings
        Generate a hypothetical answer, then search for documents similar to it

        Args:
            query: Original query

        Returns:
            Hypothetical document
        """
        prompt = f"""Write a detailed, factual answer to this question:

Question: {query}

Answer (be specific and detailed):"""

        hypothetical_doc = await self._generate_single_variation(prompt)

        return QueryVariation(
            text=hypothetical_doc,
            variation_type="hyde",
            confidence=0.75
        )

    async def _generate_variations(
        self,
        prompt: str,
        num_variations: int
    ) -> List[str]:
        """Generate multiple variations using LLM"""
        # Placeholder - would use actual LLM client
        # For now, return simple variations
        import random

        base_variations = [
            "What is the main concept?",
            "Can you explain the process?",
            "What are the key details?",
            "How does this work?",
            "What are the important aspects?"
        ]

        return random.sample(base_variations, min(num_variations, len(base_variations)))

    async def _generate_single_variation(self, prompt: str) -> str:
        """Generate a single variation"""
        variations = await self._generate_variations(prompt, 1)
        return variations[0] if variations else ""


class RAGFusion:
    """
    RAG-Fusion: Combine results from multiple query variations
    Uses Reciprocal Rank Fusion (RRF)
    """

    def __init__(
        self,
        retriever,
        query_generator: Optional[QueryGenerator] = None,
        rrf_k: int = 60
    ):
        """
        Initialize RAG-Fusion

        Args:
            retriever: Document retriever (vector store client)
            query_generator: Query variation generator
            rrf_k: RRF constant (typically 60)
        """
        self.retriever = retriever
        self.query_generator = query_generator or QueryGenerator()
        self.rrf_k = rrf_k

    async def retrieve_with_fusion(
        self,
        query: str,
        num_variations: int = 3,
        top_k_per_query: int = 20,
        final_top_k: int = 10,
        enable_hyde: bool = False,
        enable_decomposition: bool = False
    ) -> List[FusedResult]:
        """
        Main RAG-Fusion retrieval pipeline

        Steps:
        1. Generate query variations
        2. Retrieve documents for each variation
        3. Fuse results with RRF
        4. Return top-k documents

        Args:
            query: Original query
            num_variations: Number of paraphrases to generate
            top_k_per_query: Documents to retrieve per query
            final_top_k: Final number of documents to return
            enable_hyde: Whether to use HyDE
            enable_decomposition: Whether to decompose query

        Returns:
            Fused and ranked documents
        """
        logger.info(f"RAG-Fusion retrieval for query: '{query[:100]}...'")

        # Step 1: Generate query variations
        all_queries = [
            QueryVariation(text=query, variation_type="original", confidence=1.0)
        ]

        # Add paraphrases
        paraphrases = await self.query_generator.generate_paraphrases(
            query,
            num_variations
        )
        all_queries.extend(paraphrases)

        # Add HyDE if enabled
        if enable_hyde:
            hyde_query = await self.query_generator.generate_hyde(query)
            all_queries.append(hyde_query)

        # Add decomposition if enabled
        if enable_decomposition:
            sub_queries = await self.query_generator.generate_decomposition(query)
            all_queries.extend(sub_queries)

        logger.info(f"Generated {len(all_queries)} query variations")

        # Step 2: Retrieve documents for each query variation
        all_results = {}
        for query_var in all_queries:
            try:
                results = await self.retriever.search(
                    query=query_var.text,
                    top_k=top_k_per_query
                )

                all_results[query_var.text] = {
                    "query_variation": query_var,
                    "results": results
                }

                logger.info(
                    f"Retrieved {len(results)} docs for variation: "
                    f"'{query_var.text[:50]}...'"
                )

            except Exception as e:
                logger.warning(f"Retrieval failed for variation: {e}")
                continue

        # Step 3: Fuse results using RRF
        fused_results = await self._fuse_results(all_results, final_top_k)

        logger.info(f"RAG-Fusion complete. Returning {len(fused_results)} documents")

        return fused_results

    async def _fuse_results(
        self,
        all_results: Dict[str, Dict],
        top_k: int
    ) -> List[FusedResult]:
        """
        Fuse results from multiple queries using RRF

        RRF Formula: score(d) = Σ 1/(k + rank(d))

        Args:
            all_results: Results from each query variation
            top_k: Number of top documents to return

        Returns:
            Fused and ranked documents
        """
        # Build document score map
        doc_scores = {}
        doc_ranks = {}
        doc_original_scores = {}
        doc_content_map = {}

        for query_text, result_data in all_results.items():
            query_var = result_data["query_variation"]
            results = result_data["results"]

            for rank, doc in enumerate(results):
                doc_id = doc.get("id")

                if not doc_id:
                    continue

                # Store document content
                if doc_id not in doc_content_map:
                    doc_content_map[doc_id] = doc

                # RRF score
                rrf_score = 1.0 / (self.rrf_k + rank + 1)

                # Weight by query variation confidence
                weighted_score = rrf_score * query_var.confidence

                # Accumulate scores
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_ranks[doc_id] = {}
                    doc_original_scores[doc_id] = {}

                doc_scores[doc_id] += weighted_score
                doc_ranks[doc_id][query_text] = rank + 1
                doc_original_scores[doc_id][query_text] = doc.get("score", 0.0)

        # Create fused results
        fused_results = []
        for doc_id, fused_score in doc_scores.items():
            doc = doc_content_map[doc_id]

            fused_results.append(FusedResult(
                document_id=doc_id,
                content=doc.get("content", ""),
                fused_score=fused_score,
                source_scores=doc_original_scores[doc_id],
                rank_positions=doc_ranks[doc_id],
                metadata={
                    "title": doc.get("title", ""),
                    "num_queries_found_in": len(doc_ranks[doc_id]),
                    "avg_rank": sum(doc_ranks[doc_id].values()) / len(doc_ranks[doc_id]),
                    "rrf_k": self.rrf_k
                }
            ))

        # Sort by fused score
        fused_results.sort(key=lambda x: x.fused_score, reverse=True)

        return fused_results[:top_k]

    async def retrieve_with_weighted_fusion(
        self,
        query: str,
        strategy_weights: Dict[str, float],
        top_k: int = 10
    ) -> List[FusedResult]:
        """
        Fusion with custom weights for different strategies

        Args:
            query: Original query
            strategy_weights: Weights for each strategy
                Example: {"original": 0.5, "paraphrase": 0.3, "hyde": 0.2}
            top_k: Number of documents to return

        Returns:
            Weighted fused results
        """
        # Generate variations based on strategies
        all_queries = []

        if "original" in strategy_weights:
            all_queries.append(
                QueryVariation(
                    text=query,
                    variation_type="original",
                    confidence=strategy_weights["original"]
                )
            )

        if "paraphrase" in strategy_weights:
            paraphrases = await self.query_generator.generate_paraphrases(query, 2)
            for p in paraphrases:
                p.confidence = strategy_weights["paraphrase"]
            all_queries.extend(paraphrases)

        if "hyde" in strategy_weights:
            hyde = await self.query_generator.generate_hyde(query)
            hyde.confidence = strategy_weights["hyde"]
            all_queries.append(hyde)

        if "step_back" in strategy_weights:
            step_back = await self.query_generator.generate_step_back(query)
            step_back.confidence = strategy_weights["step_back"]
            all_queries.append(step_back)

        # Retrieve and fuse
        all_results = {}
        for query_var in all_queries:
            results = await self.retriever.search(query_var.text, top_k=20)
            all_results[query_var.text] = {
                "query_variation": query_var,
                "results": results
            }

        return await self._fuse_results(all_results, top_k)
