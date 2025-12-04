"""
Intelligent Query Routing System
Routes queries to optimal retrieval pipelines based on query characteristics
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
import re
import asyncio

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class PipelineType(str, Enum):
    """Available retrieval pipelines"""
    STANDARD_RAG = "standard_rag"  # Vector search → Rerank → Generate
    MATH_FORMULA = "math_formula"  # LaTeX parse → Formula DB → Symbolic solver
    CODE_SEARCH = "code_search"  # Code-specific embeddings → Syntax parse
    MULTI_HOP = "multi_hop"  # Decomposition → Multiple retrievals → Synthesis
    FACTUAL_QA = "factual_qa"  # Dense retrieval → Cross-encoder → Direct answer
    COMPARATIVE = "comparative"  # Multi-query → Parallel retrieve → Compare
    STRUCTURED_DATA = "structured_data"  # Text-to-SQL → Database query
    HYBRID_SEARCH = "hybrid_search"  # Vector + BM25 + Metadata filters


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    pipeline: PipelineType
    confidence: float
    reasoning: str
    parameters: Dict[str, Any]
    alternative_pipelines: List[tuple[PipelineType, float]]  # (pipeline, score)


@dataclass
class RouteConfig:
    """Configuration for a route"""
    pipeline: PipelineType
    description: str
    keywords: List[str]
    patterns: List[str]  # Regex patterns
    embedding: Optional[List[float]] = None
    score_threshold: float = 0.7


class QueryAnalyzer:
    """Analyze query characteristics for routing"""

    def __init__(self):
        self.latex_patterns = [
            r'\$.*?\$',
            r'\\\[.*?\\\]',
            r'\\frac',
            r'\\int',
            r'\\sum',
            r'\\alpha|\\beta|\\gamma'
        ]

        self.code_patterns = [
            r'\bdef\b',
            r'\bclass\b',
            r'\bimport\b',
            r'\bfunction\b',
            r'=>',
            r'{\s*\}',
            r'```'
        ]

        self.comparative_keywords = [
            'compare', 'difference', 'versus', 'vs', 'better',
            'contrast', 'comparison'
        ]

        self.multi_hop_keywords = [
            'and then', 'after that', 'followed by', 'stages', 'steps',
            'process', 'how does', 'explain how'
        ]

    def detect_latex(self, query: str) -> bool:
        """Detect if query contains LaTeX formulas"""
        for pattern in self.latex_patterns:
            if re.search(pattern, query):
                return True
        return False

    def detect_code(self, query: str) -> bool:
        """Detect if query is about code"""
        for pattern in self.code_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        # Check for code-related keywords
        code_keywords = ['code', 'implement', 'function', 'algorithm', 'syntax']
        query_lower = query.lower()
        return any(kw in query_lower for kw in code_keywords)

    def detect_comparative(self, query: str) -> bool:
        """Detect comparative queries"""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.comparative_keywords)

    def detect_multi_hop(self, query: str) -> bool:
        """Detect multi-hop reasoning queries"""
        query_lower = query.lower()

        # Check for multi-hop keywords
        if any(kw in query_lower for kw in self.multi_hop_keywords):
            return True

        # Check for multiple questions
        question_marks = query.count('?')
        if question_marks > 1:
            return True

        # Check for conjunctions indicating complexity
        conjunctions = [' and ', ' or ', ' but ', ' also ']
        conjunction_count = sum(1 for c in conjunctions if c in query_lower)
        return conjunction_count >= 2

    def detect_structured_query(self, query: str) -> bool:
        """Detect queries that might benefit from structured data"""
        structured_keywords = [
            'filter', 'where', 'between', 'before', 'after',
            'from year', 'published in', 'author', 'type'
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in structured_keywords)

    def extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract features from query for routing"""
        return {
            "has_latex": self.detect_latex(query),
            "has_code": self.detect_code(query),
            "is_comparative": self.detect_comparative(query),
            "is_multi_hop": self.detect_multi_hop(query),
            "needs_structured_data": self.detect_structured_query(query),
            "query_length": len(query),
            "num_questions": query.count('?'),
            "has_numbers": bool(re.search(r'\d+', query)),
            "complexity_score": self._estimate_complexity(query)
        }

    def _estimate_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1)"""
        score = 0.0

        # Length factor
        if len(query) > 100:
            score += 0.3
        elif len(query) > 50:
            score += 0.1

        # Multiple questions
        if query.count('?') > 1:
            score += 0.3

        # Complex words
        complex_words = ['moreover', 'furthermore', 'consequently', 'specifically']
        if any(w in query.lower() for w in complex_words):
            score += 0.2

        # Nested clauses
        nested_indicators = [' which ', ' that ', ' where ', ' when ']
        nested_count = sum(1 for ind in nested_indicators if ind in query.lower())
        score += min(nested_count * 0.1, 0.2)

        return min(score, 1.0)


class IntelligentRouter:
    """
    Route queries to optimal pipelines
    Supports: Rule-based, Semantic, and LLM-based routing
    """

    def __init__(self, embeddings_model=None):
        """
        Initialize router

        Args:
            embeddings_model: Model for semantic routing
        """
        self.query_analyzer = QueryAnalyzer()
        self.embeddings_model = embeddings_model

        # Define route configurations
        self.routes = self._initialize_routes()

        logger.info("IntelligentRouter initialized")

    def _initialize_routes(self) -> Dict[PipelineType, RouteConfig]:
        """Initialize routing configurations"""
        return {
            PipelineType.MATH_FORMULA: RouteConfig(
                pipeline=PipelineType.MATH_FORMULA,
                description="Mathematical formulas and equations",
                keywords=["solve", "equation", "formula", "calculate", "integral", "derivative"],
                patterns=[r'\$.*?\$', r'\\frac', r'\\int'],
                score_threshold=0.8
            ),

            PipelineType.CODE_SEARCH: RouteConfig(
                pipeline=PipelineType.CODE_SEARCH,
                description="Code and programming queries",
                keywords=["code", "implement", "function", "algorithm", "syntax"],
                patterns=[r'\bdef\b', r'\bclass\b', r'```'],
                score_threshold=0.75
            ),

            PipelineType.COMPARATIVE: RouteConfig(
                pipeline=PipelineType.COMPARATIVE,
                description="Comparison and contrast queries",
                keywords=["compare", "difference", "versus", "vs", "better"],
                patterns=[r'\bvs\b', r'\bversus\b'],
                score_threshold=0.7
            ),

            PipelineType.MULTI_HOP: RouteConfig(
                pipeline=PipelineType.MULTI_HOP,
                description="Complex multi-step reasoning",
                keywords=["steps", "process", "stages", "how does", "explain"],
                patterns=[],
                score_threshold=0.65
            ),

            PipelineType.STRUCTURED_DATA: RouteConfig(
                pipeline=PipelineType.STRUCTURED_DATA,
                description="Structured data queries with filters",
                keywords=["filter", "where", "from year", "published", "author"],
                patterns=[],
                score_threshold=0.7
            ),

            PipelineType.FACTUAL_QA: RouteConfig(
                pipeline=PipelineType.FACTUAL_QA,
                description="Simple factual questions",
                keywords=["what is", "who is", "when did", "define"],
                patterns=[],
                score_threshold=0.6
            ),

            PipelineType.STANDARD_RAG: RouteConfig(
                pipeline=PipelineType.STANDARD_RAG,
                description="Standard RAG pipeline (default)",
                keywords=[],
                patterns=[],
                score_threshold=0.0
            ),
        }

    async def route_query(
        self,
        query: str,
        method: str = "hybrid"  # "rule_based", "semantic", "llm", "hybrid"
    ) -> RoutingDecision:
        """
        Route query to optimal pipeline

        Args:
            query: User query
            method: Routing method to use

        Returns:
            Routing decision
        """
        logger.info(f"Routing query using method: {method}")

        if method == "rule_based":
            return await self._route_rule_based(query)
        elif method == "semantic":
            return await self._route_semantic(query)
        elif method == "llm":
            return await self._route_llm_based(query)
        elif method == "hybrid":
            return await self._route_hybrid(query)
        else:
            raise ValueError(f"Unknown routing method: {method}")

    async def _route_rule_based(self, query: str) -> RoutingDecision:
        """
        Rule-based routing using patterns and keywords
        Fast but less flexible
        """
        features = self.query_analyzer.extract_query_features(query)
        scores = {}

        # Score each pipeline
        for pipeline_type, config in self.routes.items():
            score = 0.0

            # Pattern matching
            for pattern in config.patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 0.5

            # Keyword matching
            query_lower = query.lower()
            for keyword in config.keywords:
                if keyword in query_lower:
                    score += 0.3

            scores[pipeline_type] = min(score, 1.0)

        # Feature-based routing
        if features["has_latex"]:
            scores[PipelineType.MATH_FORMULA] = max(
                scores.get(PipelineType.MATH_FORMULA, 0), 0.9
            )

        if features["has_code"]:
            scores[PipelineType.CODE_SEARCH] = max(
                scores.get(PipelineType.CODE_SEARCH, 0), 0.85
            )

        if features["is_comparative"]:
            scores[PipelineType.COMPARATIVE] = max(
                scores.get(PipelineType.COMPARATIVE, 0), 0.8
            )

        if features["is_multi_hop"]:
            scores[PipelineType.MULTI_HOP] = max(
                scores.get(PipelineType.MULTI_HOP, 0), 0.75
            )

        # Select best pipeline
        best_pipeline = max(scores.items(), key=lambda x: x[1])
        pipeline_type, confidence = best_pipeline

        # If no confident match, use standard RAG
        if confidence < 0.5:
            pipeline_type = PipelineType.STANDARD_RAG
            confidence = 0.6

        # Get alternatives
        alternatives = sorted(
            [(p, s) for p, s in scores.items() if p != pipeline_type],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        return RoutingDecision(
            pipeline=pipeline_type,
            confidence=confidence,
            reasoning=f"Rule-based routing: {features}",
            parameters=self._get_pipeline_parameters(pipeline_type, features),
            alternative_pipelines=alternatives
        )

    async def _route_semantic(self, query: str) -> RoutingDecision:
        """
        Semantic routing using embeddings
        Compare query embedding to pipeline description embeddings
        """
        if not self.embeddings_model:
            logger.warning("No embeddings model, falling back to rule-based")
            return await self._route_rule_based(query)

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        # Compare to each pipeline's description embedding
        scores = {}
        for pipeline_type, config in self.routes.items():
            if config.embedding is None:
                # Generate embedding for description
                config.embedding = await self._get_embedding(config.description)

            # Cosine similarity
            similarity = cosine_similarity(
                [query_embedding],
                [config.embedding]
            )[0][0]

            scores[pipeline_type] = float(similarity)

        # Select best
        best_pipeline = max(scores.items(), key=lambda x: x[1])
        pipeline_type, confidence = best_pipeline

        alternatives = sorted(
            [(p, s) for p, s in scores.items() if p != pipeline_type],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        features = self.query_analyzer.extract_query_features(query)

        return RoutingDecision(
            pipeline=pipeline_type,
            confidence=confidence,
            reasoning=f"Semantic routing: similarity={confidence:.3f}",
            parameters=self._get_pipeline_parameters(pipeline_type, features),
            alternative_pipelines=alternatives
        )

    async def _route_llm_based(self, query: str) -> RoutingDecision:
        """
        LLM-based routing
        Let LLM decide which pipeline to use
        Most accurate but slowest
        """
        # Create routing prompt
        pipeline_descriptions = "\n".join([
            f"{i+1}. {p.value}: {config.description}"
            for i, (p, config) in enumerate(self.routes.items())
        ])

        prompt = f"""Given this query, select the most appropriate retrieval pipeline.

Query: {query}

Available pipelines:
{pipeline_descriptions}

Return JSON with:
{{
    "pipeline": "pipeline_name",
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}

Response:"""

        # Simulate LLM response (would use actual LLM)
        # For now, fallback to rule-based
        return await self._route_rule_based(query)

    async def _route_hybrid(self, query: str) -> RoutingDecision:
        """
        Hybrid routing: Combine rule-based and semantic
        Best balance of speed and accuracy
        """
        # Get both decisions
        rule_decision = await self._route_rule_based(query)
        semantic_decision = await self._route_semantic(query)

        # Weight and combine
        # Rule-based: 0.6, Semantic: 0.4
        if rule_decision.pipeline == semantic_decision.pipeline:
            # Agreement - high confidence
            confidence = (rule_decision.confidence * 0.6 +
                         semantic_decision.confidence * 0.4)

            return RoutingDecision(
                pipeline=rule_decision.pipeline,
                confidence=min(confidence * 1.2, 1.0),  # Boost for agreement
                reasoning=f"Hybrid routing (agreement): {rule_decision.reasoning}",
                parameters=rule_decision.parameters,
                alternative_pipelines=rule_decision.alternative_pipelines
            )
        else:
            # Disagreement - go with higher confidence
            if rule_decision.confidence > semantic_decision.confidence:
                return rule_decision
            else:
                return semantic_decision

    def _get_pipeline_parameters(
        self,
        pipeline: PipelineType,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get optimal parameters for selected pipeline"""

        params = {
            "top_k_retrieval": 20,
            "top_k_final": 10,
            "enable_reranking": True
        }

        if pipeline == PipelineType.MATH_FORMULA:
            params.update({
                "parse_latex": True,
                "symbolic_solver": True,
                "top_k_retrieval": 10
            })

        elif pipeline == PipelineType.COMPARATIVE:
            params.update({
                "multi_query": True,
                "num_variations": 3,
                "enable_fusion": True
            })

        elif pipeline == PipelineType.MULTI_HOP:
            params.update({
                "decompose_query": True,
                "max_hops": 3,
                "enable_synthesis": True
            })

        elif pipeline == PipelineType.CODE_SEARCH:
            params.update({
                "code_embeddings": True,
                "syntax_parse": True
            })

        return params

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        if self.embeddings_model:
            return await self.embeddings_model.embed_query(text)
        else:
            # Return dummy embedding
            return np.random.rand(384).tolist()

    async def route_with_query_construction(
        self,
        constructed_query: Any,  # ConstructedQuery from query_constructor.py
        method: str = "hybrid"
    ) -> RoutingDecision:
        """
        Enhanced routing that uses query construction information

        Args:
            constructed_query: ConstructedQuery from QueryConstructor
            method: Routing method to use

        Returns:
            Routing decision enhanced with query construction metadata
        """
        logger.info(f"Routing with query construction. Strategy: {constructed_query.retrieval_strategy}")

        # Start with basic routing
        routing_decision = await self.route_query(
            query=constructed_query.primary_query,
            method=method
        )

        # Override pipeline if query constructor suggests formula-aware
        if constructed_query.retrieval_strategy == "formula_aware":
            routing_decision = RoutingDecision(
                pipeline=PipelineType.MATH_FORMULA,
                confidence=0.95,  # High confidence from query construction
                reasoning="Query contains LaTeX formulas - using math formula pipeline",
                parameters=self._get_pipeline_parameters(
                    PipelineType.MATH_FORMULA,
                    {"has_latex": True}
                ),
                alternative_pipelines=routing_decision.alternative_pipelines
            )

        # Enhance parameters with query construction info
        routing_decision.parameters.update({
            "expansion_queries": constructed_query.expansion_queries,
            "structured_filters": constructed_query.structured_filters,
            "reranking_hints": constructed_query.reranking_hints,
            "query_metadata": constructed_query.metadata
        })

        logger.info(f"Enhanced routing complete. Pipeline: {routing_decision.pipeline}")

        return routing_decision
