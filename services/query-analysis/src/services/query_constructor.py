"""
Query Constructor - Builds optimized search queries for different retrieval strategies
"""
import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict
from enum import Enum

if TYPE_CHECKING:
    from .latex_query_analyzer import LatexQueryAnalysis, LatexFormula

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    """Retrieval strategies for different query types"""
    DENSE_VECTOR = "dense_vector"  # Standard semantic search
    HYBRID = "hybrid"  # Dense + Sparse (BM25)
    FORMULA_AWARE = "formula_aware"  # Specialized for math formulas
    MULTI_QUERY = "multi_query"  # RAG Fusion style
    STRUCTURED = "structured"  # Metadata-based search


@dataclass
class ConstructedQuery:
    """A constructed query ready for retrieval"""
    original_query: str
    retrieval_strategy: RetrievalStrategy
    primary_query: str  # Main semantic search query
    expansion_queries: List[str]  # Additional queries for multi-query retrieval
    structured_filters: Dict[str, Any]  # Metadata filters
    reranking_hints: Dict[str, Any]  # Hints for reranking
    metadata: Dict[str, Any]


class QueryConstructor:
    """
    Constructs optimized queries for different retrieval strategies

    Features:
    - Formula-aware query construction
    - Multi-query generation (RAG Fusion style)
    - Structured filter generation from formulas
    - Reranking hints for formula similarity
    """

    def __init__(self):
        """Initialize query constructor"""
        self.strategy_map = {
            "general": RetrievalStrategy.DENSE_VECTOR,
            "formula_query": RetrievalStrategy.FORMULA_AWARE,
            "solve_equation": RetrievalStrategy.FORMULA_AWARE,
            "integrate": RetrievalStrategy.FORMULA_AWARE,
            "differentiate": RetrievalStrategy.FORMULA_AWARE,
            "simplify": RetrievalStrategy.FORMULA_AWARE,
            "prove": RetrievalStrategy.HYBRID,
            "evaluate": RetrievalStrategy.FORMULA_AWARE,
        }

    def construct_query(
        self,
        latex_analysis: "LatexQueryAnalysis",
        enable_multi_query: bool = True,
        max_expansion_queries: int = 5
    ) -> ConstructedQuery:
        """
        Construct optimized query from LaTeX analysis

        Args:
            latex_analysis: Result from LatexQueryAnalyzer
            enable_multi_query: Whether to generate multiple queries
            max_expansion_queries: Maximum number of expansion queries

        Returns:
            Constructed query ready for retrieval
        """
        logger.info(f"Constructing query for type: {latex_analysis.query_type}")

        # Determine retrieval strategy
        strategy = self._select_strategy(latex_analysis)

        # Build primary query
        primary_query = self._build_primary_query(latex_analysis)

        # Generate expansion queries
        expansion_queries = []
        if enable_multi_query:
            expansion_queries = self._generate_expansion_queries(
                latex_analysis,
                max_queries=max_expansion_queries
            )

        # Generate structured filters
        structured_filters = self._generate_structured_filters(latex_analysis)

        # Generate reranking hints
        reranking_hints = self._generate_reranking_hints(latex_analysis)

        result = ConstructedQuery(
            original_query=latex_analysis.original_query,
            retrieval_strategy=strategy,
            primary_query=primary_query,
            expansion_queries=expansion_queries,
            structured_filters=structured_filters,
            reranking_hints=reranking_hints,
            metadata={
                "has_latex": latex_analysis.has_latex,
                "query_type": latex_analysis.query_type,
                "num_formulas": len(latex_analysis.formulas),
                "num_expansion_queries": len(expansion_queries),
            }
        )

        logger.info(f"Query constructed. Strategy: {strategy}, Expansion queries: {len(expansion_queries)}")

        return result

    def _select_strategy(self, analysis: "LatexQueryAnalysis") -> RetrievalStrategy:
        """Select appropriate retrieval strategy"""
        if analysis.has_latex and len(analysis.formulas) > 0:
            # Math queries use formula-aware strategy
            return self.strategy_map.get(
                analysis.query_type,
                RetrievalStrategy.FORMULA_AWARE
            )
        else:
            # Regular queries use dense vector or hybrid
            if any(keyword in analysis.text_query.lower() for keyword in ["compare", "difference", "vs"]):
                return RetrievalStrategy.HYBRID
            return RetrievalStrategy.DENSE_VECTOR

    def _build_primary_query(self, analysis: "LatexQueryAnalysis") -> str:
        """Build the primary search query"""
        if not analysis.has_latex:
            return analysis.text_query

        # For formula queries, combine text with formula context
        primary_parts = []

        # Add text query
        if analysis.text_query:
            primary_parts.append(analysis.text_query)

        # Add formula type context
        formula_types = set(f.formula_type for f in analysis.formulas)
        if formula_types and "unknown" not in formula_types and "general" not in formula_types:
            types_str = ", ".join(formula_types)
            primary_parts.append(f"({types_str})")

        # Add key variables
        all_vars = analysis.metadata.get("variables", [])
        if all_vars:
            vars_str = ", ".join(all_vars[:3])  # Top 3 variables
            primary_parts.append(f"variables: {vars_str}")

        primary = " ".join(primary_parts)
        return primary.strip()

    def _generate_expansion_queries(
        self,
        analysis: "LatexQueryAnalysis",
        max_queries: int = 5
    ) -> List[str]:
        """
        Generate expansion queries for multi-query retrieval

        Similar to RAG Fusion, generates diverse queries to improve recall
        """
        expansion_queries = []

        if not analysis.has_latex:
            # For regular queries, generate semantic variations
            base = analysis.text_query
            expansion_queries = [
                f"What is {base}?",
                f"Explain {base}",
                f"Information about {base}",
                f"{base} definition and examples",
            ]
        else:
            # For formula queries, generate diverse formula-related queries
            text = analysis.text_query
            formulas = analysis.formulas

            # Query 1: Focus on the mathematical concept
            if formulas:
                formula_types = [f.formula_type for f in formulas if f.formula_type not in ["unknown", "general"]]
                if formula_types:
                    expansion_queries.append(f"{formula_types[0]} {text}")

            # Query 2: Focus on solving/computing
            if analysis.query_type in ["solve_equation", "evaluate"]:
                expansion_queries.append(f"how to solve {text}")
                expansion_queries.append(f"step by step solution for {text}")

            # Query 3: Focus on theory
            expansion_queries.append(f"theory and explanation of {text}")

            # Query 4: Focus on examples
            expansion_queries.append(f"examples of {text}")

            # Query 5: Include formula text representations
            for formula in formulas[:2]:  # First 2 formulas
                if formula.text_representation:
                    expansion_queries.append(f"{text} {formula.text_representation}")

            # Query 6: Include variables
            all_vars = analysis.metadata.get("variables", [])
            if all_vars:
                vars_str = " ".join(all_vars[:3])
                expansion_queries.append(f"{text} involving {vars_str}")

        # Remove duplicates and limit
        seen = set()
        unique_queries = []
        for query in expansion_queries:
            q_normalized = query.lower().strip()
            if q_normalized not in seen and q_normalized != analysis.original_query.lower():
                seen.add(q_normalized)
                unique_queries.append(query)

        return unique_queries[:max_queries]

    def _generate_structured_filters(self, analysis: "LatexQueryAnalysis") -> Dict[str, Any]:
        """
        Generate structured filters for metadata-based search

        These filters can be used to narrow down search results
        """
        filters = {}

        if not analysis.has_latex:
            return filters

        # Filter by formula types
        formula_types = [f.formula_type for f in analysis.formulas if f.formula_type not in ["unknown", "general"]]
        if formula_types:
            filters["formula_types"] = formula_types

        # Filter by variables
        all_vars = analysis.metadata.get("variables", [])
        if all_vars:
            filters["variables"] = all_vars

        # Filter by operators
        all_operators = analysis.metadata.get("operators", [])
        if all_operators:
            filters["operators"] = all_operators

        # Filter by query type
        if analysis.query_type != "general":
            filters["query_type"] = analysis.query_type

        # Filter by complexity (number of formulas, variables, operators)
        complexity_score = (
            len(analysis.formulas) * 2 +
            len(all_vars) +
            len(all_operators)
        )
        if complexity_score > 0:
            if complexity_score <= 3:
                filters["complexity"] = "simple"
            elif complexity_score <= 7:
                filters["complexity"] = "moderate"
            else:
                filters["complexity"] = "complex"

        return filters

    def _generate_reranking_hints(self, analysis: "LatexQueryAnalysis") -> Dict[str, Any]:
        """
        Generate hints for reranking

        These hints help the reranker prioritize relevant results
        """
        hints = {
            "prefer_formula_match": analysis.has_latex,
            "query_type": analysis.query_type,
        }

        if not analysis.has_latex:
            return hints

        # Prioritize exact formula matches
        hints["exact_formulas"] = [f.raw_latex for f in analysis.formulas]

        # Prioritize simplified formula matches
        simplified_formulas = [f.simplified for f in analysis.formulas if f.simplified]
        if simplified_formulas:
            hints["simplified_formulas"] = simplified_formulas

        # Prioritize documents with matching variables
        all_vars = analysis.metadata.get("variables", [])
        if all_vars:
            hints["required_variables"] = all_vars

        # Prioritize documents with matching formula types
        formula_types = [f.formula_type for f in analysis.formulas if f.formula_type not in ["unknown", "general"]]
        if formula_types:
            hints["preferred_formula_types"] = formula_types

        # Weight factor for formula matching (0-1)
        # Higher for queries that are primarily about formulas
        formula_ratio = len(analysis.formulas) / max(len(analysis.text_query.split()), 1)
        hints["formula_weight"] = min(0.8, 0.3 + formula_ratio * 0.5)

        return hints

    def to_retrieval_request(self, constructed_query: ConstructedQuery) -> Dict[str, Any]:
        """
        Convert constructed query to retrieval service request format

        Args:
            constructed_query: Constructed query

        Returns:
            Dictionary ready for retrieval API
        """
        request = {
            "query": constructed_query.primary_query,
            "strategy": constructed_query.retrieval_strategy.value,
            "filters": constructed_query.structured_filters,
            "metadata": constructed_query.metadata
        }

        # Add expansion queries if multi-query strategy
        if constructed_query.expansion_queries:
            request["expansion_queries"] = constructed_query.expansion_queries

        # Add reranking configuration
        if constructed_query.reranking_hints:
            request["reranking"] = {
                "enabled": True,
                "hints": constructed_query.reranking_hints
            }

        return request

    def to_dict(self, constructed_query: ConstructedQuery) -> Dict[str, Any]:
        """Convert constructed query to dictionary"""
        result = asdict(constructed_query)
        result["retrieval_strategy"] = constructed_query.retrieval_strategy.value
        return result


class FormulaSearchQueryGenerator:
    """
    Specialized generator for formula-specific search queries

    Generates queries optimized for finding similar formulas
    """

    @staticmethod
    def generate_similar_formula_queries(formula: "LatexFormula") -> List[str]:
        """Generate queries to find similar formulas"""
        queries = []

        # Query by formula type
        if formula.formula_type not in ["unknown", "general"]:
            queries.append(f"{formula.formula_type}")

        # Query by variables and operators
        if formula.variables and formula.operators:
            vars_str = " ".join(formula.variables)
            ops_str = " ".join(formula.operators[:3])
            queries.append(f"{vars_str} with {ops_str}")

        # Query by text representation
        if formula.text_representation:
            queries.append(formula.text_representation)

        # Query by raw LaTeX (for exact matches)
        queries.append(formula.raw_latex)

        return queries

    @staticmethod
    def generate_formula_explanation_queries(formula: "LatexFormula") -> List[str]:
        """Generate queries to find explanations of formulas"""
        queries = []

        # Explanation queries
        if formula.formula_type not in ["unknown", "general"]:
            queries.append(f"what is {formula.formula_type}")
            queries.append(f"explain {formula.formula_type}")

        # How-to queries
        if formula.variables:
            vars_str = " ".join(formula.variables[:2])
            queries.append(f"how to work with {vars_str}")

        return queries
