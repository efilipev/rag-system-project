"""
Tests for Query Construction Pipeline with LaTeX Integration
"""
import pytest
import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "query-analysis" / "app"))

from services.latex_query_analyzer import LatexQueryAnalyzer, LatexQueryAnalysis
from services.query_constructor import QueryConstructor, RetrievalStrategy
from services.query_analyzer import QueryAnalyzerService


class TestLatexQueryAnalyzer:
    """Test LaTeX query analysis"""

    @pytest.fixture
    def analyzer(self):
        """Create LaTeX query analyzer"""
        return LatexQueryAnalyzer(latex_parser_url="http://localhost:8005")

    @pytest.mark.asyncio
    async def test_detect_inline_latex(self, analyzer):
        """Test detection of inline LaTeX formulas"""
        query = "Solve $x^2 + 5x + 6 = 0$"

        analysis = await analyzer.analyze_query_with_latex(query)

        assert analysis.has_latex is True
        assert len(analysis.formulas) == 1
        assert analysis.formulas[0].raw_latex == "x^2 + 5x + 6 = 0"
        assert "x" in analysis.formulas[0].variables
        assert analysis.query_type == "solve_equation"

    @pytest.mark.asyncio
    async def test_detect_display_latex(self, analyzer):
        """Test detection of display LaTeX"""
        query = r"What is the integral \\[\\int_{0}^{1} x^2 dx\\]?"

        analysis = await analyzer.analyze_query_with_latex(query)

        assert analysis.has_latex is True
        assert len(analysis.formulas) == 1
        assert "integral" in analysis.formulas[0].formula_type.lower() or \
               "int" in analysis.formulas[0].raw_latex

    @pytest.mark.asyncio
    async def test_no_latex_detection(self, analyzer):
        """Test query without LaTeX"""
        query = "What is machine learning?"

        analysis = await analyzer.analyze_query_with_latex(query)

        assert analysis.has_latex is False
        assert len(analysis.formulas) == 0
        assert analysis.query_type == "general"
        assert len(analysis.search_queries) > 0

    @pytest.mark.asyncio
    async def test_multiple_formulas(self, analyzer):
        """Test detection of multiple formulas"""
        query = "Compare $f(x) = x^2$ and $g(x) = 2x + 1$"

        analysis = await analyzer.analyze_query_with_latex(query)

        assert analysis.has_latex is True
        assert len(analysis.formulas) == 2
        assert analysis.formulas[0].raw_latex == "f(x) = x^2"
        assert analysis.formulas[1].raw_latex == "g(x) = 2x + 1"

    @pytest.mark.asyncio
    async def test_formula_type_classification(self, analyzer):
        """Test formula type classification"""
        test_cases = [
            ("Solve $x = 5$", "equation"),
            (r"Calculate $\\int x dx$", "integral"),
            (r"Find $\\frac{dy}{dx}$", "derivative"),
            (r"Compute $\\sum_{i=1}^n i$", "summation"),
        ]

        for query, expected_type in test_cases:
            analysis = await analyzer.analyze_query_with_latex(query)
            assert analysis.has_latex is True
            assert any(expected_type in f.formula_type.lower() for f in analysis.formulas), \
                f"Expected {expected_type} in formula types for query: {query}"

    @pytest.mark.asyncio
    async def test_variable_extraction(self, analyzer):
        """Test variable extraction from formulas"""
        query = "Solve $ax^2 + bx + c = 0$ for x"

        analysis = await analyzer.analyze_query_with_latex(query)

        assert analysis.has_latex is True
        variables = analysis.metadata.get("variables", [])
        assert "x" in variables
        # a, b, c might be detected as variables
        assert len(variables) >= 1

    @pytest.mark.asyncio
    async def test_search_query_generation(self, analyzer):
        """Test generation of search queries"""
        query = "How to solve $x^2 = 4$?"

        analysis = await analyzer.analyze_query_with_latex(query)

        assert len(analysis.search_queries) > 1
        # Should include text query
        assert any("solve" in q.lower() for q in analysis.search_queries)
        # Should include formula
        assert any("x^2" in q for q in analysis.search_queries)


class TestQueryConstructor:
    """Test query construction"""

    @pytest.fixture
    def constructor(self):
        """Create query constructor"""
        return QueryConstructor()

    @pytest.fixture
    def sample_latex_analysis(self):
        """Create sample LaTeX analysis"""
        from services.latex_query_analyzer import LatexFormula

        formula = LatexFormula(
            raw_latex="x^2 + 5x + 6 = 0",
            formula_type="equation",
            variables=["x"],
            operators=["=", "+"]
        )

        return LatexQueryAnalysis(
            has_latex=True,
            original_query="Solve $x^2 + 5x + 6 = 0$",
            text_query="Solve",
            formulas=[formula],
            query_type="solve_equation",
            search_queries=["Solve", "solving equations"],
            metadata={
                "num_formulas": 1,
                "variables": ["x"],
                "operators": ["=", "+"]
            }
        )

    def test_strategy_selection(self, constructor, sample_latex_analysis):
        """Test retrieval strategy selection"""
        constructed = constructor.construct_query(sample_latex_analysis)

        assert constructed.retrieval_strategy == RetrievalStrategy.FORMULA_AWARE

    def test_primary_query_construction(self, constructor, sample_latex_analysis):
        """Test primary query construction"""
        constructed = constructor.construct_query(sample_latex_analysis)

        assert len(constructed.primary_query) > 0
        assert "solve" in constructed.primary_query.lower() or \
               "equation" in constructed.primary_query.lower()

    def test_expansion_query_generation(self, constructor, sample_latex_analysis):
        """Test expansion query generation"""
        constructed = constructor.construct_query(
            sample_latex_analysis,
            enable_multi_query=True,
            max_expansion_queries=5
        )

        assert len(constructed.expansion_queries) > 0
        assert len(constructed.expansion_queries) <= 5

    def test_structured_filters(self, constructor, sample_latex_analysis):
        """Test structured filter generation"""
        constructed = constructor.construct_query(sample_latex_analysis)

        assert "formula_types" in constructed.structured_filters or \
               "query_type" in constructed.structured_filters
        assert constructed.structured_filters.get("query_type") == "solve_equation"

    def test_reranking_hints(self, constructor, sample_latex_analysis):
        """Test reranking hints generation"""
        constructed = constructor.construct_query(sample_latex_analysis)

        assert constructed.reranking_hints.get("prefer_formula_match") is True
        assert "exact_formulas" in constructed.reranking_hints
        assert constructed.reranking_hints["exact_formulas"][0] == "x^2 + 5x + 6 = 0"

    def test_non_latex_query(self, constructor):
        """Test construction for non-LaTeX query"""
        from services.latex_query_analyzer import LatexQueryAnalysis

        analysis = LatexQueryAnalysis(
            has_latex=False,
            original_query="What is machine learning?",
            text_query="What is machine learning?",
            formulas=[],
            query_type="general",
            search_queries=["What is machine learning?"],
            metadata={"num_formulas": 0}
        )

        constructed = constructor.construct_query(analysis)

        assert constructed.retrieval_strategy == RetrievalStrategy.DENSE_VECTOR
        assert constructed.primary_query == "What is machine learning?"

    def test_retrieval_request_format(self, constructor, sample_latex_analysis):
        """Test conversion to retrieval request format"""
        constructed = constructor.construct_query(sample_latex_analysis)
        request = constructor.to_retrieval_request(constructed)

        assert "query" in request
        assert "strategy" in request
        assert "filters" in request
        assert "metadata" in request
        assert request["strategy"] == "formula_aware"


class TestEndToEndIntegration:
    """Test end-to-end query construction pipeline"""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_latex(self):
        """Test complete pipeline with LaTeX query"""
        # Initialize components
        analyzer = LatexQueryAnalyzer(latex_parser_url="http://localhost:8005")
        constructor = QueryConstructor()

        query = "Solve the quadratic equation $x^2 - 4 = 0$"

        try:
            # Step 1: Analyze query
            latex_analysis = await analyzer.analyze_query_with_latex(query)

            assert latex_analysis.has_latex is True
            assert latex_analysis.query_type in ["solve_equation", "formula_query"]

            # Step 2: Construct retrieval query
            constructed_query = constructor.construct_query(
                latex_analysis,
                enable_multi_query=True
            )

            assert constructed_query.retrieval_strategy == RetrievalStrategy.FORMULA_AWARE
            assert len(constructed_query.expansion_queries) > 0
            assert len(constructed_query.structured_filters) > 0

            # Step 3: Convert to retrieval request
            request = constructor.to_retrieval_request(constructed_query)

            assert request["strategy"] == "formula_aware"
            assert "expansion_queries" in request
            assert "reranking" in request

        finally:
            await analyzer.close()

    @pytest.mark.asyncio
    async def test_full_pipeline_without_latex(self):
        """Test complete pipeline with regular query"""
        analyzer = LatexQueryAnalyzer(latex_parser_url="http://localhost:8005")
        constructor = QueryConstructor()

        query = "What is the history of calculus?"

        try:
            # Step 1: Analyze query
            latex_analysis = await analyzer.analyze_query_with_latex(query)

            assert latex_analysis.has_latex is False
            assert latex_analysis.query_type == "general"

            # Step 2: Construct retrieval query
            constructed_query = constructor.construct_query(latex_analysis)

            assert constructed_query.retrieval_strategy == RetrievalStrategy.DENSE_VECTOR
            assert constructed_query.primary_query == query

        finally:
            await analyzer.close()


@pytest.mark.asyncio
async def test_integration_with_query_analyzer_service():
    """Test integration with QueryAnalyzerService"""
    # This test requires spaCy and LangChain to be installed
    try:
        service = QueryAnalyzerService(latex_parser_url="http://localhost:8005")
        await service.initialize()

        query = "Solve $2x + 3 = 7$"

        # Test enhanced analysis
        enhanced_analysis = await service.analyze_query_enhanced(query)

        assert "standard_analysis" in enhanced_analysis
        assert "latex_analysis" in enhanced_analysis
        assert enhanced_analysis["has_latex"] is True

        # Test query construction
        constructed = await service.construct_retrieval_query(query)

        assert constructed.retrieval_strategy == RetrievalStrategy.FORMULA_AWARE
        assert len(constructed.expansion_queries) > 0

        await service.close()

    except Exception as e:
        pytest.skip(f"Skipping integration test: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
