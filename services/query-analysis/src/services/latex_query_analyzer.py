"""
LaTeX Query Analyzer - Detects and analyzes queries containing LaTeX formulas
"""
import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import httpx

logger = logging.getLogger(__name__)


@dataclass
class LatexFormula:
    """Detected LaTeX formula"""
    raw_latex: str
    formula_type: str  # "equation", "integral", "derivative", "matrix", etc.
    variables: List[str]
    operators: List[str]
    simplified: Optional[str] = None
    mathml: Optional[str] = None
    text_representation: Optional[str] = None


@dataclass
class LatexQueryAnalysis:
    """Result of analyzing a query with LaTeX"""
    has_latex: bool
    original_query: str
    text_query: str  # Query with LaTeX removed
    formulas: List[LatexFormula]
    query_type: str  # "solve_equation", "integrate", "differentiate", "formula_query", "general"
    search_queries: List[str]
    metadata: Dict[str, Any]


class LatexQueryAnalyzer:
    """
    Analyze queries containing LaTeX formulas

    Features:
    - Detect LaTeX patterns ($...$, \\[...\\], \\begin{equation}...\\end{equation})
    - Extract and parse formulas
    - Classify formula types (equation, integral, derivative, etc.)
    - Generate formula-aware search queries
    - Route to appropriate retrieval strategy
    """

    LATEX_PATTERNS = {
        "inline_math": r'\$([^\$]+)\$',
        "display_math": r'\\\[(.+?)\\\]',
        "equation_env": r'\\begin\{equation\}(.+?)\\end\{equation\}',
        "align_env": r'\\begin\{align\}(.+?)\\end\{align\}',
    }

    FORMULA_TYPES = {
        "equation": [r'=', r'\\eq'],
        "integral": [r'\\int', r'\\iint', r'\\iiint'],
        "derivative": [r'\\frac\{d', r'\\partial', r"'"],
        "summation": [r'\\sum'],
        "limit": [r'\\lim'],
        "matrix": [r'\\begin\{matrix\}', r'\\begin\{pmatrix\}', r'\\begin\{bmatrix\}'],
        "vector": [r'\\vec', r'\\mathbf'],
        "trigonometric": [r'\\sin', r'\\cos', r'\\tan', r'\\arcsin', r'\\arccos', r'\\arctan'],
        "logarithmic": [r'\\log', r'\\ln', r'\\lg'],
    }

    QUERY_KEYWORDS = {
        "solve": ["solve", "find solution", "calculate", "compute"],
        "simplify": ["simplify", "reduce", "expand"],
        "integrate": ["integrate", "integration", "antiderivative"],
        "differentiate": ["differentiate", "derivative", "derive"],
        "evaluate": ["evaluate", "what is", "calculate"],
        "prove": ["prove", "show that", "demonstrate"],
    }

    def __init__(self, latex_parser_url: str = "http://localhost:8005"):
        """
        Initialize analyzer

        Args:
            latex_parser_url: URL of LaTeX parser service
        """
        self.latex_parser_url = latex_parser_url
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

    async def analyze_query_with_latex(
        self,
        query: str
    ) -> LatexQueryAnalysis:
        """
        Analyze query containing LaTeX formulas

        Args:
            query: User query potentially containing LaTeX

        Returns:
            Analysis result with formula information
        """
        logger.info(f"Analyzing query for LaTeX: {query[:100]}...")

        # 1. Detect LaTeX formulas
        formulas_raw = self._extract_formulas(query)

        if not formulas_raw:
            # No LaTeX detected, return standard analysis
            logger.info("No LaTeX formulas detected in query")
            return LatexQueryAnalysis(
                has_latex=False,
                original_query=query,
                text_query=query,
                formulas=[],
                query_type="general",
                search_queries=[query],
                metadata={"num_formulas": 0}
            )

        logger.info(f"Detected {len(formulas_raw)} LaTeX formulas")

        # 2. Parse each formula
        parsed_formulas = []
        for formula in formulas_raw:
            try:
                parsed = await self._parse_formula(formula)
                parsed_formulas.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse formula '{formula}': {e}")
                # Add unparsed formula
                parsed_formulas.append(LatexFormula(
                    raw_latex=formula,
                    formula_type="unknown",
                    variables=[],
                    operators=[]
                ))

        # 3. Extract text query (remove LaTeX)
        text_query = self._extract_text_query(query, formulas_raw)

        # 4. Classify query type
        query_type = self._classify_math_query(text_query, parsed_formulas)

        # 5. Generate search queries
        search_queries = self._generate_search_queries(
            text_query,
            parsed_formulas,
            query_type
        )

        logger.info(f"Query analysis complete. Type: {query_type}, Search queries: {len(search_queries)}")

        return LatexQueryAnalysis(
            has_latex=True,
            original_query=query,
            text_query=text_query,
            formulas=parsed_formulas,
            query_type=query_type,
            search_queries=search_queries,
            metadata={
                "num_formulas": len(parsed_formulas),
                "formula_types": [f.formula_type for f in parsed_formulas],
                "variables": list(set(v for f in parsed_formulas for v in f.variables)),
                "operators": list(set(op for f in parsed_formulas for op in f.operators))
            }
        )

    def _extract_formulas(self, query: str) -> List[str]:
        """Extract LaTeX formulas from query"""
        formulas = []

        for pattern_name, pattern in self.LATEX_PATTERNS.items():
            matches = re.findall(pattern, query, re.DOTALL)
            formulas.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_formulas = []
        for formula in formulas:
            if formula not in seen:
                seen.add(formula)
                unique_formulas.append(formula)

        return unique_formulas

    async def _parse_formula(self, latex_string: str) -> LatexFormula:
        """
        Parse LaTeX formula using LaTeX parser service

        Args:
            latex_string: Raw LaTeX string

        Returns:
            Parsed formula object
        """
        try:
            # Call LaTeX parser service
            response = await self.http_client.post(
                f"{self.latex_parser_url}/api/v1/parse",
                json={
                    "latex_string": latex_string,
                    "output_format": "mathml",
                    "simplify": True
                }
            )
            response.raise_for_status()
            parsed = response.json()

            # Also get text representation
            text_response = await self.http_client.post(
                f"{self.latex_parser_url}/api/v1/parse",
                json={
                    "latex_string": latex_string,
                    "output_format": "text",
                    "simplify": False
                }
            )
            text_response.raise_for_status()
            text_parsed = text_response.json()

        except Exception as e:
            logger.warning(f"LaTeX parser service unavailable: {e}. Using fallback parsing.")
            # Fallback to local parsing
            return self._parse_formula_fallback(latex_string)

        # Classify formula type
        formula_type = self._classify_formula_type(latex_string)

        # Extract variables and operators
        variables = self._extract_variables(latex_string)
        operators = self._extract_operators(latex_string)

        return LatexFormula(
            raw_latex=latex_string,
            formula_type=formula_type,
            variables=variables,
            operators=operators,
            simplified=parsed.get("simplified_form"),
            mathml=parsed.get("parsed_output"),
            text_representation=text_parsed.get("parsed_output")
        )

    def _parse_formula_fallback(self, latex_string: str) -> LatexFormula:
        """Fallback formula parsing without service"""
        formula_type = self._classify_formula_type(latex_string)
        variables = self._extract_variables(latex_string)
        operators = self._extract_operators(latex_string)

        return LatexFormula(
            raw_latex=latex_string,
            formula_type=formula_type,
            variables=variables,
            operators=operators
        )

    def _classify_formula_type(self, latex: str) -> str:
        """Classify formula type based on LaTeX commands"""
        for formula_type, patterns in self.FORMULA_TYPES.items():
            for pattern in patterns:
                if re.search(pattern, latex):
                    return formula_type

        return "general"

    def _extract_variables(self, latex: str) -> List[str]:
        """Extract variables from LaTeX"""
        # Find single letters that are likely variables
        # Exclude common LaTeX commands
        variables = re.findall(r'(?<![\\a-zA-Z])[a-z](?![a-zA-Z])', latex)

        # Also find Greek letters
        greek_letters = re.findall(r'\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega)', latex)

        all_vars = list(set(variables + greek_letters))
        return sorted(all_vars)

    def _extract_operators(self, latex: str) -> List[str]:
        """Extract operators from LaTeX"""
        operators = []

        operator_patterns = {
            r'\+': '+',
            r'-': '-',
            r'\*': '*',
            r'/': '/',
            r'=': '=',
            r'\\cdot': '·',
            r'\\times': '×',
            r'\\div': '÷',
            r'\\frac': 'fraction',
            r'\\sqrt': 'sqrt',
            r'\\sum': 'sum',
            r'\\int': 'integral',
            r'\\partial': 'partial',
            r'\\lim': 'limit',
        }

        for pattern, name in operator_patterns.items():
            if re.search(pattern, latex):
                operators.append(name)

        return list(set(operators))

    def _extract_text_query(
        self,
        query: str,
        formulas: List[str]
    ) -> str:
        """Remove LaTeX formulas and return text query"""
        text_query = query

        for pattern in self.LATEX_PATTERNS.values():
            text_query = re.sub(pattern, ' [FORMULA] ', text_query, flags=re.DOTALL)

        # Clean up extra spaces
        text_query = re.sub(r'\s+', ' ', text_query).strip()
        # Remove [FORMULA] placeholders if they're at the edges
        text_query = text_query.replace('[FORMULA]', '').strip()

        return text_query

    def _classify_math_query(
        self,
        text_query: str,
        formulas: List[LatexFormula]
    ) -> str:
        """
        Classify the mathematical query type

        Returns:
            Query type: "solve_equation", "simplify", "integrate", etc.
        """
        if not formulas:
            return "general"

        text_lower = text_query.lower()

        # Check for specific query keywords
        if any(kw in text_lower for kw in self.QUERY_KEYWORDS["solve"]):
            if any(f.formula_type == "equation" for f in formulas):
                return "solve_equation"
            return "evaluate"

        if any(kw in text_lower for kw in self.QUERY_KEYWORDS["simplify"]):
            return "simplify"

        if any(kw in text_lower for kw in self.QUERY_KEYWORDS["integrate"]):
            return "integrate"

        if any(kw in text_lower for kw in self.QUERY_KEYWORDS["differentiate"]):
            return "differentiate"

        if any(kw in text_lower for kw in self.QUERY_KEYWORDS["prove"]):
            return "prove"

        # Classify based on formula types
        if any(f.formula_type == "equation" for f in formulas):
            return "solve_equation"

        if any(f.formula_type == "integral" for f in formulas):
            return "integrate"

        if any(f.formula_type == "derivative" for f in formulas):
            return "differentiate"

        # Default
        return "formula_query"

    def _generate_search_queries(
        self,
        text_query: str,
        formulas: List[LatexFormula],
        query_type: str
    ) -> List[str]:
        """
        Generate multiple search queries for formula queries

        Args:
            text_query: Text portion of query
            formulas: Parsed formulas
            query_type: Type of math query

        Returns:
            List of search queries for retrieval
        """
        search_queries = []

        # 1. Original text query
        if text_query:
            search_queries.append(text_query)

        # 2. Query with formula types
        if formulas:
            formula_types = [f.formula_type for f in formulas]
            type_query = f"{text_query} {' '.join(formula_types)}"
            search_queries.append(type_query)

        # 3. Query with variables
        all_variables = list(set(v for f in formulas for v in f.variables))
        if all_variables:
            var_query = f"{text_query} variables: {' '.join(all_variables)}"
            search_queries.append(var_query)

        # 4. Query type specific queries
        if query_type == "solve_equation":
            search_queries.append(f"solving equations {text_query}")
            search_queries.append(f"how to solve {' '.join(all_variables)} equation")

        elif query_type == "integrate":
            search_queries.append(f"integration {text_query}")
            search_queries.append(f"integral calculus {' '.join(all_variables)}")

        elif query_type == "differentiate":
            search_queries.append(f"differentiation {text_query}")
            search_queries.append(f"derivative {' '.join(all_variables)}")

        elif query_type == "simplify":
            search_queries.append(f"simplification {text_query}")
            search_queries.append(f"algebraic simplification")

        # 5. Add text representations of formulas
        for formula in formulas:
            if formula.text_representation:
                search_queries.append(f"{text_query} {formula.text_representation}")

        # 6. Add LaTeX formulas for exact matching
        for formula in formulas:
            search_queries.append(formula.raw_latex)
            if formula.simplified:
                search_queries.append(formula.simplified)

        # Remove duplicates and empty queries
        search_queries = [q.strip() for q in search_queries if q and q.strip()]
        seen = set()
        unique_queries = []
        for query in search_queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)

        return unique_queries[:10]  # Limit to top 10 queries

    def to_dict(self, analysis: LatexQueryAnalysis) -> Dict[str, Any]:
        """Convert analysis to dictionary"""
        result = asdict(analysis)
        # Convert formulas to dicts
        result["formulas"] = [asdict(f) for f in analysis.formulas]
        return result
