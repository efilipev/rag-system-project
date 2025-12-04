"""
Query Analyzer Service using LangChain and spaCy
Enhanced with LaTeX formula detection and analysis
"""
import re
from typing import List, Optional, Dict, Any

import spacy
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.core.config import settings
from src.core.logging import logger
from src.models.schemas import Entity, QueryAnalysis, QueryIntent, LatexAnalysis, LatexFormula
from src.services.latex_query_analyzer import LatexQueryAnalyzer, LatexQueryAnalysis
from src.services.query_constructor import QueryConstructor, ConstructedQuery
from src.services.intent_classifier import get_intent_classifier, IntentClassifier
from src.services.query_expander import get_query_expander, QueryExpander
from src.services.cache_service import CacheService


class QueryAnalyzerService:
    """
    Service for analyzing user queries using NLP techniques
    """

    def __init__(self, latex_parser_url: Optional[str] = None, cache_service: Optional[CacheService] = None):
        self.nlp: Optional[spacy.language.Language] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self.intent_classifier: Optional[IntentClassifier] = None
        self.query_expander: Optional[QueryExpander] = None
        self.cache_service = cache_service

        # LaTeX analysis components
        self.latex_analyzer: Optional[LatexQueryAnalyzer] = None
        self.query_constructor: Optional[QueryConstructor] = None
        self.latex_parser_url = latex_parser_url or "http://localhost:8005"

    async def initialize(self) -> None:
        """
        Initialize NLP models and embeddings
        """
        try:
            logger.info("Initializing Query Analyzer Service")

            # Load spaCy model
            logger.info(f"Loading spaCy model: {settings.SPACY_MODEL}")
            self.nlp = spacy.load(settings.SPACY_MODEL)

            # Initialize LangChain embeddings
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.MAX_QUERY_LENGTH,
                chunk_overlap=50,
                length_function=len,
            )

            # Initialize ML-based intent classifier
            logger.info("Initializing ML-based intent classifier")
            self.intent_classifier = await get_intent_classifier()

            # Initialize query expander
            logger.info("Initializing query expander")
            self.query_expander = get_query_expander(nlp_model=self.nlp)

            # Initialize LaTeX components
            logger.info(f"Initializing LaTeX analyzer with parser at {self.latex_parser_url}")
            self.latex_analyzer = LatexQueryAnalyzer(latex_parser_url=self.latex_parser_url)
            self.query_constructor = QueryConstructor()

            logger.info("Query Analyzer Service initialized successfully (with LaTeX, ML intent classification, and query expansion)")

        except Exception as e:
            logger.error(f"Failed to initialize Query Analyzer Service: {e}")
            raise

    async def close(self):
        """Close and cleanup resources"""
        if self.latex_analyzer:
            await self.latex_analyzer.close()
        logger.info("Query Analyzer Service closed")

    def normalize_query(self, query: str) -> str:
        """
        Normalize query text
        """
        # Convert to lowercase
        normalized = query.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Remove special characters but keep important ones
        normalized = re.sub(r"[^\w\s\-?!.]", "", normalized)

        return normalized

    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query using spaCy
        """
        if not self.nlp:
            return []

        doc = self.nlp(query)

        keywords = []

        # Extract nouns, proper nouns, and verbs
        for token in doc:
            if (
                token.pos_ in ["NOUN", "PROPN", "VERB"]
                and not token.is_stop
                and len(token.text) > 2
            ):
                keywords.append(token.lemma_)

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 2:
                keywords.append(chunk.text.lower())

        # Remove duplicates and limit
        keywords = list(dict.fromkeys(keywords))
        return keywords[: settings.MAX_KEYWORDS]

    def extract_entities(self, query: str) -> List[Entity]:
        """
        Extract named entities from query
        """
        if not self.nlp or not settings.ENABLE_ENTITY_EXTRACTION:
            return []

        doc = self.nlp(query)

        entities = []
        for ent in doc.ents:
            entities.append(
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )

        return entities

    def classify_intent(self, query: str) -> Optional[QueryIntent]:
        """
        Classify query intent using ML-based classifier
        """
        if not settings.ENABLE_INTENT_CLASSIFICATION:
            return None

        try:
            if self.intent_classifier:
                # Use ML-based intent classifier
                result = self.intent_classifier.classify_intent(query)
                return QueryIntent(
                    intent=result["intent"],
                    confidence=result["confidence"]
                )
            else:
                # Fallback to simple classification if ML classifier not available
                logger.warning("Intent classifier not available, using fallback")
                return QueryIntent(intent="informational", confidence=0.5)

        except Exception as e:
            logger.error(f"Error in intent classification: {e}", exc_info=True)
            return QueryIntent(intent="informational", confidence=0.5)

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using LangChain
        Includes caching for performance
        """
        if not self.embeddings:
            raise ValueError("Embeddings not initialized")

        # Check cache first
        if self.cache_service:
            cached_embedding = await self.cache_service.get_embedding(text)
            if cached_embedding:
                logger.debug("Using cached embedding")
                return cached_embedding

        # Generate embedding
        embedding = self.embeddings.embed_query(text)

        # Cache the result
        if self.cache_service:
            await self.cache_service.set_embedding(text, embedding)

        return embedding

    async def analyze_query(self, query: str, use_cache: bool = True) -> QueryAnalysis:
        """
        Perform comprehensive query analysis
        Includes caching for performance
        """
        logger.info(f"Analyzing query: {query[:100]}...")

        # Validate query length
        if len(query) < settings.MIN_QUERY_LENGTH:
            raise ValueError(
                f"Query too short. Minimum length: {settings.MIN_QUERY_LENGTH}"
            )

        if len(query) > settings.MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long. Maximum length: {settings.MAX_QUERY_LENGTH}"
            )

        # Check cache first
        if use_cache and self.cache_service:
            cached_analysis = await self.cache_service.get_analysis(query)
            if cached_analysis:
                logger.info("Using cached query analysis")
                # Convert cached dict back to QueryAnalysis object
                cached_data = cached_analysis.get("analysis", cached_analysis)
                return QueryAnalysis(**cached_data)

        # Normalize query
        normalized_query = self.normalize_query(query)

        # Extract keywords
        keywords = self.extract_keywords(query)

        # Extract entities
        entities = self.extract_entities(query)

        # Classify intent
        intent = self.classify_intent(query)

        # Generate embedding (with caching)
        embedding = await self.generate_embedding(normalized_query)

        # LaTeX analysis
        latex_analysis_result = None
        if self.latex_analyzer:
            try:
                latex_result = await self.latex_analyzer.analyze_query_with_latex(query)
                if latex_result.has_latex:
                    # Convert to Pydantic model
                    formulas = [
                        LatexFormula(
                            raw_latex=f.raw_latex,
                            formula_type=f.formula_type,
                            variables=f.variables,
                            operators=f.operators,
                            simplified=f.simplified,
                            mathml=f.mathml,
                            text_representation=f.text_representation
                        )
                        for f in latex_result.formulas
                    ]
                    latex_analysis_result = LatexAnalysis(
                        has_latex=True,
                        formulas=formulas,
                        query_type=latex_result.query_type,
                        search_queries=latex_result.search_queries
                    )
                    logger.info(f"LaTeX detected: {len(formulas)} formulas, type: {latex_result.query_type}")
            except Exception as e:
                logger.warning(f"LaTeX analysis failed: {e}")

        # Build analysis result
        analysis = QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            keywords=keywords,
            entities=entities,
            intent=intent,
            language="en",  # Could be enhanced with language detection
            embedding=embedding,
            latex_analysis=latex_analysis_result,
            metadata={
                "num_keywords": len(keywords),
                "num_entities": len(entities),
                "query_length": len(query),
                "has_latex": latex_analysis_result.has_latex if latex_analysis_result else False,
            },
        )

        # Cache the result
        if use_cache and self.cache_service:
            await self.cache_service.set_analysis(query, analysis.model_dump())

        logger.info(f"Query analysis completed. Keywords: {len(keywords)}, Entities: {len(entities)}")

        return analysis

    async def expand_query(self, query: str, max_expansions: int = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Expand query using multiple strategies
        Includes caching for performance

        Args:
            query: Original query
            max_expansions: Maximum number of expansions
            use_cache: Whether to use caching

        Returns:
            Dictionary with query expansions
        """
        if not self.query_expander:
            logger.warning("Query expander not initialized")
            return {
                "original_query": query,
                "all_expansions": [],
                "total_expansions": 0,
                "expansion_methods": []
            }

        # Check cache first
        if use_cache and self.cache_service:
            cached_expansion = await self.cache_service.get_expansion(query)
            if cached_expansion:
                logger.info("Using cached query expansion")
                return cached_expansion.get("expansion", cached_expansion)

        # Generate expansion
        expansion_result = self.query_expander.expand_query(
            query=query,
            max_expansions=max_expansions
        )

        # Cache the result
        if use_cache and self.cache_service:
            await self.cache_service.set_expansion(query, expansion_result)

        return expansion_result

    async def analyze_query_enhanced(
        self,
        query: str,
        enable_latex: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced query analysis with LaTeX detection

        Args:
            query: User query
            enable_latex: Whether to enable LaTeX detection

        Returns:
            Combined analysis result with standard NLP + LaTeX analysis
        """
        logger.info(f"Enhanced query analysis for: {query[:100]}...")

        # Standard query analysis
        standard_analysis = await self.analyze_query(query)

        # LaTeX analysis
        latex_analysis = None
        if enable_latex and self.latex_analyzer:
            try:
                latex_analysis = await self.latex_analyzer.analyze_query_with_latex(query)
            except Exception as e:
                logger.warning(f"LaTeX analysis failed: {e}")

        # Combine results
        result = {
            "standard_analysis": {
                "original_query": standard_analysis.original_query,
                "normalized_query": standard_analysis.normalized_query,
                "keywords": standard_analysis.keywords,
                "entities": [
                    {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
                    for e in standard_analysis.entities
                ],
                "intent": {
                    "intent": standard_analysis.intent.intent if standard_analysis.intent else None,
                    "confidence": standard_analysis.intent.confidence if standard_analysis.intent else None
                } if standard_analysis.intent else None,
                "language": standard_analysis.language,
                "metadata": standard_analysis.metadata
            },
            "latex_analysis": None,
            "has_latex": False
        }

        if latex_analysis:
            result["latex_analysis"] = self.latex_analyzer.to_dict(latex_analysis)
            result["has_latex"] = latex_analysis.has_latex

        logger.info(f"Enhanced analysis complete. Has LaTeX: {result['has_latex']}")

        return result

    async def construct_retrieval_query(
        self,
        query: str,
        enable_multi_query: bool = True,
        max_expansion_queries: int = 5
    ) -> ConstructedQuery:
        """
        Construct optimized query for retrieval

        Args:
            query: User query
            enable_multi_query: Whether to generate multiple expansion queries
            max_expansion_queries: Maximum number of expansion queries

        Returns:
            Constructed query ready for retrieval
        """
        logger.info(f"Constructing retrieval query for: {query[:100]}...")

        # Analyze query with LaTeX support
        if self.latex_analyzer and self.query_constructor:
            latex_analysis = await self.latex_analyzer.analyze_query_with_latex(query)

            # Construct optimized query
            constructed_query = self.query_constructor.construct_query(
                latex_analysis=latex_analysis,
                enable_multi_query=enable_multi_query,
                max_expansion_queries=max_expansion_queries
            )

            logger.info(
                f"Query construction complete. Strategy: {constructed_query.retrieval_strategy}, "
                f"Expansion queries: {len(constructed_query.expansion_queries)}"
            )

            return constructed_query

        else:
            # Fallback: basic construction without LaTeX support
            logger.warning("LaTeX analyzer not available, using basic query construction")
            from src.services.query_constructor import ConstructedQuery, RetrievalStrategy

            return ConstructedQuery(
                original_query=query,
                retrieval_strategy=RetrievalStrategy.DENSE_VECTOR,
                primary_query=query,
                expansion_queries=[],
                structured_filters={},
                reranking_hints={},
                metadata={"has_latex": False}
            )
