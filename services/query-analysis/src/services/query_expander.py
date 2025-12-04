"""
Query Expansion and Reformulation Service
Generates alternative phrasings and expansions of user queries
"""
from typing import List, Dict, Any, Optional, Set
import re
from itertools import combinations

import spacy
from nltk.corpus import wordnet

from src.core.config import settings
from src.core.logging import logger


class QueryExpander:
    """
    Service for expanding and reformulating queries to improve retrieval
    """

    def __init__(self, nlp_model: Optional[spacy.language.Language] = None):
        self.nlp = nlp_model
        self._wordnet_available = False
        self._initialize_wordnet()

    def _initialize_wordnet(self):
        """Initialize WordNet for synonym expansion"""
        try:
            import nltk
            try:
                wordnet.synsets('test')
                self._wordnet_available = True
                logger.info("WordNet initialized successfully")
            except LookupError:
                logger.info("Downloading WordNet corpus...")
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                self._wordnet_available = True
                logger.info("WordNet downloaded and initialized")
        except Exception as e:
            logger.warning(f"WordNet initialization failed: {e}. Synonym expansion disabled.")
            self._wordnet_available = False

    def get_synonyms(self, word: str, pos: Optional[str] = None) -> Set[str]:
        """
        Get synonyms for a word using WordNet

        Args:
            word: The word to find synonyms for
            pos: Part of speech (optional): 'n' (noun), 'v' (verb), 'a' (adjective), 'r' (adverb)

        Returns:
            Set of synonyms
        """
        if not self._wordnet_available:
            return set()

        try:
            synonyms = set()
            for syn in wordnet.synsets(word, pos=pos):
                for lemma in syn.lemmas():
                    # Get lemma name and clean it
                    synonym = lemma.name().replace('_', ' ').lower()
                    # Avoid adding the same word or very similar words
                    if synonym != word.lower() and len(synonym) > 2:
                        synonyms.add(synonym)

            # Limit to most relevant synonyms (top 3)
            return set(list(synonyms)[:3])

        except Exception as e:
            logger.debug(f"Error getting synonyms for '{word}': {e}")
            return set()

    def expand_with_synonyms(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query by replacing keywords with synonyms

        Args:
            query: Original query
            max_expansions: Maximum number of synonym-based expansions

        Returns:
            List of expanded queries
        """
        if not self.nlp or not self._wordnet_available:
            return []

        try:
            doc = self.nlp(query)
            expansions = []

            # Find important words (nouns, verbs, adjectives)
            important_words = [
                (token.text, token.pos_)
                for token in doc
                if token.pos_ in ['NOUN', 'VERB', 'ADJ']
                and not token.is_stop
                and len(token.text) > 3
            ]

            # Get synonyms for each important word
            word_synonyms = {}
            for word, pos in important_words[:3]:  # Limit to first 3 important words
                # Map spaCy POS to WordNet POS
                wordnet_pos = None
                if pos == 'NOUN':
                    wordnet_pos = 'n'
                elif pos == 'VERB':
                    wordnet_pos = 'v'
                elif pos == 'ADJ':
                    wordnet_pos = 'a'

                syns = self.get_synonyms(word.lower(), wordnet_pos)
                if syns:
                    word_synonyms[word.lower()] = list(syns)

            # Generate expansions by replacing words with synonyms
            if word_synonyms:
                query_lower = query.lower()
                for word, synonyms in word_synonyms.items():
                    for synonym in synonyms[:2]:  # Use top 2 synonyms per word
                        if len(expansions) >= max_expansions:
                            break

                        # Replace word with synonym (case-insensitive)
                        expanded = re.sub(
                            r'\b' + re.escape(word) + r'\b',
                            synonym,
                            query_lower,
                            count=1
                        )

                        if expanded != query_lower and expanded not in expansions:
                            expansions.append(expanded)

                    if len(expansions) >= max_expansions:
                        break

            return expansions[:max_expansions]

        except Exception as e:
            logger.error(f"Error in synonym expansion: {e}", exc_info=True)
            return []

    def reformulate_question(self, query: str) -> List[str]:
        """
        Reformulate questions into different forms

        Args:
            query: Original query

        Returns:
            List of reformulated queries
        """
        reformulations = []
        query_lower = query.lower().strip()

        # Detect question patterns and reformulate
        patterns = [
            # "How do I X?" -> "How to X", "X tutorial", "X guide"
            (r'how (do|can|should) i (.+)', [
                lambda m: f"how to {m.group(2)}",
                lambda m: f"{m.group(2)} tutorial",
                lambda m: f"{m.group(2)} guide",
            ]),
            # "What is X?" -> "X definition", "X meaning", "define X"
            (r'what is (.+)', [
                lambda m: f"{m.group(1)} definition",
                lambda m: f"{m.group(1)} meaning",
                lambda m: f"define {m.group(1)}",
            ]),
            # "Why does X?" -> "X reason", "X explanation", "cause of X"
            (r'why does (.+)', [
                lambda m: f"{m.group(1)} reason",
                lambda m: f"{m.group(1)} explanation",
                lambda m: f"cause of {m.group(1)}",
            ]),
            # "How does X work?" -> "X mechanism", "X explanation", "X process"
            (r'how does (.+) work', [
                lambda m: f"{m.group(1)} mechanism",
                lambda m: f"{m.group(1)} explanation",
                lambda m: f"{m.group(1)} process",
            ]),
            # "What are X?" -> "X list", "X examples", "types of X"
            (r'what are (.+)', [
                lambda m: f"{m.group(1)} list",
                lambda m: f"{m.group(1)} examples",
                lambda m: f"types of {m.group(1)}",
            ]),
            # "Which X is better?" -> "X comparison", "best X", "X vs"
            (r'which (.+) is better', [
                lambda m: f"{m.group(1)} comparison",
                lambda m: f"best {m.group(1)}",
                lambda m: f"compare {m.group(1)}",
            ]),
        ]

        for pattern, reformulation_funcs in patterns:
            match = re.search(pattern, query_lower)
            if match:
                for func in reformulation_funcs:
                    try:
                        reformulation = func(match)
                        if reformulation and reformulation != query_lower:
                            reformulations.append(reformulation)
                    except Exception as e:
                        logger.debug(f"Error applying reformulation: {e}")
                break  # Use only the first matching pattern

        return reformulations[:3]  # Return top 3 reformulations

    def expand_with_related_terms(self, query: str) -> List[str]:
        """
        Expand query by adding related technical terms

        Args:
            query: Original query

        Returns:
            List of queries with related terms added
        """
        if not self.nlp:
            return []

        try:
            doc = self.nlp(query)
            expansions = []

            # Define domain-specific related terms
            related_terms = {
                'authentication': ['auth', 'login', 'security', 'credentials'],
                'database': ['db', 'storage', 'persistence', 'data store'],
                'api': ['endpoint', 'rest', 'graphql', 'interface'],
                'framework': ['library', 'toolkit', 'package'],
                'error': ['exception', 'bug', 'issue', 'problem'],
                'performance': ['optimization', 'speed', 'efficiency'],
                'deploy': ['deployment', 'release', 'production'],
                'test': ['testing', 'unit test', 'integration test'],
            }

            # Find if any related terms apply
            query_lower = query.lower()
            for term, related in related_terms.items():
                if term in query_lower:
                    for related_term in related[:2]:  # Use top 2 related terms
                        # Add related term to query
                        expanded = f"{query_lower} {related_term}"
                        if expanded not in expansions:
                            expansions.append(expanded)

                        if len(expansions) >= 2:
                            break

                if len(expansions) >= 2:
                    break

            return expansions

        except Exception as e:
            logger.error(f"Error in related terms expansion: {e}", exc_info=True)
            return []

    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex queries into simpler sub-queries

        Args:
            query: Original query

        Returns:
            List of sub-queries
        """
        if not self.nlp:
            return []

        try:
            sub_queries = []

            # Split on conjunctions
            parts = re.split(r'\s+and\s+|\s+or\s+', query, flags=re.IGNORECASE)
            if len(parts) > 1:
                for part in parts:
                    part = part.strip()
                    if len(part) > settings.MIN_QUERY_LENGTH:
                        sub_queries.append(part)

            # Split on question marks (multiple questions)
            questions = [q.strip() + '?' for q in query.split('?') if q.strip()]
            if len(questions) > 1:
                sub_queries.extend([q for q in questions if len(q) > settings.MIN_QUERY_LENGTH])

            return sub_queries[:3]  # Return top 3 sub-queries

        except Exception as e:
            logger.error(f"Error in query decomposition: {e}", exc_info=True)
            return []

    def expand_query(
        self,
        query: str,
        max_expansions: int = None,
        enable_synonyms: bool = True,
        enable_reformulation: bool = True,
        enable_related_terms: bool = True,
        enable_decomposition: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive query expansion using multiple strategies

        Args:
            query: Original query
            max_expansions: Maximum number of expansions (defaults to config)
            enable_synonyms: Enable synonym expansion
            enable_reformulation: Enable question reformulation
            enable_related_terms: Enable related terms expansion
            enable_decomposition: Enable query decomposition

        Returns:
            Dictionary with original query and various expansions
        """
        if max_expansions is None:
            max_expansions = settings.QUERY_EXPANSION_COUNT

        try:
            result = {
                "original_query": query,
                "expansions": [],
                "reformulations": [],
                "sub_queries": [],
                "expansion_methods": []
            }

            # Synonym expansion
            if enable_synonyms and settings.ENABLE_QUERY_EXPANSION:
                synonym_expansions = self.expand_with_synonyms(query, max_expansions=2)
                if synonym_expansions:
                    result["expansions"].extend(synonym_expansions)
                    result["expansion_methods"].append("synonyms")

            # Question reformulation
            if enable_reformulation and settings.ENABLE_QUERY_EXPANSION:
                reformulations = self.reformulate_question(query)
                if reformulations:
                    result["reformulations"].extend(reformulations)
                    result["expansion_methods"].append("reformulation")

            # Related terms expansion
            if enable_related_terms and settings.ENABLE_QUERY_EXPANSION:
                related_expansions = self.expand_with_related_terms(query)
                if related_expansions:
                    result["expansions"].extend(related_expansions)
                    result["expansion_methods"].append("related_terms")

            # Query decomposition
            if enable_decomposition:
                sub_queries = self.decompose_query(query)
                if sub_queries:
                    result["sub_queries"].extend(sub_queries)
                    result["expansion_methods"].append("decomposition")

            # Combine all expansions and limit to max_expansions
            all_expansions = (
                result["expansions"] +
                result["reformulations"] +
                result["sub_queries"]
            )

            # Remove duplicates while preserving order
            seen = set([query.lower()])
            unique_expansions = []
            for exp in all_expansions:
                if exp.lower() not in seen:
                    seen.add(exp.lower())
                    unique_expansions.append(exp)

            # Limit total expansions
            result["all_expansions"] = unique_expansions[:max_expansions]
            result["total_expansions"] = len(result["all_expansions"])

            logger.info(
                f"Query expansion generated {result['total_expansions']} expansions "
                f"using methods: {', '.join(result['expansion_methods'])}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in query expansion: {e}", exc_info=True)
            return {
                "original_query": query,
                "expansions": [],
                "reformulations": [],
                "sub_queries": [],
                "all_expansions": [],
                "total_expansions": 0,
                "expansion_methods": [],
                "error": str(e)
            }


# Global instance
_query_expander: Optional[QueryExpander] = None


def get_query_expander(nlp_model: Optional[spacy.language.Language] = None) -> QueryExpander:
    """Get or create the query expander singleton"""
    global _query_expander

    if _query_expander is None:
        _query_expander = QueryExpander(nlp_model=nlp_model)

    return _query_expander
