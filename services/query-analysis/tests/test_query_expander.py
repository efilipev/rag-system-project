"""
Unit tests for Query Expander Service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.query_expander import QueryExpander


class TestQueryExpander:
    """Test suite for QueryExpander"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        with patch('app.services.query_expander.spacy.load') as mock_spacy:
            # Mock spaCy model
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.__iter__ = Mock(return_value=iter([
                Mock(text="machine", pos_="NOUN", lemma_="machine"),
                Mock(text="learning", pos_="NOUN", lemma_="learning")
            ]))
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp

            self.expander = QueryExpander()
            yield

    def test_initialization(self):
        """Test that QueryExpander initializes correctly"""
        assert self.expander is not None
        assert hasattr(self.expander, 'nlp')

    @patch('app.services.query_expander.wordnet')
    def test_expand_query_basic(self, mock_wordnet):
        """Test basic query expansion"""
        # Mock WordNet synonyms
        mock_synset = MagicMock()
        mock_synset.lemmas.return_value = [
            Mock(name=lambda: "device"),
            Mock(name=lambda: "computer")
        ]
        mock_wordnet.synsets.return_value = [mock_synset]

        query = "What is machine learning?"
        result = self.expander.expand_query(query, max_expansions=3)

        assert result is not None
        assert 'original_query' in result
        assert 'all_expansions' in result
        assert 'expansion_methods' in result
        assert 'total_expansions' in result
        assert result['original_query'] == query

    @patch('app.services.query_expander.wordnet')
    def test_synonym_expansion(self, mock_wordnet):
        """Test synonym-based expansion"""
        mock_synset = MagicMock()
        mock_synset.lemmas.return_value = [
            Mock(name=lambda: "implementation"),
            Mock(name=lambda: "execution")
        ]
        mock_wordnet.synsets.return_value = [mock_synset]

        query = "How to implement OAuth2?"
        result = self.expander.expand_query(query, max_expansions=5)

        assert len(result['all_expansions']) <= 5
        assert 'synonyms' in result['expansion_methods']

    def test_question_reformulation(self):
        """Test question reformulation expansion"""
        query = "What is Python?"
        result = self.expander.expand_query(query, max_expansions=3)

        # Should generate reformulated questions
        assert result is not None
        assert len(result['all_expansions']) > 0

        # Check if reformulation method was used
        if 'reformulation' in result['expansion_methods']:
            # Reformulated queries should maintain the core meaning
            expansions = result['all_expansions']
            assert any('python' in exp.lower() for exp in expansions)

    def test_max_expansions_limit(self):
        """Test that max_expansions parameter is respected"""
        query = "How do I use asyncio in Python?"
        max_exp = 3

        result = self.expander.expand_query(query, max_expansions=max_exp)

        assert len(result['all_expansions']) <= max_exp

    def test_empty_query_handling(self):
        """Test handling of empty query"""
        query = ""
        result = self.expander.expand_query(query, max_expansions=3)

        assert result is not None
        assert result['original_query'] == query
        assert isinstance(result['all_expansions'], list)
        # Empty query should return few or no expansions
        assert len(result['all_expansions']) <= 1

    def test_short_query_handling(self):
        """Test handling of very short queries"""
        query = "API"
        result = self.expander.expand_query(query, max_expansions=3)

        assert result is not None
        assert result['original_query'] == query
        assert isinstance(result['all_expansions'], list)

    @patch('app.services.query_expander.wordnet')
    def test_no_synonyms_found(self, mock_wordnet):
        """Test handling when no synonyms are found"""
        # Mock WordNet to return no synonyms
        mock_wordnet.synsets.return_value = []

        query = "What is xyz123?"
        result = self.expander.expand_query(query, max_expansions=3)

        assert result is not None
        assert result['original_query'] == query
        # Should still return some expansions (through reformulation)
        assert isinstance(result['all_expansions'], list)

    def test_expansion_uniqueness(self):
        """Test that expansions are unique"""
        query = "What is Python programming?"
        result = self.expander.expand_query(query, max_expansions=10)

        expansions = result['all_expansions']
        # Check for uniqueness
        unique_expansions = set(expansions)
        assert len(expansions) == len(unique_expansions), "Expansions should be unique"

    def test_expansion_methods_accuracy(self):
        """Test that expansion_methods accurately reflects methods used"""
        query = "How to implement authentication?"
        result = self.expander.expand_query(query, max_expansions=5)

        # Should have used at least one method
        assert len(result['expansion_methods']) > 0

        # Methods should be from expected set
        expected_methods = ['synonyms', 'reformulation', 'related_terms', 'decomposition']
        for method in result['expansion_methods']:
            assert method in expected_methods

    def test_preserve_original_meaning(self):
        """Test that expansions preserve key terms from original query"""
        query = "What is OAuth2 authentication?"
        result = self.expander.expand_query(query, max_expansions=5)

        # At least some expansions should contain key terms
        key_terms = ['oauth', 'auth']
        expansions_text = ' '.join(result['all_expansions']).lower()

        # Should preserve at least one key concept
        assert any(term in expansions_text for term in key_terms)

    @patch('app.services.query_expander.wordnet')
    def test_multiple_word_synonyms(self, mock_wordnet):
        """Test handling of multi-word phrases"""
        mock_synset = MagicMock()
        mock_synset.lemmas.return_value = [
            Mock(name=lambda: "deep_learning"),
            Mock(name=lambda: "neural_network")
        ]
        mock_wordnet.synsets.return_value = [mock_synset]

        query = "Explain machine learning algorithms"
        result = self.expander.expand_query(query, max_expansions=4)

        assert result is not None
        assert len(result['all_expansions']) <= 4

    def test_case_sensitivity(self):
        """Test that case is handled appropriately"""
        query = "What is PYTHON?"
        result = self.expander.expand_query(query, max_expansions=3)

        assert result is not None
        # Original query case should be preserved in the result
        assert result['original_query'] == query

    def test_special_characters_handling(self):
        """Test handling of queries with special characters"""
        query = "How to use @decorator in Python?"
        result = self.expander.expand_query(query, max_expansions=3)

        assert result is not None
        assert result['original_query'] == query
        assert isinstance(result['all_expansions'], list)

    def test_long_query_expansion(self):
        """Test expansion of very long queries"""
        query = "How do I implement a distributed caching system using Redis with automatic failover and replication across multiple data centers?"
        result = self.expander.expand_query(query, max_expansions=3)

        assert result is not None
        assert len(result['all_expansions']) <= 3

    def test_technical_terms_preservation(self):
        """Test that technical terms are preserved in expansions"""
        query = "How to configure Kubernetes ingress controller?"
        result = self.expander.expand_query(query, max_expansions=3)

        # Technical terms like Kubernetes should appear in at least one expansion
        expansions_text = ' '.join(result['all_expansions']).lower()
        assert 'kubernetes' in expansions_text or 'k8s' in expansions_text or 'ingress' in expansions_text

    def test_zero_max_expansions(self):
        """Test behavior when max_expansions is 0"""
        query = "What is Python?"
        result = self.expander.expand_query(query, max_expansions=0)

        assert result is not None
        assert len(result['all_expansions']) == 0
        assert result['total_expansions'] == 0

    def test_negative_max_expansions(self):
        """Test behavior when max_expansions is negative"""
        query = "What is Python?"
        result = self.expander.expand_query(query, max_expansions=-1)

        # Should handle gracefully, either returning empty or defaulting to reasonable number
        assert result is not None
        assert isinstance(result['all_expansions'], list)

    def test_expansion_diversity(self):
        """Test that expansions have some diversity"""
        query = "What is machine learning?"
        result = self.expander.expand_query(query, max_expansions=5)

        if len(result['all_expansions']) >= 2:
            # Check that not all expansions are identical
            expansions = result['all_expansions']
            assert len(set(expansions)) > 1, "Expansions should have diversity"

    def test_result_structure(self):
        """Test that result has correct structure"""
        query = "How to use FastAPI?"
        result = self.expander.expand_query(query, max_expansions=3)

        # Check all required fields are present
        required_fields = ['original_query', 'all_expansions', 'expansion_methods', 'total_expansions']
        for field in required_fields:
            assert field in result, f"Result should contain '{field}'"

        # Check data types
        assert isinstance(result['original_query'], str)
        assert isinstance(result['all_expansions'], list)
        assert isinstance(result['expansion_methods'], list)
        assert isinstance(result['total_expansions'], int)

    def test_total_expansions_count(self):
        """Test that total_expansions matches actual count"""
        query = "What is asyncio?"
        result = self.expander.expand_query(query, max_expansions=4)

        assert result['total_expansions'] == len(result['all_expansions'])
