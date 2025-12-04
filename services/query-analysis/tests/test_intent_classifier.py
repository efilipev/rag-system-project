"""
Unit tests for Intent Classifier Service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.intent_classifier import IntentClassifier


class TestIntentClassifier:
    """Test suite for IntentClassifier"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        # Mock the transformers pipeline to avoid loading the actual model
        with patch('app.services.intent_classifier.pipeline') as mock_pipeline:
            self.mock_classifier = MagicMock()
            mock_pipeline.return_value = self.mock_classifier
            self.classifier = IntentClassifier()
            yield

    def test_initialization(self):
        """Test that IntentClassifier initializes correctly"""
        assert self.classifier is not None
        assert self.classifier.intent_labels is not None
        assert len(self.classifier.intent_labels) == 8
        assert "informational" in self.classifier.intent_labels
        assert "procedural" in self.classifier.intent_labels

    def test_classify_informational_query(self):
        """Test classification of informational query"""
        # Mock the classifier response
        self.mock_classifier.return_value = {
            'labels': ['informational', 'procedural', 'definitional'],
            'scores': [0.75, 0.15, 0.10]
        }

        query = "What is machine learning?"
        result = self.classifier.classify_intent(query)

        assert result['intent'] == 'informational'
        assert result['confidence'] == 0.75
        assert 'all_scores' in result
        assert len(result['all_scores']) == 3

    def test_classify_procedural_query(self):
        """Test classification of procedural/how-to query"""
        self.mock_classifier.return_value = {
            'labels': ['procedural', 'informational', 'definitional'],
            'scores': [0.85, 0.10, 0.05]
        }

        query = "How do I implement OAuth2 in Python?"
        result = self.classifier.classify_intent(query)

        assert result['intent'] == 'procedural'
        assert result['confidence'] == 0.85

    def test_classify_comparative_query(self):
        """Test classification of comparative query"""
        self.mock_classifier.return_value = {
            'labels': ['comparative', 'informational', 'procedural'],
            'scores': [0.80, 0.12, 0.08]
        }

        query = "What is the difference between REST and GraphQL?"
        result = self.classifier.classify_intent(query)

        assert result['intent'] == 'comparative'
        assert result['confidence'] == 0.80

    def test_classify_recommendation_query(self):
        """Test classification of recommendation query"""
        self.mock_classifier.return_value = {
            'labels': ['recommendation', 'informational', 'procedural'],
            'scores': [0.70, 0.20, 0.10]
        }

        query = "What's the best Python framework for web development?"
        result = self.classifier.classify_intent(query)

        assert result['intent'] == 'recommendation'
        assert result['confidence'] == 0.70

    def test_classify_navigational_query(self):
        """Test classification of navigational query"""
        self.mock_classifier.return_value = {
            'labels': ['navigational', 'informational', 'procedural'],
            'scores': [0.65, 0.25, 0.10]
        }

        query = "Where can I find the Python documentation for asyncio?"
        result = self.classifier.classify_intent(query)

        assert result['intent'] == 'navigational'
        assert result['confidence'] == 0.65

    def test_classify_definitional_query(self):
        """Test classification of definitional query"""
        self.mock_classifier.return_value = {
            'labels': ['definitional', 'informational', 'procedural'],
            'scores': [0.90, 0.07, 0.03]
        }

        query = "What does API mean?"
        result = self.classifier.classify_intent(query)

        assert result['intent'] == 'definitional'
        assert result['confidence'] == 0.90

    def test_classify_causal_query(self):
        """Test classification of causal/why query"""
        self.mock_classifier.return_value = {
            'labels': ['causal', 'informational', 'procedural'],
            'scores': [0.78, 0.15, 0.07]
        }

        query = "Why does Python use indentation for blocks?"
        result = self.classifier.classify_intent(query)

        assert result['intent'] == 'causal'
        assert result['confidence'] == 0.78

    def test_classify_transactional_query(self):
        """Test classification of transactional query"""
        self.mock_classifier.return_value = {
            'labels': ['transactional', 'procedural', 'informational'],
            'scores': [0.72, 0.20, 0.08]
        }

        query = "Deploy my application to production"
        result = self.classifier.classify_intent(query)

        assert result['intent'] == 'transactional'
        assert result['confidence'] == 0.72

    def test_classify_empty_query(self):
        """Test classification of empty query"""
        self.mock_classifier.return_value = {
            'labels': ['informational', 'procedural', 'definitional'],
            'scores': [0.40, 0.35, 0.25]
        }

        query = ""
        result = self.classifier.classify_intent(query)

        # Should still return a result, even if confidence is lower
        assert 'intent' in result
        assert 'confidence' in result

    def test_classify_low_confidence(self):
        """Test classification with low confidence scores"""
        self.mock_classifier.return_value = {
            'labels': ['informational', 'procedural', 'definitional'],
            'scores': [0.35, 0.33, 0.32]
        }

        query = "Some ambiguous query text"
        result = self.classifier.classify_intent(query)

        assert result['intent'] == 'informational'  # Should still return highest score
        assert result['confidence'] == 0.35
        assert result['confidence'] < 0.5  # But it's low

    def test_confidence_threshold(self):
        """Test that confidence scores are properly returned"""
        self.mock_classifier.return_value = {
            'labels': ['informational', 'procedural', 'definitional'],
            'scores': [0.95, 0.03, 0.02]
        }

        query = "What is Python?"
        result = self.classifier.classify_intent(query)

        assert result['confidence'] > 0.9  # High confidence
        assert result['intent'] == 'informational'

    def test_intent_labels_structure(self):
        """Test that intent labels are properly structured"""
        assert isinstance(self.classifier.intent_labels, dict)

        # Check all expected intents are present
        expected_intents = [
            'informational', 'procedural', 'causal', 'comparative',
            'recommendation', 'navigational', 'transactional', 'definitional'
        ]

        for intent in expected_intents:
            assert intent in self.classifier.intent_labels
            assert isinstance(self.classifier.intent_labels[intent], str)
            assert len(self.classifier.intent_labels[intent]) > 0

    def test_classifier_called_with_correct_params(self):
        """Test that the classifier is called with correct parameters"""
        self.mock_classifier.return_value = {
            'labels': ['informational'],
            'scores': [0.75]
        }

        query = "Test query"
        self.classifier.classify_intent(query)

        # Verify the classifier was called with the query and candidate labels
        self.mock_classifier.assert_called_once()
        call_args = self.mock_classifier.call_args
        assert call_args[0][0] == query  # First positional arg is the query
        assert 'candidate_labels' in call_args[1]  # candidate_labels in kwargs

    def test_result_includes_all_scores(self):
        """Test that result includes all intent scores"""
        self.mock_classifier.return_value = {
            'labels': ['informational', 'procedural', 'definitional'],
            'scores': [0.60, 0.25, 0.15]
        }

        query = "What is Python?"
        result = self.classifier.classify_intent(query)

        assert 'all_scores' in result
        assert len(result['all_scores']) == 3
        assert result['all_scores'][0]['intent'] == 'informational'
        assert result['all_scores'][0]['score'] == 0.60

    def test_query_preprocessing(self):
        """Test that queries with special characters are handled correctly"""
        self.mock_classifier.return_value = {
            'labels': ['informational'],
            'scores': [0.75]
        }

        # Test with special characters
        query = "What's the difference between `async` and `await`?"
        result = self.classifier.classify_intent(query)

        assert result is not None
        assert 'intent' in result
        assert 'confidence' in result

    def test_long_query_classification(self):
        """Test classification of very long queries"""
        self.mock_classifier.return_value = {
            'labels': ['procedural'],
            'scores': [0.68]
        }

        # Very long query
        query = " ".join(["word"] * 100)
        result = self.classifier.classify_intent(query)

        assert result is not None
        assert result['intent'] == 'procedural'

    def test_multilanguage_query_fallback(self):
        """Test handling of non-English queries"""
        self.mock_classifier.return_value = {
            'labels': ['informational'],
            'scores': [0.55]
        }

        # Non-English query
        query = "¿Qué es Python?"
        result = self.classifier.classify_intent(query)

        # Should still return a result (model might handle it)
        assert result is not None
        assert 'intent' in result
