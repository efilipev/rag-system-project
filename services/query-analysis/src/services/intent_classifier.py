"""
ML-based Intent Classification Service
Improved intent detection using transformer models
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from transformers import pipeline

from src.core.config import settings
from src.core.logging import logger


class IntentClassifier:
    """
    ML-based intent classifier using zero-shot classification or trained models
    """

    def __init__(self):
        self.classifier = None
        self.model_type = settings.INTENT_MODEL_TYPE

        # Define intent labels and descriptions
        self.intent_labels = {
            "informational": "seeking information, facts, or explanations about a topic",
            "procedural": "asking how to do something, step-by-step instructions, or procedures",
            "causal": "asking why something happens, causes, or reasons",
            "comparative": "comparing different options, entities, or concepts",
            "recommendation": "seeking suggestions, recommendations, or best practices",
            "navigational": "looking for a specific resource, document, or location",
            "transactional": "intending to perform an action, make a purchase, or complete a task",
            "definitional": "asking for the definition or meaning of a term",
        }

        # Keywords for fallback classification (improved version)
        self.intent_keywords = {
            "informational": [
                "what", "who", "when", "where", "which", "is", "are", "does",
                "tell me", "information", "about", "explain", "describe"
            ],
            "procedural": [
                "how", "guide", "tutorial", "steps", "process", "way to",
                "instructions", "method", "procedure", "demonstrate"
            ],
            "causal": [
                "why", "reason", "cause", "because", "explanation", "purpose",
                "motivation", "rationale", "why does", "why is"
            ],
            "comparative": [
                "compare", "difference", "versus", "vs", "better", "best",
                "contrast", "alternative", "or", "which is", "either"
            ],
            "recommendation": [
                "recommend", "suggest", "should", "advice", "best", "good",
                "prefer", "choose", "selection", "opinion", "what would"
            ],
            "navigational": [
                "find", "locate", "where is", "link to", "navigate", "go to",
                "access", "reach", "page", "search for specific"
            ],
            "transactional": [
                "buy", "purchase", "order", "download", "install", "create",
                "register", "sign up", "submit", "apply"
            ],
            "definitional": [
                "define", "definition", "meaning", "what is", "what are",
                "means", "refers to", "terminology", "term"
            ],
        }

    async def initialize(self) -> None:
        """Initialize the intent classifier model"""
        try:
            if self.model_type == "zero-shot":
                logger.info("Initializing zero-shot intent classifier")
                # Use facebook/bart-large-mnli for zero-shot classification
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=-1  # Use CPU (change to 0 for GPU)
                )
                logger.info("Zero-shot intent classifier initialized successfully")

            elif self.model_type == "sklearn" and settings.INTENT_MODEL_PATH:
                logger.info(f"Loading sklearn intent classifier from {settings.INTENT_MODEL_PATH}")
                # TODO: Implement loading of trained sklearn model
                # import joblib
                # self.classifier = joblib.load(settings.INTENT_MODEL_PATH)
                raise NotImplementedError("Sklearn model loading not implemented yet")

            elif self.model_type == "custom" and settings.INTENT_MODEL_PATH:
                logger.info(f"Loading custom intent classifier from {settings.INTENT_MODEL_PATH}")
                # TODO: Implement loading of custom model
                raise NotImplementedError("Custom model loading not implemented yet")

            else:
                logger.warning(f"Unknown model type: {self.model_type}. Using keyword-based fallback.")
                self.classifier = None

        except Exception as e:
            logger.error(f"Failed to initialize intent classifier: {e}. Using keyword-based fallback.", exc_info=True)
            self.classifier = None

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify the intent of a query

        Args:
            query: The query text to classify

        Returns:
            Dictionary with intent and confidence score
        """
        try:
            if self.classifier and self.model_type == "zero-shot":
                return self._zero_shot_classify(query)
            else:
                return self._keyword_based_classify(query)

        except Exception as e:
            logger.error(f"Error classifying intent: {e}. Falling back to keyword-based.", exc_info=True)
            return self._keyword_based_classify(query)

    def _zero_shot_classify(self, query: str) -> Dict[str, Any]:
        """
        Classify intent using zero-shot classification

        Args:
            query: The query text

        Returns:
            Dictionary with intent and confidence
        """
        # Get label names
        candidate_labels = list(self.intent_labels.keys())

        # Use hypothesis template to improve accuracy
        hypothesis_template = "This query is {} in nature."

        # Perform zero-shot classification
        result = self.classifier(
            query,
            candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False  # Single intent classification
        )

        # Extract top prediction
        intent = result['labels'][0]
        confidence = result['scores'][0]

        # Get top 3 predictions for additional context
        predictions = [
            {"intent": label, "confidence": float(score)}
            for label, score in zip(result['labels'][:3], result['scores'][:3])
        ]

        return {
            "intent": intent,
            "confidence": float(confidence),
            "predictions": predictions,
            "method": "zero-shot"
        }

    def _keyword_based_classify(self, query: str) -> Dict[str, Any]:
        """
        Fallback keyword-based intent classification with improved scoring

        Args:
            query: The query text

        Returns:
            Dictionary with intent and confidence
        """
        query_lower = query.lower()
        query_words = query_lower.split()

        # Calculate scores for each intent
        intent_scores = {}

        for intent, keywords in self.intent_keywords.items():
            score = 0
            matched_keywords = []

            for keyword in keywords:
                if keyword in query_lower:
                    # Multi-word keywords get higher weight
                    weight = len(keyword.split())
                    score += weight

                    # Check if keyword is at the beginning (higher weight)
                    if query_lower.startswith(keyword):
                        score += 0.5

                    matched_keywords.append(keyword)

            intent_scores[intent] = score

        # Get intent with highest score
        if intent_scores and max(intent_scores.values()) > 0:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            top_intent, top_score = sorted_intents[0]

            # Calculate confidence based on score difference
            total_score = sum(intent_scores.values())
            confidence = top_score / total_score if total_score > 0 else 0.5

            # Normalize confidence to 0-1 range
            confidence = min(max(confidence, 0.3), 0.9)  # Cap between 0.3 and 0.9

            # Get top 3 predictions
            predictions = [
                {"intent": intent, "confidence": score / total_score if total_score > 0 else 0}
                for intent, score in sorted_intents[:3]
                if score > 0
            ]

            return {
                "intent": top_intent,
                "confidence": confidence,
                "predictions": predictions,
                "method": "keyword-based"
            }

        # Default to informational if no matches
        return {
            "intent": "informational",
            "confidence": 0.5,
            "predictions": [{"intent": "informational", "confidence": 0.5}],
            "method": "default"
        }

    def get_intent_description(self, intent: str) -> Optional[str]:
        """Get human-readable description of an intent"""
        return self.intent_labels.get(intent)

    async def classify_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple queries in batch for better performance

        Args:
            queries: List of query texts

        Returns:
            List of intent classification results
        """
        if not queries:
            return []

        try:
            if self.classifier and self.model_type == "zero-shot":
                candidate_labels = list(self.intent_labels.keys())
                hypothesis_template = "This query is {} in nature."

                results = []
                for query in queries:
                    result = self.classifier(
                        query,
                        candidate_labels,
                        hypothesis_template=hypothesis_template,
                        multi_label=False
                    )

                    results.append({
                        "intent": result['labels'][0],
                        "confidence": float(result['scores'][0]),
                        "predictions": [
                            {"intent": label, "confidence": float(score)}
                            for label, score in zip(result['labels'][:3], result['scores'][:3])
                        ],
                        "method": "zero-shot"
                    })

                return results
            else:
                # Use keyword-based for batch
                return [self._keyword_based_classify(query) for query in queries]

        except Exception as e:
            logger.error(f"Error in batch intent classification: {e}", exc_info=True)
            return [self._keyword_based_classify(query) for query in queries]


# Singleton instance
_intent_classifier: Optional[IntentClassifier] = None


async def get_intent_classifier() -> IntentClassifier:
    """Get or create the intent classifier singleton"""
    global _intent_classifier

    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
        await _intent_classifier.initialize()

    return _intent_classifier
