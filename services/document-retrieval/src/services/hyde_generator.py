"""
HyDE (Hypothetical Document Embeddings) Generator using Ollama

This module implements hypothetical document generation using Ollama
to bridge the vocabulary gap between queries and documents in specialized domains.

Based on the HyDE-ColBERT paper implementation:
- Generates diverse hypothetical answer documents
- Domain-specific prompts for optimal generation
- Quality filtering to remove low-quality hypotheticals
- Adaptive temperature sampling for diversity
"""

from typing import Dict, List, Optional, Tuple, Union

import httpx

from src.core.config import settings
from src.core.logging import logger


# Domain-specific system prompts for HyDE generation
DOMAIN_SYSTEM_PROMPTS: Dict[str, str] = {
    "general": """You are a helpful assistant providing informative answers to general knowledge questions.
Write in a clear, encyclopedic style similar to Wikipedia articles.
Include specific facts, examples, and relevant details in 2-4 well-structured sentences.""",

    "biomedical": """You are a medical researcher writing a scientific paper abstract for PubMed.
Write in the formal, evidence-based style of medical journal abstracts.

Your answer MUST include:
- Specific medical terminology and clinical concepts
- Mechanisms of action or pathophysiology when relevant
- Clinical significance and patient outcomes
- Quantitative data when applicable (percentages, dosages, statistical measures)
- References to research findings ("studies show", "clinical trials demonstrate", "research indicates")

Use technical medical vocabulary. Be precise and clinical.""",

    "scientific": """You are a researcher writing a technical abstract for an arxiv scientific paper.
Write in the formal, precise style of peer-reviewed scientific literature.

Your answer MUST include:
- Precise scientific terminology and technical concepts
- Specific methodologies or experimental approaches
- Quantitative results and measurements when possible
- Theoretical frameworks or scientific principles
- References to scientific concepts ("research demonstrates", "experiments show", "analysis reveals")

Use technical scientific jargon appropriately. Focus on measurable, observable facts.""",

    "financial": """You are a financial analyst writing a market research report excerpt for Bloomberg or Financial Times.
Write in the professional, data-driven style of financial analysis.

Your answer MUST include:
- Specific financial metrics (P/E ratios, returns, yields, volatility measures)
- Market mechanisms and economic relationships
- Regulatory or institutional context when relevant
- Concrete examples of instruments, strategies, or market events
- Quantitative data (percentages, dollar amounts, basis points)

Use financial industry terminology. Include specific numbers and metrics.""",

    "argumentative": """You are a philosophy researcher writing an entry for the Stanford Encyclopedia of Philosophy.
Write in the balanced, analytical style of academic philosophical discourse.

Your answer MUST include:
- Multiple perspectives on the issue
- Logical reasoning chains and argumentation structure
- Different types of evidence (empirical, logical, ethical, philosophical)
- Explicit counterarguments or alternative viewpoints
- Acknowledgment of complexity and nuance

Present balanced argumentation. Use philosophical terminology.""",
}

# Prompt variations for generating diverse hypotheticals
PROMPT_VARIATIONS = [
    "Focus on the underlying mechanisms, processes, and how things work. Include technical details and specific methodologies.",
    "Focus on practical implications, real-world examples, and concrete outcomes. Include quantitative data and specific cases.",
    "Focus on the broader context, related concepts, and theoretical frameworks. Include comparisons and key distinctions.",
    "Focus on recent findings, current understanding, and state-of-the-art knowledge. Include specific research results.",
    "Focus on causal relationships, factors, and explanatory principles. Include logical reasoning and evidence.",
]


class HyDEGenerator:
    """
    Generate hypothetical answer documents using Ollama.

    The HyDE technique generates synthetic documents that resemble potential answers
    to the query, helping bridge the vocabulary gap between queries and documents.

    Optimized based on hyde-colbert-paper benchmark results (Phase 5):
    - quality_threshold=0.7: filters to ~40% high-quality hypotheticals
    - n_hypotheticals=3: optimal balance of diversity and quality
    - temperatures=[0.3, 0.5, 0.7]: diversity through temperature variation
    """

    def __init__(
        self,
        ollama_base_url: Optional[str] = None,
        model: Optional[str] = None,
        n_hypotheticals: int = 3,
        temperatures: Optional[List[float]] = None,
        max_tokens: int = 256,
        quality_threshold: float = 0.7,
        timeout: float = 60.0,
    ):
        """
        Initialize HyDE generator with Ollama.

        Args:
            ollama_base_url: Ollama API base URL
            model: Ollama model to use
            n_hypotheticals: Number of hypothetical documents to generate
            temperatures: List of temperatures for diverse generation
            max_tokens: Maximum tokens per generated document
            quality_threshold: Minimum quality score to keep a hypothetical
            timeout: HTTP timeout in seconds
        """
        self.ollama_base_url = ollama_base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self.n_hypotheticals = n_hypotheticals
        self.temperatures = temperatures or settings.HYDE_TEMPERATURES
        self.max_tokens = max_tokens
        self.quality_threshold = quality_threshold
        self.timeout = timeout

        # Ensure we have enough temperatures
        while len(self.temperatures) < n_hypotheticals:
            self.temperatures.append(self.temperatures[-1])

        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "quality_filtered": 0,
        }

        logger.info(
            f"HyDE Generator initialized: model={self.model}, "
            f"n_hypotheticals={n_hypotheticals}, temperatures={self.temperatures[:n_hypotheticals]}"
        )

    async def _call_ollama(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.5,
    ) -> Optional[str]:
        """
        Call Ollama API to generate text.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Generation temperature

        Returns:
            Generated text or None if failed
        """
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "").strip()
        except httpx.TimeoutException:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return None

    def _get_prompt_for_domain(self, query: str, domain: str) -> Dict[str, str]:
        """
        Get the system and user prompts for a specific domain.

        Args:
            query: The user's question/query
            domain: The domain type

        Returns:
            Dictionary with 'system' and 'user' prompt strings
        """
        if domain not in DOMAIN_SYSTEM_PROMPTS:
            domain = "general"

        user_prompt = f"""Question: {query}

Please provide a direct, concise answer to this question. Write as if you are answering in a document that would be retrieved for this query.
Keep your response focused and informative, using 2-4 sentences."""

        return {
            "system": DOMAIN_SYSTEM_PROMPTS[domain],
            "user": user_prompt,
        }

    def _get_multi_hypothetical_prompts(
        self,
        query: str,
        domain: str,
        n_variations: int,
    ) -> List[Dict[str, str]]:
        """
        Generate multiple prompt variations to encourage diverse hypothetical documents.

        Args:
            query: The user's question/query
            domain: The domain type
            n_variations: Number of prompt variations to generate

        Returns:
            List of prompt dictionaries with different formulations
        """
        base_prompt = self._get_prompt_for_domain(query, domain)
        prompts = []

        for i in range(n_variations):
            variation_instruction = PROMPT_VARIATIONS[i % len(PROMPT_VARIATIONS)]
            modified_user_prompt = f"""Question: {query}

{variation_instruction}

Provide a focused, informative answer in 2-4 well-structured sentences. Use domain-appropriate terminology."""

            prompts.append({
                "system": base_prompt["system"],
                "user": modified_user_prompt,
            })

        return prompts

    def _compute_adaptive_temperatures(
        self,
        query: str,
        domain: str,
        n_temps: int,
    ) -> List[float]:
        """
        Compute adaptive temperature values based on query complexity and domain.

        Args:
            query: The user's query
            domain: Domain type
            n_temps: Number of temperature values to generate

        Returns:
            List of adaptive temperature values
        """
        words = query.split()
        word_count = len(words)

        # Factors indicating complexity
        has_technical_terms = any(word.istitle() or word.isupper() for word in words)
        has_numbers = any(char.isdigit() for char in query)
        has_multiple_clauses = query.count(',') + query.count(';') + query.count(' and ')

        # Complexity score (0.0 = simple, 1.0 = complex)
        complexity = 0.0

        if word_count < 5:
            complexity += 0.0
        elif word_count < 10:
            complexity += 0.3
        elif word_count < 20:
            complexity += 0.6
        else:
            complexity += 0.9

        if has_technical_terms:
            complexity += 0.1
        if has_numbers:
            complexity += 0.05
        if has_multiple_clauses >= 2:
            complexity += 0.15
        elif has_multiple_clauses == 1:
            complexity += 0.05

        complexity = min(1.0, complexity)

        # Domain-specific temperature adjustments
        domain_temp_adjustments = {
            'biomedical': -0.05,
            'scientific': -0.05,
            'financial': -0.1,
            'argumentative': +0.1,
            'general': 0.0,
        }

        base_adjustment = domain_temp_adjustments.get(domain, 0.0)

        # Compute temperature range based on complexity
        min_temp = 0.2 + complexity * 0.1 + base_adjustment
        max_temp = 0.4 + complexity * 0.4 + base_adjustment

        min_temp = max(0.1, min_temp)
        max_temp = min(0.9, max(min_temp + 0.2, max_temp))

        if n_temps == 1:
            temperatures = [(min_temp + max_temp) / 2]
        else:
            step = (max_temp - min_temp) / (n_temps - 1)
            temperatures = [min_temp + i * step for i in range(n_temps)]

        return temperatures

    def _score_hypothetical_quality(
        self,
        query: str,
        hypothetical: str,
        domain: str,
    ) -> float:
        """
        Score hypothetical quality using multiple heuristics.

        Quality indicators:
        1. Length (too short = generic/incomplete)
        2. Repetition (high repetition = degenerate)
        3. Generic phrases (indicates fallback-like quality)
        4. Domain-specific terminology (indicates relevance)
        5. Query overlap (too much = parroting, too little = off-topic)

        Args:
            query: Original query text
            hypothetical: Generated hypothetical document
            domain: Domain type

        Returns:
            Quality score 0.0-1.0 (higher = better quality)
        """
        score = 1.0

        # 1. Length check
        words = hypothetical.split()
        word_count = len(words)

        if word_count < 20:
            score *= 0.5
        if word_count < 10:
            score *= 0.3

        # 2. Repetition detection
        if word_count > 0:
            unique_words = set(w.lower() for w in words)
            unique_ratio = len(unique_words) / word_count

            if unique_ratio < 0.6:
                score *= 0.6
            if unique_ratio < 0.4:
                score *= 0.4

        # 3. Generic phrase detection
        hypo_lower = hypothetical.lower()
        generic_phrases = [
            'important aspects', 'relevant details', 'key considerations',
            'various factors', 'multiple perspectives', 'several aspects',
            'encompasses', 'involves', 'includes', 'regarding',
            'with respect to', 'in terms of', 'can be described',
            'it is important to note', 'worth noting', 'should be considered'
        ]

        generic_count = sum(1 for phrase in generic_phrases if phrase in hypo_lower)

        if generic_count > 3:
            score *= 0.7
        if generic_count > 5:
            score *= 0.5

        # 4. Domain-specific quality indicators
        domain_indicators = {
            'biomedical': [
                'study', 'studies', 'patient', 'patients', 'treatment', 'clinical',
                'medical', 'disease', 'symptom', 'symptoms', 'drug', 'medication',
                'therapy', 'diagnosis', 'condition', '%', 'mg', 'ml', 'dose',
            ],
            'scientific': [
                'research', 'study', 'experiment', 'data', 'result', 'results',
                'analysis', 'hypothesis', 'theory', 'observation', 'observations',
                'method', 'methods', 'process', 'mechanism', 'evidence', 'findings',
            ],
            'financial': [
                'market', 'markets', 'investment', 'return', 'returns', 'risk',
                'portfolio', 'price', 'prices', '%', 'percent', 'rate', 'rates',
                'cost', 'profit', 'loss', 'value', 'asset', 'assets', 'stock',
            ],
            'argumentative': [
                'argue', 'argues', 'argument', 'claim', 'claims', 'evidence',
                'however', 'although', 'despite', 'perspective', 'viewpoint',
                'contend', 'assert', 'maintain', 'support', 'oppose', 'believe',
            ],
            'general': [
                'example', 'such as', 'including', 'because', 'therefore',
                'however', 'although', 'additionally', 'furthermore', 'moreover'
            ]
        }

        indicators = domain_indicators.get(domain, domain_indicators['general'])
        if indicators:
            indicator_count = sum(1 for ind in indicators if ind in hypo_lower)
            indicator_ratio = indicator_count / len(indicators)
            domain_boost = 0.5 + 0.5 * min(indicator_ratio * 3, 1.0)
            score *= domain_boost

        # 5. Query overlap check
        query_words = set(query.lower().split())
        hypo_words = set(hypo_lower.split())

        if len(query_words) > 0:
            overlap = len(query_words & hypo_words) / len(query_words)

            if overlap > 0.8:
                score *= 0.5
            elif overlap < 0.1:
                score *= 0.6

        return max(0.0, min(1.0, score))

    def _fallback_response(self, query: str, domain: str = "general") -> str:
        """
        Generate a fallback response when generation fails.

        Args:
            query: Original query
            domain: Query domain

        Returns:
            Domain-appropriate fallback hypothetical
        """
        import random

        fallback_templates = {
            "biomedical": [
                f"Medical information regarding {query} includes various clinical considerations and treatment approaches that healthcare professionals should evaluate.",
                f"The medical topic of {query} involves important clinical factors, potential treatments, and evidence-based approaches used in healthcare settings.",
            ],
            "scientific": [
                f"Scientific research on {query} encompasses multiple theoretical frameworks, experimental methodologies, and empirical findings from peer-reviewed studies.",
                f"The scientific understanding of {query} involves various hypotheses, research methodologies, and data-driven conclusions.",
            ],
            "financial": [
                f"Financial considerations regarding {query} include market dynamics, investment strategies, and risk assessment factors that investors should evaluate.",
                f"The financial aspects of {query} involve economic principles, market mechanisms, and strategic investment considerations.",
            ],
            "argumentative": [
                f"The debate surrounding {query} involves multiple perspectives, logical arguments, and evidence-based reasoning from different stakeholders.",
                f"Arguments regarding {query} encompass various viewpoints, supporting evidence, and counterarguments that merit consideration.",
            ],
            "general": [
                f"Information about {query} includes several important aspects, relevant details, and key considerations that help answer this question.",
                f"The topic of {query} involves multiple factors, relevant information, and important details worth exploring.",
            ],
        }

        templates = fallback_templates.get(domain, fallback_templates["general"])
        return random.choice(templates)

    async def generate(
        self,
        query: str,
        domain: str = "general",
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a single hypothetical document for a query.

        Args:
            query: The user's question/query
            domain: Domain type (general, biomedical, scientific, financial, argumentative)
            temperature: Generation temperature

        Returns:
            Generated hypothetical document text
        """
        self.stats["total_requests"] += 1

        prompts = self._get_prompt_for_domain(query, domain)

        if temperature is None:
            adaptive_temps = self._compute_adaptive_temperatures(query, domain, 1)
            temperature = adaptive_temps[0]

        response = await self._call_ollama(
            prompt=prompts["user"],
            system_prompt=prompts["system"],
            temperature=temperature,
        )

        if response:
            self.stats["successful_generations"] += 1
            return response
        else:
            self.stats["failed_generations"] += 1
            return self._fallback_response(query, domain)

    async def generate_multiple(
        self,
        query: str,
        domain: str = "general",
        n_docs: Optional[int] = None,
        enable_filtering: bool = True,
    ) -> List[str]:
        """
        Generate multiple diverse hypothetical documents for a query with quality filtering.

        Args:
            query: The user's question/query
            domain: Domain type
            n_docs: Number of documents to return
            enable_filtering: Whether to apply quality filtering

        Returns:
            List of high-quality hypothetical document strings
        """
        n_docs = n_docs or self.n_hypotheticals
        n_generate = min(n_docs * 2, 6) if enable_filtering else n_docs

        logger.info(f"Generating {n_generate} hypotheticals for query: {query[:50]}...")

        hypotheticals_with_scores: List[Tuple[str, float]] = []
        prompt_variations = self._get_multi_hypothetical_prompts(query, domain, n_generate)
        adaptive_temps = self._compute_adaptive_temperatures(query, domain, n_generate)

        for i in range(n_generate):
            temp = adaptive_temps[i]
            prompt_dict = prompt_variations[i % len(prompt_variations)]

            self.stats["total_requests"] += 1
            response = await self._call_ollama(
                prompt=prompt_dict["user"],
                system_prompt=prompt_dict["system"],
                temperature=temp,
            )

            if response:
                self.stats["successful_generations"] += 1
                if enable_filtering:
                    quality_score = self._score_hypothetical_quality(query, response, domain)
                    hypotheticals_with_scores.append((response, quality_score))
                    logger.debug(
                        f"Generated hypothetical {i+1}/{n_generate} "
                        f"(temp={temp:.1f}, quality={quality_score:.3f})"
                    )
                else:
                    hypotheticals_with_scores.append((response, 1.0))
            else:
                self.stats["failed_generations"] += 1

        if not hypotheticals_with_scores:
            logger.error("No hypotheticals generated successfully! Using fallback.")
            return [self._fallback_response(query, domain) for _ in range(n_docs)]

        # Filter by quality threshold
        if enable_filtering:
            filtered = [(h, s) for h, s in hypotheticals_with_scores if s >= self.quality_threshold]

            if len(filtered) < n_docs:
                logger.warning(
                    f"Only {len(filtered)}/{n_generate} hypotheticals passed quality threshold "
                    f"({self.quality_threshold:.2f}). Requested {n_docs}."
                )
                if len(filtered) == 0:
                    filtered = hypotheticals_with_scores

            self.stats["quality_filtered"] += len(hypotheticals_with_scores) - len(filtered)

            filtered.sort(key=lambda x: x[1], reverse=True)
            final_hypotheticals = [h for h, s in filtered[:n_docs]]

            avg_quality = sum(s for _, s in filtered[:n_docs]) / len(filtered[:n_docs])
            logger.info(
                f"Quality filtering: Generated {n_generate}, "
                f"filtered to {len(final_hypotheticals)} (avg quality: {avg_quality:.3f})"
            )
        else:
            final_hypotheticals = [h for h, _ in hypotheticals_with_scores[:n_docs]]

        return final_hypotheticals

    async def generate_with_fusion(
        self,
        query: str,
        domain: str = "general",
        n_docs: Optional[int] = None,
    ) -> Dict[str, Union[str, List[str], int]]:
        """
        Generate hypothetical documents and return both query and hypotheticals.

        This method returns a dictionary suitable for fusion-based retrieval.

        Args:
            query: The user's question/query
            domain: Domain type
            n_docs: Number of hypothetical documents

        Returns:
            Dictionary with 'query', 'hypotheticals', 'domain', and 'n_generated' keys
        """
        hypotheticals = await self.generate_multiple(query, domain, n_docs)

        return {
            "query": query,
            "hypotheticals": hypotheticals,
            "domain": domain,
            "n_generated": len(hypotheticals),
        }

    def get_stats(self) -> Dict[str, int]:
        """Get generation statistics."""
        return dict(self.stats)

    async def health_check(self) -> bool:
        """
        Check if Ollama is available and the model is loaded.

        Returns:
            True if healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_base_url}/api/tags")
                response.raise_for_status()
                data = response.json()

                models = [m.get("name", "") for m in data.get("models", [])]
                model_base = self.model.split(":")[0]

                for model in models:
                    if model.startswith(model_base):
                        logger.info(f"Ollama health check passed: model {model} available")
                        return True

                logger.warning(f"Model {self.model} not found in Ollama. Available: {models}")
                return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False


# Singleton instance
_hyde_generator: Optional[HyDEGenerator] = None


async def get_hyde_generator() -> HyDEGenerator:
    """Get or create the HyDE generator singleton."""
    global _hyde_generator
    if _hyde_generator is None:
        _hyde_generator = HyDEGenerator()
    return _hyde_generator
