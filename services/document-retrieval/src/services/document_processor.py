"""
Document Processor Service

Handles document processing, chunking, and preparation for embedding.
Optimized settings based on benchmark results:
- Chunk size: 1024 tokens
- Chunk overlap: 10% (102 tokens)
- Preserves mathematical formulas (LaTeX)
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import logging

from src.core.config import settings

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Supported document types"""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    WIKIPEDIA = "wikipedia"
    LATEX = "latex"


@dataclass
class ProcessedChunk:
    """Represents a processed document chunk ready for embedding"""
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Content analysis
    has_math: bool = False
    math_formulas: List[str] = field(default_factory=list)
    has_code: bool = False
    code_blocks: List[str] = field(default_factory=list)

    # Position information
    start_char: int = 0
    end_char: int = 0
    word_count: int = 0
    token_estimate: int = 0


@dataclass
class ProcessedDocument:
    """Represents a fully processed document"""
    document_id: str
    title: str
    source: str
    document_type: DocumentType
    chunks: List[ProcessedChunk]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    total_chunks: int = 0
    total_words: int = 0
    total_math_formulas: int = 0
    total_code_blocks: int = 0
    processing_time_ms: float = 0


class MathExtractor:
    """Extracts and analyzes mathematical content"""

    # LaTeX/Math patterns
    PATTERNS = {
        "display_math": re.compile(r'\$\$([^$]+)\$\$', re.DOTALL),
        "inline_math": re.compile(r'\$([^$\n]+)\$'),
        "latex_equation": re.compile(r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}', re.DOTALL),
        "latex_align": re.compile(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', re.DOTALL),
        "latex_math": re.compile(r'\\begin\{math\}(.*?)\\end\{math\}', re.DOTALL),
        "mediawiki_math": re.compile(r'<math[^>]*>(.*?)</math>', re.DOTALL | re.IGNORECASE),
        "fraction": re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}'),
        "sqrt": re.compile(r'\\sqrt(?:\[([^\]]+)\])?\{([^}]+)\}'),
        "sum": re.compile(r'\\sum(?:_\{([^}]*)\})?(?:\^\{([^}]*)\})?'),
        "integral": re.compile(r'\\int(?:_\{([^}]*)\})?(?:\^\{([^}]*)\})?'),
        "matrix": re.compile(r'\\begin\{[pbvBV]?matrix\}(.*?)\\end\{[pbvBV]?matrix\}', re.DOTALL),
    }

    # Common math symbols that indicate mathematical content
    MATH_INDICATORS = [
        '∫', '∑', '∏', '√', '∞', '≠', '≤', '≥', '±', '×', '÷',
        '∂', '∇', '∈', '∉', '⊂', '⊃', '∪', '∩', '∧', '∨',
        'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ω',
        '→', '←', '↔', '⇒', '⇐', '⇔',
    ]

    def extract_all(self, text: str) -> Tuple[List[str], bool]:
        """
        Extract all mathematical formulas from text

        Returns:
            Tuple of (list of formulas, has_math boolean)
        """
        formulas = []

        for name, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                if isinstance(matches[0], tuple):
                    # Some patterns return groups
                    formulas.extend([m[0] if m[0] else m[1] for m in matches if any(m)])
                else:
                    formulas.extend(matches)

        # Check for Unicode math symbols
        has_symbols = any(symbol in text for symbol in self.MATH_INDICATORS)

        # Deduplicate
        formulas = list(set(formulas))

        has_math = len(formulas) > 0 or has_symbols

        return formulas, has_math

    def preserve_math_markers(self, text: str) -> str:
        """
        Add special markers around math content to preserve during chunking
        """
        # Mark display math
        text = self.PATTERNS["display_math"].sub(r'[MATH_START]\1[MATH_END]', text)
        text = self.PATTERNS["latex_equation"].sub(r'[MATH_START]\1[MATH_END]', text)
        text = self.PATTERNS["latex_align"].sub(r'[MATH_START]\1[MATH_END]', text)

        return text


class CodeExtractor:
    """Extracts code blocks from text"""

    PATTERNS = {
        "fenced_code": re.compile(r'```(\w*)\n?(.*?)```', re.DOTALL),
        "indented_code": re.compile(r'^(?:    |\t)(.+)$', re.MULTILINE),
        "inline_code": re.compile(r'`([^`]+)`'),
    }

    def extract_all(self, text: str) -> Tuple[List[str], bool]:
        """
        Extract all code blocks from text

        Returns:
            Tuple of (list of code blocks, has_code boolean)
        """
        code_blocks = []

        # Fenced code blocks
        fenced = self.PATTERNS["fenced_code"].findall(text)
        code_blocks.extend([code for _, code in fenced if code.strip()])

        has_code = len(code_blocks) > 0

        return code_blocks, has_code


class TextChunker:
    """
    Chunks text into optimal sizes for embedding

    Optimized settings from benchmarks:
    - chunk_size: 1024 tokens
    - chunk_overlap: 102 tokens (10%)
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        preserve_sentences: bool = True,
        preserve_paragraphs: bool = True
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.preserve_sentences = preserve_sentences
        self.preserve_paragraphs = preserve_paragraphs

        # Approximate tokens to characters ratio
        self.chars_per_token = 4

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text"""
        return len(text) // self.chars_per_token

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Handle common abbreviations
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g)\.\s', r'\1[DOT] ', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore abbreviations
        sentences = [s.replace('[DOT]', '.') for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Chunk text into overlapping segments

        Returns:
            List of tuples (chunk_text, start_char, end_char)
        """
        if self.estimate_tokens(text) <= self.chunk_size:
            return [(text, 0, len(text))]

        chunks = []

        if self.preserve_paragraphs:
            units = self._split_into_paragraphs(text)
            unit_type = "paragraph"
        elif self.preserve_sentences:
            units = self._split_into_sentences(text)
            unit_type = "sentence"
        else:
            # Character-based chunking
            return self._chunk_by_chars(text)

        current_chunk = []
        current_tokens = 0
        current_start = 0

        char_pos = 0
        for unit in units:
            unit_tokens = self.estimate_tokens(unit)

            # If single unit exceeds chunk size, split it further
            if unit_tokens > self.chunk_size:
                if current_chunk:
                    chunk_text = self._join_units(current_chunk, unit_type)
                    chunks.append((chunk_text, current_start, char_pos))
                    current_chunk = []
                    current_tokens = 0
                    current_start = char_pos

                # Recursively chunk the large unit
                if unit_type == "paragraph":
                    sub_chunks = self._chunk_sentences(unit)
                else:
                    sub_chunks = self._chunk_by_chars(unit)

                for sub_text, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_text, char_pos + sub_start, char_pos + sub_end))

                char_pos += len(unit) + 1
                continue

            # Check if adding this unit exceeds chunk size
            if current_tokens + unit_tokens > self.chunk_size and current_chunk:
                chunk_text = self._join_units(current_chunk, unit_type)
                chunks.append((chunk_text, current_start, char_pos))

                # Calculate overlap
                overlap_units = []
                overlap_tokens = 0
                for u in reversed(current_chunk):
                    u_tokens = self.estimate_tokens(u)
                    if overlap_tokens + u_tokens <= self.chunk_overlap:
                        overlap_units.insert(0, u)
                        overlap_tokens += u_tokens
                    else:
                        break

                current_chunk = overlap_units
                current_tokens = overlap_tokens
                current_start = char_pos - sum(len(u) + 1 for u in overlap_units)

            current_chunk.append(unit)
            current_tokens += unit_tokens
            char_pos += len(unit) + 1

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = self._join_units(current_chunk, unit_type)
            chunks.append((chunk_text, current_start, char_pos))

        return chunks

    def _chunk_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text by sentences"""
        sentences = self._split_into_sentences(text)
        return self._chunk_units(sentences, " ")

    def _chunk_by_chars(self, text: str) -> List[Tuple[str, int, int]]:
        """Simple character-based chunking"""
        char_chunk_size = self.chunk_size * self.chars_per_token
        char_overlap = self.chunk_overlap * self.chars_per_token

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + char_chunk_size, len(text))

            # Try to break at word boundary
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunks.append((text[start:end].strip(), start, end))
            start = end - char_overlap

        return chunks

    def _chunk_units(self, units: List[str], separator: str) -> List[Tuple[str, int, int]]:
        """Chunk a list of units with overlap"""
        chunks = []
        current = []
        current_tokens = 0
        char_pos = 0

        for unit in units:
            unit_tokens = self.estimate_tokens(unit)

            if current_tokens + unit_tokens > self.chunk_size and current:
                chunk_text = separator.join(current)
                chunks.append((chunk_text, char_pos - len(chunk_text), char_pos))

                # Overlap
                overlap = []
                overlap_tokens = 0
                for u in reversed(current):
                    if overlap_tokens + self.estimate_tokens(u) <= self.chunk_overlap:
                        overlap.insert(0, u)
                        overlap_tokens += self.estimate_tokens(u)
                    else:
                        break
                current = overlap
                current_tokens = overlap_tokens

            current.append(unit)
            current_tokens += unit_tokens
            char_pos += len(unit) + len(separator)

        if current:
            chunk_text = separator.join(current)
            chunks.append((chunk_text, char_pos - len(chunk_text), char_pos))

        return chunks

    def _join_units(self, units: List[str], unit_type: str) -> str:
        """Join units with appropriate separator"""
        if unit_type == "paragraph":
            return "\n\n".join(units)
        return " ".join(units)


class DocumentProcessor:
    """
    Main document processing service

    Processes documents through:
    1. Content extraction
    2. Math/code detection
    3. Chunking with overlap
    4. Metadata enrichment
    """

    def __init__(self):
        self.chunker = TextChunker()
        self.math_extractor = MathExtractor()
        self.code_extractor = CodeExtractor()

    def process(
        self,
        text: str,
        title: str = "",
        source: str = "",
        document_type: DocumentType = DocumentType.TEXT,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """
        Process a document into chunks

        Args:
            text: Document text
            title: Document title
            source: Source identifier (URL, filename, etc.)
            document_type: Type of document
            metadata: Additional metadata

        Returns:
            ProcessedDocument with chunks ready for embedding
        """
        start_time = datetime.now()
        metadata = metadata or {}

        # Generate document ID
        doc_id = hashlib.md5(f"{title}:{source}:{text[:100]}".encode()).hexdigest()

        # Extract math and code
        math_formulas, has_math = self.math_extractor.extract_all(text)
        code_blocks, has_code = self.code_extractor.extract_all(text)

        # Chunk the document
        raw_chunks = self.chunker.chunk(text)

        # Create processed chunks
        chunks = []
        for i, (chunk_text, start_char, end_char) in enumerate(raw_chunks):
            # Analyze chunk content
            chunk_math, chunk_has_math = self.math_extractor.extract_all(chunk_text)
            chunk_code, chunk_has_code = self.code_extractor.extract_all(chunk_text)

            chunk = ProcessedChunk(
                chunk_id=f"{doc_id}_chunk_{i}",
                document_id=doc_id,
                text=chunk_text,
                chunk_index=i,
                total_chunks=len(raw_chunks),
                has_math=chunk_has_math,
                math_formulas=chunk_math,
                has_code=chunk_has_code,
                code_blocks=chunk_code,
                start_char=start_char,
                end_char=end_char,
                word_count=len(chunk_text.split()),
                token_estimate=self.chunker.estimate_tokens(chunk_text),
                metadata={
                    "title": title,
                    "source": source,
                    "document_type": document_type.value,
                    **metadata
                }
            )
            chunks.append(chunk)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return ProcessedDocument(
            document_id=doc_id,
            title=title,
            source=source,
            document_type=document_type,
            chunks=chunks,
            metadata={
                "has_math": has_math,
                "has_code": has_code,
                **metadata
            },
            total_chunks=len(chunks),
            total_words=len(text.split()),
            total_math_formulas=len(math_formulas),
            total_code_blocks=len(code_blocks),
            processing_time_ms=processing_time
        )

    def process_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[ProcessedDocument]:
        """
        Process multiple documents

        Args:
            documents: List of dicts with 'text', 'title', 'source', etc.

        Returns:
            List of ProcessedDocuments
        """
        results = []

        for doc in documents:
            processed = self.process(
                text=doc.get("text", ""),
                title=doc.get("title", ""),
                source=doc.get("source", ""),
                document_type=DocumentType(doc.get("type", "text")),
                metadata=doc.get("metadata", {})
            )
            results.append(processed)

        return results


# Singleton instance
_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """Get or create the document processor singleton"""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor
