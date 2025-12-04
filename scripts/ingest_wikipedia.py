#!/usr/bin/env python3
"""
Wikipedia Dataset Ingestion Script for RAG System

Downloads English Wikipedia from Hugging Face and ingests into the RAG system.
Supports filtering for articles with mathematical formulas (LaTeX).

Usage:
    python scripts/ingest_wikipedia.py --sample 100000
    python scripts/ingest_wikipedia.py --sample 100000 --math-only
    python scripts/ingest_wikipedia.py --full  # Warning: Very large!
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Generator
import time

# Third-party imports
try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Required packages not installed. Run:")
    print("  pip install datasets tqdm")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed. Run:")
    print("  pip install sentence-transformers")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:
    print("qdrant-client not installed. Run:")
    print("  pip install qdrant-client")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
@dataclass
class IngestionConfig:
    """Configuration for Wikipedia ingestion"""
    # Dataset settings
    dataset_name: str = "wikimedia/wikipedia"
    dataset_version: str = "20231101.en"  # English Wikipedia, Nov 2023
    sample_size: Optional[int] = 100000  # None for full dataset
    math_only: bool = False  # Filter for articles with math formulas

    # Chunking settings (optimized from benchmarks)
    chunk_size: int = 1024  # tokens
    chunk_overlap: int = 102  # ~10% overlap

    # Embedding settings (optimized from benchmarks)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768
    batch_size: int = 32

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6335
    collection_name: str = "wikipedia"

    # Output settings
    output_dir: str = "./data/wikipedia"
    save_raw: bool = True  # Save raw articles as JSON


@dataclass
class WikiArticle:
    """Represents a Wikipedia article"""
    id: str
    title: str
    text: str
    url: str
    has_math: bool
    math_formulas: List[str]
    categories: List[str]
    word_count: int


@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    chunk_id: str
    article_id: str
    article_title: str
    text: str
    chunk_index: int
    total_chunks: int
    has_math: bool
    math_formulas: List[str]
    metadata: Dict[str, Any]


class MathFormulaExtractor:
    """Extracts and processes mathematical formulas from text"""

    # LaTeX patterns commonly found in Wikipedia
    LATEX_PATTERNS = [
        r'\$\$([^$]+)\$\$',  # Display math: $$...$$
        r'\$([^$]+)\$',  # Inline math: $...$
        r'\\begin\{equation\}(.*?)\\end\{equation\}',  # equation environment
        r'\\begin\{align\}(.*?)\\end\{align\}',  # align environment
        r'\\begin\{math\}(.*?)\\end\{math\}',  # math environment
        r'<math[^>]*>(.*?)</math>',  # MediaWiki math tags
        r'\\frac\{[^}]+\}\{[^}]+\}',  # fractions
        r'\\sum_\{[^}]*\}',  # summations
        r'\\int_\{[^}]*\}',  # integrals
        r'\\sqrt\{[^}]+\}',  # square roots
        r'\\[a-zA-Z]+\{[^}]*\}',  # General LaTeX commands
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.DOTALL | re.IGNORECASE) for p in self.LATEX_PATTERNS]

    def extract_formulas(self, text: str) -> List[str]:
        """Extract all mathematical formulas from text"""
        formulas = []
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            formulas.extend(matches)
        return list(set(formulas))  # Remove duplicates

    def has_math(self, text: str) -> bool:
        """Check if text contains mathematical formulas"""
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        return False

    def clean_text_preserve_math(self, text: str) -> str:
        """Clean text while preserving mathematical notation"""
        # Remove HTML tags except math
        text = re.sub(r'<(?!math|/math)[^>]+>', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class TextChunker:
    """Chunks text into optimal sizes for embedding"""

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 102):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 chars)"""
        return len(text) // 4

    def chunk_text(self, text: str, preserve_sentences: bool = True) -> List[str]:
        """Split text into chunks with overlap"""
        if self._estimate_tokens(text) <= self.chunk_size:
            return [text]

        chunks = []

        if preserve_sentences:
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = []
            current_size = 0

            for sentence in sentences:
                sentence_size = self._estimate_tokens(sentence)

                if current_size + sentence_size > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(' '.join(current_chunk))

                    # Start new chunk with overlap
                    overlap_sentences = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        s_size = self._estimate_tokens(s)
                        if overlap_size + s_size <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += s_size
                        else:
                            break

                    current_chunk = overlap_sentences
                    current_size = overlap_size

                current_chunk.append(sentence)
                current_size += sentence_size

            if current_chunk:
                chunks.append(' '.join(current_chunk))
        else:
            # Simple character-based chunking
            char_chunk_size = self.chunk_size * 4
            char_overlap = self.chunk_overlap * 4

            start = 0
            while start < len(text):
                end = min(start + char_chunk_size, len(text))
                chunks.append(text[start:end])
                start = end - char_overlap

        return chunks


class WikipediaIngester:
    """Main class for ingesting Wikipedia into RAG system"""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.math_extractor = MathFormulaExtractor()
        self.chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.embedding_model = None
        self.qdrant_client = None

        # Statistics
        self.stats = {
            "total_articles": 0,
            "articles_with_math": 0,
            "total_chunks": 0,
            "total_formulas": 0,
            "start_time": None,
            "end_time": None
        }

    def initialize(self):
        """Initialize embedding model and Qdrant client"""
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)

        logger.info(f"Connecting to Qdrant: {self.config.qdrant_host}:{self.config.qdrant_port}")
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            timeout=60
        )

        # Create collection if needed
        self._ensure_collection()

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.config.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.config.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.embedding_dimension,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000
                )
            )
        else:
            logger.info(f"Collection {self.config.collection_name} already exists")

    def load_dataset(self) -> Iterator[Dict[str, Any]]:
        """Load Wikipedia dataset from Hugging Face"""
        logger.info(f"Loading dataset: {self.config.dataset_name}/{self.config.dataset_version}")

        # Load dataset with streaming for memory efficiency
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_version,
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        return iter(dataset)

    def process_article(self, raw_article: Dict[str, Any]) -> Optional[WikiArticle]:
        """Process a raw article into WikiArticle format"""
        try:
            text = raw_article.get("text", "")
            title = raw_article.get("title", "")

            if not text or not title:
                return None

            # Extract math formulas
            formulas = self.math_extractor.extract_formulas(text)
            has_math = len(formulas) > 0

            # Skip non-math articles if filtering
            if self.config.math_only and not has_math:
                return None

            # Clean text
            cleaned_text = self.math_extractor.clean_text_preserve_math(text)

            # Generate ID
            article_id = hashlib.md5(f"{title}:{raw_article.get('id', '')}".encode()).hexdigest()

            return WikiArticle(
                id=article_id,
                title=title,
                text=cleaned_text,
                url=raw_article.get("url", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
                has_math=has_math,
                math_formulas=formulas,
                categories=[],  # Categories not always in dataset
                word_count=len(cleaned_text.split())
            )
        except Exception as e:
            logger.warning(f"Error processing article: {e}")
            return None

    def chunk_article(self, article: WikiArticle) -> List[DocumentChunk]:
        """Chunk an article into document chunks"""
        text_chunks = self.chunker.chunk_text(article.text)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Check if this chunk contains math
            chunk_formulas = self.math_extractor.extract_formulas(chunk_text)

            chunk = DocumentChunk(
                chunk_id=f"{article.id}_chunk_{i}",
                article_id=article.id,
                article_title=article.title,
                text=chunk_text,
                chunk_index=i,
                total_chunks=len(text_chunks),
                has_math=len(chunk_formulas) > 0,
                math_formulas=chunk_formulas,
                metadata={
                    "source": "wikipedia",
                    "url": article.url,
                    "word_count": len(chunk_text.split()),
                    "article_has_math": article.has_math,
                    "total_article_formulas": len(article.math_formulas)
                }
            )
            chunks.append(chunk)

        return chunks

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """Generate embeddings for chunks"""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def store_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Store chunks and embeddings in Qdrant"""
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = models.PointStruct(
                id=hashlib.md5(chunk.chunk_id.encode()).hexdigest(),
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "article_id": chunk.article_id,
                    "title": chunk.article_title,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "has_math": chunk.has_math,
                    "math_formulas": chunk.math_formulas,
                    **chunk.metadata
                }
            )
            points.append(point)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.config.collection_name,
                points=batch
            )

    def save_article_json(self, article: WikiArticle, output_dir: Path):
        """Save article as JSON file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{article.id}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(article), f, ensure_ascii=False, indent=2)

    def ingest(self):
        """Main ingestion pipeline"""
        self.stats["start_time"] = datetime.now()
        logger.info("Starting Wikipedia ingestion...")

        # Initialize
        self.initialize()

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        dataset_iter = self.load_dataset()

        # Process articles
        chunk_buffer = []
        processed = 0

        pbar = tqdm(
            total=self.config.sample_size,
            desc="Processing articles",
            unit="articles"
        )

        for raw_article in dataset_iter:
            if self.config.sample_size and processed >= self.config.sample_size:
                break

            # Process article
            article = self.process_article(raw_article)
            if article is None:
                continue

            self.stats["total_articles"] += 1
            if article.has_math:
                self.stats["articles_with_math"] += 1
                self.stats["total_formulas"] += len(article.math_formulas)

            # Save raw article if configured
            if self.config.save_raw:
                self.save_article_json(article, output_dir / "raw")

            # Chunk article
            chunks = self.chunk_article(article)
            chunk_buffer.extend(chunks)
            self.stats["total_chunks"] += len(chunks)

            # Process chunks in batches
            if len(chunk_buffer) >= self.config.batch_size * 10:
                embeddings = self.embed_chunks(chunk_buffer)
                self.store_chunks(chunk_buffer, embeddings)
                chunk_buffer = []

            processed += 1
            pbar.update(1)

        # Process remaining chunks
        if chunk_buffer:
            embeddings = self.embed_chunks(chunk_buffer)
            self.store_chunks(chunk_buffer, embeddings)

        pbar.close()

        self.stats["end_time"] = datetime.now()
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        # Print summary
        logger.info("=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total articles processed: {self.stats['total_articles']}")
        logger.info(f"Articles with math: {self.stats['articles_with_math']}")
        logger.info(f"Total math formulas: {self.stats['total_formulas']}")
        logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Rate: {self.stats['total_articles'] / duration:.2f} articles/second")

        # Save stats
        stats_file = output_dir / "ingestion_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                **self.stats,
                "start_time": self.stats["start_time"].isoformat(),
                "end_time": self.stats["end_time"].isoformat(),
                "duration_seconds": duration,
                "config": asdict(self.config)
            }, f, indent=2)

        logger.info(f"Stats saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia from Hugging Face into RAG system"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=100000,
        help="Number of articles to process (default: 100000)"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Process full dataset (WARNING: ~6M articles, very large!)"
    )

    parser.add_argument(
        "--math-only",
        action="store_true",
        help="Only process articles containing mathematical formulas"
    )

    parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Qdrant host (default: localhost)"
    )

    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6335,
        help="Qdrant port (default: 6335)"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="wikipedia",
        help="Qdrant collection name (default: wikipedia)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/wikipedia",
        help="Output directory for raw data (default: ./data/wikipedia)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)"
    )

    parser.add_argument(
        "--no-save-raw",
        action="store_true",
        help="Don't save raw article JSON files"
    )

    args = parser.parse_args()

    # Build config
    config = IngestionConfig(
        sample_size=None if args.full else args.sample,
        math_only=args.math_only,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        save_raw=not args.no_save_raw
    )

    # Run ingestion
    ingester = WikipediaIngester(config)
    ingester.ingest()


if __name__ == "__main__":
    main()
