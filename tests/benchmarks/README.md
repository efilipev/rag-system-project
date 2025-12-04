# RAG System Benchmarks

Comprehensive benchmarking suite for evaluating and optimizing RAG (Retrieval-Augmented Generation) systems.

## Overview

This benchmarking suite provides production-grade tests for:

- ✅ **Chunking Sensitivity** - Test how chunk size affects retrieval quality
- ✅ **Embedding Quantization** - Compare quality vs storage cost tradeoffs
- ✅ **BEIR Evaluation** - Standard IR benchmark datasets
- ✅ **Advanced Techniques** - Hybrid search, index types, multi-representation
- ✅ **Ablation Studies** - Measure cumulative impact of optimizations

## Quick Start

### 1. Install Dependencies

```bash
pip install pytest pytest-asyncio pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run Benchmarks

```bash
# Run all quick tests (5 minutes)
python run_benchmarks.py --quick

# Run specific benchmarks
python run_benchmarks.py --chunking       # Chunking sensitivity
python run_benchmarks.py --quantization   # Embedding quantization
python run_benchmarks.py --advanced       # Advanced techniques
python run_benchmarks.py --ablation       # Ablation study

# Run full benchmark suite (2-4 hours)
python run_benchmarks.py --full
```

### 3. View Results

```bash
# Results are saved to:
test_results/
├── chunking/
│   ├── chunk_size_sensitivity.png
│   └── chunk_size_sensitivity.csv
├── quantization/
│   ├── quality_vs_storage.png
│   └── quantization_results.csv
└── advanced/
    ├── hybrid_search.png
    └── hybrid_search.csv
```

## Benchmark Descriptions

### 1. Chunking Sensitivity Benchmark

Tests how chunk size affects retrieval quality across different embedding models.

**File:** `chunking_sensitivity_benchmark.py`

**Tests:**
- Chunk sizes: 64, 128, 256, 512, 1024, 2048 tokens
- Overlap percentages: 0%, 10%, 25%, 50%
- Chunking strategies: fixed, sentence, semantic
- Context-aware vs standard embeddings

**Key Metrics:**
- nDCG@10 variance across chunk sizes
- Optimal chunk size per model
- Stability analysis

**Example Output:**
```
voyage-context-3: Variance = 2.06% (STABLE)
voyage-3-large:   Variance = 4.34% (SENSITIVE)

At 64-token chunks: voyage-context-3 outperforms by 6.63%
```

**Usage:**
```python
from tests.benchmarks.chunking_sensitivity_benchmark import ChunkingSensitivityBenchmark

benchmark = ChunkingSensitivityBenchmark(
    retriever=your_retriever,
    embedding_models=["voyage-context-3", "voyage-3-large"]
)

results = await benchmark.test_chunk_size_sensitivity(
    corpus=corpus,
    queries=queries,
    qrels=qrels
)

benchmark.plot_chunk_size_sensitivity(results)
```

### 2. Embedding Quantization Benchmark

Compares quality vs storage cost for different quantization methods.

**File:** `embedding_quantization_benchmark.py`

**Quantization Types:**
- Float32 (baseline)
- Float16 (2x compression)
- Int8 (4x compression)
- Binary (32x compression)
- Product Quantization (10-100x compression)

**Key Metrics:**
- Compression ratio
- Quality loss percentage
- Storage cost (MB)
- Efficiency score (quality per storage unit)

**Example Output:**
```
voyage-context-3:
  float:   NDCG@10 = 79.5%, Storage = 400 MB
  int8:    NDCG@10 = 77.8%, Storage = 100 MB (2.1% loss)
  binary:  NDCG@10 = 71.5%, Storage = 12.5 MB (10.1% loss)
```

**Usage:**
```python
from tests.benchmarks.embedding_quantization_benchmark import EmbeddingQuantizationBenchmark

benchmark = EmbeddingQuantizationBenchmark(
    retriever=your_retriever,
    embedding_models=["voyage-context-3"],
    num_documents=100000
)

results = await benchmark.test_all_quantization_methods(
    corpus=corpus,
    queries=queries,
    qrels=qrels
)

benchmark.plot_quality_vs_storage(results)
```

### 3. Advanced Benchmarks

Tests advanced RAG techniques beyond basic retrieval.

**File:** `advanced_benchmarks.py`

**Benchmarks:**
- **Hybrid Search**: Dense + Sparse (BM25) combinations
- **Index Types**: HNSW, IVF, Flat comparison
- **Multi-Representation**: Parent-child, late chunking
- **Context Length**: Impact of context window size
- **Matryoshka Embeddings**: Multi-resolution embeddings

**Usage:**
```python
from tests.benchmarks.advanced_benchmarks import AdvancedBenchmarks

benchmark = AdvancedBenchmarks(retriever=your_retriever)

# Hybrid search
hybrid_results = await benchmark.benchmark_hybrid_search(
    corpus=corpus,
    queries=queries,
    qrels=qrels
)

# Index types
index_results = await benchmark.benchmark_index_types(
    corpus=corpus,
    queries=queries,
    qrels=qrels
)
```

### 4. BEIR Evaluation

Standard benchmark using 13 information retrieval datasets.

**File:** `beir_evaluation.py`

**Datasets:**
- MS MARCO
- TREC-COVID
- NFCorpus
- Natural Questions
- HotpotQA
- FiQA
- ArguAna
- Touché-2020
- SciFact
- SCIDOCS
- DBPedia
- Quora
- Climate-FEVER

**Usage:**
```python
from tests.benchmarks.beir_evaluation import BEIRBenchmark

benchmark = BEIRBenchmark(retriever=your_retriever)

results = await benchmark.evaluate_on_beir(
    datasets=["msmarco", "nfcorpus"],
    embedding_model="voyage-context-3"
)
```

## Running Tests

### Quick Test (5 minutes)

Tests with mock data to validate implementation:

```bash
pytest tests/benchmarks/test_comprehensive_benchmarks.py -v
```

### Individual Tests

```bash
# Chunking sensitivity
pytest tests/benchmarks/test_comprehensive_benchmarks.py::TestComprehensiveBenchmarks::test_chunking_sensitivity -v -s

# Embedding quantization
pytest tests/benchmarks/test_comprehensive_benchmarks.py::TestComprehensiveBenchmarks::test_embedding_quantization -v -s

# Advanced benchmarks
pytest tests/benchmarks/test_comprehensive_benchmarks.py::TestComprehensiveBenchmarks::test_advanced_benchmarks -v -s

# Ablation study
pytest tests/benchmarks/test_comprehensive_benchmarks.py::test_ablation_study -v -s
```

### Full Benchmark Suite (2-4 hours)

```bash
pytest tests/benchmarks/test_comprehensive_benchmarks.py::TestComprehensiveBenchmarks::test_full_benchmark_suite -v -s -m slow
```

## Understanding Results

### Chunking Sensitivity

**Low Variance (< 3%)**: Model is stable across chunk sizes
**High Variance (> 4%)**: Model is sensitive to chunk size

**Recommendation**: Use context-aware models (voyage-context-3) for stable performance across chunk sizes.

### Embedding Quantization

**Best Overall**: Int8 quantization (4x compression, <5% quality loss)
**Best Compression**: Binary quantization (32x compression, ~10% quality loss)
**Production Use**: Int8 for most cases, binary for latency-critical applications

### Hybrid Search

**Dense Only**: Best for semantic similarity
**Sparse Only (BM25)**: Best for keyword matching
**Hybrid 70/30**: Best overall performance (+10-15% over dense-only)
**Hybrid RRF**: Best when combining diverse signals

## Expected Performance Improvements

| Optimization | nDCG@10 Improvement | Storage Impact | Latency Impact |
|--------------|---------------------|----------------|----------------|
| Optimal Chunking (256→512) | +5-10% | No change | No change |
| Context-Aware Model | +3-7% | No change | +10% |
| Hybrid Search (70/30) | +10-15% | No change | +20% |
| Cross-Encoder Rerank | +15-25% | No change | +200ms |
| RAG-Fusion | +10-20% | No change | +3x |
| Int8 Quantization | -2-5% | -75% | -10% |
| Binary Quantization | -10-15% | -96.9% | -30% |

**Combined Optimal Config**: +30-40% quality improvement with 75% storage reduction

## Ablation Study

Measures cumulative impact of optimizations:

```bash
python run_benchmarks.py --ablation
```

**Expected Output:**
```
Baseline                 : nDCG@10 = 0.420 ( +0.0%)
+ Optimal Chunking       : nDCG@10 = 0.480 (+14.3%)
+ Hybrid Search          : nDCG@10 = 0.540 (+28.6%)
+ Quantization           : nDCG@10 = 0.530 (+26.2%)
+ Reranking              : nDCG@10 = 0.630 (+50.0%)
```

## Files

```
tests/benchmarks/
├── README.md                                  # This file
├── chunking_sensitivity_benchmark.py          # Chunking benchmarks (650 lines)
├── embedding_quantization_benchmark.py        # Quantization benchmarks (550 lines)
├── advanced_benchmarks.py                     # Advanced techniques (500 lines)
├── beir_evaluation.py                         # BEIR integration (600 lines)
├── test_rag_techniques.py                     # RAG technique tests (400 lines)
├── test_comprehensive_benchmarks.py           # Integration tests (450 lines)
└── visualization_dashboard.py                 # Visualization tools (450 lines)

docs/
├── COMPREHENSIVE_BENCHMARK_GUIDE.md           # Complete guide (600 lines)
├── BENCHMARKING_COMPLETE.md                   # Implementation summary (500 lines)
├── ADVANCED_RAG_GUIDE.md                      # Advanced RAG guide (500 lines)
└── LATEX_QUERY_INTEGRATION_GUIDE.md          # LaTeX integration (400 lines)

run_benchmarks.py                              # Convenience runner script
```

## Troubleshooting

### Import Errors

Ensure `__init__.py` files exist:
```bash
touch tests/__init__.py
touch tests/benchmarks/__init__.py
```

### Missing Dependencies

```bash
pip install pytest pytest-asyncio pandas numpy matplotlib seaborn scikit-learn
```

### Slow Tests

Use `--quick` flag for fast validation:
```bash
python run_benchmarks.py --quick
```

### Memory Issues

For large benchmarks, reduce batch size or number of documents in the test configuration.

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@software{rag_benchmarks_2025,
  title = {Comprehensive RAG System Benchmarks},
  author = {Your Team},
  year = {2025},
  url = {https://github.com/your-repo}
}
```

## References

- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
- [Voyage AI: Context-Aware Embeddings](https://www.voyageai.com)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Cohere Embeddings](https://docs.cohere.com/docs/embeddings)

## License

MIT License - See LICENSE file for details
