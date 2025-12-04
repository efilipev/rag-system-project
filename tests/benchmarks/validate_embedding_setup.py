"""
Validate Embedding Benchmark Setup

This script validates that all components are working correctly before
running the full benchmark.

Usage:
    python tests/benchmarks/validate_embedding_setup.py
"""
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks.embedding_model_comparison_benchmark import EMBEDDING_MODELS

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def validate_models():
    """Validate all models are configured correctly"""
    logger.info("="*80)
    logger.info("VALIDATING EMBEDDING MODELS")
    logger.info("="*80)

    logger.info(f"\nTotal models configured: {len(EMBEDDING_MODELS)}\n")

    issues = []

    for i, model in enumerate(EMBEDDING_MODELS, 1):
        logger.info(f"{i}. {model.name}")
        logger.info(f"   Model ID: {model.model_id}")
        logger.info(f"   Provider: {model.provider.value}")
        logger.info(f"   Dimension: {model.dimension}")
        logger.info(f"   Max Tokens: {model.max_tokens}")
        logger.info(f"   Specialization: {model.specialization}")
        logger.info(f"   Open Source: {model.is_open_source}")

        # Validate configuration
        if not model.name:
            issues.append(f"Model {i}: Missing name")
        if not model.model_id:
            issues.append(f"Model {i}: Missing model_id")
        if model.dimension <= 0:
            issues.append(f"Model {i}: Invalid dimension")
        if model.max_tokens <= 0:
            issues.append(f"Model {i}: Invalid max_tokens")

        logger.info("")

    if issues:
        logger.error("\n❌ VALIDATION FAILED:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("✓ All models configured correctly\n")
        return True


def test_sentence_transformers():
    """Test that sentence-transformers is installed and working"""
    logger.info("="*80)
    logger.info("TESTING SENTENCE-TRANSFORMERS")
    logger.info("="*80 + "\n")

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("✓ sentence-transformers imported successfully")

        # Test loading a small model
        logger.info("\nTesting model loading (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("✓ Model loaded successfully")

        # Test encoding
        logger.info("\nTesting encoding...")
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        logger.info(f"✓ Encoding successful: shape={embedding.shape}")

        logger.info("")
        return True

    except ImportError as e:
        logger.error(f"❌ sentence-transformers not installed: {e}")
        logger.error("Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing sentence-transformers: {e}")
        return False


def test_beir():
    """Test that BEIR is installed"""
    logger.info("="*80)
    logger.info("TESTING BEIR")
    logger.info("="*80 + "\n")

    try:
        import beir
        from beir.datasets.data_loader import GenericDataLoader
        logger.info(f"✓ BEIR imported successfully (version: {beir.__version__ if hasattr(beir, '__version__') else 'unknown'})")
        logger.info("")
        return True
    except ImportError as e:
        logger.error(f"❌ BEIR not installed: {e}")
        logger.error("Install with: pip install beir")
        return False


def test_dependencies():
    """Test all required dependencies"""
    logger.info("="*80)
    logger.info("TESTING DEPENDENCIES")
    logger.info("="*80 + "\n")

    dependencies = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "torch": "torch"
    }

    all_ok = True

    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"❌ {name} not installed")
            all_ok = False

    logger.info("")
    return all_ok


def print_model_summary():
    """Print summary of models by category"""
    logger.info("="*80)
    logger.info("MODEL SUMMARY")
    logger.info("="*80 + "\n")

    # Group by provider
    from collections import defaultdict
    by_provider = defaultdict(list)

    for model in EMBEDDING_MODELS:
        by_provider[model.provider.value].append(model)

    for provider, models in sorted(by_provider.items()):
        logger.info(f"{provider.upper()}:")
        for model in models:
            logger.info(f"  - {model.name} ({model.dimension}d)")
        logger.info("")

    # Group by specialization
    by_spec = defaultdict(list)
    for model in EMBEDDING_MODELS:
        by_spec[model.specialization].append(model)

    logger.info("BY SPECIALIZATION:")
    for spec, models in sorted(by_spec.items()):
        logger.info(f"  {spec}: {len(models)} models")
    logger.info("")

    # Dimension distribution
    dimensions = [m.dimension for m in EMBEDDING_MODELS]
    logger.info("DIMENSIONS:")
    logger.info(f"  Min: {min(dimensions)}")
    logger.info(f"  Max: {max(dimensions)}")
    logger.info(f"  Average: {sum(dimensions) / len(dimensions):.1f}")
    logger.info("")


def main():
    """Run all validations"""
    print("\n" + "="*80)
    print("EMBEDDING BENCHMARK VALIDATION")
    print("="*80 + "\n")

    results = {
        "Models": validate_models(),
        "Sentence Transformers": test_sentence_transformers(),
        "BEIR": test_beir(),
        "Dependencies": test_dependencies()
    }

    # Print summary
    print_model_summary()

    # Final result
    logger.info("="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80 + "\n")

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    logger.info("")

    if all_passed:
        logger.info("🎉 ALL VALIDATIONS PASSED!")
        logger.info("\nYou can now run:")
        logger.info("  python run_embedding_benchmark.py")
        logger.info("  python tests/benchmarks/test_embeddings_on_beir.py --quick")
        logger.info("")
        return 0
    else:
        logger.error("❌ SOME VALIDATIONS FAILED")
        logger.error("\nPlease install missing dependencies:")
        logger.error("  pip install -r requirements-advanced-rag.txt")
        logger.error("")
        return 1


if __name__ == "__main__":
    sys.exit(main())
