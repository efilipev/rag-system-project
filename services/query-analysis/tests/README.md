# Query Analysis Service - Unit Tests

Comprehensive unit test suite for the Query Analysis microservice.

## Test Coverage

### 1. Intent Classifier Tests (`test_intent_classifier.py`)
- **27 tests** covering ML-based intent classification
- Tests all 8 intent categories: informational, procedural, causal, comparative, recommendation, navigational, transactional, definitional
- Edge cases: empty queries, low confidence, multilanguage queries

### 2. Query Expander Tests (`test_query_expander.py`)
- **29 tests** covering query expansion functionality
- Synonym expansion via WordNet
- Question reformulation strategies
- Query uniqueness and diversity
- Edge cases: empty queries, special characters, technical terms

### 3. Cache Service Tests (`test_cache_service.py`)
- **30 tests** covering Redis caching
- Embedding caching (384-dim vectors)
- Analysis result caching
- Query expansion caching
- Cache statistics and hit rate calculation
- Error handling and edge cases

### 4. Consumer Tests (`test_consumers.py`)
- **22 tests** covering RabbitMQ consumers
- QueryHistoryConsumer: storing query history in Redis
- QueryAnalyticsConsumer: aggregating analytics data
- ConsumerManager: lifecycle management
- TTL management (30 days for history, 90 days for analytics)

### 5. API Routes Tests (`test_routes.py`)
- **23 tests** covering FastAPI endpoints
- `/analyze` endpoint testing
- `/expand` endpoint testing
- `/cache/stats`, `/cache/invalidate`, `/cache/clear` endpoints
- `/analytics` and `/history` endpoints
- Integration scenarios

**Total: 131 comprehensive unit tests**

## Running Tests

### Prerequisites

```bash
# Install dev dependencies
cd services/query-analysis
poetry install --with dev
```

### Run All Tests

```bash
# Run all tests with verbose output
poetry run pytest

# Run with coverage report
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_intent_classifier.py

# Run specific test class
poetry run pytest tests/test_cache_service.py::TestCacheService

# Run specific test function
poetry run pytest tests/test_intent_classifier.py::TestIntentClassifier::test_classify_informational_query
```

### Run Tests by Category

```bash
# Run only async tests
poetry run pytest -m asyncio

# Run only unit tests
poetry run pytest -m unit

# Run only integration tests
poetry run pytest -m integration
```

### Test Output Options

```bash
# Detailed output with print statements
poetry run pytest -v -s

# Short traceback
poetry run pytest --tb=short

# Stop on first failure
poetry run pytest -x

# Run tests in parallel (requires pytest-xdist)
poetry run pytest -n auto
```

### Coverage Reports

```bash
# Generate HTML coverage report
poetry run pytest --cov=app --cov-report=html
# View report: open htmlcov/index.html

# Generate terminal coverage report
poetry run pytest --cov=app --cov-report=term-missing

# Generate XML coverage report (for CI/CD)
poetry run pytest --cov=app --cov-report=xml
```

## Test Structure

```
tests/
├── __init__.py                 # Test package init
├── conftest.py                 # Shared fixtures
├── test_intent_classifier.py  # Intent classification tests
├── test_query_expander.py     # Query expansion tests
├── test_cache_service.py      # Redis caching tests
├── test_consumers.py          # RabbitMQ consumer tests
└── test_routes.py             # FastAPI route tests
```

## Shared Fixtures (`conftest.py`)

- `mock_redis`: Mock Redis client
- `mock_rabbitmq_connection`: Mock RabbitMQ connection
- `mock_rabbitmq_message`: Mock message for consumer testing
- `sample_query`: Sample test query
- `sample_analysis_result`: Sample analysis result
- `sample_expansion_result`: Sample expansion result
- `mock_cache_service`: Mock cache service instance

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install Poetry
        run: curl -sSL https://install.python-poetry.org | python3 -
      - name: Install Dependencies
        run: poetry install --with dev
      - name: Run Tests
        run: poetry run pytest --cov=app --cov-report=xml
      - name: Upload Coverage
        uses: codecov/codecov-action@v2
```

## Writing New Tests

### Test Naming Convention

- File: `test_<module_name>.py`
- Class: `Test<ClassName>`
- Function: `test_<what_it_tests>`

### Example Test

```python
import pytest
from app.services.my_service import MyService

class TestMyService:
    """Test suite for MyService"""

    @pytest.fixture
    def service(self):
        """Create service instance for testing"""
        return MyService()

    def test_basic_functionality(self, service):
        """Test basic functionality"""
        result = service.do_something("input")
        assert result == "expected"

    @pytest.mark.asyncio
    async def test_async_functionality(self, service):
        """Test async functionality"""
        result = await service.do_something_async("input")
        assert result is not None
```

## Mocking Guidelines

### Mock External Dependencies

```python
from unittest.mock import Mock, AsyncMock, patch

# Mock synchronous function
mock_func = Mock(return_value="result")

# Mock asynchronous function
mock_async_func = AsyncMock(return_value="result")

# Mock with side effects
mock_func.side_effect = Exception("Error")

# Patch during test
with patch('app.services.my_service.external_call') as mock:
    mock.return_value = "mocked"
    result = my_service.call_external()
```

### Mock Redis

```python
@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    return redis
```

## Test Best Practices

1. **Test one thing at a time** - Each test should verify a single behavior
2. **Use descriptive names** - Test names should clearly indicate what they test
3. **Arrange-Act-Assert** - Structure tests clearly
4. **Mock external dependencies** - Don't hit real Redis/RabbitMQ in unit tests
5. **Test edge cases** - Empty inputs, None values, errors
6. **Keep tests fast** - Unit tests should run in milliseconds
7. **Make tests independent** - Tests should not depend on each other

## Troubleshooting

### Common Issues

**Issue: `ImportError: cannot import name 'X'`**
- Solution: Ensure you're running tests from the service root directory
- Use: `cd services/query-analysis && poetry run pytest`

**Issue: Async tests not running**
- Solution: Install pytest-asyncio: `poetry add --group dev pytest-asyncio`
- Ensure `asyncio_mode = "auto"` in pyproject.toml

**Issue: Redis/RabbitMQ connection errors**
- Solution: Ensure you're using mocked clients (check fixtures in conftest.py)
- Don't connect to real services in unit tests

**Issue: Model loading timeouts**
- Solution: Mock the model loading in setUp/fixture
- Use `@patch` decorator to avoid loading actual ML models

## Performance

Unit tests should be fast:
- Target: < 1 second for all 131 tests
- Actual: ~0.5-2 seconds (depends on mocking effectiveness)

If tests are slow:
1. Check if external services are being called
2. Ensure ML models are properly mocked
3. Use `pytest-timeout` to identify slow tests

## Coverage Goals

- **Target: >80% code coverage**
- Focus on critical paths: analyzers, consumers, API routes
- Don't obsess over 100% - focus on meaningful tests

Check current coverage:
```bash
poetry run pytest --cov=app --cov-report=term-missing
```
