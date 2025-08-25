# Bioarxiv Pipeline Testing

This directory contains comprehensive tests for the bioarxiv pipeline, covering all major components from paper download to similarity search.

## Test Structure

```
tests/
├── __init__.py                 # Package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── test_download_biorxiv_papers.py    # Tests for paper download
├── test_chunk_arxiv_papers.py         # Tests for paper chunking
├── test_chunk_evaluation.py           # Tests for chunk evaluation
├── test_similarity_search.py          # Tests for similarity search
├── test_full_pipeline.py              # Integration tests
├── run_tests.py                       # Test runner script
├── requirements-test.txt               # Testing dependencies
└── README.md                          # This file
```

## Test Categories

### 1. Unit Tests
- **Download Tests** (`test_download_biorxiv_papers.py`): Test individual functions for downloading papers
- **Chunking Tests** (`test_chunk_arxiv_papers.py`): Test text chunking functionality
- **Evaluation Tests** (`test_chunk_evaluation.py`): Test chunk quality metrics
- **Search Tests** (`test_similarity_search.py`): Test similarity search components

### 2. Integration Tests
- **Full Pipeline Tests** (`test_full_pipeline.py`): Test complete workflow from download to search
- **Data Flow Tests**: Verify data integrity through the pipeline
- **Performance Tests**: Test performance characteristics

### 3. Component Tests
- **MinIO Operations**: Test paper storage and retrieval
- **Qdrant Operations**: Test vector database operations
- **OpenAI Integration**: Test embedding generation
- **PDF Processing**: Test text extraction and chunking

## Installation

### 1. Install Testing Dependencies

```bash
pip install -r tests/requirements-test.txt
```

### 2. Install Main Dependencies

```bash
pip install -r requirements.txt
```

## Running Tests

### Quick Start

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/test_download_biorxiv_papers.py
```

### Using the Test Runner

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --mode unit
python tests/run_tests.py --mode integration
python tests/run_tests.py --mode download
python tests/run_tests.py --mode chunking
python tests/run_tests.py --mode evaluation
python tests/run_tests.py --mode search

# Run with coverage and HTML reports
python tests/run_tests.py --coverage --html

# Run tests in parallel
python tests/run_tests.py --parallel

# Verbose output
python tests/run_tests.py --verbose
```

### Pytest Commands

```bash
# Run with markers
pytest -m "unit"
pytest -m "integration"
pytest -m "download"
pytest -m "chunking"
pytest -m "evaluation"
pytest -m "search"

# Run with coverage
pytest --cov=. --cov-report=html:htmlcov --cov-report=term-missing

# Run specific test
pytest tests/test_download_biorxiv_papers.py::TestRxivEntry::test_rxiv_entry_creation

# Run tests in parallel
pytest -n auto

# Generate HTML report
pytest --html=test-report.html --self-contained-html
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)

- **Test Discovery**: Automatically finds tests in `tests/` directory
- **Coverage**: Generates HTML and XML coverage reports
- **Markers**: Defines test categories for selective execution
- **Warnings**: Filters out common warnings

### Fixtures (`conftest.py`)

Common test fixtures include:
- **Mock Clients**: MinIO, Qdrant, OpenAI clients
- **Sample Data**: PDF content, API responses, chunk data
- **Environment Variables**: Mocked configuration
- **Temporary Directories**: For file operations

## Test Examples

### Example 1: Testing Paper Download

```python
def test_download_paper_success(self, mock_minio_client, temp_dir):
    """Test successful paper download."""
    entry = RxivEntry(
        doi="10.1101/2023.01.01.123456",
        title="Test Paper",
        version=1,
        server="biorxiv"
    )
    
    result = download_paper(
        entry, mock_minio_client, "test-bucket", "papers/", temp_dir
    )
    
    assert result == True
    mock_minio_client.put_object.assert_called_once()
```

### Example 2: Testing Chunking

```python
def test_create_chunks_recursive_splitter(self):
    """Test chunk creation using RecursiveCharacterTextSplitter."""
    sample_text = "This is a sample text that will be split into chunks. " * 50
    
    chunks = create_chunks(sample_text, chunk_size=100, chunk_overlap=20)
    
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
```

### Example 3: Testing Similarity Search

```python
def test_search_success(self, mock_qdrant, mock_embedding_provider):
    """Test successful similarity search."""
    searcher = SimilaritySearcher(
        mock_qdrant, mock_embedding_provider, "test_collection"
    )
    
    results = searcher.search("machine learning", top_k=2)
    
    assert len(results) == 2
    assert results[0].score > results[1].score
```

## Coverage Reports

### HTML Coverage Report

```bash
pytest --cov=. --cov-report=html:htmlcov
# Open htmlcov/index.html in your browser
```

### Terminal Coverage Report

```bash
pytest --cov=. --cov-report=term-missing
```

### XML Coverage Report

```bash
pytest --cov=. --cov-report=xml
# Useful for CI/CD integration
```

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
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r tests/requirements-test.txt
    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml --junitxml=junit.xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running tests from the project root directory
2. **Missing Dependencies**: Install all requirements from `requirements-test.txt`
3. **Mock Issues**: Check that mocks are properly configured in `conftest.py`
4. **Environment Variables**: Ensure test environment is properly mocked

### Debug Mode

```bash
# Run with debug output
pytest -v -s --tb=long

# Run single test with debug
pytest tests/test_download_biorxiv_papers.py::TestRxivEntry::test_rxiv_entry_creation -v -s
```

### Test Isolation

```bash
# Run tests in isolation
pytest --dist=no

# Run tests sequentially
pytest -n 0
```

## Performance Testing

### Benchmark Tests

```bash
# Run performance tests
pytest -m "slow"

# Run with timing
pytest --durations=10
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Run memory-intensive tests
python -m memory_profiler tests/test_full_pipeline.py::TestPipelinePerformance::test_memory_usage
```

## Contributing

### Adding New Tests

1. **Follow Naming Convention**: `test_*.py` for test files
2. **Use Descriptive Names**: Test methods should clearly describe what they test
3. **Add Markers**: Use appropriate pytest markers for categorization
4. **Mock External Dependencies**: Don't make real API calls in tests
5. **Test Edge Cases**: Include tests for error conditions and boundary cases

### Test Data

- **Sample PDFs**: Use minimal PDF content for testing
- **Mock Responses**: Create realistic but minimal mock data
- **Fixtures**: Reuse common test data through fixtures

### Code Coverage

- **Aim for High Coverage**: Target >90% code coverage
- **Test Error Paths**: Include tests for exception handling
- **Integration Coverage**: Ensure pipeline components work together

## Support

For testing-related issues:
1. Check the pytest documentation
2. Review existing test examples
3. Check test configuration files
4. Ensure all dependencies are installed

## License

Tests are covered under the same license as the main project.

