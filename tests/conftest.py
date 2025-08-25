"""
Pytest configuration and common fixtures for bioarxiv pipeline tests.
"""
import pytest
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Mock environment variables
@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-api-key',
        'QDRANT_URL': 'http://localhost:6333',
        'QDRANT_API_KEY': 'test-qdrant-key',
        'MINIO_ACCESS_KEY': 'test-minio-key',
        'MINIO_SECRET_KEY': 'test-minio-secret',
        'MINIO_SECURE': 'false'
    }):
        yield

@pytest.fixture
def mock_minio_client():
    """Mock MinIO client for testing."""
    mock_client = Mock()
    mock_client.make_bucket.return_value = None
    mock_client.stat_object.return_value = Mock()
    mock_client.put_object.return_value = None
    mock_client.get_object.return_value = Mock()
    return mock_client

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = Mock()
    mock_client.get_collection.return_value = Mock()
    mock_client.upsert.return_value = Mock()
    mock_client.search.return_value = Mock()
    return mock_client

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )
    return mock_client

@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Sample PDF content for testing) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000111 00000 n \n0000000206 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"

@pytest.fixture
def sample_biorxiv_response():
    """Sample bioRxiv API response for testing."""
    return {
        "messages": [
            {
                "doi": "10.1101/2023.01.01.123456",
                "title": "Sample Research Paper Title",
                "version": 1,
                "server": "biorxiv",
                "date": "2023-01-01"
            }
        ],
        "cursor": "next_cursor_token"
    }

@pytest.fixture
def sample_chunk_data():
    """Sample chunk data for testing."""
    return {
        "paper_id": "test_paper_123",
        "chunk_id": 0,
        "text": "This is a sample chunk of text for testing purposes.",
        "source": "test_paper.pdf",
        "bucket": "test-bucket",
        "object": "papers/test_paper.pdf",
        "arxiv_id": "1234.5678",
        "etag": "test-etag",
        "bytes": 1024,
        "pages": 5,
        "ingested_at": "2023-01-01T00:00:00Z"
    }

