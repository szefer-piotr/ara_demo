"""
Tests for arxiv paper chunking functionality.
"""
import pytest
import tempfile
import os
import uuid
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunk_arxiv_papers import (
    log_setup, parse_args, extract_text, chunk_text,
    main
)


class TestLogSetup:
    """Test logging setup functionality."""
    
    def test_log_setup_default(self):
        """Test default logging setup."""
        with patch('chunk_arxiv_papers.logging.basicConfig') as mock_logging:
            log_setup()
            mock_logging.assert_called_once()
    
    def test_log_setup_custom_level(self):
        """Test custom logging level setup."""
        with patch('chunk_arxiv_papers.logging.basicConfig') as mock_logging:
            log_setup("DEBUG")
            mock_logging.assert_called_once()


class TestParseArgs:
    """Test argument parsing functionality."""
    
    def test_parse_args_required(self):
        """Test parsing of required arguments."""
        test_args = [
            'chunk_arxiv_papers.py',
            '--minio-endpoint', 'localhost:9000',
            '--minio-bucket', 'test-bucket',
            '--qdrant-url', 'http://localhost:6333'
        ]
        
        with patch('sys.argv', test_args):
            args = parse_args()
            
            assert args.minio_endpoint == 'localhost:9000'
            assert args.minio_bucket == 'test-bucket'
            assert args.qdrant_url == 'http://localhost:6333'
    
    def test_parse_args_with_optional(self):
        """Test parsing of optional arguments."""
        test_args = [
            'chunk_arxiv_papers.py',
            '--minio-endpoint', 'localhost:9000',
            '--minio-bucket', 'test-bucket',
            '--qdrant-url', 'http://localhost:6333',
            '--chunk-size', '1000',
            '--chunk-overlap', '100',
            '--prefix', 'papers/',
            '--collection', 'test_collection'
        ]
        
        with patch('sys.argv', test_args):
            args = parse_args()
            
            assert args.chunk_size == 1000
            assert args.chunk_overlap == 100
            assert args.prefix == 'papers/'
            assert args.collection == 'test_collection'
    
    def test_parse_args_environment_defaults(self):
        """Test that environment variables are used as defaults."""
        test_args = [
            'chunk_arxiv_papers.py',
            '--minio-endpoint', 'localhost:9000',
            '--minio-bucket', 'test-bucket'
        ]
        
        with patch('sys.argv', test_args), \
             patch.dict(os.environ, {
                 'QDRANT_URL': 'http://env:6333',
                 'OPENAI_API_KEY': 'env-key',
                 'MINIO_ACCESS_KEY': 'env-access'
             }):
            args = parse_args()
            
            assert args.qdrant_url == 'http://env:6333'
            assert args.openai_api_key == 'env-key'
            assert args.minio_access_key == 'env-access'


class TestPDFTextExtraction:
    """Test PDF text extraction functionality."""
    
    def test_extract_text_success(self):
        """Test successful text extraction."""
        # Mock PDF content
        pdf_bytes = b"%PDF-1.4\nSample PDF content"
        
        with patch('chunk_arxiv_papers.fitz.open') as mock_fitz_open:
            mock_doc = Mock()
            mock_page = Mock()
            mock_page.get_text.return_value = "Sample text from PyMuPDF"
            mock_doc.__iter__.return_value = [mock_page]
            mock_doc.__len__.return_value = 1
            mock_fitz_open.return_value = mock_doc
            
            result = extract_text(pdf_bytes)
            
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] == "Sample text from PyMuPDF"  # text
            assert result[1] == 1  # pages


class TestTextChunking:
    """Test text chunking functionality."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        sample_text = "This is a sample text that will be split into chunks. " * 10
        
        chunks = chunk_text(sample_text, size=100, overlap=20)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Should have some overlap
            assert any(word in next_chunk for word in current_chunk.split()[-5:])
    
    def test_chunk_text_empty_text(self):
        """Test chunking with empty text."""
        chunks = chunk_text("", size=100, overlap=20)
        assert len(chunks) == 0
    
    def test_chunk_text_small_text(self):
        """Test chunking with text smaller than chunk size."""
        small_text = "This is a small text."
        chunks = chunk_text(small_text, size=100, overlap=20)
        
        assert len(chunks) == 1
        assert chunks[0] == small_text
    
    def test_chunk_text_different_sizes(self):
        """Test chunking with different chunk sizes."""
        text = "This is a sample text that will be split into chunks. " * 20
        
        small_chunks = chunk_text(text, size=50, overlap=10)
        large_chunks = chunk_text(text, size=200, overlap=50)
        
        assert len(small_chunks) > len(large_chunks)
        assert all(len(chunk) <= 50 for chunk in small_chunks)
        assert all(len(chunk) <= 200 for chunk in large_chunks)


class TestMainFunction:
    """Test main function functionality."""
    
    @patch('chunk_arxiv_papers.Minio')
    @patch('chunk_arxiv_papers.QdrantClient')
    @patch('chunk_arxiv_papers.extract_text')
    @patch('chunk_arxiv_papers.chunk_text')
    def test_main_function_success(self, mock_chunk_text, mock_extract_text, mock_qdrant_class, mock_minio_class):
        """Test successful main function execution."""
        # Mock clients
        mock_minio = Mock()
        mock_minio.list_objects.return_value = [
            Mock(object_name="papers/test1.pdf"),
            Mock(object_name="papers/test2.pdf")
        ]
        mock_minio_class.return_value = mock_minio
        
        mock_qdrant = Mock()
        mock_qdrant_class.return_value = mock_qdrant
        
        # Mock text extraction
        mock_extract_text.return_value = ("Sample text content", 5)
        
        # Mock chunking
        mock_chunks = ["Chunk 1", "Chunk 2"]
        mock_chunk_text.return_value = mock_chunks
        
        # Mock other functions
        with patch('chunk_arxiv_papers.get_pdf_bytes') as mock_get_pdf:
            mock_get_pdf.return_value = (b"pdf_content", "etag", 1024)
            
            with patch('chunk_arxiv_papers.embed_texts_openai') as mock_embed:
                mock_embed.return_value = [[0.1] * 1536, [0.1] * 1536]
                
                with patch('chunk_arxiv_papers.upsert_points') as mock_upsert:
                    # Test with minimal arguments
                    test_args = [
                        'chunk_arxiv_papers.py',
                        '--minio-endpoint', 'localhost:9000',
                        '--minio-bucket', 'test-bucket',
                        '--qdrant-url', 'http://localhost:6333'
                    ]
                    
                    with patch('sys.argv', test_args):
                        main()
                    
                    # Verify components were called
                    mock_minio_class.assert_called_once()
                    mock_qdrant_class.assert_called_once()
                    mock_extract_text.assert_called()
                    mock_chunk_text.assert_called()
                    mock_upsert.assert_called()
    
    @patch('chunk_arxiv_papers.Minio')
    @patch('chunk_arxiv_papers.QdrantClient')
    def test_main_function_minio_error(self, mock_qdrant_class, mock_minio_class):
        """Test main function with MinIO error."""
        mock_minio_class.side_effect = Exception("MinIO connection failed")
        
        test_args = [
            'chunk_arxiv_papers.py',
            '--minio-endpoint', 'localhost:9000',
            '--minio-bucket', 'test-bucket',
            '--qdrant-url', 'http://localhost:6333'
        ]
        
        with patch('sys.argv', test_args), \
             pytest.raises(Exception, match="MinIO connection failed"):
            main()
    
    @patch('chunk_arxiv_papers.Minio')
    @patch('chunk_arxiv_papers.QdrantClient')
    def test_main_function_qdrant_error(self, mock_qdrant_class, mock_minio_class):
        """Test main function with Qdrant error."""
        mock_minio = Mock()
        mock_minio.list_objects.return_value = []
        mock_minio_class.return_value = mock_minio
        
        mock_qdrant_class.side_effect = Exception("Qdrant connection failed")
        
        test_args = [
            'chunk_arxiv_papers.py',
            '--minio-endpoint', 'localhost:9000',
            '--minio-bucket', 'test-bucket',
            '--qdrant-url', 'http://localhost:6333'
        ]
        
        with patch('sys.argv', test_args), \
             pytest.raises(Exception, match="Qdrant connection failed"):
            main()


class TestIntegration:
    """Test integration between components."""
    
    @patch('chunk_arxiv_papers.Minio')
    @patch('chunk_arxiv_papers.QdrantClient')
    @patch('chunk_arxiv_papers.extract_text')
    @patch('chunk_arxiv_papers.chunk_text')
    def test_full_pipeline_integration(self, mock_chunk_text, mock_extract_text, 
                                     mock_qdrant_class, mock_minio_class):
        """Test full pipeline integration."""
        # Setup mocks
        mock_minio = Mock()
        mock_minio.list_objects.return_value = [
            Mock(object_name="papers/test.pdf")
        ]
        mock_minio_class.return_value = mock_minio
        
        mock_qdrant = Mock()
        mock_qdrant_class.return_value = mock_qdrant
        
        mock_extract_text.return_value = ("Sample paper content for testing chunking", 5)
        
        mock_chunks = ["Sample paper content", "for testing chunking"]
        mock_chunk_text.return_value = mock_chunks
        
        # Mock other functions
        with patch('chunk_arxiv_papers.get_pdf_bytes') as mock_get_pdf:
            mock_get_pdf.return_value = (b"pdf_content", "etag", 1024)
            
            with patch('chunk_arxiv_papers.embed_texts_openai') as mock_embed:
                mock_embed.return_value = [[0.1] * 1536, [0.1] * 1536]
                
                with patch('chunk_arxiv_papers.upsert_points') as mock_upsert:
                    # Run pipeline
                    test_args = [
                        'chunk_arxiv_papers.py',
                        '--minio-endpoint', 'localhost:9000',
                        '--minio-bucket', 'test-bucket',
                        '--qdrant-url', 'http://localhost:6333',
                        '--chunk-size', '50',
                        '--chunk-overlap', '10'
                    ]
                    
                    with patch('sys.argv', test_args):
                        main()
                    
                    # Verify all components were called
                    mock_minio_class.assert_called_once()
                    mock_qdrant_class.assert_called_once()
                    mock_extract_text.assert_called_once()
                    mock_chunk_text.assert_called_once()
                    mock_upsert.assert_called_once()

