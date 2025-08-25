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
    log_setup, parse_args, extract_text_from_pdf, create_chunks,
    process_paper, main
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
    
    @patch('chunk_arxiv_papers.fitz.open')
    def test_extract_text_with_pymupdf(self, mock_fitz_open):
        """Test text extraction using PyMuPDF."""
        # Mock PyMuPDF
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample text from PyMuPDF"
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc
        
        with patch('chunk_arxiv_papers.HAVE_PYMUPDF', True):
            result = extract_text_from_pdf("dummy_path.pdf")
            
            assert result == "Sample text from PyMuPDF"
            mock_fitz_open.assert_called_once_with("dummy_path.pdf")
    
    @patch('chunk_arxiv_papers.pdfminer_extract_text')
    def test_extract_text_with_pdfminer(self, mock_pdfminer):
        """Test text extraction using pdfminer."""
        mock_pdfminer.return_value = "Sample text from pdfminer"
        
        with patch('chunk_arxiv_papers.HAVE_PYMUPDF', False), \
             patch('chunk_arxiv_papers.HAVE_PDFMINER', True):
            result = extract_text_from_pdf("dummy_path.pdf")
            
            assert result == "Sample text from pdfminer"
            mock_pdfminer.assert_called_once_with("dummy_path.pdf")
    
    def test_extract_text_no_library(self):
        """Test text extraction when no PDF library is available."""
        with patch('chunk_arxiv_papers.HAVE_PYMUPDF', False), \
             patch('chunk_arxiv_papers.HAVE_PDFMINER', False):
            
            with pytest.raises(RuntimeError, match="No PDF library available"):
                extract_text_from_pdf("dummy_path.pdf")


class TestChunkCreation:
    """Test chunk creation functionality."""
    
    def test_create_chunks_recursive_splitter(self):
        """Test chunk creation using RecursiveCharacterTextSplitter."""
        sample_text = "This is a sample text that will be split into chunks. " * 50
        
        chunks = create_chunks(sample_text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 100 for chunk in chunks)
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i].page_content
            next_chunk = chunks[i + 1].page_content
            
            # Should have some overlap
            assert any(word in next_chunk for word in current_chunk.split()[-5:])
    
    def test_create_chunks_semantic_splitter(self):
        """Test chunk creation using SemanticChunker."""
        sample_text = "This is a sample text that will be split into chunks. " * 50
        
        with patch('chunk_arxiv_papers.OpenAIEmbeddings') as mock_embeddings:
            mock_embeddings.return_value = Mock()
            
            chunks = create_chunks(
                sample_text, 
                chunk_size=100, 
                chunk_overlap=20,
                use_semantic=True
            )
            
            assert len(chunks) > 1
            mock_embeddings.assert_called_once()
    
    def test_create_chunks_empty_text(self):
        """Test chunk creation with empty text."""
        chunks = create_chunks("", chunk_size=100, chunk_overlap=20)
        assert len(chunks) == 0
    
    def test_create_chunks_small_text(self):
        """Test chunk creation with text smaller than chunk size."""
        small_text = "This is a small text."
        chunks = create_chunks(small_text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) == 1
        assert chunks[0].page_content == small_text


class TestPaperProcessing:
    """Test paper processing functionality."""
    
    @patch('chunk_arxiv_papers.extract_text_from_pdf')
    @patch('chunk_arxiv_papers.create_chunks')
    def test_process_paper_success(self, mock_create_chunks, mock_extract_text):
        """Test successful paper processing."""
        # Mock text extraction
        mock_extract_text.return_value = "Sample paper text content"
        
        # Mock chunk creation
        mock_chunks = [
            Mock(page_content="Chunk 1", metadata={}),
            Mock(page_content="Chunk 2", metadata={})
        ]
        mock_create_chunks.return_value = mock_chunks
        
        # Mock MinIO object info
        mock_object_info = Mock()
        mock_object_info.etag = "test-etag"
        mock_object_info.size = 1024
        
        # Mock Qdrant client
        mock_qdrant = Mock()
        
        result = process_paper(
            "test_paper.pdf",
            "test-bucket",
            mock_object_info,
            mock_qdrant,
            "test_collection",
            chunk_size=100,
            chunk_overlap=20,
            use_semantic=False
        )
        
        assert result == True
        mock_extract_text.assert_called_once()
        mock_create_chunks.assert_called_once()
        mock_qdrant.upsert.assert_called_once()
    
    @patch('chunk_arxiv_papers.extract_text_from_pdf')
    def test_process_paper_extraction_failure(self, mock_extract_text):
        """Test paper processing when text extraction fails."""
        mock_extract_text.side_effect = Exception("Extraction failed")
        
        mock_object_info = Mock()
        mock_qdrant = Mock()
        
        result = process_paper(
            "test_paper.pdf",
            "test-bucket",
            mock_object_info,
            mock_qdrant,
            "test_collection"
        )
        
        assert result == False
        mock_qdrant.upsert.assert_not_called()
    
    @patch('chunk_arxiv_papers.extract_text_from_pdf')
    @patch('chunk_arxiv_papers.create_chunks')
    def test_process_paper_empty_text(self, mock_create_chunks, mock_extract_text):
        """Test paper processing with empty extracted text."""
        mock_extract_text.return_value = ""
        
        mock_object_info = Mock()
        mock_qdrant = Mock()
        
        result = process_paper(
            "test_paper.pdf",
            "test-bucket",
            mock_object_info,
            mock_qdrant,
            "test_collection"
        )
        
        assert result == False
        mock_create_chunks.assert_not_called()
        mock_qdrant.upsert.assert_not_called()


class TestMainFunction:
    """Test main function functionality."""
    
    @patch('chunk_arxiv_papers.Minio')
    @patch('chunk_arxiv_papers.QdrantClient')
    @patch('chunk_arxiv_papers.process_paper')
    def test_main_function_success(self, mock_process, mock_qdrant_class, mock_minio_class):
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
        
        # Mock object info
        mock_object_info = Mock()
        mock_object_info.etag = "test-etag"
        mock_object_info.size = 1024
        mock_minio.stat_object.return_value = mock_object_info
        
        # Mock processing results
        mock_process.side_effect = [True, True]
        
        # Test with minimal arguments
        test_args = [
            'chunk_arxiv_papers.py',
            '--minio-endpoint', 'localhost:9000',
            '--minio-bucket', 'test-bucket',
            '--qdrant-url', 'http://localhost:6333'
        ]
        
        with patch('sys.argv', test_args):
            main()
        
        # Verify clients were created
        mock_minio_class.assert_called_once()
        mock_qdrant_class.assert_called_once()
        
        # Verify papers were processed
        assert mock_process.call_count == 2
    
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
    @patch('chunk_arxiv_papers.extract_text_from_pdf')
    @patch('chunk_arxiv_papers.create_chunks')
    def test_full_pipeline_integration(self, mock_create_chunks, mock_extract_text, 
                                     mock_qdrant_class, mock_minio_class):
        """Test full pipeline integration."""
        # Setup mocks
        mock_minio = Mock()
        mock_minio.list_objects.return_value = [
            Mock(object_name="papers/test.pdf")
        ]
        mock_minio.stat_object.return_value = Mock(etag="test-etag", size=1024)
        mock_minio_class.return_value = mock_minio
        
        mock_qdrant = Mock()
        mock_qdrant_class.return_value = mock_qdrant
        
        mock_extract_text.return_value = "Sample paper content for testing chunking"
        
        mock_chunks = [
            Mock(page_content="Sample paper content", metadata={}),
            Mock(page_content="for testing chunking", metadata={})
        ]
        mock_create_chunks.return_value = mock_chunks
        
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
        mock_create_chunks.assert_called_once()
        mock_qdrant.upsert.assert_called_once()

