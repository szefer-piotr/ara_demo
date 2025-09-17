"""
Tests for similarity search functionality.
"""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similarity_search_arxiv_papers import (
    EmbeddingResults, EmbeddingProvider, OpenAIEmbeddingProvider,
    main
)


class TestEmbeddingResults:
    """Test EmbeddingResults dataclass."""
    
    def test_embedding_results_creation(self):
        """Test creating EmbeddingResults instance."""
        vector = [0.1, 0.2, 0.3]
        model = "text-embedding-3-small"
        
        result = EmbeddingResults(vector=vector, model=model)
        
        assert result.vector == vector
        assert result.model == model


class TestEmbeddingProvider:
    """Test EmbeddingProvider base class."""
    
    def test_embedding_provider_abstract(self):
        """Test that EmbeddingProvider is abstract."""
        provider = EmbeddingProvider()
        
        with pytest.raises(NotImplementedError):
            provider.embed("test text")


class TestOpenAIEmbeddingProvider:
    """Test OpenAI embedding provider."""
    
    def test_openai_provider_creation(self):
        """Test OpenAI provider creation."""
        api_key = "test-api-key"
        model = "text-embedding-3-small"
        
        with patch('similarity_search_arxiv_papers.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            provider = OpenAIEmbeddingProvider(api_key, model)
            
            assert provider.api_key == api_key
            assert provider.model == model
            assert provider.timeout == 30
    
    def test_openai_provider_creation_defaults(self):
        """Test OpenAI provider creation with defaults."""
        api_key = "test-api-key"
        
        with patch('similarity_search_arxiv_papers.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            provider = OpenAIEmbeddingProvider(api_key)
            
            assert provider.model == "text-embedding-3-small"
            assert provider.timeout == 30
    
    def test_openai_provider_no_api_key(self):
        """Test OpenAI provider creation without API key."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY not provided"):
            OpenAIEmbeddingProvider("")
    
    def test_openai_provider_embed_success(self):
        """Test successful embedding generation."""
        api_key = "test-api-key"
        text = "Sample text for embedding"
        
        with patch('similarity_search_arxiv_papers.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_embeddings = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            
            mock_embeddings.create.return_value = mock_response
            mock_client.embeddings = mock_embeddings
            mock_openai.return_value = mock_client
            
            provider = OpenAIEmbeddingProvider(api_key)
            result = provider.embed(text)
            
            assert isinstance(result, EmbeddingResults)
            assert result.vector == [0.1, 0.2, 0.3]
            assert result.model == "text-embedding-3-small"
            
            # Verify API call
            mock_embeddings.create.assert_called_once()
            call_args = mock_embeddings.create.call_args
            assert call_args[1]['input'] == "Sample text for embedding"
            assert call_args[1]['model'] == "text-embedding-3-small"
    
    def test_openai_provider_embed_text_cleaning(self):
        """Test that text is cleaned before embedding."""
        api_key = "test-api-key"
        text = "  Multiple    spaces\nand\nnewlines  "
        
        with patch('similarity_search_arxiv_papers.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_embeddings = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            
            mock_embeddings.create.return_value = mock_response
            mock_client.embeddings = mock_embeddings
            mock_openai.return_value = mock_client
            
            provider = OpenAIEmbeddingProvider(api_key)
            provider.embed(text)
            
            # Verify text was cleaned
            call_args = mock_embeddings.create.call_args
            assert call_args[1]['input'] == "Multiple spaces and newlines"
    
    def test_openai_provider_embed_error(self):
        """Test embedding generation with API error."""
        api_key = "test-api-key"
        text = "Sample text"
        
        with patch('similarity_search_arxiv_papers.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_embeddings = Mock()
            mock_embeddings.create.side_effect = Exception("API Error")
            mock_client.embeddings = mock_embeddings
            mock_openai.return_value = mock_client
            
            provider = OpenAIEmbeddingProvider(api_key)
            
            with pytest.raises(Exception, match="API Error"):
                provider.embed(text)


class TestMainFunction:
    """Test main function functionality."""
    
    def test_main_function_missing_query(self):
        """Test main function without query."""
        test_args = [
            'similarity_search_arxiv_papers.py',
            '--collection', 'test_collection'
        ]
        
        with patch('sys.argv', test_args), \
             pytest.raises(SystemExit):
            main()
    
    def test_main_function_missing_collection(self):
        """Test main function without collection."""
        test_args = [
            'similarity_search_arxiv_papers.py',
            '--query', 'test query'
        ]
        
        with patch('sys.argv', test_args), \
             pytest.raises(SystemExit):
            main()


class TestIntegration:
    """Test integration between components."""
    
    def test_embedding_dimensions(self):
        """Test that embeddings have correct dimensions."""
        # Mock OpenAI client
        with patch('similarity_search_arxiv_papers.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_embeddings = Mock()
            mock_response = Mock()
            
            # Test with text-embedding-3-small (1536 dimensions)
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_embeddings.create.return_value = mock_response
            mock_client.embeddings = mock_embeddings
            mock_openai.return_value = mock_client
            
            provider = OpenAIEmbeddingProvider("test-key")
            result = provider.embed("test text")
            
            assert len(result.vector) == 1536
            assert all(isinstance(x, float) for x in result.vector)
            assert result.model == "text-embedding-3-small"

