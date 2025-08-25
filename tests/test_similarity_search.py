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
    SearchResult, SimilaritySearcher, main
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


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating SearchResult instance."""
        result = SearchResult(
            score=0.95,
            payload={"text": "Sample text", "paper_id": "paper_1"},
            id="point_123"
        )
        
        assert result.score == 0.95
        assert result.payload == {"text": "Sample text", "paper_id": "paper_1"}
        assert result.id == "point_123"


class TestSimilaritySearcher:
    """Test SimilaritySearcher class."""
    
    def test_searcher_creation(self):
        """Test searcher creation."""
        mock_qdrant = Mock()
        mock_embedding_provider = Mock()
        collection = "test_collection"
        
        searcher = SimilaritySearcher(
            mock_qdrant, mock_embedding_provider, collection
        )
        
        assert searcher.qdrant == mock_qdrant
        assert searcher.embedding_provider == mock_embedding_provider
        assert searcher.collection == collection
    
    @patch('similarity_search_arxiv_papers.SimilaritySearcher._build_filter')
    def test_search_success(self, mock_build_filter):
        """Test successful similarity search."""
        # Mock components
        mock_qdrant = Mock()
        mock_embedding_provider = Mock()
        mock_embedding_provider.embed.return_value = EmbeddingResults(
            vector=[0.1] * 1536, model="test-model"
        )
        
        # Mock search results
        mock_search_results = [
            Mock(
                score=0.95,
                payload={"text": "Result 1", "paper_id": "paper_1"},
                id="point_1"
            ),
            Mock(
                score=0.87,
                payload={"text": "Result 2", "paper_id": "paper_2"},
                id="point_2"
            )
        ]
        mock_qdrant.search.return_value = mock_search_results
        
        # Mock filter
        mock_filter = Mock()
        mock_build_filter.return_value = mock_filter
        
        # Create searcher and search
        searcher = SimilaritySearcher(
            mock_qdrant, mock_embedding_provider, "test_collection"
        )
        
        query = "machine learning"
        results = searcher.search(query, top_k=2)
        
        # Verify results
        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[0].payload["text"] == "Result 1"
        assert results[1].score == 0.87
        assert results[1].payload["text"] == "Result 2"
        
        # Verify API calls
        mock_embedding_provider.embed.assert_called_once_with(query)
        mock_qdrant.search.assert_called_once()
        
        # Verify search parameters
        search_call = mock_qdrant.search.call_args
        assert search_call[1]['collection_name'] == "test_collection"
        assert search_call[1]['query_vector'] == [0.1] * 1536
        assert search_call[1]['limit'] == 2
    
    def test_search_with_filters(self):
        """Test search with metadata filters."""
        # Mock components
        mock_qdrant = Mock()
        mock_embedding_provider = Mock()
        mock_embedding_provider.embed.return_value = EmbeddingResults(
            vector=[0.1] * 1536, model="test-model"
        )
        
        # Mock search results
        mock_search_results = [
            Mock(
                score=0.92,
                payload={"text": "Filtered result", "paper_id": "paper_1"},
                id="point_1"
            )
        ]
        mock_qdrant.search.return_value = mock_search_results
        
        # Create searcher and search with filters
        searcher = SimilaritySearcher(
            mock_qdrant, mock_embedding_provider, "test_collection"
        )
        
        query = "neural networks"
        filters = {"paper_id": "paper_1"}
        results = searcher.search(query, top_k=1, filters=filters)
        
        # Verify filter was applied
        search_call = mock_qdrant.search.call_args
        assert 'query_filter' in search_call[1]
    
    def test_search_embedding_error(self):
        """Test search when embedding generation fails."""
        mock_qdrant = Mock()
        mock_embedding_provider = Mock()
        mock_embedding_provider.embed.side_effect = Exception("Embedding failed")
        
        searcher = SimilaritySearcher(
            mock_qdrant, mock_embedding_provider, "test_collection"
        )
        
        with pytest.raises(Exception, match="Embedding failed"):
            searcher.search("test query")
    
    def test_search_qdrant_error(self):
        """Test search when Qdrant search fails."""
        mock_qdrant = Mock()
        mock_embedding_provider = Mock()
        mock_embedding_provider.embed.return_value = EmbeddingResults(
            vector=[0.1] * 1536, model="test-model"
        )
        mock_qdrant.search.side_effect = Exception("Qdrant search failed")
        
        searcher = SimilaritySearcher(
            mock_qdrant, mock_embedding_provider, "test_collection"
        )
        
        with pytest.raises(Exception, match="Qdrant search failed"):
            searcher.search("test query")
    
    def test_build_filter_simple(self):
        """Test building simple filter."""
        mock_qdrant = Mock()
        mock_embedding_provider = Mock()
        
        searcher = SimilaritySearcher(
            mock_qdrant, mock_embedding_provider, "test_collection"
        )
        
        filters = {"paper_id": "paper_1"}
        result_filter = searcher._build_filter(filters)
        
        assert result_filter is not None
    
    def test_build_filter_complex(self):
        """Test building complex filter."""
        mock_qdrant = Mock()
        mock_embedding_provider = Mock()
        
        searcher = SimilaritySearcher(
            mock_qdrant, mock_embedding_provider, "test_collection"
        )
        
        filters = {
            "paper_id": "paper_1",
            "chunk_id": 0,
            "source": "test.pdf"
        }
        result_filter = searcher._build_filter(filters)
        
        assert result_filter is not None
    
    def test_build_filter_empty(self):
        """Test building filter with empty filters."""
        mock_qdrant = Mock()
        mock_embedding_provider = Mock()
        
        searcher = SimilaritySearcher(
            mock_qdrant, mock_embedding_provider, "test_collection"
        )
        
        result_filter = searcher._build_filter({})
        assert result_filter is None


class TestMainFunction:
    """Test main function functionality."""
    
    @patch('similarity_search_arxiv_papers.QdrantClient')
    @patch('similarity_search_arxiv_papers.OpenAIEmbeddingProvider')
    @patch('similarity_search_arxiv_papers.SimilaritySearcher')
    def test_main_function_success(self, mock_searcher_class, mock_provider_class, mock_qdrant_class):
        """Test successful main function execution."""
        # Mock components
        mock_qdrant = Mock()
        mock_qdrant_class.return_value = mock_qdrant
        
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        
        mock_searcher = Mock()
        mock_searcher.search.return_value = [
            Mock(
                score=0.95,
                payload={"text": "Result text", "paper_id": "paper_1"},
                id="point_1"
            )
        ]
        mock_searcher_class.return_value = mock_searcher
        
        # Test with minimal arguments
        test_args = [
            'similarity_search_arxiv_papers.py',
            '--query', 'machine learning',
            '--collection', 'test_collection'
        ]
        
        with patch('sys.argv', test_args):
            main()
        
        # Verify components were created
        mock_qdrant_class.assert_called_once()
        mock_provider_class.assert_called_once()
        mock_searcher_class.assert_called_once()
        
        # Verify search was performed
        mock_searcher.search.assert_called_once_with("machine learning", top_k=5)
    
    @patch('similarity_search_arxiv_papers.QdrantClient')
    @patch('similarity_search_arxiv_papers.OpenAIEmbeddingProvider')
    def test_main_function_qdrant_error(self, mock_provider_class, mock_qdrant_class):
        """Test main function with Qdrant error."""
        mock_qdrant_class.side_effect = Exception("Qdrant connection failed")
        
        test_args = [
            'similarity_search_arxiv_papers.py',
            '--query', 'test query',
            '--collection', 'test_collection'
        ]
        
        with patch('sys.argv', test_args), \
             pytest.raises(Exception, match="Qdrant connection failed"):
            main()
    
    @patch('similarity_search_arxiv_papers.QdrantClient')
    def test_main_function_provider_error(self, mock_qdrant_class):
        """Test main function with embedding provider error."""
        mock_qdrant = Mock()
        mock_qdrant_class.return_value = mock_qdrant
        
        with patch('similarity_search_arxiv_papers.OpenAIEmbeddingProvider') as mock_provider_class:
            mock_provider_class.side_effect = Exception("Provider creation failed")
            
            test_args = [
                'similarity_search_arxiv_papers.py',
                '--query', 'test query',
                '--collection', 'test_collection'
            ]
            
            with patch('sys.argv', test_args), \
                 pytest.raises(Exception, match="Provider creation failed"):
                main()
    
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
    
    @patch('similarity_search_arxiv_papers.QdrantClient')
    @patch('similarity_search_arxiv_papers.OpenAIEmbeddingProvider')
    def test_full_search_pipeline(self, mock_provider_class, mock_qdrant_class):
        """Test full search pipeline integration."""
        # Mock Qdrant client
        mock_qdrant = Mock()
        mock_qdrant_class.return_value = mock_qdrant
        
        # Mock embedding provider
        mock_provider = Mock()
        mock_provider.embed.return_value = EmbeddingResults(
            vector=[0.1] * 1536, model="text-embedding-3-small"
        )
        mock_provider_class.return_value = mock_provider
        
        # Mock search results
        mock_search_results = [
            Mock(
                score=0.95,
                payload={
                    "text": "Machine learning algorithms have shown remarkable performance",
                    "paper_id": "paper_1",
                    "chunk_id": 0,
                    "source": "ml_paper.pdf"
                },
                id="point_1"
            ),
            Mock(
                score=0.87,
                payload={
                    "text": "Deep learning approaches in computer vision",
                    "paper_id": "paper_2",
                    "chunk_id": 1,
                    "source": "dl_paper.pdf"
                },
                id="point_2"
            )
        ]
        mock_qdrant.search.return_value = mock_search_results
        
        # Create searcher and perform search
        from similarity_search_arxiv_papers import SimilaritySearcher
        
        searcher = SimilaritySearcher(mock_qdrant, mock_provider, "test_collection")
        
        query = "machine learning algorithms"
        results = searcher.search(query, top_k=2)
        
        # Verify results
        assert len(results) == 2
        assert results[0].score == 0.95
        assert "Machine learning algorithms" in results[0].payload["text"]
        assert results[1].score == 0.87
        assert "Deep learning approaches" in results[1].payload["text"]
        
        # Verify API calls
        mock_provider.embed.assert_called_once_with(query)
        mock_qdrant.search.assert_called_once()
    
    def test_search_result_serialization(self):
        """Test that search results can be serialized for output."""
        # Create sample search result
        result = SearchResult(
            score=0.95,
            payload={
                "text": "Sample text content",
                "paper_id": "paper_1",
                "chunk_id": 0,
                "source": "test.pdf"
            },
            id="point_123"
        )
        
        # Test JSON serialization
        import json
        
        # Convert to dict for JSON serialization
        result_dict = {
            "score": result.score,
            "payload": result.payload,
            "id": result.id
        }
        
        json_str = json.dumps(result_dict)
        assert "0.95" in json_str
        assert "Sample text content" in json_str
        assert "paper_1" in json_str
    
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

