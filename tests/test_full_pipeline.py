"""
Integration tests for the complete bioarxiv pipeline.
"""
import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download_biorxiv_papers import RxivEntry
from chunk_arxiv_papers import create_chunks
from chunk_evaluation import quality_metrics
from similarity_search_arxiv_papers import SimilaritySearcher, OpenAIEmbeddingProvider


class TestFullPipelineIntegration:
    """Test the complete pipeline integration."""
    
    @patch('download_biorxiv_papers.Minio')
    @patch('chunk_arxiv_papers.QdrantClient')
    @patch('similarity_search_arxiv_papers.OpenAI')
    def test_complete_pipeline_workflow(self, mock_openai, mock_qdrant_class, mock_minio_class):
        """Test the complete pipeline from download to search."""
        # Setup mocks for the entire pipeline
        
        # 1. MinIO setup for paper storage
        mock_minio = Mock()
        mock_minio_class.return_value = mock_minio
        mock_minio.list_objects.return_value = [
            Mock(object_name="papers/test_paper.pdf")
        ]
        mock_minio.stat_object.return_value = Mock(etag="test-etag", size=1024)
        
        # 2. Qdrant setup for vector storage
        mock_qdrant = Mock()
        mock_qdrant_class.return_value = mock_qdrant
        mock_collection = Mock()
        mock_collection.points_count = 100
        mock_qdrant.get_collection.return_value = mock_collection
        
        # 3. OpenAI setup for embeddings
        mock_openai_client = Mock()
        mock_embeddings = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_embeddings.create.return_value = mock_response
        mock_openai_client.embeddings = mock_embeddings
        mock_openai.return_value = mock_openai_client
        
        # Test the pipeline components
        
        # Step 1: Simulate paper download
        test_paper = RxivEntry(
            doi="10.1101/2023.01.01.123456",
            title="Machine Learning in Bioinformatics",
            version=1,
            server="biorxiv"
        )
        
        assert test_paper.doi == "10.1101/2023.01.01.123456"
        assert test_paper.paper_id == "10.1101_2023.01.01.123456"
        assert "biorxiv.org" in test_paper.pdf_url()
        
        # Step 2: Simulate text extraction and chunking
        sample_text = """
        Machine learning has revolutionized bioinformatics in recent years. 
        Deep learning approaches have shown remarkable performance in protein structure prediction.
        The AlphaFold2 model demonstrated unprecedented accuracy in predicting protein structures.
        This breakthrough has significant implications for drug discovery and understanding disease mechanisms.
        """
        
        # Create chunks
        chunks = create_chunks(sample_text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 100 for chunk in chunks)
        
        # Step 3: Simulate chunk quality evaluation
        for chunk in chunks:
            metrics = quality_metrics(chunk.page_content)
            assert metrics["char_len"] > 0
            assert metrics["token_est"] > 0
            assert metrics["pct_ascii"] > 0.8  # Most should be ASCII
        
        # Step 4: Simulate vector storage in Qdrant
        mock_points = []
        for i, chunk in enumerate(chunks):
            mock_point = Mock()
            mock_point.payload = {
                "text": chunk.page_content,
                "paper_id": test_paper.paper_id,
                "chunk_id": i,
                "source": "test_paper.pdf",
                "quality_metrics": quality_metrics(chunk.page_content)
            }
            mock_points.append(mock_point)
        
        # Step 5: Simulate similarity search
        mock_search_results = [
            Mock(
                score=0.95,
                payload=mock_points[0].payload,
                id="point_1"
            ),
            Mock(
                score=0.87,
                payload=mock_points[1].payload,
                id="point_2"
            )
        ]
        mock_qdrant.search.return_value = mock_search_results
        
        # Create searcher and perform search
        embedding_provider = OpenAIEmbeddingProvider("test-key")
        searcher = SimilaritySearcher(mock_qdrant, embedding_provider, "test_collection")
        
        query = "protein structure prediction"
        results = searcher.search(query, top_k=2)
        
        # Verify search results
        assert len(results) == 2
        assert results[0].score == 0.95
        assert "Machine learning" in results[0].payload["text"]
        assert results[1].score == 0.87
        
        # Verify all components were called
        mock_minio_class.assert_called_once()
        mock_qdrant_class.assert_called_once()
        mock_openai.assert_called_once()
        mock_qdrant.search.assert_called_once()
    
    def test_pipeline_data_flow(self):
        """Test data flow through the pipeline."""
        # Test data transformation through each stage
        
        # Stage 1: Paper metadata
        paper = RxivEntry(
            doi="10.1101/2023.01.01.789012",
            title="Deep Learning for Drug Discovery",
            version=1,
            server="biorxiv"
        )
        
        # Stage 2: Text content
        paper_text = """
        Drug discovery is a complex and expensive process that can benefit significantly from machine learning approaches.
        Deep learning models can analyze molecular structures and predict drug-target interactions with high accuracy.
        Recent advances in transformer architectures have improved the performance of molecular property prediction.
        """
        
        # Stage 3: Chunking
        chunks = create_chunks(paper_text, chunk_size=150, chunk_overlap=30)
        
        # Stage 4: Quality metrics
        chunk_metrics = []
        for chunk in chunks:
            metrics = quality_metrics(chunk.page_content)
            chunk_metrics.append(metrics)
            
            # Verify metrics are reasonable
            assert metrics["char_len"] > 0
            assert metrics["token_est"] > 0
            assert metrics["pct_ascii"] > 0.8
            assert metrics["lines"] >= 1
        
        # Stage 5: Simulate vector representation
        # (In real pipeline, this would use OpenAI embeddings)
        chunk_vectors = []
        for chunk in chunks:
            # Mock vector (1536 dimensions for text-embedding-3-small)
            vector = [0.1] * 1536
            chunk_vectors.append({
                "text": chunk.page_content,
                "vector": vector,
                "metadata": {
                    "paper_id": paper.paper_id,
                    "chunk_id": len(chunk_vectors),
                    "quality_metrics": chunk_metrics[len(chunk_vectors)]
                }
            })
        
        # Verify data integrity through pipeline
        assert len(chunks) == len(chunk_metrics) == len(chunk_vectors)
        assert all(len(chunk_vector["vector"]) == 1536 for chunk_vector in chunk_vectors)
        
        # Verify text content is preserved
        for i, chunk in enumerate(chunks):
            assert chunk.page_content == chunk_vectors[i]["text"]
    
    def test_pipeline_error_handling(self):
        """Test error handling throughout the pipeline."""
        
        # Test 1: Invalid paper data
        with pytest.raises(TypeError):
            invalid_paper = RxivEntry(
                doi=None,  # Invalid DOI
                title="Test Paper",
                version=1,
                server="biorxiv"
            )
        
        # Test 2: Empty text chunking
        empty_chunks = create_chunks("", chunk_size=100, chunk_overlap=20)
        assert len(empty_chunks) == 0
        
        # Test 3: Quality metrics for empty text
        empty_metrics = quality_metrics("")
        assert empty_metrics["char_len"] == 0
        assert empty_metrics["token_est"] == 0
        
        # Test 4: Very short text
        short_chunks = create_chunks("Short", chunk_size=100, chunk_overlap=20)
        assert len(short_chunks) == 1
        assert short_chunks[0].page_content == "Short"
    
    def test_pipeline_configuration_options(self):
        """Test different pipeline configuration options."""
        
        # Test different chunk sizes
        text = "This is a sample text that will be split into chunks of different sizes. " * 10
        
        small_chunks = create_chunks(text, chunk_size=50, chunk_overlap=10)
        large_chunks = create_chunks(text, chunk_size=200, chunk_overlap=50)
        
        assert len(small_chunks) > len(large_chunks)
        assert all(len(chunk.page_content) <= 50 for chunk in small_chunks)
        assert all(len(chunk.page_content) <= 200 for chunk in large_chunks)
        
        # Test different overlap settings
        no_overlap_chunks = create_chunks(text, chunk_size=100, chunk_overlap=0)
        high_overlap_chunks = create_chunks(text, chunk_size=100, chunk_overlap=50)
        
        # High overlap should result in more chunks
        assert len(high_overlap_chunks) >= len(no_overlap_chunks)
    
    def test_pipeline_output_formats(self):
        """Test different output formats in the pipeline."""
        
        # Test chunk output format
        text = "Sample text for testing output formats."
        chunks = create_chunks(text, chunk_size=100, chunk_overlap=20)
        
        for chunk in chunks:
            # Verify chunk has required attributes
            assert hasattr(chunk, 'page_content')
            assert hasattr(chunk, 'metadata')
            assert isinstance(chunk.page_content, str)
        
        # Test quality metrics output format
        metrics = quality_metrics(text)
        expected_keys = [
            "char_len", "token_est", "lines", "avg_line_len",
            "pct_ascii", "pct_alnum", "pct_ws", "pct_punct",
            "uniq_char_ratio", "longest_run", "has_refs", "looks_like_table"
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        # Test data types
        assert isinstance(metrics["char_len"], int)
        assert isinstance(metrics["pct_ascii"], float)
        assert isinstance(metrics["has_refs"], bool)
    
    @patch('similarity_search_arxiv_papers.QdrantClient')
    @patch('similarity_search_arxiv_papers.OpenAI')
    def test_end_to_end_search_scenario(self, mock_openai, mock_qdrant_class):
        """Test a complete end-to-end search scenario."""
        
        # Setup mock Qdrant
        mock_qdrant = Mock()
        mock_qdrant_class.return_value = mock_qdrant
        
        # Setup mock OpenAI
        mock_openai_client = Mock()
        mock_embeddings = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_embeddings.create.return_value = mock_response
        mock_openai_client.embeddings = mock_embeddings
        mock_openai.return_value = mock_openai_client
        
        # Simulate stored papers
        stored_papers = [
            {
                "paper_id": "paper_1",
                "title": "Machine Learning in Genomics",
                "chunks": [
                    "Machine learning algorithms have revolutionized genomic analysis.",
                    "Deep learning approaches show superior performance in sequence prediction."
                ]
            },
            {
                "paper_id": "paper_2", 
                "title": "Protein Structure Prediction",
                "chunks": [
                    "Protein structure prediction is a fundamental problem in bioinformatics.",
                    "Recent advances in deep learning have improved prediction accuracy."
                ]
            }
        ]
        
        # Simulate search query
        query = "machine learning genomics"
        
        # Mock search results
        mock_search_results = [
            Mock(
                score=0.95,
                payload={
                    "text": stored_papers[0]["chunks"][0],
                    "paper_id": "paper_1",
                    "chunk_id": 0,
                    "title": stored_papers[0]["title"]
                },
                id="point_1"
            ),
            Mock(
                score=0.87,
                payload={
                    "text": stored_papers[1]["chunks"][1],
                    "paper_id": "paper_2",
                    "chunk_id": 1,
                    "title": stored_papers[1]["title"]
                },
                id="point_2"
            )
        ]
        mock_qdrant.search.return_value = mock_search_results
        
        # Perform search
        embedding_provider = OpenAIEmbeddingProvider("test-key")
        searcher = SimilaritySearcher(mock_qdrant, embedding_provider, "test_collection")
        
        results = searcher.search(query, top_k=2)
        
        # Verify results
        assert len(results) == 2
        
        # First result should be most relevant
        assert results[0].score > results[1].score
        assert "genomics" in results[0].payload["text"].lower()
        
        # Verify search was performed with correct parameters
        mock_qdrant.search.assert_called_once()
        search_call = mock_qdrant.search.call_args
        assert search_call[1]['limit'] == 2
        assert search_call[1]['collection_name'] == "test_collection"
        
        # Verify embedding was generated
        mock_embeddings.create.assert_called_once()
        embedding_call = mock_embeddings.create.call_args
        assert embedding_call[1]['input'] == query
        assert embedding_call[1]['model'] == "text-embedding-3-small"


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    def test_chunking_performance(self):
        """Test chunking performance with different text sizes."""
        import time
        
        # Small text
        small_text = "Short text. " * 10
        start_time = time.time()
        small_chunks = create_chunks(small_text, chunk_size=100, chunk_overlap=20)
        small_time = time.time() - start_time
        
        # Large text
        large_text = "Longer text content. " * 1000
        start_time = time.time()
        large_chunks = create_chunks(large_text, chunk_size=100, chunk_overlap=20)
        large_time = time.time() - start_time
        
        # Verify performance is reasonable
        assert small_time < 1.0  # Should be very fast
        assert large_time < 5.0  # Should still be reasonable
        
        # Verify chunk counts are reasonable
        assert len(small_chunks) <= 5
        assert len(large_chunks) <= 1000
    
    def test_quality_metrics_performance(self):
        """Test quality metrics calculation performance."""
        import time
        
        # Test with different text sizes
        text_sizes = [100, 1000, 10000]
        
        for size in text_sizes:
            text = "Sample text content. " * (size // 20)
            
            start_time = time.time()
            metrics = quality_metrics(text)
            calc_time = time.time() - start_time
            
            # Should be fast even for large texts
            assert calc_time < 1.0
            
            # Verify metrics are correct
            assert metrics["char_len"] == len(text)
            assert metrics["token_est"] > 0
    
    def test_memory_usage(self):
        """Test memory usage characteristics."""
        import gc
        import sys
        
        # Test chunking memory usage
        large_text = "Large text content. " * 10000
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Measure memory before
        before_memory = sys.getsizeof(large_text)
        
        # Create chunks
        chunks = create_chunks(large_text, chunk_size=1000, chunk_overlap=100)
        
        # Measure memory after
        after_memory = sys.getsizeof(chunks) + sum(sys.getsizeof(chunk.page_content) for chunk in chunks)
        
        # Memory usage should be reasonable (not more than 10x original)
        assert after_memory < before_memory * 10
        
        # Clean up
        del chunks
        gc.collect()

