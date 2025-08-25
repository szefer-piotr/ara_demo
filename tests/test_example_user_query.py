"""
Example test demonstrating the pipeline with a real user query.
This test shows how the complete pipeline would work for a user searching
for information about machine learning in bioinformatics.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download_biorxiv_papers import RxivEntry
from chunk_arxiv_papers import create_chunks
from chunk_evaluation import quality_metrics
from similarity_search_arxiv_papers import SimilaritySearcher, OpenAIEmbeddingProvider


class TestExampleUserQuery:
    """Test the pipeline with an example user query."""
    
    def test_user_query_machine_learning_bioinformatics(self):
        """
        Test the complete pipeline for a user query:
        "What are the latest advances in machine learning for bioinformatics?"
        """
        
        # Step 1: Simulate papers that would be downloaded
        sample_papers = [
            RxivEntry(
                doi="10.1101/2023.01.01.123456",
                title="Deep Learning Approaches in Genomic Sequence Analysis",
                version=1,
                server="biorxiv"
            ),
            RxivEntry(
                doi="10.1101/2023.01.02.789012",
                title="Machine Learning for Protein Structure Prediction: A Comprehensive Review",
                version=1,
                server="biorxiv"
            ),
            RxivEntry(
                doi="10.1101/2023.01.03.345678",
                title="Transformer Models in Bioinformatics: Applications and Challenges",
                version=1,
                server="biorxiv"
            )
        ]
        
        # Verify papers are properly structured
        for paper in sample_papers:
            assert paper.doi.startswith("10.1101/")
            assert len(paper.title) > 0
            assert paper.server in ["biorxiv", "medrxiv"]
            assert paper.version > 0
        
        # Step 2: Simulate text content from these papers
        paper_contents = {
            "10.1101/2023.01.01.123456": """
            Deep learning has revolutionized genomic sequence analysis in recent years. 
            Convolutional neural networks and recurrent neural networks have shown remarkable 
            performance in predicting gene function and regulatory elements. The application 
            of transformer architectures to genomic data has opened new possibilities for 
            understanding complex biological relationships.
            """,
            
            "10.1101/2023.01.02.789012": """
            Machine learning approaches have significantly advanced protein structure prediction. 
            AlphaFold2 demonstrated unprecedented accuracy in predicting protein structures 
            from amino acid sequences. This breakthrough has implications for drug discovery, 
            understanding disease mechanisms, and protein engineering.
            """,
            
            "10.1101/2023.01.03.345678": """
            Transformer models have shown exceptional performance in various bioinformatics tasks. 
            BERT-based models trained on protein sequences can predict protein function and 
            structure. Attention mechanisms allow these models to capture long-range dependencies 
            in biological sequences.
            """
        }
        
        # Step 3: Create chunks from each paper
        all_chunks = []
        chunk_metadata = []
        
        for paper in sample_papers:
            content = paper_contents[paper.doi]
            chunks = create_chunks(content, chunk_size=150, chunk_overlap=30)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk.page_content)
                chunk_metadata.append({
                    "paper_id": paper.paper_id,
                    "paper_title": paper.title,
                    "chunk_id": i,
                    "doi": paper.doi,
                    "source": f"{paper.paper_id}.pdf"
                })
        
        # Verify chunking worked properly
        assert len(all_chunks) > 0
        assert len(all_chunks) == len(chunk_metadata)
        
        # Step 4: Evaluate chunk quality
        chunk_qualities = []
        for chunk_text in all_chunks:
            quality = quality_metrics(chunk_text)
            chunk_qualities.append(quality)
            
            # Verify quality metrics are reasonable
            assert quality["char_len"] > 0
            assert quality["token_est"] > 0
            assert quality["pct_ascii"] > 0.8  # Most should be ASCII
            assert quality["lines"] >= 1
        
        # Step 5: Simulate vector embeddings (mock OpenAI)
        with patch('similarity_search_arxiv_papers.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_embeddings = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_embeddings.create.return_value = mock_response
            mock_client.embeddings = mock_embeddings
            mock_openai.return_value = mock_client
            
            # Create embedding provider
            embedding_provider = OpenAIEmbeddingProvider("test-key")
            
            # Test embedding generation
            query = "What are the latest advances in machine learning for bioinformatics?"
            embedding_result = embedding_provider.embed(query)
            
            assert len(embedding_result.vector) == 1536
            assert embedding_result.model == "text-embedding-3-small"
        
        # Step 6: Simulate similarity search
        with patch('similarity_search_arxiv_papers.QdrantClient') as mock_qdrant_class:
            mock_qdrant = Mock()
            mock_qdrant_class.return_value = mock_qdrant
            
            # Mock search results
            mock_search_results = [
                Mock(
                    score=0.95,
                    payload={
                        "text": all_chunks[0],
                        "paper_id": chunk_metadata[0]["paper_id"],
                        "paper_title": chunk_metadata[0]["paper_title"],
                        "chunk_id": chunk_metadata[0]["chunk_id"],
                        "doi": chunk_metadata[0]["doi"],
                        "source": chunk_metadata[0]["source"]
                    },
                    id="point_1"
                ),
                Mock(
                    score=0.92,
                    payload={
                        "text": all_chunks[1],
                        "paper_id": chunk_metadata[1]["paper_id"],
                        "paper_title": chunk_metadata[1]["paper_title"],
                        "chunk_id": chunk_metadata[1]["chunk_id"],
                        "doi": chunk_metadata[1]["doi"],
                        "source": chunk_metadata[1]["source"]
                    },
                    id="point_2"
                ),
                Mock(
                    score=0.88,
                    payload={
                        "text": all_chunks[2],
                        "paper_id": chunk_metadata[2]["paper_id"],
                        "paper_title": chunk_metadata[2]["paper_title"],
                        "chunk_id": chunk_metadata[2]["chunk_id"],
                        "doi": chunk_metadata[2]["doi"],
                        "source": chunk_metadata[2]["source"]
                    },
                    id="point_3"
                )
            ]
            mock_qdrant.search.return_value = mock_search_results
            
            # Create searcher and perform search
            searcher = SimilaritySearcher(mock_qdrant, embedding_provider, "bioarxiv_chunks")
            
            results = searcher.search(query, top_k=3)
            
            # Verify search results
            assert len(results) == 3
            
            # Results should be ordered by relevance score
            assert results[0].score >= results[1].score >= results[2].score
            
            # Verify content relevance
            first_result = results[0]
            assert "deep learning" in first_result.payload["text"].lower()
            assert "genomic" in first_result.payload["text"].lower()
            assert first_result.payload["score"] == 0.95
            
            # Verify metadata is preserved
            assert "paper_title" in first_result.payload
            assert "doi" in first_result.payload
            assert "source" in first_result.payload
    
    def test_user_query_protein_structure_prediction(self):
        """
        Test the pipeline for another user query:
        "How has machine learning improved protein structure prediction?"
        """
        
        # Simulate relevant paper content
        protein_paper_content = """
        Protein structure prediction has been revolutionized by machine learning approaches.
        AlphaFold2, developed by DeepMind, achieved unprecedented accuracy in the Critical 
        Assessment of protein Structure Prediction (CASP) competition. The model uses 
        attention mechanisms and multiple sequence alignments to predict protein structures 
        with remarkable precision. This breakthrough has significant implications for 
        drug discovery, as understanding protein structure is crucial for designing 
        effective therapeutics.
        """
        
        # Create chunks
        chunks = create_chunks(protein_paper_content, chunk_size=200, chunk_overlap=50)
        
        # Verify chunking
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 200 for chunk in chunks)
        
        # Check that key concepts are preserved across chunks
        all_text = " ".join(chunk.page_content for chunk in chunks)
        assert "AlphaFold2" in all_text
        assert "protein structure" in all_text.lower()
        assert "machine learning" in all_text.lower()
        
        # Evaluate chunk quality
        for chunk in chunks:
            quality = quality_metrics(chunk.page_content)
            assert quality["char_len"] > 0
            assert quality["token_est"] > 0
            assert quality["pct_alnum"] > 0.5  # Should have substantial alphanumeric content
    
    def test_user_query_transformer_models_bioinformatics(self):
        """
        Test the pipeline for a specific technical query:
        "What are the applications of transformer models in bioinformatics?"
        """
        
        # Simulate technical paper content
        transformer_content = """
        Transformer models have demonstrated exceptional performance in bioinformatics applications.
        BERT-based models trained on protein sequences can predict protein function, structure,
        and interactions. The attention mechanism allows these models to capture long-range
        dependencies in biological sequences, which is crucial for understanding protein
        folding and function. Recent advances include ProtBERT, which was pre-trained on
        millions of protein sequences and fine-tuned for specific tasks.
        """
        
        # Create chunks with different sizes to test flexibility
        small_chunks = create_chunks(transformer_content, chunk_size=100, chunk_overlap=20)
        large_chunks = create_chunks(transformer_content, chunk_size=300, chunk_overlap=50)
        
        # Verify different chunking strategies
        assert len(small_chunks) > len(large_chunks)
        
        # Check content preservation
        small_text = " ".join(chunk.page_content for chunk in small_chunks)
        large_text = " ".join(chunk.page_content for chunk in large_chunks)
        
        assert "transformer" in small_text.lower()
        assert "BERT" in large_text
        assert "protein" in small_text.lower()
        
        # Test quality metrics for different chunk sizes
        for chunk in small_chunks:
            quality = quality_metrics(chunk.page_content)
            assert quality["char_len"] <= 100
            assert quality["pct_ascii"] > 0.8
        
        for chunk in large_chunks:
            quality = quality_metrics(chunk.page_content)
            assert quality["char_len"] <= 300
            assert quality["token_est"] > 0
    
    def test_pipeline_data_integrity(self):
        """Test that data integrity is maintained throughout the pipeline."""
        
        # Original paper
        paper = RxivEntry(
            doi="10.1101/2023.01.01.test123",
            title="Test Paper for Data Integrity",
            version=1,
            server="biorxiv"
        )
        
        # Original content
        original_content = """
        This is a test paper about machine learning in bioinformatics.
        It contains important information about deep learning approaches.
        The content should be preserved accurately through the pipeline.
        """
        
        # Create chunks
        chunks = create_chunks(original_content, chunk_size=100, chunk_overlap=20)
        
        # Verify chunks contain original content
        all_chunk_text = " ".join(chunk.page_content for chunk in chunks)
        assert "machine learning" in all_chunk_text.lower()
        assert "bioinformatics" in all_chunk_text.lower()
        assert "deep learning" in all_chunk_text.lower()
        
        # Verify metadata consistency
        for i, chunk in enumerate(chunks):
            # In a real pipeline, each chunk would have metadata
            chunk_info = {
                "paper_id": paper.paper_id,
                "paper_title": paper.title,
                "chunk_id": i,
                "doi": paper.doi,
                "source": f"{paper.paper_id}.pdf"
            }
            
            # Verify chunk info is consistent
            assert chunk_info["paper_id"] == paper.paper_id
            assert chunk_info["chunk_id"] == i
            assert chunk_info["doi"] == paper.doi
        
        # Verify quality metrics are consistent
        for chunk in chunks:
            quality = quality_metrics(chunk.page_content)
            
            # Basic sanity checks
            assert quality["char_len"] == len(chunk.page_content)
            assert quality["token_est"] > 0
            assert quality["pct_ascii"] > 0.8
            assert quality["lines"] >= 1
            
            # Verify percentages sum to approximately 100%
            total_pct = (quality["pct_ascii"] + quality["pct_alnum"] + 
                        quality["pct_ws"] + quality["pct_punct"])
            assert 0.95 <= total_pct <= 1.05  # Allow for small rounding errors

