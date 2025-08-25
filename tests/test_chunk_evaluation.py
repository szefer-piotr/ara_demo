"""
Tests for chunk evaluation functionality.
"""
import pytest
import tempfile
import os
import csv
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunk_evaluation import (
    extract_text, longest_run_len, quality_metrics, 
    process_collection, main
)


class TestTextExtraction:
    """Test text extraction functionality."""
    
    def test_extract_text_from_string_field(self):
        """Test extracting text from a string field."""
        payload = {"text": "Sample text content"}
        fields = ["text"]
        
        result = extract_text(payload, fields)
        assert result == "Sample text content"
    
    def test_extract_text_from_list_field(self):
        """Test extracting text from a list field."""
        payload = {"content": ["part1", "part2", "part3"]}
        fields = ["content"]
        
        result = extract_text(payload, fields)
        assert result == "part1\npart2\npart3"
    
    def test_extract_text_from_numeric_field(self):
        """Test extracting text from a numeric field."""
        payload = {"value": 42}
        fields = ["value"]
        
        result = extract_text(payload, fields)
        assert result == "42"
    
    def test_extract_text_from_multiple_fields(self):
        """Test extracting text from multiple fields, using first non-empty."""
        payload = {
            "empty_field": "",
            "text_field": "Sample text",
            "another_field": "Another text"
        }
        fields = ["empty_field", "text_field", "another_field"]
        
        result = extract_text(payload, fields)
        assert result == "Sample text"
    
    def test_extract_text_no_valid_fields(self):
        """Test extracting text when no valid fields exist."""
        payload = {"empty_field": "", "null_field": None}
        fields = ["empty_field", "null_field"]
        
        result = extract_text(payload, fields)
        assert result == ""
    
    def test_extract_text_missing_fields(self):
        """Test extracting text when fields don't exist in payload."""
        payload = {"existing_field": "text"}
        fields = ["missing_field", "existing_field"]
        
        result = extract_text(payload, fields)
        assert result == "text"


class TestLongestRunLength:
    """Test longest run length calculation."""
    
    def test_longest_run_single_character(self):
        """Test longest run with single character."""
        text = "aaaa"
        result = longest_run_len(text)
        assert result == 4
    
    def test_longest_run_multiple_characters(self):
        """Test longest run with multiple characters."""
        text = "aaabbbcccc"
        result = longest_run_len(text)
        assert result == 4  # 'c' has the longest run
    
    def test_longest_run_no_repeats(self):
        """Test longest run with no repeated characters."""
        text = "abcdef"
        result = longest_run_len(text)
        assert result == 1
    
    def test_longest_run_empty_string(self):
        """Test longest run with empty string."""
        text = ""
        result = longest_run_len(text)
        assert result == 0
    
    def test_longest_run_single_character_string(self):
        """Test longest run with single character string."""
        text = "a"
        result = longest_run_len(text)
        assert result == 1


class TestQualityMetrics:
    """Test quality metrics calculation."""
    
    def test_quality_metrics_empty_text(self):
        """Test quality metrics with empty text."""
        result = quality_metrics("")
        
        assert result["char_len"] == 0
        assert result["token_est"] == 0
        assert result["lines"] == 0
        assert result["avg_line_len"] == 0.0
        assert result["pct_ascii"] == 0.0
        assert result["pct_alnum"] == 0.0
        assert result["pct_ws"] == 0.0
        assert result["pct_punct"] == 0.0
        assert result["uniq_char_ratio"] == 0.0
        assert result["longest_run"] == 0
        assert result["has_refs"] == False
        assert result["looks_like_table"] == False
    
    def test_quality_metrics_basic_text(self):
        """Test quality metrics with basic text."""
        text = "Hello world! This is a test."
        result = quality_metrics(text)
        
        assert result["char_len"] == 28
        assert result["token_est"] == 6
        assert result["lines"] == 1
        assert result["avg_line_len"] == 28.0
        assert result["pct_ascii"] == 1.0
        assert result["pct_alnum"] > 0.5
        assert result["pct_ws"] > 0.1
        assert result["pct_punct"] > 0.05
        assert result["uniq_char_ratio"] > 0.5
        assert result["longest_run"] == 1
        assert result["has_refs"] == False
        assert result["looks_like_table"] == False
    
    def test_quality_metrics_multiline_text(self):
        """Test quality metrics with multiline text."""
        text = "Line 1\nLine 2\nLine 3"
        result = quality_metrics(text)
        
        assert result["char_len"] == 20
        assert result["token_est"] == 6
        assert result["lines"] == 3
        assert result["avg_line_len"] == 20.0 / 3
    
    def test_quality_metrics_with_references(self):
        """Test quality metrics with reference-like text."""
        text = "This paper discusses machine learning. References: [1] Smith et al."
        result = quality_metrics(text)
        
        assert result["has_refs"] == True
    
    def test_quality_metrics_with_table_like_content(self):
        """Test quality metrics with table-like content."""
        text = "Table 1: Results\n| Col1 | Col2 |\n|------|------|"
        result = quality_metrics(text)
        
        assert result["looks_like_table"] == True
    
    def test_quality_metrics_unicode_text(self):
        """Test quality metrics with unicode text."""
        text = "Café résumé naïve"
        result = quality_metrics(text)
        
        assert result["pct_ascii"] < 1.0
        assert result["char_len"] == 20
    
    def test_quality_metrics_repeated_characters(self):
        """Test quality metrics with repeated characters."""
        text = "aaaaa bbbbb ccccc"
        result = quality_metrics(text)
        
        assert result["longest_run"] == 5


class TestCollectionProcessing:
    """Test collection processing functionality."""
    
    @patch('chunk_evaluation.QdrantClient')
    def test_process_collection_success(self, mock_qdrant_class):
        """Test successful collection processing."""
        # Mock Qdrant client
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        
        # Mock collection info
        mock_collection = Mock()
        mock_collection.points_count = 100
        mock_client.get_collection.return_value = mock_collection
        
        # Mock search results
        mock_points = []
        for i in range(10):
            mock_point = Mock()
            mock_point.payload = {
                "text": f"Sample text {i}",
                "paper_id": f"paper_{i}",
                "chunk_id": i
            }
            mock_points.append(mock_point)
        
        mock_client.scroll.return_value = (mock_points, None)
        
        # Process collection
        result = process_collection(
            mock_client, "test_collection", ["text"], max_points=10, batch_size=5
        )
        
        assert len(result) == 10
        assert all("quality_metrics" in point for point in result)
        assert all("extracted_text" in point for point in result)
    
    @patch('chunk_evaluation.QdrantClient')
    def test_process_collection_empty(self, mock_qdrant_class):
        """Test processing empty collection."""
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        
        mock_collection = Mock()
        mock_collection.points_count = 0
        mock_client.get_collection.return_value = mock_collection
        
        result = process_collection(
            mock_client, "test_collection", ["text"]
        )
        
        assert result == []
    
    @patch('chunk_evaluation.QdrantClient')
    def test_process_collection_with_limit(self, mock_qdrant_class):
        """Test collection processing with point limit."""
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        
        mock_collection = Mock()
        mock_collection.points_count = 100
        mock_client.get_collection.return_value = mock_collection
        
        # Mock search results
        mock_points = []
        for i in range(20):
            mock_point = Mock()
            mock_point.payload = {"text": f"Sample text {i}"}
            mock_points.append(mock_point)
        
        mock_client.scroll.return_value = (mock_points, None)
        
        # Process with limit
        result = process_collection(
            mock_client, "test_collection", ["text"], max_points=10
        )
        
        assert len(result) == 10
    
    @patch('chunk_evaluation.QdrantClient')
    def test_process_collection_batch_processing(self, mock_qdrant_class):
        """Test collection processing with batching."""
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        
        mock_collection = Mock()
        mock_collection.points_count = 100
        mock_client.get_collection.return_value = mock_collection
        
        # Mock multiple scroll calls for batching
        mock_points_batch1 = [
            Mock(payload={"text": f"Text {i}"}) for i in range(5)
        ]
        mock_points_batch2 = [
            Mock(payload={"text": f"Text {i}"}) for i in range(5, 10)
        ]
        
        mock_client.scroll.side_effect = [
            (mock_points_batch1, "offset1"),
            (mock_points_batch2, None)
        ]
        
        result = process_collection(
            mock_client, "test_collection", ["text"], max_points=10, batch_size=5
        )
        
        assert len(result) == 10
        assert mock_client.scroll.call_count == 2


class TestCSVOutput:
    """Test CSV output functionality."""
    
    def test_csv_output_format(self, temp_dir):
        """Test CSV output format and content."""
        sample_data = [
            {
                "paper_id": "paper_1",
                "chunk_id": 0,
                "extracted_text": "Sample text 1",
                "quality_metrics": {
                    "char_len": 13,
                    "token_est": 2,
                    "lines": 1,
                    "avg_line_len": 13.0,
                    "pct_ascii": 1.0,
                    "pct_alnum": 0.77,
                    "pct_ws": 0.23,
                    "pct_punct": 0.0,
                    "uniq_char_ratio": 0.85,
                    "longest_run": 1,
                    "has_refs": False,
                    "looks_like_table": False
                }
            }
        ]
        
        csv_file = temp_dir / "test_output.csv"
        
        # Write CSV
        with open(csv_file, 'w', newline='') as f:
            if sample_data:
                fieldnames = ['paper_id', 'chunk_id', 'extracted_text']
                fieldnames.extend(sample_data[0]['quality_metrics'].keys())
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in sample_data:
                    row = {
                        'paper_id': item['paper_id'],
                        'chunk_id': item['chunk_id'],
                        'extracted_text': item['extracted_text']
                    }
                    row.update(item['quality_metrics'])
                    writer.writerow(row)
        
        # Verify CSV was created and has content
        assert csv_file.exists()
        
        with open(csv_file, 'r') as f:
            content = f.read()
            assert "paper_id" in content
            assert "char_len" in content
            assert "Sample text 1" in content


class TestMainFunction:
    """Test main function functionality."""
    
    @patch('chunk_evaluation.QdrantClient')
    @patch('chunk_evaluation.process_collection')
    def test_main_function_success(self, mock_process, mock_qdrant_class):
        """Test successful main function execution."""
        # Mock Qdrant client
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        
        # Mock collection info
        mock_collection = Mock()
        mock_collection.points_count = 100
        mock_client.get_collection.return_value = mock_collection
        
        # Mock processed data
        mock_data = [
            {
                "paper_id": "paper_1",
                "chunk_id": 0,
                "extracted_text": "Sample text",
                "quality_metrics": {"char_len": 11, "token_est": 2}
            }
        ]
        mock_process.return_value = mock_data
        
        # Test with minimal arguments
        test_args = [
            'chunk_evaluation.py',
            '-c', 'test_collection'
        ]
        
        with patch('sys.argv', test_args):
            main()
        
        # Verify client was created
        mock_qdrant_class.assert_called_once()
        
        # Verify collection was processed
        mock_process.assert_called_once()
    
    @patch('chunk_evaluation.QdrantClient')
    def test_main_function_collection_not_found(self, mock_qdrant_class):
        """Test main function when collection doesn't exist."""
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        
        # Mock collection not found
        mock_client.get_collection.side_effect = Exception("Collection not found")
        
        test_args = [
            'chunk_evaluation.py',
            '-c', 'nonexistent_collection'
        ]
        
        with patch('sys.argv', test_args), \
             pytest.raises(Exception, match="Collection not found"):
            main()
    
    @patch('chunk_evaluation.QdrantClient')
    def test_main_function_with_output_file(self, mock_qdrant_class, temp_dir):
        """Test main function with output file specified."""
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        
        mock_collection = Mock()
        mock_collection.points_count = 100
        mock_client.get_collection.return_value = mock_collection
        
        # Mock empty data
        with patch('chunk_evaluation.process_collection') as mock_process:
            mock_process.return_value = []
            
            output_file = temp_dir / "test_output.csv"
            test_args = [
                'chunk_evaluation.py',
                '-c', 'test_collection',
                '--out', str(output_file)
            ]
            
            with patch('sys.argv', test_args):
                main()
            
            # Verify output file was created (even if empty)
            assert output_file.exists()


class TestIntegration:
    """Test integration between components."""
    
    def test_full_evaluation_pipeline(self, temp_dir):
        """Test full evaluation pipeline integration."""
        # Create sample data
        sample_data = [
            {
                "paper_id": "paper_1",
                "chunk_id": 0,
                "extracted_text": "This is a sample text for evaluation.",
                "quality_metrics": quality_metrics("This is a sample text for evaluation.")
            },
            {
                "paper_id": "paper_1",
                "chunk_id": 1,
                "extracted_text": "Another chunk with different content.",
                "quality_metrics": quality_metrics("Another chunk with different content.")
            }
        ]
        
        # Test CSV output
        csv_file = temp_dir / "evaluation_output.csv"
        
        with open(csv_file, 'w', newline='') as f:
            if sample_data:
                fieldnames = ['paper_id', 'chunk_id', 'extracted_text']
                fieldnames.extend(sample_data[0]['quality_metrics'].keys())
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in sample_data:
                    row = {
                        'paper_id': item['paper_id'],
                        'chunk_id': item['chunk_id'],
                        'extracted_text': item['extracted_text']
                    }
                    row.update(item['quality_metrics'])
                    writer.writerow(row)
        
        # Verify output
        assert csv_file.exists()
        
        with open(csv_file, 'r') as f:
            content = f.read()
            assert "paper_1" in content
            assert "sample text" in content
            assert "char_len" in content
            assert "token_est" in content
    
    def test_quality_metrics_consistency(self):
        """Test that quality metrics are consistent across different text types."""
        # Test with scientific text
        scientific_text = "The results demonstrate a significant improvement (p < 0.05) in performance."
        scientific_metrics = quality_metrics(scientific_text)
        
        # Test with table-like text
        table_text = "Table 1: Results\n| Metric | Value |\n|--------|-------|"
        table_metrics = quality_metrics(table_text)
        
        # Test with reference text
        ref_text = "Previous work [1, 2] has shown similar results. References: [1] Smith et al."
        ref_metrics = quality_metrics(ref_text)
        
        # Verify metrics are reasonable
        assert scientific_metrics["char_len"] > 0
        assert table_metrics["looks_like_table"] == True
        assert ref_metrics["has_refs"] == True
        
        # Verify all metrics have expected types
        for metrics in [scientific_metrics, table_metrics, ref_metrics]:
            assert isinstance(metrics["char_len"], int)
            assert isinstance(metrics["pct_ascii"], float)
            assert isinstance(metrics["has_refs"], bool)
            assert isinstance(metrics["looks_like_table"], bool)

