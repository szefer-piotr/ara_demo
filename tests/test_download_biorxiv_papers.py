"""
Tests for bioarxiv paper download functionality.
"""
import pytest
import datetime as dt
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download_biorxiv_papers import (
    RxivEntry, build_query, match_all_terms, date_range_windows,
    ensure_bucket, object_exists, sha256_bytes, fetch_papers,
    download_paper, main
)


class TestRxivEntry:
    """Test RxivEntry dataclass functionality."""
    
    def test_rxiv_entry_creation(self):
        """Test creating a RxivEntry instance."""
        entry = RxivEntry(
            doi="10.1101/2023.01.01.123456",
            title="Test Paper",
            version=1,
            server="biorxiv"
        )
        
        assert entry.doi == "10.1101/2023.01.01.123456"
        assert entry.title == "Test Paper"
        assert entry.version == 1
        assert entry.server == "biorxiv"
    
    def test_paper_id_property(self):
        """Test paper_id property converts DOI correctly."""
        entry = RxivEntry(
            doi="10.1101/2023.01.01.123456",
            title="Test Paper",
            version=1,
            server="biorxiv"
        )
        
        assert entry.paper_id == "10.1101_2023.01.01.123456"
    
    def test_pdf_url_biorxiv(self):
        """Test PDF URL generation for bioRxiv papers."""
        entry = RxivEntry(
            doi="10.1101/2023.01.01.123456",
            title="Test Paper",
            version=1,
            server="biorxiv"
        )
        
        expected_url = "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1.full.pdf"
        assert entry.pdf_url() == expected_url
    
    def test_pdf_url_medrxiv(self):
        """Test PDF URL generation for medRxiv papers."""
        entry = RxivEntry(
            doi="10.1101/2023.01.01.123456",
            title="Test Paper",
            version=1,
            server="medrxiv"
        )
        
        expected_url = "https://www.medrxiv.org/content/10.1101/2023.01.01.123456v1.full.pdf"
        assert entry.pdf_url() == expected_url


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_build_query(self):
        """Test query building from keywords."""
        keywords = ["  machine learning  ", "  ", "deep learning", ""]
        result = build_query(keywords)
        
        assert result == ["machine learning", "deep learning"]
    
    def test_match_all_terms(self):
        """Test term matching functionality."""
        text = "This is a sample text about machine learning and deep learning"
        
        # All terms should match
        assert match_all_terms(text, ["machine", "learning"]) == True
        assert match_all_terms(text, ["deep", "learning"]) == True
        
        # Not all terms should match
        assert match_all_terms(text, ["machine", "python"]) == False
        assert match_all_terms(text, ["neural", "networks"]) == False
    
    def test_date_range_windows(self):
        """Test date range window generation."""
        start_date = dt.date(2023, 1, 1)
        end_date = dt.date(2023, 1, 10)
        windows = list(date_range_windows(start_date, end_date, 3))
        
        assert len(windows) == 4
        assert windows[0] == (dt.date(2023, 1, 1), dt.date(2023, 1, 3))
        assert windows[1] == (dt.date(2023, 1, 4), dt.date(2023, 1, 6))
        assert windows[2] == (dt.date(2023, 1, 7), dt.date(2023, 1, 9))
        assert windows[3] == (dt.date(2023, 1, 10), dt.date(2023, 1, 10))


class TestMinioOperations:
    """Test MinIO operations."""
    
    def test_ensure_bucket_creates_new(self, mock_minio_client):
        """Test bucket creation when it doesn't exist."""
        ensure_bucket(mock_minio_client, "new-bucket")
        mock_minio_client.make_bucket.assert_called_once_with("new-bucket")
    
    def test_ensure_bucket_exists(self, mock_minio_client):
        """Test bucket creation when it already exists."""
        # Mock S3Error for existing bucket
        from minio.error import S3Error
        mock_minio_client.make_bucket.side_effect = S3Error(
            "BucketAlreadyOwnedByYou", "Bucket already exists"
        )
        
        # Should not raise exception
        ensure_bucket(mock_minio_client, "existing-bucket")
    
    def test_object_exists_true(self, mock_minio_client):
        """Test object existence check when object exists."""
        result = object_exists(mock_minio_client, "bucket", "key")
        assert result == True
        mock_minio_client.stat_object.assert_called_once_with("bucket", "key")
    
    def test_object_exists_false(self, mock_minio_client):
        """Test object existence check when object doesn't exist."""
        from minio.error import S3Error
        mock_minio_client.stat_object.side_effect = S3Error(
            "NoSuchKey", "Object not found"
        )
        
        result = object_exists(mock_minio_client, "bucket", "key")
        assert result == False
    
    def test_sha256_bytes(self):
        """Test SHA256 hash calculation."""
        test_data = b"test data"
        expected_hash = "916f0cbfcb9a2ce135187640b9c8a63a75cc0fd4e4f3a5f76231b8e33d47c248"
        
        result = sha256_bytes(test_data)
        assert result == expected_hash


class TestPaperFetching:
    """Test paper fetching functionality."""
    
    @patch('download_biorxiv_papers.requests.get')
    def test_fetch_papers_success(self, mock_get, sample_biorxiv_response):
        """Test successful paper fetching."""
        mock_response = Mock()
        mock_response.json.return_value = sample_biorxiv_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        papers = list(fetch_papers("biorxiv", dt.date(2023, 1, 1), dt.date(2023, 1, 1)))
        
        assert len(papers) == 1
        assert papers[0].doi == "10.1101/2023.01.01.123456"
        assert papers[0].title == "Sample Research Paper Title"
        assert papers[0].server == "biorxiv"
    
    @patch('download_biorxiv_papers.requests.get')
    def test_fetch_papers_api_error(self, mock_get):
        """Test paper fetching with API error."""
        mock_get.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            list(fetch_papers("biorxiv", dt.date(2023, 1, 1), dt.date(2023, 1, 1)))


class TestPaperDownload:
    """Test paper download functionality."""
    
    @patch('download_biorxiv_papers.requests.get')
    def test_download_paper_success(self, mock_get, mock_minio_client, temp_dir):
        """Test successful paper download."""
        # Mock PDF content
        pdf_content = b"%PDF-1.4\nSample PDF content"
        mock_response = Mock()
        mock_response.content = pdf_content
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock object doesn't exist
        mock_minio_client.stat_object.side_effect = Exception("NoSuchKey")
        
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
    
    @patch('download_biorxiv_papers.requests.get')
    def test_download_paper_already_exists(self, mock_get, mock_minio_client, temp_dir):
        """Test paper download when paper already exists."""
        entry = RxivEntry(
            doi="10.1101/2023.01.01.123456",
            title="Test Paper",
            version=1,
            server="biorxiv"
        )
        
        result = download_paper(
            entry, mock_minio_client, "test-bucket", "papers/", temp_dir
        )
        
        assert result == False
        mock_minio_client.put_object.assert_not_called()


class TestMainFunction:
    """Test main function functionality."""
    
    @patch('download_biorxiv_papers.Minio')
    @patch('download_biorxiv_papers.fetch_papers')
    @patch('download_biorxiv_papers.download_paper')
    def test_main_function(self, mock_download, mock_fetch, mock_minio_class):
        """Test main function execution."""
        # Mock MinIO client
        mock_minio = Mock()
        mock_minio_class.return_value = mock_minio
        
        # Mock paper fetching
        mock_papers = [
            RxivEntry(
                doi="10.1101/2023.01.01.123456",
                title="Test Paper 1",
                version=1,
                server="biorxiv"
            ),
            RxivEntry(
                doi="10.1101/2023.01.01.789012",
                title="Test Paper 2",
                version=1,
                server="biorxiv"
            )
        ]
        mock_fetch.return_value = mock_papers
        
        # Mock download results
        mock_download.side_effect = [True, False]  # First succeeds, second already exists
        
        # Test with minimal arguments
        with patch('sys.argv', ['download_biorxiv_papers.py', '--server', 'biorxiv']):
            main()
        
        # Verify MinIO client was created
        mock_minio_class.assert_called_once()
        
        # Verify papers were fetched
        mock_fetch.assert_called_once()
        
        # Verify download was attempted for both papers
        assert mock_download.call_count == 2

