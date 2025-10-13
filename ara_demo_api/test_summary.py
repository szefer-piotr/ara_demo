#!/usr/bin/env python3
"""
Test script for data summarization feature

Usage:
    python3 test_summary.py
    python3 test_summary.py --no-llm  # Skip LLM descriptions
"""

import requests
import json
import sys
import argparse
from datetime import datetime

API_URL = "http://localhost:8000"
TEST_FILE = "test_data.csv"


def print_section(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)


def print_success(msg):
    """Print success message"""
    print(f"✓ {msg}")


def print_error(msg):
    """Print error message"""
    print(f"✗ {msg}")


def test_health():
    """Test API health"""
    print_section("1. Health Check")
    
    response = requests.get(f"{API_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print_success(f"API is {data.get('data', {}).get('status', 'unknown')}")
        return True
    else:
        print_error(f"Health check failed: {response.status_code}")
        return False


def create_session(session_id):
    """Create a test session"""
    print_section("2. Create Session")
    
    response = requests.post(
        f"{API_URL}/sessions",
        headers={"X-Session-ID": session_id}
    )
    
    if response.status_code == 200:
        data = response.json()
        print_success(f"Session: {data.get('session_id')}")
        print(f"   Created: {data.get('created_at')}")
        return True
    else:
        print_error(f"Session creation failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return False


def upload_file(session_id, filepath):
    """Upload CSV file"""
    print_section("3. Upload File")
    
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (filepath, f, 'text/csv')}
            headers = {'X-Session-ID': session_id}
            
            response = requests.post(
                f"{API_URL}/files",
                files=files,
                headers=headers
            )
        
        if response.status_code == 200:
            data = response.json()
            file_id = data.get('file_id')
            print_success(f"File uploaded: {file_id}")
            print(f"   Filename: {data.get('file_name')}")
            print(f"   Size: {data.get('file_size')} bytes")
            print(f"   Rows: {data.get('metadata', {}).get('rows')}")
            print(f"   Columns: {data.get('metadata', {}).get('columns')}")
            print(f"   Encoding: {data.get('metadata', {}).get('encoding')}")
            return file_id
        else:
            print_error(f"Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except FileNotFoundError:
        print_error(f"File not found: {filepath}")
        return None


def get_summary(file_id, with_descriptions=True):
    """Get data summary"""
    section_title = "4. Get Summary " + ("WITH" if with_descriptions else "WITHOUT") + " LLM Descriptions"
    print_section(section_title)
    
    response = requests.get(
        f"{API_URL}/files/{file_id}/summary",
        params={"infer_descriptions": with_descriptions}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        print_success("Summary generated")
        print(f"   Rows: {data.get('row_count')}")
        print(f"   Columns: {data.get('column_count')}")
        
        # Display columns
        print("\n   Column Information:")
        for col in data.get('columns', [])[:5]:  # Show first 5
            print(f"   - {col['name']} ({col['type']})")
            print(f"     Unique: {col['unique_values']}, Missing: {col['null_count']}")
            
            if 'description' in col and with_descriptions:
                print(f"     Description: {col['description']}")
            
            if 'statistics' in col:
                stats = col['statistics']
                print(f"     Stats: mean={stats.get('mean', 'N/A'):.2f}, "
                      f"min={stats.get('min', 'N/A'):.2f}, "
                      f"max={stats.get('max', 'N/A'):.2f}")
        
        # Show descriptions summary
        if with_descriptions and 'column_descriptions' in data:
            print("\n   LLM-Inferred Descriptions:")
            for col_name, desc in data['column_descriptions'].items():
                print(f"   - {col_name}: {desc}")
        
        return data
    else:
        print_error(f"Summary failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return None


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test data summarization")
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM descriptions')
    parser.add_argument('--file', default=TEST_FILE, help='CSV file to upload')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ARA Demo API - Data Summarization Test")
    print("="*60)
    print(f"API URL: {API_URL}")
    print(f"Test File: {args.file}")
    print(f"LLM Descriptions: {'Disabled' if args.no_llm else 'Enabled'}")
    
    # Generate unique session ID
    session_id = f"test-{int(datetime.now().timestamp())}"
    
    # Run tests
    if not test_health():
        print_error("API is not healthy. Exiting.")
        sys.exit(1)
    
    if not create_session(session_id):
        print_error("Failed to create session. Exiting.")
        sys.exit(1)
    
    file_id = upload_file(session_id, args.file)
    if not file_id:
        print_error("Failed to upload file. Exiting.")
        sys.exit(1)
    
    # Get summary with or without LLM
    summary = get_summary(file_id, with_descriptions=not args.no_llm)
    
    if summary:
        print_section("Test Complete")
        print_success("All tests passed!")
        print(f"\nFile ID: {file_id}")
        print(f"Session ID: {session_id}")
        print("\nTo get the summary again:")
        print(f"  curl http://localhost:8000/files/{file_id}/summary")
    else:
        print_error("Summary generation failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)

