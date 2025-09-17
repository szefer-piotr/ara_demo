#!/usr/bin/env python3
"""
Wrapper script for downloading BioRxiv papers with environment variable support.

This script loads configuration from a .env file and then calls the main download script.
Create a .env file based on biorxiv_config.env.example with your AWS credentials.

Usage:
    python download_biorxiv_env.py
    # or
    python download_biorxiv_env.py --minio-bucket my-custom-bucket
    # or with unpacking
    python download_biorxiv_env.py --unpack-meca
"""

import os
import sys
import subprocess
import argparse
import tempfile
import zipfile
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List
from minio import Minio
from minio.error import S3Error


def load_env_file(env_path: str = ".env") -> None:
    """Load environment variables from .env file."""
    if not Path(env_path).exists():
        print(f"⚠️  Warning: {env_path} file not found. Using system environment variables.")
        return
    
    print(f"📁 Loading configuration from {env_path}")
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
                print(f"   ✅ Loaded: {key}")


def create_minio_client() -> Minio:
    """Create MinIO client using environment variables."""
    endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9002')
    access_key = os.getenv('MINIO_ACCESS_KEY', 'piotrminio')
    secret_key = os.getenv('MINIO_SECRET_KEY', 'piotrminio')
    
    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )


def extract_metadata_from_meca(zip_path: Path) -> Dict[str, Any]:
    """Extract metadata from a .meca file (which is actually a zip file)."""
    metadata = {
        'title': 'Unknown',
        'authors': 'Unknown',
        'doi': 'Unknown',
        'abstract': 'No abstract available',
        'category': 'Unknown',
        'files': []
    }
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Look for manifest.xml
            manifest_files = [f for f in zip_file.namelist() if f.endswith('manifest.xml')]
            
            if manifest_files:
                with zip_file.open(manifest_files[0]) as manifest:
                    try:
                        tree = ET.parse(manifest)
                        root = tree.getroot()
                        
                        # Extract title
                        title_elem = root.find('.//title')
                        if title_elem is not None:
                            metadata['title'] = title_elem.text or 'Unknown'
                        
                        # Extract authors
                        author_elems = root.findall('.//author')
                        if author_elems:
                            authors = [elem.text for elem in author_elems if elem.text]
                            metadata['authors'] = '; '.join(authors) if authors else 'Unknown'
                        
                        # Extract DOI
                        doi_elem = root.find('.//doi')
                        if doi_elem is not None:
                            metadata['doi'] = doi_elem.text or 'Unknown'
                        
                        # Extract category
                        category_elem = root.find('.//category')
                        if category_elem is not None:
                            metadata['category'] = category_elem.text or 'Unknown'
                            
                    except ET.ParseError as e:
                        print(f"Warning: Failed to parse manifest.xml: {e}")
            
            # List all files in the package
            metadata['files'] = zip_file.namelist()
            
    except zipfile.BadZipFile as e:
        print(f"Warning: Failed to read .meca file as zip: {e}")
    
    return metadata


def unpack_meca_file(
    minio_client: Minio,
    bucket_name: str,
    meca_key: str,
    unpack_prefix: str = "unpacked"
) -> bool:
    """
    Unpack a .meca file from MinIO and upload individual components.
    
    Args:
        minio_client: MinIO client instance
        bucket_name: Name of the MinIO bucket
        meca_key: Key of the .meca file in MinIO
        unpack_prefix: Prefix for unpacked files
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📦 Unpacking: {meca_key}")
        
        # Create temporary file for the .meca file
        with tempfile.NamedTemporaryFile(suffix='.meca', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Download .meca file from MinIO
            print(f"   📥 Downloading from MinIO...")
            minio_client.fget_object(bucket_name, meca_key, str(tmp_path))
            
            # Extract metadata
            metadata = extract_metadata_from_meca(tmp_path)
            print(f"   📋 Extracted metadata: {metadata['title'][:60]}...")
            
            # Unpack the .meca file
            with zipfile.ZipFile(tmp_path, 'r') as zip_file:
                # Create base path for unpacked files
                base_name = Path(meca_key).stem.replace('Current_Content_', '').replace('_', '/')
                base_path = f"{unpack_prefix}/{base_name}"
                
                # Upload each file individually
                for file_name in zip_file.namelist():
                    if file_name.endswith('/'):  # Skip directories
                        continue
                    
                    # Create clean filename
                    clean_name = file_name.replace('/', '_')
                    minio_key = f"{base_path}/{clean_name}"
                    
                    # Extract and upload file
                    with zip_file.open(file_name) as file_content:
                        file_data = file_content.read()
                        
                        # Determine content type
                        content_type = 'application/octet-stream'
                        if file_name.endswith('.pdf'):
                            content_type = 'application/pdf'
                        elif file_name.endswith('.xml'):
                            content_type = 'application/xml'
                        elif file_name.endswith('.txt'):
                            content_type = 'text/plain'
                        elif file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                            content_type = 'image/jpeg'
                        elif file_name.endswith('.png'):
                            content_type = 'image/png'
                        
                        # Upload to MinIO
                        minio_client.put_object(
                            bucket_name,
                            minio_key,
                            file_data,
                            length=len(file_data),
                            content_type=content_type
                        )
                        
                        print(f"      ✅ Uploaded: {clean_name}")
                
                # Upload enhanced metadata
                metadata['unpacked_files'] = [f for f in zip_file.namelist() if not f.endswith('/')]
                metadata['unpacked_path'] = base_path
                
                metadata_key = f"{base_path}/metadata.json"
                metadata_json = json.dumps(metadata, indent=2)
                minio_client.put_object(
                    bucket_name,
                    metadata_key,
                    metadata_json.encode('utf-8'),
                    length=len(metadata_json.encode('utf-8')),
                    content_type='application/json'
                )
                
                print(f"      ✅ Uploaded: metadata.json")
            
            print(f"   ✅ Successfully unpacked: {meca_key}")
            return True
            
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()
                
    except Exception as e:
        print(f"   ❌ Failed to unpack {meca_key}: {e}")
        return False


def list_meca_files(minio_client: Minio, bucket_name: str, month_folder: str) -> List[str]:
    """List all .meca files in the specified month folder."""
    try:
        prefix = f"{month_folder}/"
        objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
        
        meca_files = []
        for obj in objects:
            if obj.object_name.endswith('.meca'):
                meca_files.append(obj.object_name)
        
        return meca_files
    except Exception as e:
        print(f"Error listing .meca files: {e}")
        return []


def unpack_all_meca_files(
    minio_client: Minio,
    bucket_name: str,
    month_folder: str,
    max_files: int = None
) -> None:
    """
    Unpack all .meca files in the specified month folder.
    
    Args:
        minio_client: MinIO client instance
        bucket_name: Name of the MinIO bucket
        month_folder: Month folder to process
        max_files: Maximum number of files to process (None for all)
    """
    print(f"🚀 Starting unpacking of .meca files in {month_folder}")
    
    # List all .meca files
    meca_files = list_meca_files(minio_client, bucket_name, month_folder)
    
    if not meca_files:
        print(f"⚠️  No .meca files found in {month_folder}")
        return
    
    # Limit files if specified
    if max_files:
        meca_files = meca_files[:max_files]
    
    print(f"📁 Found {len(meca_files)} .meca files to unpack")
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, meca_key in enumerate(meca_files, 1):
        print(f"\n📦 Processing file {i}/{len(meca_files)}")
        
        if unpack_meca_file(minio_client, bucket_name, meca_key):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 UNPACKING SUMMARY")
    print("=" * 60)
    print(f"Month folder: {month_folder}")
    print(f"Total files: {len(meca_files)}")
    print(f"Successfully unpacked: {successful}")
    print(f"Failed: {failed}")
    print("=" * 60)
    
    if successful > 0:
        print("✅ Unpacking completed successfully!")
        print(f"📁 Unpacked files are available in the '{bucket_name}' bucket")
        print(f"   under the 'unpacked/{month_folder}/' prefix")
    else:
        print("⚠️  No files were successfully unpacked")


def main():
    """Main function that loads environment and calls the download script."""
    parser = argparse.ArgumentParser(
        description="Download BioRxiv papers and optionally unpack .meca files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download papers from last month
    python download_biorxiv_env.py
    
    # Download and unpack .meca files
    python download_biorxiv_env.py --unpack-meca
    
    # Download to custom bucket and unpack
    python download_biorxiv_env.py --minio-bucket my-papers --unpack-meca
    
    # Only unpack existing .meca files (no download)
    python download_biorxiv_env.py --unpack-only --month-folder July_2025
        """
    )
    
    # Main options
    parser.add_argument(
        '--unpack-meca',
        action='store_true',
        help='Download papers and then unpack .meca files'
    )
    
    parser.add_argument(
        '--unpack-only',
        action='store_true',
        help='Only unpack existing .meca files (skip download)'
    )
    
    parser.add_argument(
        '--month-folder',
        help='Month folder to unpack (e.g., July_2025). Required with --unpack-only'
    )
    
    parser.add_argument(
        '--max-unpack',
        type=int,
        help='Maximum number of .meca files to unpack'
    )
    
    # Add any additional command line arguments
    args, remaining_args = parser.parse_known_args()
    
    # Load environment variables
    load_env_file()
    
    # Check if AWS credentials are available (only needed for download)
    if not args.unpack_only:
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not aws_key or not aws_secret:
            print("❌ Error: AWS credentials not found!")
            print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            print("or create a .env file based on biorxiv_config.env.example")
            return 1
    
    # Get MinIO bucket from environment or command line
    minio_bucket = os.getenv('MINIO_BUCKET', 'biorxiv-papers')
    
    if args.unpack_only:
        # Only unpack existing files
        if not args.month_folder:
            print("❌ Error: --month-folder is required with --unpack-only")
            print("Example: --month-folder July_2025")
            return 1
        
        print(f"🔧 Unpacking only mode for month: {args.month_folder}")
        
        try:
            minio_client = create_minio_client()
            unpack_all_meca_files(
                minio_client,
                minio_bucket,
                args.month_folder,
                args.max_unpack
            )
            return 0
        except Exception as e:
            print(f"❌ Error during unpacking: {e}")
            return 1
    
    # Build command for the main download script
    cmd = [
        sys.executable, 'download_biorxiv_last_month.py',
        '--minio-bucket', minio_bucket
    ]
    
    # Add optional environment-based arguments
    if os.getenv('MINIO_ENDPOINT'):
        cmd.extend(['--minio-endpoint', os.getenv('MINIO_ENDPOINT')])
    
    if os.getenv('MINIO_ACCESS_KEY'):
        cmd.extend(['--minio-access-key', os.getenv('MINIO_ACCESS_KEY')])
    
    if os.getenv('MINIO_SECRET_KEY'):
        cmd.extend(['--minio-secret-key', os.getenv('MINIO_SECRET_KEY')])
    
    if os.getenv('MAX_FILES'):
        cmd.extend(['--max-files', os.getenv('MAX_FILES')])
    
    # Add any additional command line arguments
    cmd.extend(remaining_args)
    
    print(f"🚀 Running download: {' '.join(cmd)}")
    print()
    
    # Run the main download script
    try:
        result = subprocess.run(cmd, check=True)
        
        # If download was successful and unpacking is requested
        if result.returncode == 0 and args.unpack_meca:
            print("\n" + "=" * 60)
            print("🔧 DOWNLOAD COMPLETED - STARTING UNPACKING")
            print("=" * 60)
            
            # Get the month folder from the download
            from download_biorxiv_last_month import get_last_month
            month_folder, _, _ = get_last_month()
            
            try:
                minio_client = create_minio_client()
                unpack_all_meca_files(
                    minio_client,
                    minio_bucket,
                    month_folder,
                    args.max_unpack
                )
            except Exception as e:
                print(f"❌ Error during unpacking: {e}")
                return 1
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download script failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("❌ Error: download_biorxiv_last_month.py not found!")
        print("Make sure the script is in the same directory.")
        return 1


if __name__ == "__main__":
    exit(main())
