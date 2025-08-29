#!/usr/bin/env python3
"""
Download BioRxiv papers from the last month using the official TDM S3 bucket
and upload them to a MinIO bucket.

This script uses the official BioRxiv TDM access:
- S3 bucket: s3://biorxiv-src-monthly (requester-pays)
- Location: us-east-1
- Format: .meca files (actually zip files containing PDF, XML, and metadata)

Usage:
    python download_biorxiv_last_month.py --minio-bucket my-bucket --max-files 100
"""

import argparse
import boto3
import datetime
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import xml.etree.ElementTree as ET

from minio import Minio
from minio.error import S3Error
from botocore.config import Config
from botocore.exceptions import ClientError


# Configuration
BIORXIV_BUCKET = "biorxiv-src-monthly"
BIORXIV_REGION = "us-east-1"
DEFAULT_MINIO_ENDPOINT = "localhost:9002"
DEFAULT_MINIO_ACCESS_KEY = "piotrminio"
DEFAULT_MINIO_SECRET_KEY = "piotrminio"

# Month name mapping
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_last_month() -> tuple[str, int, int]:
    """Get the last month in the format used by BioRxiv."""
    today = datetime.datetime.now()
    
    # Go back one month
    if today.month == 1:
        last_month = 12
        last_year = today.year - 1
    else:
        last_month = today.month - 1
        last_year = today.year
    
    month_name = MONTH_NAMES[last_month - 1]
    folder_name = f"{month_name}_{last_year}"
    
    return folder_name, last_year, last_month


def create_s3_client(aws_access_key: str, aws_secret_key: str, region: str = BIORXIV_REGION):
    """Create an S3 client for accessing the BioRxiv bucket."""
    config = Config(
        region_name=region,
        retries={'max_attempts': 3, 'mode': 'adaptive'}
    )
    
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )
    
    return session.client('s3', config=config)


def create_minio_client(
    endpoint: str,
    access_key: str,
    secret_key: str,
    secure: bool = False
) -> Minio:
    """Create a MinIO client."""
    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )


def ensure_minio_bucket_exists(minio_client: Minio, bucket_name: str) -> None:
    """Ensure the MinIO bucket exists, create if it doesn't."""
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logging.info(f"Created MinIO bucket: {bucket_name}")
        else:
            logging.info(f"MinIO bucket already exists: {bucket_name}")
    except S3Error as e:
        logging.error(f"Failed to create/check MinIO bucket {bucket_name}: {e}")
        raise


def list_biorxiv_month_files(s3_client, month_folder: str, max_files: Optional[int] = None) -> List[Dict[str, Any]]:
    """List all .meca files in the specified month folder."""
    prefix = f"Current_Content/{month_folder}/"
    
    logging.info(f"Listing files in {prefix}")
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=BIORXIV_BUCKET,
            Prefix=prefix,
            RequestPayer='requester'
        )
        
        files = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith('.meca'):
                files.append({
                    'key': key,
                    'size': obj['Size'],
                    'last_modified': obj['LastModified']
                })
                
                if max_files and len(files) >= max_files:
                    break
        
        logging.info(f"Found {len(files)} .meca files in {month_folder}")
        return files
        
    except ClientError as e:
        logging.error(f"Failed to list files in {prefix}: {e}")
        raise


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
                        logging.warning(f"Failed to parse manifest.xml: {e}")
            
            # List all files in the package
            metadata['files'] = zip_file.namelist()
            
    except zipfile.BadZipFile as e:
        logging.warning(f"Failed to read .meca file as zip: {e}")
    
    return metadata


def download_and_upload_paper(
    s3_client,
    minio_client: Minio,
    minio_bucket: str,
    paper_info: Dict[str, Any],
    month_folder: str
) -> bool:
    """Download a paper from BioRxiv S3 and upload to MinIO."""
    s3_key = paper_info['key']
    file_size = paper_info['size']
    
    logging.info(f"Processing: {s3_key} ({file_size} bytes)")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.meca', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Download from BioRxiv S3
            logging.debug(f"Downloading {s3_key} to {tmp_path}")
            s3_client.download_file(
                BIORXIV_BUCKET,
                s3_key,
                str(tmp_path),
                ExtraArgs={'RequestPayer': 'requester'}
            )
            
            # Extract metadata
            metadata = extract_metadata_from_meca(tmp_path)
            
            # Create a clean filename for MinIO
            clean_filename = s3_key.replace('/', '_').replace(' ', '_')
            minio_key = f"{month_folder}/{clean_filename}"
            
            # Upload to MinIO
            logging.debug(f"Uploading to MinIO: {minio_key}")
            minio_client.fput_object(
                minio_bucket,
                minio_key,
                str(tmp_path),
                content_type='application/zip'
            )
            
            # Upload metadata as JSON
            metadata_key = f"{month_folder}/{clean_filename}.metadata.json"
            metadata_json = json.dumps(metadata, indent=2)
            minio_client.put_object(
                minio_bucket,
                metadata_key,
                io.BytesIO(metadata_json.encode('utf-8')),
                length=len(metadata_json.encode('utf-8')),
                content_type='application/json'
            )
            
            logging.info(f"✅ Successfully processed: {metadata.get('title', 'Unknown')[:60]}...")
            return True
            
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()
                
    except Exception as e:
        logging.error(f"❌ Failed to process {s3_key}: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download BioRxiv papers from the last month and upload to MinIO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download up to 100 papers from last month
    python download_biorxiv_last_month.py --minio-bucket biorxiv-papers --max-files 100
    
    # Use custom AWS credentials
    python download_biorxiv_last_month.py --aws-access-key YOUR_KEY --aws-secret-key YOUR_SECRET
    
    # Use custom MinIO endpoint
    python download_biorxiv_last_month.py --minio-endpoint my-minio:9000 --minio-bucket papers
        """
    )
    
    # AWS credentials
    parser.add_argument(
        '--aws-access-key',
        default=os.getenv('AWS_ACCESS_KEY_ID'),
        help='AWS access key (default: AWS_ACCESS_KEY_ID env var)'
    )
    parser.add_argument(
        '--aws-secret-key',
        default=os.getenv('AWS_SECRET_ACCESS_KEY'),
        help='AWS secret key (default: AWS_SECRET_ACCESS_KEY env var)'
    )
    
    # MinIO configuration
    parser.add_argument(
        '--minio-endpoint',
        default=DEFAULT_MINIO_ENDPOINT,
        help=f'MinIO endpoint (default: {DEFAULT_MINIO_ENDPOINT})'
    )
    parser.add_argument(
        '--minio-access-key',
        default=DEFAULT_MINIO_ACCESS_KEY,
        help=f'MinIO access key (default: {DEFAULT_MINIO_ACCESS_KEY})'
    )
    parser.add_argument(
        '--minio-secret-key',
        default=DEFAULT_MINIO_SECRET_KEY,
        help=f'MinIO secret key (default: {DEFAULT_MINIO_SECRET_KEY})'
    )
    parser.add_argument(
        '--minio-bucket',
        required=True,
        help='MinIO bucket name to upload papers to'
    )
    
    # Download options
    parser.add_argument(
        '--max-files',
        type=int,
        default=100,
        help='Maximum number of files to download (default: 100)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate AWS credentials
    if not args.aws_access_key or not args.aws_secret_key:
        logging.error("AWS credentials are required. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables or use --aws-access-key and --aws-secret-key arguments.")
        return 1
    
    # Get last month
    month_folder, year, month = get_last_month()
    logging.info(f"📅 Processing papers from: {month_folder} ({year}-{month:02d})")
    
    try:
        # Create S3 client for BioRxiv
        logging.info("🔗 Connecting to BioRxiv S3 bucket...")
        s3_client = create_s3_client(args.aws_access_key, args.aws_secret_key)
        
        # Create MinIO client
        logging.info("🔗 Connecting to MinIO...")
        minio_client = create_minio_client(
            args.minio_endpoint,
            args.minio_access_key,
            args.minio_secret_key
        )
        
        # Ensure MinIO bucket exists
        ensure_minio_bucket_exists(minio_client, args.minio_bucket)
        
        # List files in the month folder
        files = list_biorxiv_month_files(s3_client, month_folder, args.max_files)
        
        if not files:
            logging.warning(f"No .meca files found in {month_folder}")
            return 0
        
        # Download and upload papers
        logging.info(f"🚀 Starting download of {len(files)} papers...")
        
        successful = 0
        failed = 0
        
        for i, file_info in enumerate(files, 1):
            logging.info(f"📄 Processing paper {i}/{len(files)}")
            
            if download_and_upload_paper(s3_client, minio_client, args.minio_bucket, file_info, month_folder):
                successful += 1
            else:
                failed += 1
        
        # Summary
        logging.info("=" * 60)
        logging.info("📊 DOWNLOAD SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Month: {month_folder}")
        logging.info(f"Total files found: {len(files)}")
        logging.info(f"Successfully processed: {successful}")
        logging.info(f"Failed: {failed}")
        logging.info(f"MinIO bucket: {args.minio_bucket}")
        logging.info("=" * 60)
        
        if successful > 0:
            logging.info("✅ Download completed successfully!")
        else:
            logging.warning("⚠️  No papers were successfully processed")
            
        return 0 if failed == 0 else 1
        
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    import io  # Import here to avoid circular import
    exit(main())
