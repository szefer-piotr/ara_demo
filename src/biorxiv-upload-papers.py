#! /usr/bin/env python3

import zipfile
import logging
import requests
import json
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ParamValidationError
from datetime import datetime, timedelta
from typing import List, Optional
import os
from minio import Minio

from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

BUCKET = "biorxiv-src-monthly"  # Updated to correct bucket name
DEFAULT_REGION = "us-east-1"

# MinIO configuration - use port 9002 for API
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9002")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "biorxiv-papers")

print(MINIO_ENDPOINT)
print(MINIO_ACCESS_KEY)
print(MINIO_SECRET_KEY)
print(MINIO_SECURE)
print(MINIO_BUCKET)


@dataclass
class PaperInfo:
    doi: str
    title: str
    authors: str
    author_corresponding: str = ""
    author_corresponding_institution: str = ""
    date: str = ""
    version: str = ""
    type: str = ""
    license: str = ""
    category: str = ""
    abstract: str = ""
    pdf_url: str = ""
    s3_key: Optional[str] = None


def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def create_s3_client(aws_access_key_id: str = None, aws_secret_access_key: str = None, 
                     aws_region: str = None, profile: str = None):
    """
    Create an S3 client for accessing bioRxiv bucket.
    
    Args:
        aws_access_key_id: AWS access key (or use environment variable AWS_ACCESS_KEY_ID)
        aws_secret_access_key: AWS secret key (or use environment variable AWS_SECRET_ACCESS_KEY)
        aws_region: AWS region (defaults to us-east-1)
        profile: AWS profile name (alternative to raw keys)
    
    Returns:
        boto3 S3 client configured for requester-pays access
    """
    cfg = Config(region_name=aws_region or DEFAULT_REGION, signature_version="s3v4")
    
    if profile:
        session = boto3.Session(profile_name=profile, region_name=aws_region or DEFAULT_REGION)
    else:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=aws_region or DEFAULT_REGION,
        )
    
    return session.client("s3", config=cfg)


def find_MECA_packages_in_s3(s3_client, doi: Optional[List[str]], target_month: str = "June_2025") -> Optional[List[str]]:
    """
    Find MECA packages in the bioRxiv S3 bucket by month or a list of DOIs.
    
    Args:
        s3_client: Configured boto3 S3 client
        doi: Paper DOI list (e.g., ["10.1101/2024.01.15.123456", "10.1101/2024.01.15.123457"])
        target_month: Optional month to search in (e.g., "August_2025"), default is June 2025
    
    Returns:
        List of S3 keys if found, None otherwise
    """
    s3_keys = []
    try:
        months_to_search = [f"Current_Content/{target_month}/"]
        
        for month_prefix in months_to_search:
            print(f"   📅 Searching in {month_prefix}...")
            
            try:
                # Use paginator to handle more than 1000 objects
                paginator = s3_client.get_paginator('list_objects_v2')
                page_iterator = paginator.paginate(
                    Bucket=BUCKET,
                    Prefix=month_prefix,
                    RequestPayer='requester'
                )

                total_objects = 0
                for page in page_iterator:
                    if 'Contents' in page:
                        page_objects = len(page['Contents'])
                        total_objects += page_objects
                        print(f"   📄 Found {page_objects} objects in this page...")
                        for obj in page['Contents']:
                            key = obj['Key']
                            s3_keys.append(key)

                print(f"   📄 Found {total_objects} objects in {month_prefix}")
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchBucket':
                    print(f"   ⚠️  Month folder {month_prefix} not found")
                else:
                    print(f"   ❌ Error accessing {month_prefix}: {e}")
                continue
        
    except Exception as e:
        print(f"❌ Error searching for paper: {e}")
        return None

    return s3_keys


def download_MECA_packages_from_s3(
    s3_client, 
    n_packages: int | None, 
    s3_keys: List[str], 
    output_dir: str = "./MECA_packages"
) -> None:
    """
    Download MECA packages from the bioRxiv S3 bucket.
    
    Args:
        s3_client: Configured boto3 S3 client
        n_packages: Number of packages to download, or None to download all
        s3_keys: List of S3 keys to download
        output_dir: Local directory to save the packages
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Limit the number of packages to download
    if n_packages is not None:
        s3_keys = s3_keys[:n_packages]
    
    print(f"📥 Downloading {len(s3_keys)} MECA packages...")
    
    downloaded_packages = 0
    skipped_packages = 0
    
    for i, s3_key in enumerate(s3_keys, 1):
        try:
            # Create a safe filename from the S3 key
            safe_filename = os.path.basename(s3_key).replace('/', '_')
            output_path = os.path.join(output_dir, safe_filename)
            
            # Check if file already exists locally
            if os.path.exists(output_path):
                print(f"   ⏭️  Package {i}/{len(s3_keys)}: {safe_filename} already exists, skipping...")
                skipped_packages += 1
                continue
            
            print(f"   📦 Downloading package {i}/{len(s3_keys)}: {safe_filename}")
            
            s3_client.download_file(
                Bucket=BUCKET,
                Key=s3_key,
                Filename=output_path,
                ExtraArgs={"RequestPayer": "requester"}
            )
            downloaded_packages += 1
            
        except Exception as e:
            print(f"   ❌ Failed to download {s3_key}: {e}")
            continue
    
    print(f"✅ Download summary:")
    print(f"   📥 Newly downloaded: {downloaded_packages} packages")
    print(f"   ⏭️  Skipped (already exist): {skipped_packages} packages")
    print(f"   📊 Total processed: {len(s3_keys)} packages")


def unzip_MECA_packages(output_dir: str = "./MECA_packages") -> None:
    """
    Unzip MECA packages from the local directory.
    
    Args:
        output_dir: Local directory to save the packages
    """
    extracted_dirs = []

    if not os.path.exists(output_dir):
        print(f"   ❌ Output directory {output_dir} does not exist")
        return extracted_dirs

    meca_files = [f for f in os.listdir(output_dir) if f.endswith(".meca")]

    if not meca_files:
        print(f"   ❌ No MECA files found in {output_dir}")
        return extracted_dirs

    print(f"   ✅ Found {len(meca_files)} MECA files in {output_dir}")

    for i, meca_file in enumerate(meca_files, 1): 
        try:
            meca_path = os.path.join(output_dir, meca_file)
            extract_dir = os.path.join(output_dir, meca_file.replace(".meca", ""))

            print(f"   📦 Extracting MECA package {i}/{len(meca_files)}: {meca_file}")

            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(meca_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            extracted_dirs.append(extract_dir)
            print(f"   ✅ Successfully extracted to {extract_dir}")

        except Exception as e:
            print(f"   ❌ Error extracting {meca_file}: {e}")
            continue

    print(f"   ✅ Successfully extracted {len(extracted_dirs)} MECA packages")
    return extracted_dirs


def print_MECA_summary(extracted_dirs: List[str]) -> None:
    """
    Print a summary of extracted MECA packages.

    Args:
    extracted_dirs: List of paths to extracted directories
    """
    print("\n📊 MECA Packages Summary")
    print("=" * 50)

    for i, extracted_dir in enumerate(extracted_dirs, 1):
        print(f"\n📦 Package {i}: {os.path.basename(extracted_dir)}")
        print(f"   📁 Directory: {extracted_dir}")

        # Count files by type
        total_files = 0
        pdf_files = 0
        xml_files = 0
        other_files = 0

        for root, dirs, files in os.walk(extracted_dir):
            for file in files:
                total_files += 1
                if file.endswith('.pdf'):
                    pdf_files += 1
                elif file.endswith('.xml'):
                    xml_files += 1
                else:
                    other_files += 1

        print(f"   📄 Total files: {total_files}")
        print(f"   📖 PDF files: {pdf_files}")
        print(f"   📋 XML files: {xml_files}")
        print(f"   🔧 Other files: {other_files}")   


def test_minio_connection(endpoint: str = None, access_key: str = None, secret_key: str = None, secure: bool = None) -> bool:
    """
    Test the connection to the MinIO server by listing buckets.
    
    Args:
        endpoint: MinIO server endpoint (defaults to MINIO_ENDPOINT)
        access_key: Access key for MinIO (defaults to MINIO_ACCESS_KEY)
        secret_key: Secret key for MinIO (defaults to MINIO_SECRET_KEY)
        secure: Set to True if using HTTPS (defaults to MINIO_SECURE)
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        # Use provided parameters or fall back to global configuration
        test_endpoint = endpoint or MINIO_ENDPOINT
        test_access_key = access_key or MINIO_ACCESS_KEY
        test_secret_key = secret_key or MINIO_SECRET_KEY
        test_secure = secure if secure is not None else MINIO_SECURE
        
        print(f"🔍 Testing MinIO connection to: {test_endpoint}")
        print(f"   🔑 Access Key: {test_access_key}")
        print(f"   🔒 Secure: {test_secure}")
        
        # Initialize the MinIO client
        client = Minio(
            test_endpoint,
            access_key=test_access_key,
            secret_key=test_secret_key,
            secure=test_secure
        )
        
        # Attempt to list buckets to verify the connection
        buckets = client.list_buckets()
        print(f"✅ Successfully connected to MinIO!")
        print(f"   📦 Found {len(buckets)} buckets:")
        for bucket in buckets:
            print(f"      - {bucket.name} (created: {bucket.creation_date})")
        
        # Test if our target bucket exists
        if client.bucket_exists(MINIO_BUCKET):
            print(f"   🎯 Target bucket '{MINIO_BUCKET}' exists")
        else:
            print(f"   ⚠️  Target bucket '{MINIO_BUCKET}' does not exist")
        
        return True
        
    except Exception as e:
        print(f"❌ MinIO connection test failed: {e}")
        return False


def create_minio_client() -> Minio:
    """
    Create a MinIO client for uploading files.
    
    Returns:
        Configured MinIO client
    """
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        
        # Ensure the bucket exists
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
            print(f"✅ Created MinIO bucket: {MINIO_BUCKET}")
            
        return client
        
    except Exception as e:
        print(f"❌ Failed to create MinIO client: {e}")
        raise
    

def upload_files_to_minio(extracted_dirs: List[str], file_type: str = "pdf") -> None:
    """
    Upload files from extracted MECA directories to MinIO.
    
    Args:
        extracted_dirs: List of paths to extracted MECA directories
        file_type: Type of files to upload ('pdf' for PDF files only, 'all' for all files)
    """
    if not extracted_dirs:
        print("⚠️  No extracted directories provided for upload")
        return
    
    try:
        minio_client = create_minio_client()
        print(f"✅ MinIO client created successfully")
        
        total_files = 0
        uploaded_files = 0
        skipped_files = 0
        
        print(f"📤 Starting upload to MinIO bucket: {MINIO_BUCKET}")
        print(f"🎯 File type filter: {file_type}")
        
        for extracted_dir in extracted_dirs:
            if not os.path.exists(extracted_dir):
                print(f"   ⚠️  Directory not found: {extracted_dir}")
                continue
            
            print(f"\n📁 Processing directory: {os.path.basename(extracted_dir)}")
            
            # Walk through all files in the extracted directory
            for root, dirs, files in os.walk(extracted_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_extension = os.path.splitext(file)[1].lower()
                    
                    # Apply file type filtering
                    if file_type == "pdf" and file_extension != ".pdf":
                        skipped_files += 1
                        continue
                    
                    # Create MinIO object key (maintain directory structure)
                    rel_path = os.path.relpath(file_path, extracted_dir)
                    object_key = f"{os.path.basename(extracted_dir)}/{rel_path}"
                    
                    try:
                        # Upload file to MinIO using fput_object for better performance
                        minio_client.fput_object(
                            MINIO_BUCKET,
                            object_key,
                            file_path
                        )
                        
                        uploaded_files += 1
                        print(f"   ✅ Uploaded: {file} -> {object_key}")
                        
                    except Exception as e:
                        print(f"   ❌ Failed to upload {file}: {e}")
                        continue
                    
                    total_files += 1
        
        # Print upload summary
        print(f"\n📊 Upload Summary:")
        print(f"   📁 Directories processed: {len(extracted_dirs)}")
        print(f"   📄 Total files found: {total_files}")
        print(f"   📤 Files uploaded: {uploaded_files}")
        print(f"   ⏭️  Files skipped: {skipped_files}")
        print(f"   🎯 File type filter: {file_type}")
        print(f"   ☁️  MinIO bucket: {MINIO_BUCKET}")
        
        if uploaded_files > 0:
            print(f"\n🎉 Upload completed successfully!")
        else:
            print(f"\n⚠️  No files were uploaded")
            
    except Exception as e:
        print(f"❌ Error during MinIO upload: {e}")
        logging.error(f"MinIO upload error: {e}")


def main() -> None:
    setup_logger(verbosity=1)
    
    try:
        print("🔍 BioRxiv MECA Package Workflow")
        print("=" * 50)
        
        s3_client = create_s3_client()
        print("✅ S3 client created successfully")
        
        # Step 1: Find MECA packages in S3
        print("\n1. 🔍 Searching for MECA packages...")
        s3_keys = find_MECA_packages_in_s3(s3_client, doi=None, target_month="June_2025")
        
        if s3_keys is not None:
            print(f"✅ Found 5 MECA packages in June 2025")
            
            # Show first few MECA packages as examples
            if s3_keys:
                print("\n📚 Sample MECA packages found:")
                for i, key in enumerate(s3_keys[:5], 1):
                    print(f"   {i}. {key}")
                
                if len(s3_keys) > 5:
                    print(f"   ... and {len(s3_keys) - 5} more MECA packages")

                # Step 2: Download MECA packages
                print(f"\n2. 📥 Downloading MECA packages...")
                download_MECA_packages_from_s3(s3_client, n_packages=5, s3_keys=s3_keys)
                print(f"✅ Downloaded packages to ./MECA_packages")

                # Step 3: Extract MECA packages
                print(f"\n3. 📦 Extracting MECA packages...")
                extracted_dirs = unzip_MECA_packages(output_dir="./MECA_packages")
                
                if extracted_dirs:
                    # Step 4: Analyze and summarize
                    print(f"\n4. 📊 Analyzing MECA package structure...")
                    print_MECA_summary(extracted_dirs)

                    # Step 5: Test MinIO connection first
                    print(f"\n5. 🔍 Testing MinIO connection...")
                    if test_minio_connection():
                        # Step 6: Upload to MinIO
                        print(f"\n6. 📤 Uploading files to MinIO...")
                        upload_files_to_minio(extracted_dirs)
                        print(f"✅ Files uploaded to MinIO.")
                    else:
                        print(f"⚠️  Skipping MinIO upload due to connection failure")
                        print(f"   💡 Check your MinIO server configuration and try again")
                    
                    print(f"\n🎉 MECA workflow complete!")
                    print(f"📁 Original packages: ./MECA_packages/")
                    print(f"📂 Extracted content: {len(extracted_dirs)} directories")
                else:
                    print("❌ No packages were successfully extracted")
        else:
            print("❌ No MECA packages found or error occurred")
            
    except Exception as e:
        print(f"❌ Error in main function: {e}")
        logging.error(f"Main function error: {e}")


def test_minio_only() -> None:
    """
    Standalone function to test only MinIO connection.
    Useful for debugging MinIO issues without running the full workflow.
    """
    print("🔍 MinIO Connection Test Only")
    print("=" * 40)
    
    # Test with default configuration
    print("\n📋 Testing with default configuration:")
    success = test_minio_connection()
    
    if not success:
        print(f"\n🔄 Testing alternative configurations...")
        
        # Test different ports
        test_ports = ["9001", "9003", "9004"]
        for port in test_ports:
            endpoint = f"localhost:{port}"
            print(f"\n🔍 Testing port {port}...")
            if test_minio_connection(endpoint=endpoint):
                print(f"✅ Found working MinIO on port {port}")
                print(f"💡 Update your MINIO_ENDPOINT to: {endpoint}")
                break
        else:
            print(f"❌ No working MinIO found on tested ports")
            print(f"💡 Make sure MinIO is running and accessible")


if __name__ == "__main__":
    # Check if user wants to test MinIO only
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-minio":
        test_minio_only()
    else:
        main()
