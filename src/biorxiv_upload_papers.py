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
import argparse
from minio import Minio

from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

S3_BUCKET = "biorxiv-src-monthly"  # Updated to correct bucket name
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
                    Bucket=S3_BUCKET,
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

def extract_keys_from_s3_keys(s3_keys: list[str]) -> list[str]:
    """
    Extract keys from a list of S3 keys.
    """
    return [k.replace("Current_Content/June_2025/", "").replace(".meca", "") for k in s3_keys]

def check_if_packages_exist_in_minio(minio_client: Minio, bucket_name: str, s3_keys: list[str]) -> list[str]:
    """
    Check if packages exist in the MinIO bucket.
    """
    exists_in_minio = []
    if minio_client.bucket_exists(bucket_name):
        for obj in minio_client.list_objects(bucket_name, prefix="", recursive=True):
            if obj.object_name in s3_keys:
                exists_in_minio.append(True)
            else:
                exists_in_minio.append(False)
    return exists_in_minio

def check_if_unzipped_packages_exist_locally(output_dir: str, s3_keys: list[str]) -> list[str]:
    """
    Check if packages exist locally.
    """
    exists_locally = []
    for k in s3_keys:
        if os.path.exists(os.path.join(output_dir, k)):
            exists_locally.append(True)
        else:
            exists_locally.append(False)
    return exists_locally

def check_if_meca_packages_are_downloaded(output_dir: str, s3_keys: list[str]) -> list[str]:
    """
    Check if packages are unzipped.
    """
    exists_unzipped = []
    for k in s3_keys:
        if os.path.exists(os.path.join(output_dir, k) + '.meca'):
            exists_unzipped.append(True)
        else:
            exists_unzipped.append(False)
    return exists_unzipped

def download_MECA_packages_from_s3(
    s3_client,
    s3_keys: List[str], 
    output_dir: str = "./MECA_packages"
) -> None:
    """
    Download listed MECA packages from the bioRxiv S3 bucket.
    
    Args:
        s3_client: Configured boto3 S3 client
        s3_keys: List of uuids
        output_dir: Local directory to save the packages
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"📥 Verifying {len(s3_keys)} MECA packages locally and on S3...")
    
    downloaded_packages = 0
    skipped_packages = 0
    
    for i, s3_uuid in enumerate(s3_keys, 1):
        try:
            # Reconstruct the full S3 key
            full_s3_key = f"Current_Content/June_2025/{s3_uuid}.meca"
            # Create a safe filename from the S3 key
            safe_filename = s3_uuid.replace('/', '_')
            output_path = os.path.join(output_dir, safe_filename) + ".meca"
            
            # Check if file already exists locally
            exists_packed = check_if_meca_packages_are_downloaded(output_dir, [s3_uuid])
            if exists_packed[0]:
                print(f"   ⏭️  Package {i}/{len(s3_keys)}: {safe_filename} already exists, skipping...")
                skipped_packages += 1
                continue
            
            else:
                print(f"   📦 Downloading package {i}/{len(s3_keys)}: {safe_filename}")
                s3_client.download_file(
                    Bucket=S3_BUCKET,
                    Key=full_s3_key,
                    Filename=output_path,
                    ExtraArgs={"RequestPayer": "requester"}
                )
                downloaded_packages += 1
                
        except Exception as e:
            print(f"   ❌ Failed to download {s3_uuid}: {e}")
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
    meca_files = [f for f in os.listdir(output_dir) if f.endswith(".meca")]
    print(f"   ✅ Found {len(meca_files)} MECA files in {output_dir}")

    for i, meca_file in enumerate(meca_files, 1): 
        try:
            meca_path = os.path.join(output_dir, meca_file)
            extract_dir = os.path.join(output_dir, meca_file.replace(".meca", ""))

            if os.path.exists(extract_dir):
                print(f"   ⏭️  Package {i}/{len(meca_files)}: {meca_file} already unzipped locally, skipping...")
                continue

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
    
def upload_files_to_minio(s3_keys: List[str], file_type: str = "pdf") -> None:
    """
    Upload given s3 keys, extracted locally or already downloaded from S3, to MinIO.
    
    Args:
        s3_keys: List of paths to extracted MECA directories
        file_type: Type of files to upload ('pdf' for PDF files only, 'all' for all files)
    """
    if not s3_keys:
        print("⚠️  No extracted directories provided for upload")
        print(f"   🔍 s3_keys list: {s3_keys}")
        return
    
    try:
        minio_client = create_minio_client()
        print(f"✅ MinIO client created successfully")
        
        total_files = 0
        uploaded_files = 0
        skipped_files = 0
        
        print(f"📤 Starting upload to MinIO bucket: {MINIO_BUCKET}")
        print(f"🎯 File type filter: {file_type}")
        
        # extracted_dirs = [f.replace("./MECA_packages/", "") for f in s3_keys]
        print(f"   🔍 Extracted directories LOCALLY: {s3_keys}")
        s3_keys = [f.replace(".meca", "") for f in s3_keys]
        
        print(f"   🔍 Extracted directories AFTER .meca extension replacement: {s3_keys}")
        # Debug: Check what each directory contains
        for i, dir_path in enumerate(s3_keys):
            print(f"    Directory {i+1}: {dir_path}")
            print(f"   🔍 Exists locally: {os.path.exists(dir_path)}")
            if os.path.exists(dir_path):
                print(f"    Contents: {os.listdir(dir_path)}")
            else:
                print(f"   ❌ Directory does not exist locally!")
        
        # extracted_dirs = [f.replace("./MECA_packages/", "") for f in extracted_dirs]
        
        print(f"   🔍 Extracted directories AFTER replacement: {s3_keys}")
        
        for extracted_dir in s3_keys:
            print(f"\n📁 Processing directory: {extracted_dir}")
            if not os.path.exists(extracted_dir):
                print(f"   ⚠️  Directory from EXTRACTED_DIRS not found LOCALLY: {extracted_dir}")
                continue
            
            print(f"   ✅ Directory exists, walking through files...")
            
            # Walk through all files in the extracted directory
            for root, dirs, files in os.walk(extracted_dir):
                print(f"   🔍 Files in directory: {files}")
                print(f"   🔍 Root: {root}")
                print(f"   🔍 Dirs: {dirs}")
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
        print(f"   📁 Directories processed: {len(s3_keys)}")
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

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="BioRxiv MECA Package Downloader and MinIO Uploader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python biorxiv-upload-papers.py                    # Download 5 packages (default)
            python biorxiv-upload-papers.py -n 10              # Download 10 packages
            python biorxiv-upload-papers.py --num-packages 20  # Download 20 packages
            python biorxiv-upload-papers.py --test-minio       # Test MinIO connection only
            python biorxiv-upload-papers.py -n 0               # Skip download, only process existing files
        """
    )
    
    parser.add_argument(
        "-n", "--num-packages",
        type=int,
        default=5,
        help="Number of MECA packages to download and process (default: 5, use 0 to skip download)"
    )
    
    parser.add_argument(
        "--month",
        type=str,
        default="June_2025",
        help="Target month to search for packages (default: June_2025)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./MECA_packages",
        help="Output directory for downloaded packages (default: ./MECA_packages)"
    )
    
    parser.add_argument(
        "--file-type",
        type=str,
        choices=["pdf", "all"],
        default="pdf",
        help="Type of files to upload to MinIO (default: pdf)"
    )
    
    parser.add_argument(
        "--test-minio",
        action="store_true",
        help="Test MinIO connection only and exit"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -v for INFO, -vv for DEBUG)"
    )
    
    return parser.parse_args()

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

def main(num_packages: int = 5, target_month: str = "June_2025", 
         output_dir: str = "./MECA_packages", file_type: str = "pdf") -> None:
    """
    Main function to download and process MECA packages.
    
    Args:
        num_packages: Number of MECA packages to download
        target_month: Target month to search for packages
        output_dir: Output directory for downloaded packages
        file_type: Type of files to upload to MinIO
    """
    setup_logger(verbosity=1)
    
    try:
        print("🔍 BioRxiv MECA Package Workflow")
        print("=" * 50)
        print(f" Configuration:")
        print(f"    Number of packages: {num_packages}")
        print(f"   Target month: {target_month}")
        print(f"   Output directory: {output_dir}")
        print(f"   File type filter: {file_type}")
        
        s3_client = create_s3_client()
        print("✅ S3 client created successfully")
        
        print("\n1. 🔍 Searching for MECA packages...")
        s3_keys = find_MECA_packages_in_s3(s3_client, doi=None, target_month=target_month)
        
        if s3_keys is not None:
            print(f"✅ Found {len(s3_keys)} MECA packages in {target_month}")
            if s3_keys:
                # Limit the number of packages to process
                packages_to_process = min(num_packages, len(s3_keys)) if num_packages > 0 else len(s3_keys)
                
                print(f"\n📚 Processing {packages_to_process} MECA packages found in {target_month}:")
                
                for i, key in enumerate(s3_keys[:packages_to_process], 1):
                    print(f"   {i}. {key}")
                if len(s3_keys) > packages_to_process:
                    print(f"   ... and {len(s3_keys) - packages_to_process} more MECA packages")

                # Checking if meca packages already exist in the minio bucket
                minio_client = create_minio_client()
                if minio_client.bucket_exists(MINIO_BUCKET):
                    print(f"   🔍 Checking if the packages are already in MinIO bucket...")
                    s3_uuids_to_upload_to_minio = []
                    for s3_key in s3_keys[:packages_to_process]:
                        package_name = s3_key.split("/")[-1].replace(".meca", "")
                        # Check if any object in MinIO starts with this package name
                        package_exists = any(obj.object_name.startswith(package_name + "/") \
                            for obj in minio_client.list_objects(MINIO_BUCKET, prefix="", recursive=True))
                        if package_exists:
                            print(f"    ✅ Package {package_name} already exists in MinIO bucket")
                        else:
                            print(f"    ❌ Package {package_name} does not exist in MinIO bucket")
                            s3_uuids_to_upload_to_minio.append(package_name) # Raw uuid

                # Step 2: Check if MECA packages are already downloaded and unzipped
                if num_packages > 0:  # Only download if num_packages > 0
                    print(f"\n2. Verifying locally packages missing from MinIO bucket...")

                    is_s3_key_downloaded = check_if_meca_packages_are_downloaded(
                        output_dir=output_dir,
                        s3_keys=s3_uuids_to_upload_to_minio
                    )

                    s3_uuids_to_download = [
                        s3_key for i, s3_key in enumerate(s3_uuids_to_upload_to_minio)
                        if not is_s3_key_downloaded[i]
                    ]

                    download_MECA_packages_from_s3(
                        s3_client, 
                        s3_keys=s3_uuids_to_download)
                else:
                    print(f"\n2. Skipping download (num_packages = 0)")

                # Step 3: Extract MECA packages (this will extract all .meca files in the directory)
                print(f"\n3. Extracting MECA packages...")
                extracted_dirs = unzip_MECA_packages(output_dir=output_dir)
                
                # Step 4: Upload the specific packages that need to be uploaded to MinIO
                if s3_uuids_to_upload_to_minio:
                    print(f"\n4. Uploading {len(s3_uuids_to_upload_to_minio)} packages to MinIO...")
                    local_extracted_dirs = [os.path.join(output_dir, uuid) for uuid in s3_uuids_to_upload_to_minio]
                    existing_dirs = [dir_path for dir_path in local_extracted_dirs if os.path.exists(dir_path)]

                    if existing_dirs:
                        print(f"    Found {len(existing_dirs)} existing directories to upload to MinIO:")
                        for dir_path in existing_dirs:
                            print(f"      - {dir_path}")

                        print(f"\n5. Testing MinIO connection...")
                        if test_minio_connection():
                            upload_files_to_minio(existing_dirs, file_type=file_type)
                            print(f"   Files uploaded to MinIO")
                        else:
                            print(f"   MinIO connection test failed")
                    else:
                        print(f"    No existing directories to upload to MinIO")
                else:
                    print(f"\n4. No packages need to be uploaded to MinIO (all already exist)")

                # Step 5: Analyze and summarize (if any packages were extracted)
                if extracted_dirs:
                    print(f"\n6. Analyzing MECA package structure...")
                    print_MECA_summary(extracted_dirs)

                print(f"\n MECA workflow complete!")
                print(f" Original packages: {output_dir}/")
                print(f" Extracted content: {len(extracted_dirs)} directories")
        else:
            print(" No MECA packages found or error occurred")
            
    except Exception as e:
        print(f" Error in main function: {e}")
        logging.error(f"Main function error: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.test_minio:
        test_minio_only()
    else:
        main(
            num_packages=args.num_packages,
            target_month=args.month,
            output_dir=args.output_dir,
            file_type=args.file_type
        )