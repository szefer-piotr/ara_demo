#!/usr/bin/env python3
"""
BioRxiv MECA Package Processor - Direct S3 to MinIO workflow without local storage.
"""

import boto3
import logging
import argparse
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import List, Optional
import os
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
import io
import zipfile

load_dotenv()

# S3 Configuration
S3_BUCKET = "biorxiv-src-monthly"
DEFAULT_REGION = "us-east-1"

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9002")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_MECA_BUCKET = "biorxiv-meca"
MINIO_BUCKET = "biorxiv-unpacked"

def setup_logger(verbosity: int = 1) -> None:
    """
    Setup logging configuration.
    
    Args:
        verbosity: Logging level (0=WARNING, 1=INFO, 2=DEBUG)
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    
    logging.basicConfig(
        level=level, 
        format="%(asctime)s %(levelname)s %(message)s", 
        datefmt="%H:%M:%S"
    )





# Getting MECA packages from S3
def create_s3_client(
    aws_access_key_id: str = None, 
    aws_secret_access_key: str = None, 
    aws_region: str = None, 
    profile: str = None
):
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
        
        # Ensure both buckets exist
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
            logging.info(f"Created MinIO bucket: {MINIO_BUCKET}")
        else:
            logging.info(f"MinIO bucket exists: {MINIO_BUCKET}")
            
        if not client.bucket_exists(MINIO_MECA_BUCKET):
            client.make_bucket(MINIO_MECA_BUCKET)
            logging.info(f"Created MinIO bucket: {MINIO_MECA_BUCKET}")
        else:
            logging.info(f"MinIO bucket exists: {MINIO_MECA_BUCKET}")
            
        return client
        
    except Exception as e:
        logging.error(f"Failed to create MinIO client: {e}")
        raise

def search_papers_in_s3(
    target_month: str = "June_2025", 
    n_papers: int = 10,
    s3_client=None,
    verbose: bool = True
) -> Optional[List[str]]:
    """
    Search for MECA packages in the bioRxiv S3 bucket by month and return a limited number.
    
    Args:
        target_month: Month to search in (e.g., "June_2025", "August_2025")
        n_papers: Number of papers to return (default: 10)
        s3_client: Optional pre-configured S3 client
        verbose: Whether to print progress information
    
    Returns:
        List of S3 keys for MECA packages, or None if error occurred
    """
    if s3_client is None:
        s3_client = create_s3_client()
    
    s3_keys = []
    month_prefix = f"Current_Content/{target_month}/"
    
    if verbose:
        logging.info(f"Searching for {n_papers} papers in {target_month}...")
        logging.info(f"   Searching in S3 prefix: {month_prefix}")
    
    try:
        # Use paginator to handle more than 1000 objects
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix=month_prefix,
            RequestPayer='requester'
        )

        total_objects = 0
        papers_found = 0
        
        for page in page_iterator:
            if 'Contents' in page:
                page_objects = len(page['Contents'])
                total_objects += page_objects
                
                if verbose:
                    logging.info(f"   📄 Found {page_objects} objects in this page...")
                
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Only include .meca files
                    if key.endswith('.meca'):
                        s3_keys.append(key)
                        papers_found += 1
                        
                        # Stop when we have enough papers
                        if papers_found >= n_papers:
                            break
                
                # Break outer loop if we have enough papers
                if papers_found >= n_papers:
                    break

        if verbose:
            logging.info(f"   Total objects found: {total_objects}")
            logging.info(f"   MECA packages found: {len(s3_keys)}")
            logging.info(f"   Requested papers: {n_papers}")
            
        # Limit to requested number
        s3_keys = s3_keys[:n_papers]
        
        if verbose and s3_keys:
            logging.info(f"Successfully found {len(s3_keys)} papers:")
            for i, key in enumerate(s3_keys, 1):
                paper_name = key.split("/")[-1].replace(".meca", "")
                logging.info(f"   {i}. {paper_name}")
        
        return s3_keys if s3_keys else None
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            logging.error(f"   Bucket {S3_BUCKET} not found")
        else:
            logging.error(f"   Error accessing S3: {e}")
        return None
    except Exception as e:
        logging.error(f"   Unexpected error searching for papers: {e}")
        return None

def get_paper_info(s3_keys: List[str], verbose: bool = True) -> List[dict]:
    """
    Extract paper information from S3 keys.
    
    Args:
        s3_keys: List of S3 keys for MECA packages
        verbose: Whether to print progress information
    
    Returns:
        List of dictionaries containing paper information
    """
    papers_info = []
    
    for key in s3_keys:
        # Extract paper UUID from the key
        paper_uuid = key.split("/")[-1].replace(".meca", "")
        
        # Extract month from the key
        month = key.split("/")[1] if len(key.split("/")) > 1 else "Unknown"
        
        paper_info = {
            "uuid": paper_uuid,
            "s3_key": key,
            "month": month,
            "filename": f"{paper_uuid}.meca"
        }
        
        papers_info.append(paper_info)
    
    if verbose:
        logging.info(f"Paper information extracted for {len(papers_info)} papers")
        for i, info in enumerate(papers_info, 1):
            logging.info(f"   {i}. {info['uuid']} (from {info['month']})")
    
    return papers_info

def download_meca_packages_to_minio(
    s3_keys: List[str],
    force: bool = False,
    s3_client=None,
    minio_client=None,
    minio_bucket: str = MINIO_MECA_BUCKET,
    verbose: bool = True
) -> dict:
    """
    Download MECA packages from S3 and store them directly to MinIO without processing/unzipping.
    
    Args:
        s3_keys: List of S3 keys for MECA packages
        force: Whether to force re-download packages that already exist
        s3_client: Optional pre-configured S3 client
        minio_client: Optional pre-configured MinIO client
        minio_bucket: MinIO bucket name (default: biorxiv-meca)
        verbose: Whether to show progress information
    
    Returns:
        Dictionary with processing statistics
    """
    if s3_keys is None or len(s3_keys) == 0:
        logging.error("No S3 keys provided")
        return {"processed": 0, "skipped": 0, "errors": 0}
    
    if s3_client is None:
        s3_client = create_s3_client()
    
    if minio_client is None:
        minio_client = create_minio_client()
    
    if verbose:
        logging.info(f"Starting download of {len(s3_keys)} MECA packages to MinIO...")
        logging.info(f"   Target bucket: {minio_bucket}")
        logging.info(f"   Force re-download: {force}")
    
    processed_packages = 0
    skipped_packages = 0
    error_packages = 0
    
    for i, s3_key in enumerate(s3_keys, 1):
        try:
            # Extract package UUID from S3 key
            package_uuid = s3_key.split("/")[-1].replace(".meca", "")
            minio_object_key = f"{package_uuid}.meca"
            
            if verbose:
                logging.info(f"Processing package {i}/{len(s3_keys)}: {package_uuid}")
            
            # Check if package already exists in MinIO (unless force is True)
            if not force and meca_package_exists_in_minio(minio_client, minio_object_key, minio_bucket):
                if verbose:
                    logging.info(f"   Package {package_uuid} already exists in MinIO, skipping...")
                skipped_packages += 1
                continue
            
            # Download MECA package from S3
            if verbose:
                logging.info(f"   Downloading from S3: {s3_key}")
            
            meca_data = download_meca_from_s3(s3_client, s3_key)
            if not meca_data:
                logging.error(f"   Failed to download MECA package {package_uuid}")
                error_packages += 1
                continue
            
            # Upload to MinIO
            if verbose:
                logging.info(f"   Uploading to MinIO: {minio_object_key}")
            
            upload_meca_to_minio(minio_client, minio_object_key, meca_data, minio_bucket)
            processed_packages += 1
            
            if verbose:
                logging.info(f"   Successfully downloaded {package_uuid} ({len(meca_data)} bytes)")
        
        except Exception as e:
            logging.error(f"   Error processing package {s3_key}: {e}")
            error_packages += 1
            continue
    
    # Summary
    summary = {
        "processed": processed_packages,
        "skipped": skipped_packages,
        "errors": error_packages
    }
    
    if verbose:
        logging.info(f"Download summary:")
        logging.info(f"   Packages downloaded: {processed_packages}")
        logging.info(f"   Packages skipped: {skipped_packages}")
        logging.info(f"   Packages with errors: {error_packages}")
        logging.info(f"   Target bucket: {MINIO_BUCKET}")
    
    return summary

def download_meca_from_s3(s3_client, s3_key: str) -> Optional[bytes]:
    """
    Download a MECA package from S3 to memory.
    
    Args:
        s3_client: Configured boto3 S3 client
        s3_key: Full S3 key for the MECA package
    
    Returns:
        MECA package data as bytes, or None if failed
    """
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET, 
            Key=s3_key,
            RequestPayer='requester'
        )
        return response['Body'].read()
    except Exception as e:
        logging.error(f"Failed to download MECA package {s3_key} from S3: {e}")
        return None

def upload_meca_to_minio(
    minio_client: Minio, 
    object_key: str, 
    meca_data: bytes, 
    bucket_name: str = MINIO_MECA_BUCKET) -> None:
    """
    Upload MECA package data to MinIO.
    
    Args:
        minio_client: MinIO client
        object_key: MinIO object key for the MECA package
        meca_data: MECA package data as bytes
        bucket_name: MinIO bucket name
    """
    import io
    
    try:
        minio_client.put_object(
            bucket_name,
            object_key,
            io.BytesIO(meca_data),
            len(meca_data),
            content_type="application/zip"
        )
    except Exception as e:
        logging.error(f"Failed to upload MECA package to MinIO: {e}")
        raise

def meca_package_exists_in_minio(minio_client: Minio, object_key: str, bucket_name: str = MINIO_MECA_BUCKET) -> bool:
    """
    Check if a MECA package already exists in MinIO.
    
    Args:
        minio_client: MinIO client
        object_key: MinIO object key for the MECA package
        bucket_name: MinIO bucket name
    
    Returns:
        True if package exists, False otherwise
    """
    try:
        minio_client.stat_object(bucket_name, object_key)
        return True
    except S3Error as e:
        if e.code in ("NoSuchKey", "NoSuchObject", "NoSuchBucket"):
            return False
        else:
            logging.error(f"Error checking if package exists in MinIO: {e}")
            return False
    except Exception as e:
        logging.error(f"Unexpected error checking package existence: {e}")
        return False






# Unpacking MECA packages
def extract_meca_packages_from_minio(
    minio_client: Optional[Minio] = None,
    source_bucket: str = MINIO_MECA_BUCKET,
    target_bucket: str = MINIO_BUCKET,
    force: bool = False,
    verbose: bool = True
) -> dict:
    """
    Extract MECA packages from MinIO and store them in the target bucket.
    
    """
    if minio_client is None:
        minio_client = create_minio_client()

    if verbose:
        logging.info(f"Starting extraction of MECA packages from {source_bucket} to {target_bucket}...")
        logging.info(f"   Force extraction: {force}")

    meca_packages = list_meca_packages_in_minio(minio_client, source_bucket)

    if not meca_packages:
        logging.error("No MECA packages found in the source bucket")
        return {"processed": 0, "skipped": 0, "errors": 0}

    if verbose:
        logging.info(f"Found {len(meca_packages)} MECA packages in the source bucket")

    processed_packages = 0
    skipped_packages = 0
    error_packages = 0
    total_files_extracted = 0

    for i, package_name in enumerate(meca_packages, 1):
        try:
            if verbose:
                logging.info(f"Processing package {i}/{len(meca_packages)}: {package_name}")

            package_uuid = package_name.replace(".meca", "")
            if not force and package_extracted_in_minio(minio_client, target_bucket, package_uuid):
                if verbose:
                    logging.info(f"   Package {package_uuid} already extracted in MinIO, skipping...")
                skipped_packages += 1
                continue

            if verbose:
                logging.info(f"   Downloading package from MinIO: {package_name}")
            
            meca_data = download_meca_from_minio(minio_client, source_bucket, package_name)
            if not meca_data:
                logging.error(f"   Failed to download MECA package {package_uuid}")
                error_packages += 1
                continue
            
            if verbose:
                logging.info(f"   Extracting and uploading to MinIO: {package_name} to {target_bucket}")

            files_extracted = extract_and_upload_files_to_minio(
                minio_client, 
                target_bucket,
                package_uuid,
                meca_data,
                verbose=verbose
            )

            total_files_extracted += files_extracted
            processed_packages += 1

            if verbose:
                logging.info(f"   Successfully extracted and uploaded {files_extracted} files from {package_name}")
                
        except Exception as e:
            logging.error(f"   Error processing package {package_name}: {e}")
            error_packages += 1
            continue

    summary = {
        "processed": processed_packages,
        "skipped": skipped_packages,
        "errors": error_packages,
        "files_extracted": total_files_extracted
    }
    
    if verbose:
        logging.info(f"Extraction summary:")
        logging.info(f"   Packages processed: {processed_packages}")
        logging.info(f"   Packages skipped: {skipped_packages}")
        logging.info(f"   Packages with errors: {error_packages}")
        logging.info(f"   Total files extracted: {total_files_extracted}")
        logging.info(f"   Source bucket: {source_bucket}")
        logging.info(f"   Target bucket: {target_bucket}")
    
    return summary


def list_meca_packages_in_minio(minio_client: Minio, bucket_name: str = MINIO_MECA_BUCKET) -> List[str]:
    """
    List all MECA packages in the specified MinIO bucket.
    
    Args:
        minio_client: MinIO client
        bucket_name: Name of the bucket to search
    
    Returns:
        List of MECA package names
    """
    try:
        objects = minio_client.list_objects(bucket_name, recursive=True)
        meca_packages = [obj.object_name for obj in objects if obj.object_name.endswith('.meca')]
        return meca_packages
    except Exception as e:
        logging.error(f"Error listing MECA packages in bucket {bucket_name}: {e}")
        return []


def download_meca_from_minio(minio_client, bucket_name, object_name) -> Optional[bytes]:
    """
    Download a MECA package from MinIO to memory.
    
    Args:
        minio_client: MinIO client
        bucket_name: Source bucket name
        object_name: MECA package object name
    
    Returns:
        MECA package data as bytes, or None if failed
    """
    try:
        response = minio_client.get_object(bucket_name, object_name)
        return response.read()
    except Exception as e:
        logging.error(f"Failed to download MECA package {object_name} from MinIO: {e}")
        return None

def package_extracted_in_minio(minio_client: Minio, bucket_name: str, package_uuid: str) -> bool:
    """
    Check if a package has already been extracted in MinIO.
    
    Args:
        minio_client: MinIO client
        bucket_name: Target bucket name
        package_uuid: Package UUID to check
    
    Returns:
        True if package is already extracted, False otherwise
    """
    try:
        objects = list(minio_client.list_objects(bucket_name, prefix=f"{package_uuid}/", recursive=True))
        return len(objects) > 0
    except Exception as e:
        logging.error(f"Error checking if package {package_uuid} is extracted: {e}")
        return False

def extract_and_upload_files_to_minio(
    minio_client: None,
    target_bucket: str,
    package_uuid: str,
    meca_data: bytes,
    verbose: bool = True
) -> int:
    """
    Extract files from a MECA package and upload them to MinIO.
    
    Args:
        minio_client: MinIO client
        target_bucket: Target bucket name
        package_uuid: Package UUID
        meca_data: MECA package data as bytes
        verbose: Whether to show progress information

    Returns:
        Number of files extracted
    """
    if minio_client is None:
        minio_client = create_minio_client()

    extracted_count = 0
    try:
        meca_buffer = io.BytesIO(meca_data)
        with zipfile.ZipFile(meca_buffer) as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.is_dir():
                    continue

                file_name = file_info.filename
                file_content = zip_ref.read(file_info)

                object_key = f"{package_uuid}/{file_name}"

                minio_client.put_object(
                    target_bucket,
                    object_key,
                    io.BytesIO(file_content),
                    len(file_content),
                    content_type=get_content_type(os.path.splitext(file_name)[1])
                )

                extracted_count += 1

                if verbose and extracted_count % 10 == 0:
                    logging.info(f"   Extracted {extracted_count} files from {package_uuid}")

    except Exception as e:
        logging.error(f"Error extracting MECA package {package_uuid}: {e}")

    return extracted_count


def get_content_type(file_extension: str) -> str:
    """
    Get the content type for a file extension.
    """
    content_types = {
        ".pdf": "application/pdf",
        ".xml": "application/xml",
        ".json": "application/json",
        ".txt": "text/plain",
        ".html": "text/html",
    }

    return content_types.get(file_extension.lower(), "application/octet-stream")


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="BioRxiv MECA Package Processor - Download and extract MECA packages from S3 to MinIO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python biorxiv_process_meca.py --n-papers 10                    # Process 10 papers (all steps)
            python biorxiv_process_meca.py --n-papers 5 --step 1            # Search for 5 papers only
            python biorxiv_process_meca.py --n-papers 10 --step 2           # Download 10 papers only
            python biorxiv_process_meca.py --step 3                         # Extract all papers from MinIO
            python biorxiv_process_meca.py --n-papers 20 --step 1 --step 2  # Search and download 20 papers
        """
    )
    
    parser.add_argument(
        "--n-papers",
        type=int,
        default=5,
        help="Number of papers to process (default: 5)"
    )
    
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        action="append",
        help="Pipeline step to run: 1=search papers, 2=download papers, 3=extract papers. Can be used multiple times."
    )
    
    parser.add_argument(
        "--target-month",
        type=str,
        default="June_2025",
        help="Target month to search for papers (default: June_2025)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of existing packages"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (use -v for INFO, -vv for DEBUG)"
    )
    
    return parser.parse_args()


def run_pipeline_step(step: int, n_papers: int, target_month: str, force: bool, verbose: bool):
    """
    Run a specific pipeline step.
    
    Args:
        step: Pipeline step number (1=search, 2=download, 3=extract)
        n_papers: Number of papers to process
        target_month: Target month for search
        force: Force re-processing
        verbose: Verbose logging
    """
    if step == 1:
        # Step 1: Search for papers in S3
        logging.info(f"🔍 Step 1: Searching for {n_papers} papers in {target_month}...")
        papers = search_papers_in_s3(
            target_month=target_month,
            n_papers=n_papers,
            verbose=verbose
        )
        
        if papers:
            paper_info = get_paper_info(papers, verbose=verbose)
            logging.info(f"✅ Found {len(paper_info)} papers")
            return papers
        else:
            logging.error("❌ No papers found")
            return None
    
    elif step == 2:
        # Step 2: Download papers from S3 to MinIO
        logging.info(f"📥 Step 2: Downloading papers to MinIO...")
        
        # First search for papers if not already done
        papers = search_papers_in_s3(
            target_month=target_month,
            n_papers=n_papers,
            verbose=verbose
        )
        
        if papers:
            result = download_meca_packages_to_minio(
                s3_keys=papers,
                force=force,
                verbose=verbose
            )
            logging.info(f"✅ Download complete: {result}")
            return result
        else:
            logging.error("❌ No papers found to download")
            return None
    
    elif step == 3:
        # Step 3: Extract papers from MinIO
        logging.info(f"📦 Step 3: Extracting MECA packages from MinIO...")
        result = extract_meca_packages_from_minio(
            force=force,
            verbose=verbose
        )
        logging.info(f"✅ Extraction complete: {result}")
        return result


def main():
    """
    Main function with argument parsing and pipeline execution.
    """
    args = parse_arguments()
    setup_logger(verbosity=args.verbose)
    
    try:
        logging.info(f"🚀 Starting BioRxiv MECA Pipeline")
        logging.info(f"   Number of papers: {args.n_papers}")
        logging.info(f"   Target month: {args.target_month}")
        logging.info(f"   Force re-processing: {args.force}")
        
        # Determine which steps to run
        if args.step:
            # Run specific steps
            steps_to_run = args.step
            logging.info(f"   Running steps: {steps_to_run}")
        else:
            # Run all steps by default
            steps_to_run = [1, 2, 3]
            logging.info(f"   Running all steps: {steps_to_run}")
        
        # Execute pipeline steps
        for step in sorted(steps_to_run):
            logging.info(f"\n{'='*50}")
            logging.info(f"Executing Pipeline Step {step}")
            logging.info(f"{'='*50}")
            
            try:
                result = run_pipeline_step(
                    step=step,
                    n_papers=args.n_papers,
                    target_month=args.target_month,
                    force=args.force,
                    verbose=args.verbose > 0
                )
                
                if result is None and step in [1, 2]:
                    logging.error(f"❌ Step {step} failed, stopping pipeline")
                    break
                    
            except Exception as e:
                logging.error(f"❌ Error in step {step}: {e}")
                break
        
        logging.info(f"\n🎉 Pipeline execution completed!")
        
    except Exception as e:
        logging.error(f"❌ Error in main function: {e}")


if __name__ == "__main__":
    main()
