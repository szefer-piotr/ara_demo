#! /usr/bin/env python3
 
import os
import sys
import argparse
import logging
import json
import time
import random
import string
import tempfile
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import minio
from dotenv import load_dotenv

# PDF processing imports
try:
    import PyPDF2
    PDF_LIBRARY = "PyPDF2"
except ImportError:
    try:
        import pypdf
        PDF_LIBRARY = "pypdf"
    except ImportError:
        try:
            import fitz  # PyMuPDF
            PDF_LIBRARY = "PyMuPDF"
        except ImportError:
            PDF_LIBRARY = None

load_dotenv()

# Database configuration for bioRxiv processing
DB_HOST = os.getenv("BIORXIV_DB_HOST", "localhost")
DB_PORT = int(os.getenv("BIORXIV_DB_PORT", "5433"))
DB_NAME = os.getenv("BIORXIV_DB_NAME", "biorxiv_processing")
DB_USER = os.getenv("BIORXIV_DB_USER", "biorxiv_user")
DB_PASSWORD = os.getenv("BIORXIV_DB_PASSWORD", "biorxiv_password")

# Minio configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9002")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "biorxiv-papers")

minio_client = minio.Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

@dataclass
class PDFInfo:
    """Data class to store PDF information"""
    s3_key: str
    pdf_path: str
    pdf_name: str
    local_path: Optional[str] = None
    is_supplement: bool = False
    is_processed: Optional[bool] = False


def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def fetch_pdfs_from_minio(
    bucket: str = "biorxiv-papers", 
    max_pdfs: int = 10,
    include_supplements: bool = True,
) -> List[PDFInfo]:
    if bucket is None:
        bucket = MINIO_BUCKET

    print(f"Fetching {max_pdfs} PDFs from MinIO bucket: {bucket}")
    pdfs = []
    processed_folders = set()

    try:
        if not minio_client.bucket_exists(bucket):
            raise Exception(f"Bucket {bucket} does not exist")
            return []

        objects = minio_client.list_objects(bucket, prefix="", recursive=True)

        for obj in objects:
            if max_pdfs > 0 and len(pdfs) >= max_pdfs:
                break

            if not obj.object_name.endswith(".pdf"):
                continue

            path_parts = obj.object_name.split("/")
            if len(path_parts) < 3:
                print(f"    Skipping invalid path: {obj.object_name}")
                continue

            s3_key = path_parts[0]
            folder_type = path_parts[1]

            if folder_type != "content":
                continue
            is_supplement = len(path_parts) > 3 and path_parts[2] == "supplements"
            if is_supplement and not include_supplements:
                continue

            pdf_info = PDFInfo(
                s3_key=s3_key,
                pdf_path=obj.object_name,
                pdf_name=os.path.basename(obj.object_name),
                local_path=None,
                is_supplement=is_supplement,
                is_processed=None                
            )

            pdfs.append(pdf_info)

            path_type = "supplement" if is_supplement else "main content"
            print(f"\n  Found PDF: {pdf_info.pdf_name}")
            print(f"    Path type: {pdf_info.pdf_path}")
            print(f"    S3 key: {pdf_info.s3_key}")
            print(f"    Is processed: {pdf_info.is_processed}")
            print(f"    Path type: {path_type}")

    except Exception as e:
        print(f"Error fetching PDFs from MinIO: {e}")
        return []

    unique_s3_keys = set(pdf.s3_key for pdf in pdfs)
    return pdfs


def print_pdfs(pdfs: List[PDFInfo]):
    if not pdfs:
        print("No PDFs found")
        return

    print(f"Detailed information about PDFs:")
    print("="*60)

    by_s3_key = {}
    for pdf in pdfs:
        if pdf.s3_key not in by_s3_key:
            by_s3_key[pdf.s3_key] = []
        by_s3_key[pdf.s3_key].append(pdf)
    
    for s3_key, pdf_list in by_s3_key.items():
        print(f"S3 key: {s3_key}")
    
        for pdf in pdf_list:
            location = "supplements" if pdf.is_supplement else "content"
            print(f"    {pdf.pdf_path} ({location})")

    print(f"\n\nSummary:")
    print(f"    Total PDFs: {len(pdfs)}")
    print(f"    Unique S3 keys: {len(by_s3_key)}")
    print(f"    Total supplements: {sum(1 for pdf in pdfs if pdf.is_supplement)}")
    print(f"    Total content: {sum(1 for pdf in pdfs if not pdf.is_supplement)}")

# =========== POSTGRES DATABASE FUNCTIONS ===========
# This functions will be later moved to a separate file

def get_db_connection():
    """Get a database connection to the bioRxiv processing database"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"Error getting database connection: {e}")
        return None


def create_tables():
    """Create the necessry tables in for PDF processing state management"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_processing_state (
                    id SERIAL PRIMARY KEY,
                    pdf_hash VARCHAR(64) UNIQUE NOT NULL,
                    s3_key VARCHAR(255) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    error_message TEXT,
                    metadata JSONB
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS text_chunks (
                    id SERIAL PRIMARY KEY,
                    pdf_hash VARCHAR(64) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    chunk_hash VARCHAR(64) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    FOREIGN KEY (pdf_hash) REFERENCES pdf_processing_state(pdf_hash)
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id SERIAL PRIMARY KEY,
                    pdf_hash VARCHAR(64) NOT NULL,
                    log_level VARCHAR(10) NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pdf_hash) REFERENCES pdf_processing_state(pdf_hash)
                );
            """)

            cur.execute("""CREATE INDEX IF NOT EXISTS idx_pdf_processing_status ON pdf_processing_state(status);""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_pdf_processing_created_at ON pdf_processing_state(created_at);""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_text_chunks_pdf_hash ON text_chunks(pdf_hash);""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_processing_logs_pdf_hash ON processing_logs(pdf_hash);""")
            
            conn.commit()
            logging.info("  Tables created successfully")
    
    except Exception as e:
        print(f"Error creating tables: {e}")
    finally:
        conn.close()


def test_database_connection():
    """Test the database connection and create tables if they don't exist"""
    try:
        logging.info("Testing database connection...")
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()
            logging.info(f"Connected to PostgreSQL version: {version[0]}")

        create_tables()
        logging.info("Database setup completed successfully")
    
    except Exception as e:
        logging.error(f"Error testing database connection: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

#===================================================


def main():
    setup_logger(verbosity=1)

    try:
        test_database_connection()
        pdfs = fetch_pdfs_from_minio()
        print_pdfs(pdfs)

    except Exception as e:
        logging.error(f"Main function error: {e}")
        raise


if __name__ == "__main__":
    main()