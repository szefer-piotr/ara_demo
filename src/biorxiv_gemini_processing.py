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
import re
import psycopg2
import minio
import google.generativeai as genai
import pymongo

from token import OP
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from utils.prompt_templates import gemini_processing_instructions

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
DB_PORT = int(os.getenv("BIORXIV_DB_PORT", "9999"))
# DB_NAME = os.getenv("BIORXIV_DB_NAME", "biorxiv_processing")
# DB_USER = os.getenv("BIORXIV_DB_USER", "biorxiv_user")
# DB_PASSWORD = os.getenv("BIORXIV_DB_PASSWORD", "biorxiv_password")
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

# Minio configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9002")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "biorxiv-papers")

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "gemini_responses")

# MongoDB configuration
MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGODB_PORT = int(os.getenv("MONGODB_PORT", "27017"))
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME", "admin")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD", "admin123")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "biorxiv_gemini")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "gemini_responses")

# minio_client = minio.Minio(
#         MINIO_ENDPOINT,
#         access_key=MINIO_ACCESS_KEY,
#         secret_key=MINIO_SECRET_KEY,
#         secure=MINIO_SECURE
#     )

def get_minio_client() -> minio.Minio:
    """Get a MinIO client connection"""
    try:
        client = minio.Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        
        # Test the connection by listing buckets
        client.list_buckets()
        logging.info(f"Connected to MinIO at {MINIO_ENDPOINT}")
        return client
        
    except Exception as e:
        logging.error(f"Error connecting to MinIO: {e}")
        raise

@dataclass
class PDFInfo:
    """Data class to store PDF information"""
    s3_key: str
    pdf_path: str
    pdf_name: str
    local_path: Optional[str] = None
    is_supplement: bool = False
    is_processed: Optional[bool] = False


@dataclass
class ExtractedText:
    """Data class to store extracted text"""
    pdf_hash: str
    s3_key: str
    pdf_name: str
    full_text: str
    sections: Dict[str, str]
    metadata: Dict[str, any]
    extracted_at: datetime
    text_length: int


@dataclass
class ProcessingSatus(Enum):
    """Data class to store processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


# =========== PHASE 1: BASIC TEXT EXTRACTION ===========


def calculate_pdf_hash(pdf_path: str) -> str:
    """Calculate the hash of a PDF file"""
    hash_md5 = hashlib.md5()
    try:
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logging.error(f"Error calculating hash for {pdf_path}: {e}")
        return ""


def list_folders_in_minio_bucket(minio_client: minio.Minio, bucket_name: str = "biorxiv-unpacked") -> List[str]:
    """List all folders in the specified MinIO bucket"""
    try:
        if not minio_client.bucket_exists(bucket_name):
            logging.warning(f"Bucket {bucket_name} does not exist")
            return []
        
        folders = set()
        objects = minio_client.list_objects(bucket_name, prefix="", recursive=True)
        
        for obj in objects:
            # Split the object name by '/' to get path components
            path_parts = obj.object_name.split("/")
            
            # Add each folder level to the set
            for i in range(1, len(path_parts)):  # Skip the first part (empty string from leading '/')
                folder_path = "/".join(path_parts[:i+1])
                folders.add(folder_path)
        
        folder_list = sorted(list(folders))
        logging.info(f"Found {len(folder_list)} folders in bucket {bucket_name}")
        return folder_list
        
    except Exception as e:
        logging.error(f"Error listing folders in bucket {bucket_name}: {e}")
        return []


def download_pdf_from_minio(minio_client: minio.Minio, pdf_info: PDFInfo, temp_dir: str, bucket_name: str = "biorxiv-unpacked") -> Optional[str]:
    """Download a PDF from MinIO to a temporary directory"""
    try:
        # Create safe filename
        safe_filename = pdf_info.pdf_name.replace("/", "_").replace("\\", "_")
        local_path = os.path.join(temp_dir, safe_filename)

        #Download PDF from MinIO
        minio_client.fget_object(bucket_name, pdf_info.pdf_path, local_path)
        logging.info(f"Downloaded PDF {pdf_info.pdf_name} to {local_path}")
        return local_path

    except Exception as e:
        logging.error(f"Error downloading PDF {pdf_info.pdf_name}: {e}")
        return None


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using available library"""
    if PDF_LIBRARY is None:
        raise ImportError("No PDF library available. Please install PyPDF2, pypdf, or PyMuPDF")
    
    text = ""
    
    try:
        if PDF_LIBRARY == "PyPDF2":
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text
                    except Exception as e:
                        logging.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        
        elif PDF_LIBRARY == "pypdf":
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text
                    except Exception as e:
                        logging.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        
        elif PDF_LIBRARY == "PyMuPDF":
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                except Exception as e:
                    logging.warning(f"Error extracting text from page {page_num + 1}: {e}")
            doc.close()
            
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        raise
    
    return text
        

def remove_bibliography_references(text: str) -> str:
    """Remove bibliography and references section from text"""
    # Common bibliography/references section patterns
    bibliography_patterns = [
        r'(?i)\n\s*(references|bibliography|works\s+cited|literature\s+cited)\s*\n.*$',
        r'(?i)\n\s*(acknowledgments?|acknowledgements?)\s*\n.*$',
        r'(?i)\n\s*(supplementary\s+material|supplementary\s+information)\s*\n.*$',
        r'(?i)\n\s*(author\s+contributions?)\s*\n.*$',
        r'(?i)\n\s*(competing\s+interests?|conflict\s+of\s+interest)\s*\n.*$',
        r'(?i)\n\s*(data\s+availability)\s*\n.*$',
        r'(?i)\n\s*(funding)\s*\n.*$',
    ]
    
    cleaned_text = text
    
    for pattern in bibliography_patterns:
        # Remove everything from the pattern match to the end
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)
    
    # Remove common reference patterns (e.g., [1], [2], etc.)
    reference_patterns = [
        r'\[\d+\]',  # [1], [2], etc.
        r'\(\d{4}[a-z]?\)',  # (2023), (2023a), etc.
        r'\([A-Za-z]+\s+et\s+al\.?\s*,\s*\d{4}\)',  # (Smith et al., 2023)
    ]
    
    for pattern in reference_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text)
    
    return cleaned_text.strip()


def detect_sections(text: str) -> Dict[str, str]:
    """Detect and extract main sections from the text"""
    sections = {}
    
    # Common section headers in scientific papers
    section_patterns = {
        'abstract': r'(?i)\n\s*(abstract)\s*\n',
        'introduction': r'(?i)\n\s*(introduction)\s*\n',
        'methods': r'(?i)\n\s*(methods?|methodology|materials?\s+and\s+methods?)\s*\n',
        'results': r'(?i)\n\s*(results?)\s*\n',
        'discussion': r'(?i)\n\s*(discussion)\s*\n',
        'conclusion': r'(?i)\n\s*(conclusions?|concluding\s+remarks?)\s*\n',
        'background': r'(?i)\n\s*(background|background\s+and\s+significance)\s*\n',
        'data_analysis': r'(?i)\n\s*(data\s+analysis|statistical\s+analysis)\s*\n',
        'limitations': r'(?i)\n\s*(limitations?|study\s+limitations?)\s*\n',
        'future_work': r'(?i)\n\s*(future\s+work|future\s+directions?|future\s+research)\s*\n',
    }
    
    # Split text into lines for better processing
    lines = text.split('\n')
    current_section = 'header'
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line matches any section header
        section_found = None
        for section_name, pattern in section_patterns.items():
            if re.match(pattern, f'\n{line}\n', re.IGNORECASE):
                section_found = section_name
                break
        
        if section_found:
            # Save previous section
            if current_section != 'header' and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = section_found
            current_content = []
        else:
            # Add line to current section
            current_content.append(line)
    
    # Save the last section
    if current_section != 'header' and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    # If no sections were found, put everything in a 'full_text' section
    if not sections:
        sections['full_text'] = text
    
    return sections


def process_pdf_text_extraction(minio_client: minio.Minio, pdf_info: PDFInfo) -> Optional[ExtractedText]:
    """Main function to process PDF text extraction (Phase 1)"""
    try:
        # Create temporary directory for PDF processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download PDF from MinIO
            local_pdf_path = download_pdf_from_minio(minio_client, pdf_info, temp_dir)
            if not local_pdf_path:
                return None
            
            # Calculate PDF hash
            pdf_hash = calculate_pdf_hash(local_pdf_path)
            if not pdf_hash:
                return None
            
            # Extract text from PDF
            logging.info(f"Extracting text from {pdf_info.pdf_name}...")
            full_text = extract_text_from_pdf(local_pdf_path)
            
            if not full_text.strip():
                logging.warning(f"No text extracted from {pdf_info.pdf_name}")
                return None
            
            # Remove bibliography and references
            logging.info(f"Removing bibliography from {pdf_info.pdf_name}...")
            cleaned_text = remove_bibliography_references(full_text)
            
            # Detect sections
            logging.info(f"Detecting sections in {pdf_info.pdf_name}...")
            sections = detect_sections(cleaned_text)
            
            # Create metadata
            metadata = {
                'original_text_length': len(full_text),
                'cleaned_text_length': len(cleaned_text),
                'sections_found': list(sections.keys()),
                'pdf_library_used': PDF_LIBRARY,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Create ExtractedText object
            extracted_text = ExtractedText(
                pdf_hash=pdf_hash,
                s3_key=pdf_info.s3_key,
                pdf_name=pdf_info.pdf_name,
                full_text=cleaned_text,
                sections=sections,
                metadata=metadata,
                extracted_at=datetime.now(),
                text_length=len(cleaned_text)
            )
            
            logging.info(f"Successfully processed {pdf_info.pdf_name}")
            logging.info(f"  - Text length: {len(cleaned_text)} characters")
            logging.info(f"  - Sections found: {list(sections.keys())}")
            
            return extracted_text
            
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_info.pdf_name}: {e}")
        return None


def batch_process_pdfs(minio_client: minio.Minio, pdf_list: List[PDFInfo]) -> List[ExtractedText]:
    """Process multiple PDFs in batch"""
    extracted_texts = []
    
    logging.info(f"Starting batch processing of {len(pdf_list)} PDFs...")
    
    for i, pdf_info in enumerate(pdf_list, 1):
        logging.info(f"Processing PDF {i}/{len(pdf_list)}: {pdf_info.pdf_name}")
        
        extracted_text = process_pdf_text_extraction(minio_client, pdf_info)
        if extracted_text:
            extracted_texts.append(extracted_text)
        else:
            logging.warning(f"Failed to process PDF: {pdf_info.pdf_name}")
    
    logging.info(f"Batch processing completed. Successfully processed {len(extracted_texts)}/{len(pdf_list)} PDFs")
    return extracted_texts

#========= GEMINI PROCESSING FUNCTIONS ===========

@dataclass
class GeminiProcessingResult:
    """Data class to store Gemini processing result with prompt information"""
    prompt: str
    token_count: int
    sections_used: List[str]
    context_length: int

    pdf_hash: str
    s3_key: str
    pdf_name: str
    sections_used: List[str]
    instructions: str
    prompt_template: str
    response: str
    processed_at: datetime
    metadata: Dict[str, any]

    def __str__(self):
        return self.prompt

    @property
    def is_too_long(self, max_tokens: int = 1048576):
        return self.token_count > max_tokens

    @property
    def prompt_preview(self, max_length: int = 100):
        return self.prompt[:max_length] + "..." if len(self.prompt) > max_length else self.prompt

    @classmethod
    def create_from_prompt(
        cls, prompt: str, token_count: int, sections_used: List[str], 
        context_length: int, pdf_hash: str, s3_key: str, pdf_name: str, 
        instructions: str, prompt_template: str, response: str = "", 
        processed_at: datetime = None, metadata: Dict[str, any] = None):
        """Create a GeminiProcessingResult from prompt data"""
        if processed_at is None:
            processed_at = datetime.now()
        if metadata is None:
            metadata = {}
        
        return cls(
            prompt=prompt,
            token_count=token_count,
            sections_used=sections_used,
            context_length=context_length,
            pdf_hash=pdf_hash,
            s3_key=s3_key,
            pdf_name=pdf_name,
            instructions=instructions,
            prompt_template=prompt_template,
            response=response,
            processed_at=processed_at,
            metadata=metadata
        )


def count_tokens_gemini(text: str) -> int:
    """Count tokens using Gemini's official tokenizer"""
    try:        
        model = genai.GenerativeModel('gemini-2.0-flash-exp', api_key=os.getenv('GOOGLE_API_KEY'))
        response = model.count_tokens(text)
        return response.total_tokens
    except Exception as e:
        logging.warning(f"Could not count tokens with Gemini API: {e}")
        # Fallback to word-based approximation
        return int(len(text.split()) * 1.3)


def build_gemini_prompt(
    extracted_text: ExtractedText, 
    sections: List[str], 
    instructions: str) -> GeminiProcessingResult:
    """Build a Gemini prompt for the specified sections and instructions"""
    # Handle both single ExtractedText and list of ExtractedText objects
    
    if isinstance(extracted_text, ExtractedText):
        if not extracted_text:
            logging.warning("No extracted texts provided")
            return instructions
        
        # Process all extracted texts
        all_context_parts = []
        available_sections = list(extracted_text.sections.keys())
        
        # Check if 'all' is requested
        if 'all' in sections:
            valid_sections = available_sections
            invalid_sections = []
        else:
            valid_sections = [section for section in sections if section in available_sections]
            invalid_sections = [section for section in sections if section not in available_sections]

        if invalid_sections:
            logging.warning(f"Sections not found in {extracted_text.pdf_name}: {invalid_sections}")

        if not valid_sections:
            logging.warning(f"No valid sections found in {extracted_text.pdf_name}")
            paper_context = extracted_text.full_text
        else:
            context_parts = []
            for section in valid_sections:
                context_parts.append(f"## {section.upper()}\n{extracted_text.sections[section]}")
            paper_context = "\n\n".join(context_parts)
        
        # Add paper identifier and context
        all_context_parts.append(f"# PAPER: {extracted_text.pdf_name}\n{paper_context}")
        context = "\n\n".join(all_context_parts)
    else:
        logging.warning("No extracted texts provided")
        # return instructions
        return None
    
    prompt = f"CONTEXT:\n{context}\n\nINSTRUCTIONS:\n{instructions}"
    token_count = count_tokens_gemini(prompt)
    logging.info(f"Number of tokens in prompt: {token_count}")

    return GeminiProcessingResult.create_from_prompt(
        prompt=prompt.strip(),
        token_count=token_count,
        sections_used=valid_sections if 'all' not in sections else ["all"],
        context_length=len(context),
        pdf_hash="",
        s3_key="",
        pdf_name="",
        instructions=instructions,
        prompt_template=gemini_processing_instructions,
        response="",
        metadata={}
    )


def get_response_from_gemini(prompt: GeminiProcessingResult) -> str:
    """Get a response from Gemini"""
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-2.0-flash-lite')

        logging.info(f"Sending request to gemini-2.0-flash-lite with prompt: {prompt.prompt_preview}...")

        if prompt.is_too_long:
            logging.warning(f"Prompt is too long. Truncating to {prompt.max_tokens} tokens.")
            prompt.prompt = prompt.prompt[:prompt.max_tokens]

        response = model.generate_content(
            prompt.prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=65536,
                response_mime_type="application/json"
            )
        )

        if not response.text:
            logging.warning(f"No response from Gemini. Returning empty string.")
            return ""

        logging.info(f"Response from Gemini: {response.text[:100]}...")
        return response.text

    except Exception as e:
        logging.error(f"Error getting response from Gemini: {e}")
        return ""


# ======= HELPER FUNCTIONS ===========

def fetch_pdfs_from_minio(
    minio_client: minio.Minio,
    bucket: str = "biorxiv-papers",
    max_pdfs: int = 10, 
    include_supplements: bool = True
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
    """Create the necessary tables for PDF processing state management"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check if tables already exist
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'pdf_processing_state'
                );
            """)
            tables_exist = cur.fetchone()[0]
            
            if not tables_exist:
                logging.info("Tables don't exist, they should be created by the database initialization script")
                logging.info("Please ensure the bioRxiv PostgreSQL service is running and the init script has been executed")
            else:
                logging.info("Tables already exist in the database")
            
            conn.commit()
    
    except Exception as e:
        logging.error(f"Error checking tables: {e}")
    finally:
        conn.close()


def save_extracted_text(extracted_text: ExtractedText) -> bool:
    """Save extracted text to the database"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Insert or update processing state
            cur.execute("""
                INSERT INTO public.pdf_processing_state (pdf_hash, s3_key, pdf_name, status, processed_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (pdf_hash) 
                DO UPDATE SET 
                    status = EXCLUDED.status,
                    processed_at = EXCLUDED.processed_at,
                    updated_at = CURRENT_TIMESTAMP,
                    metadata = EXCLUDED.metadata
            """, (
                extracted_text.pdf_hash,
                extracted_text.s3_key,
                extracted_text.pdf_name,
                'completed',
                extracted_text.extracted_at,
                json.dumps(extracted_text.metadata)
            ))
            
            # Insert extracted text
            cur.execute("""
                INSERT INTO extracted_texts (pdf_hash, s3_key, pdf_name, full_text, sections, metadata, text_length)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pdf_hash) 
                DO UPDATE SET 
                    full_text = EXCLUDED.full_text,
                    sections = EXCLUDED.sections,
                    metadata = EXCLUDED.metadata,
                    text_length = EXCLUDED.text_length
            """, (
                extracted_text.pdf_hash,
                extracted_text.s3_key,
                extracted_text.pdf_name,
                extracted_text.full_text,
                json.dumps(extracted_text.sections),
                json.dumps(extracted_text.metadata),
                extracted_text.text_length
            ))
            
            conn.commit()
            logging.info(f"Saved extracted text for {extracted_text.pdf_name}")
            return True
            
    except Exception as e:
        logging.error(f"Error saving extracted text: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_processing_status(pdf_hash: str) -> Optional[str]:
    """Get the processing status of a PDF"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM pdf_processing_state WHERE pdf_hash = %s", (pdf_hash,))
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        logging.error(f"Error getting processing status: {e}")
        return None
    finally:
        conn.close()


def test_database_connection():
    """Test the database connection and check if tables exist"""
    try:
        logging.info("Testing bioRxiv database connection...")
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()
            logging.info(f"Connected to PostgreSQL version: {version[0]}")

        create_tables()
        logging.info("BioRxiv database setup completed successfully")
    
    except Exception as e:
        logging.error(f"Error testing database connection: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()


def save_gemini_response_to_postgres(gemini_result: GeminiProcessingResult) -> bool:
    """Save Gemini response to PostgreSQL"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Insert into gemini_responses table
            cur.execute("""
                INSERT INTO gemini_responses (
                    pdf_hash, s3_key, pdf_name, gemini_response, 
                    sections_used, token_count, context_length,
                    instructions, prompt_template, processed_at, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pdf_hash) 
                DO UPDATE SET 
                    gemini_response = EXCLUDED.gemini_response,
                    sections_used = EXCLUDED.sections_used,
                    token_count = EXCLUDED.token_count,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                gemini_result.pdf_hash,
                gemini_result.s3_key,
                gemini_result.pdf_name,
                json.dumps(gemini_result.response) if isinstance(gemini_result.response, str) else json.dumps(gemini_result.response),
                gemini_result.sections_used,
                gemini_result.token_count,
                gemini_result.context_length,
                gemini_result.instructions,
                gemini_result.prompt_template,
                gemini_result.processed_at,
                json.dumps(gemini_result.metadata)
            ))
            
            conn.commit()
            logging.info(f"Saved Gemini response for {gemini_result.pdf_name} to PostgreSQL")
            return True
            
    except Exception as e:
        logging.error(f"Error saving Gemini response to PostgreSQL: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def batch_store_gemini_responses_in_postgres(gemini_processing_results: List[GeminiProcessingResult]) -> None:
    """Store list ofGemini responses in PostgreSQL database"""
    pass


# QDRANT FUNCTIONS
def get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client connection"""
    try:
        if QDRANT_API_KEY:
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
        else:
            client = QdrantClient(url=QDRANT_URL)
        
        # Test the connection
        client.get_collections()
        logging.info(f"Connected to Qdrant at {QDRANT_URL}")
        return client
        
    except Exception as e:
        logging.error(f"Error connecting to Qdrant: {e}")
        raise


def ensure_qdrant_collection_exists(client: QdrantClient, collection_name: str = None) -> bool:
    """Ensure the Gemini responses collection exists, create if it doesn't"""
    if collection_name is None:
        collection_name = QDRANT_COLLECTION
    
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name in collection_names:
            logging.info(f"Collection '{collection_name}' already exists")
            return True
        
        # Create collection if it doesn't exist
        logging.info(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # OpenAI text-embedding-3-small dimension
                distance=Distance.COSINE
            )
        )
        
        logging.info(f"Successfully created collection '{collection_name}'")
        return True
        
    except Exception as e:
        logging.error(f"Error ensuring collection exists: {e}")
        return False


def get_qdrant_collection_info(client: QdrantClient, collection_name: str = None) -> Dict:
    """Get information about the Gemini responses collection"""
    if collection_name is None:
        collection_name = QDRANT_COLLECTION
    
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "status": info.status,
            "optimizer_status": info.optimizer_status,
            "payload_schema": info.payload_schema
        }
    except Exception as e:
        logging.error(f"Error getting collection info: {e}")
        return {}


def delete_qdrant_collection(client: QdrantClient, collection_name: str = None) -> bool:
    """Delete the Gemini responses collection (use with caution!)"""
    if collection_name is None:
        collection_name = QDRANT_COLLECTION
    
    try:
        client.delete_collection(collection_name)
        logging.info(f"Successfully deleted collection '{collection_name}'")
        return True
    except Exception as e:
        logging.error(f"Error deleting collection: {e}")
        return False


# SEARCHING
def search_gemini_responses(client: QdrantClient, query_text: str, collection_name: str = None,) -> List[StoredGeminiResponse]:
    """Search for Gemini responses in Qdrant using vector search"""
    pass


def get_gemini_response_by_id(client: QdrantClient, point_id: str, collection_name: str = None) -> Optional[Dict[str, any]]:
    pass


def delete_gemini_response_by_id(client: QdrantClient, point_id: str, collection_name: str = None) -> bool:
    pass


@dataclass
class EmbeddingResult:
    """Data class to store embedding results"""
    vector: List[float]
    model: str
    text_length: int
    token_count: int
    created_at: datetime


class GeminiEmbeddingProvider:
    """Embedding provider for Gemini response text using OpenAI embeddings"""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not provided")

        self.model = model
        self.timeout = 30

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Error initializing OpenAI client: {e}") from e

    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for text."""
        try:
            clean_text = " ".join(text.split())
            response = self._client.embeddings.create(
                model=self.model,
                input=clean_text,
                timeout=self.timeout
            )

            vector = response.data[0].embedding
            token_count = response.usage.total_tokens
            
            return EmbeddingResult(
                vector=vector,
                model=self.model,
                text_length=len(clean_text),
                token_count=token_count,
                created_at=datetime.now()
            )
        
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            raise

    def embed_gemini_response(self, gemini_response: str, max_length: int = 20000) -> EmbeddingResult:
        """Generate embedding for Geminiresponse with text truncated if needed"""
        try:
            try:
                response_data = json.loads(gemini_response)
                if isinstance(response_data, dict):
                    text_content = self._extract_text_from_json(response_data)
                else:
                    text_content = str(response_data)
            except (json.JSONDecodeError, TypeError):
                text_content = gemini_response

            if len(text_content) > max_length:
                text_content = text_content[:max_length]
                logging.warning(f"Text content truncated to {max_length} characters")

            return self.embed_text(text_content)
        
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            raise

    def _extract_text_from_json(self, data: Dict) -> str:
        """Extract text content from JSON structure based on Gemini response template"""
        try:
            # Handle the specific Gemini response template structure
            text_parts = []
            
            # Extract metadata information
            if 'metadata' in data and isinstance(data['metadata'], dict):
                metadata = data['metadata']
                if 'title' in metadata:
                    text_parts.append(f"Title: {metadata['title']}")
                if 'abstract' in metadata:
                    text_parts.append(f"Abstract: {metadata['abstract']}")
                if 'authors' in metadata and isinstance(metadata['authors'], list):
                    authors = ", ".join(metadata['authors'])
                    text_parts.append(f"Authors: {authors}")
                if 'doi' in metadata:
                    text_parts.append(f"DOI: {metadata['doi']}")
            
            # Extract hypotheses content (main content)
            if 'extracted_content' in data and isinstance(data['extracted_content'], dict):
                extracted = data['extracted_content']
                
                if 'hypotheses' in extracted and isinstance(extracted['hypotheses'], list):
                    for i, hypothesis in enumerate(extracted['hypotheses'], 1):
                        if isinstance(hypothesis, dict):
                            text_parts.append(f"\n--- Hypothesis {i} ---")
                            
                            # Core hypothesis text
                            if 'text' in hypothesis:
                                text_parts.append(f"Text: {hypothesis['text']}")
                            
                            # Motivation
                            if 'motivation' in hypothesis:
                                text_parts.append(f"Motivation: {hypothesis['motivation']}")
                            
                            # Conceptual approaches
                            if 'conceptual_approaches' in hypothesis and isinstance(hypothesis['conceptual_approaches'], list):
                                approaches = "; ".join(hypothesis['conceptual_approaches'])
                                text_parts.append(f"Conceptual Approaches: {approaches}")
                            
                            # Validation approaches
                            if 'validation_approaches' in hypothesis and isinstance(hypothesis['validation_approaches'], dict):
                                validation = hypothesis['validation_approaches']
                                if 'experimental' in validation:
                                    text_parts.append(f"Experimental Validation: {validation['experimental']}")
                                if 'statistical' in validation and isinstance(validation['statistical'], list):
                                    stats = "; ".join(validation['statistical'])
                                    text_parts.append(f"Statistical Validation: {stats}")
                            
                            # Datasets
                            if 'datasets' in hypothesis and isinstance(hypothesis['datasets'], list):
                                datasets = "; ".join(hypothesis['datasets'])
                                text_parts.append(f"Datasets: {datasets}")
                            
                            # Results
                            if 'results' in hypothesis and isinstance(hypothesis['results'], dict):
                                results = hypothesis['results']
                                if 'outcome' in results:
                                    text_parts.append(f"Outcome: {results['outcome']}")
                                if 'explanation' in results:
                                    text_parts.append(f"Explanation: {results['explanation']}")
                            
                            # Discussion
                            if 'discussion' in hypothesis:
                                text_parts.append(f"Discussion: {hypothesis['discussion']}")
                            
                            # Future considerations
                            if 'future_considerations' in hypothesis:
                                text_parts.append(f"Future Considerations: {hypothesis['future_considerations']}")
                            
                            # Images
                            if 'images' in hypothesis and isinstance(hypothesis['images'], list):
                                for j, image in enumerate(hypothesis['images'], 1):
                                    if isinstance(image, dict):
                                        image_text = f"Image {j}"
                                        if 'figure_number' in image:
                                            image_text += f" (Figure {image['figure_number']})"
                                        if 'caption' in image:
                                            image_text += f": {image['caption']}"
                                        text_parts.append(image_text)
            
            # Add processing timestamp if available
            if 'processing_timestamp' in data:
                text_parts.append(f"Processed: {data['processing_timestamp']}")
            
            # Join all text parts
            if text_parts:
                return "\n".join(text_parts)
            
            # Fallback: try common fields from original implementation
            common_fields = ['content', 'text', 'response', 'answer', 'summary', 'analysis', 'result', 'output']
            for field in common_fields:
                if field in data and isinstance(data[field], str):
                    return data[field]
            
            # Final fallback: convert entire structure to string
            return json.dumps(data, indent=2)
            
        except Exception as e:
            logging.warning(f"Error extracting text from JSON: {e}")
            # Fallback to original structure
            return json.dumps(data, indent=2)


def get_embedding_provider() -> GeminiEmbeddingProvider:
    """Get a configured embedding provider"""
    return GeminiEmbeddingProvider()


# =========== MONGODB FUNCTIONS ===========

@dataclass
class StoredGeminiResponse:
    """Data class to store a Gemini response in Qdrant"""
    point_id: str
    pdf_hash: str
    pdf_name: str
    s3_key: str
    gemini_response: str
    sections_used: List[str]
    token_count: int
    processed_at: datetime
    metadata: Dict[str, any]

def get_mongodb_client() -> MongoClient:
    """Get a MongoDB client connection"""
    try:
        # Build connection string
        connection_string = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_HOST}:{MONGODB_PORT}/{MONGODB_DATABASE}?authSource=admin"
        
        client = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=10000,         # 10 second connection timeout
            socketTimeoutMS=20000           # 20 second socket timeout
        )
        
        # Test the connection
        client.admin.command('ping')
        logging.info(f"Connected to MongoDB at {MONGODB_HOST}:{MONGODB_PORT}")
        return client
        
    except ConnectionFailure as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        raise
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        raise


def get_mongodb_database(client: MongoClient):
    """Get the MongoDB database"""
    return client[MONGODB_DATABASE]


def get_mongodb_collection(client: MongoClient, collection_name: str = None):
    """Get the MongoDB collection"""
    if collection_name is None:
        collection_name = MONGODB_COLLECTION
    
    db = get_mongodb_database(client)
    return db[collection_name]


def setup_mongodb_indexes(client: MongoClient, collection_name: str = None):
    """Setup indexes for the MongoDB collection with versioning support"""
    try:
        collection = get_mongodb_collection(client, collection_name)
        
        # Create single field indexes
        indexes = [
            ("pdf_hash", pymongo.ASCENDING),
            ("config_key", pymongo.ASCENDING),
            ("s3_key", pymongo.ASCENDING),
            ("pdf_name", pymongo.ASCENDING),
            ("created_at", pymongo.DESCENDING),  # Important for version ordering
            ("version", pymongo.DESCENDING),
            ("processed_at", pymongo.DESCENDING),
            ("sections_used", pymongo.ASCENDING),
            ("instructions", pymongo.ASCENDING),
            ("prompt_template", pymongo.ASCENDING),
            ("token_count", pymongo.ASCENDING)
        ]
        
        for index_spec in indexes:
            collection.create_index([index_spec])
        
        # Create compound indexes for efficient queries
        compound_indexes = [
            [("pdf_hash", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)],  # For getting latest version
            [("sections_used", pymongo.ASCENDING), ("instructions", pymongo.ASCENDING), ("prompt_template", pymongo.ASCENDING)],  # For config matching
            [("config_key", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)]  # For unique configs
        ]
        
        for compound_index in compound_indexes:
            collection.create_index(compound_index)
        
        # Note: We removed the unique constraint on pdf_hash since we now allow multiple versions
        
        logging.info(f"Successfully created indexes for collection '{collection_name}'")
        return True
        
    except Exception as e:
        logging.error(f"Error creating MongoDB indexes: {e}")
        return False


def save_gemini_response_to_mongodb(gemini_result: GeminiProcessingResult, client: MongoClient = None) -> bool:
    """Save Gemini response to MongoDB only if key fields have changed"""
    try:
        # Use provided client or create new one
        if client is None:
            client = get_mongodb_client()
            close_client = True
        else:
            close_client = False
        
        collection = get_mongodb_collection(client)
        
        # Check if we need to save by comparing key fields
        existing_record = collection.find_one(
            {"pdf_hash": gemini_result.pdf_hash},
            sort=[("created_at", -1)]  # Get the most recent record
        )
        
        # Check if any of the key fields have changed
        should_save = False
        change_reason = []
        
        if existing_record is None:
            should_save = True
            change_reason.append("new record")
        else:
            # Compare sections_used (order-independent)
            existing_sections = set(existing_record.get("sections_used", []))
            new_sections = set(gemini_result.sections_used or [])
            if existing_sections != new_sections:
                should_save = True
                change_reason.append("sections_used changed")
            
            # Compare instructions
            existing_instructions = existing_record.get("instructions", "")
            if existing_instructions != gemini_result.instructions:
                should_save = True
                change_reason.append("instructions changed")
            
            # Compare prompt_template
            existing_template = existing_record.get("prompt_template", "")
            if existing_template != gemini_result.prompt_template:
                should_save = True
                change_reason.append("prompt_template changed")
        
        if not should_save:
            logging.info(f"No key field changes detected for {gemini_result.pdf_name}, skipping save")
            return True
        
        logging.info(f"Saving Gemini response for {gemini_result.pdf_name}: {', '.join(change_reason)}")
        
        # Generate a unique key for this configuration
        config_key = f"{gemini_result.pdf_hash}_{hash(str(sorted(gemini_result.sections_used or [])) + gemini_result.instructions + gemini_result.prompt_template)}"
        
        # Prepare document with version tracking
        document = {
            "pdf_hash": gemini_result.pdf_hash,
            "config_key": config_key,  # Unique key for this configuration
            "s3_key": gemini_result.s3_key,
            "pdf_name": gemini_result.pdf_name,
            "gemini_response": json.loads(gemini_result.response) if isinstance(gemini_result.response, str) else gemini_result.response,
            "sections_used": gemini_result.sections_used,
            "token_count": gemini_result.token_count,
            "context_length": gemini_result.context_length,
            "instructions": gemini_result.instructions,
            "prompt_template": gemini_result.prompt_template,
            "processed_at": gemini_result.processed_at,
            "metadata": gemini_result.metadata,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "version": existing_record.get("version", 0) + 1 if existing_record else 1,
            "change_reason": change_reason
        }
        
        # Insert new document (not upsert, always create new version)
        result = collection.insert_one(document)
        
        if result.inserted_id:
            logging.info(f"Successfully saved Gemini response (v{document['version']}) for {gemini_result.pdf_name} to MongoDB")
            return True
        else:
            logging.warning(f"Failed to save Gemini response for {gemini_result.pdf_name}")
            return False
            
    except Exception as e:
        logging.error(f"Error saving Gemini response to MongoDB: {e}")
        return False
        
    finally:
        if close_client and client:
            client.close()


def batch_store_gemini_responses_in_mongodb(gemini_processing_results: List[GeminiProcessingResult]) -> int:
    """Store list of Gemini responses in MongoDB database with change detection"""
    if not gemini_processing_results:
        logging.warning("No Gemini responses to store")
        return 0
    
    successful_saves = 0
    client = None
    
    try:
        client = get_mongodb_client()
        
        logging.info(f"Starting batch storage of {len(gemini_processing_results)} Gemini responses to MongoDB...")
        
        # Process each response individually to check for changes
        for i, gemini_result in enumerate(gemini_processing_results, 1):
            try:
                logging.info(f"Processing response {i}/{len(gemini_processing_results)}: {gemini_result.pdf_name}")
                
                if save_gemini_response_to_mongodb(gemini_result, client):
                    successful_saves += 1
                    
            except Exception as e:
                logging.error(f"Error processing response {i} ({gemini_result.pdf_name}): {e}")
                continue
        
        logging.info(f"Batch storage completed. Successfully processed {successful_saves}/{len(gemini_processing_results)} Gemini responses")
        
    except Exception as e:
        logging.error(f"Error during batch storage: {e}")
    
    finally:
        if client:
            client.close()
    
    return successful_saves


def get_gemini_response_history(pdf_hash: str, client: MongoClient = None) -> List[Dict]:
    """Get all versions of Gemini responses for a given pdf_hash"""
    try:
        if client is None:
            client = get_mongodb_client()
            close_client = True
        else:
            close_client = False
        
        collection = get_mongodb_collection(client)
        
        # Get all records for this pdf_hash, sorted by creation time (newest first)
        records = list(collection.find(
            {"pdf_hash": pdf_hash}
        ).sort("created_at", -1))
        
        logging.info(f"Found {len(records)} versions for pdf_hash: {pdf_hash}")
        return records
        
    except Exception as e:
        logging.error(f"Error retrieving Gemini response history: {e}")
        return []
        
    finally:
        if close_client and client:
            client.close()


def get_latest_gemini_response(pdf_hash: str, client: MongoClient = None) -> Optional[Dict]:
    """Get the latest Gemini response for a given pdf_hash"""
    try:
        if client is None:
            client = get_mongodb_client()
            close_client = True
        else:
            close_client = False
        
        collection = get_mongodb_collection(client)
        
        # Get the most recent record
        record = collection.find_one(
            {"pdf_hash": pdf_hash},
            sort=[("created_at", -1)]
        )
        
        if record:
            logging.info(f"Found latest response (v{record.get('version', 1)}) for pdf_hash: {pdf_hash}")
        else:
            logging.info(f"No response found for pdf_hash: {pdf_hash}")
            
        return record
        
    except Exception as e:
        logging.error(f"Error retrieving latest Gemini response: {e}")
        return None
        
    finally:
        if close_client and client:
            client.close()


def cleanup_old_versions(pdf_hash: str, keep_versions: int = 5, client: MongoClient = None) -> int:
    """Keep only the N most recent versions for a pdf_hash"""
    try:
        if client is None:
            client = get_mongodb_client()
            close_client = True
        else:
            close_client = False
        
        collection = get_mongodb_collection(client)
        
        # Get all records for this pdf_hash, sorted by creation time (newest first)
        records = list(collection.find(
            {"pdf_hash": pdf_hash}
        ).sort("created_at", -1))
        
        if len(records) <= keep_versions:
            logging.info(f"Only {len(records)} versions exist for {pdf_hash}, no cleanup needed")
            return 0
        
        # Get IDs of records to delete (older than keep_versions)
        records_to_delete = records[keep_versions:]
        ids_to_delete = [record["_id"] for record in records_to_delete]
        
        # Delete old records
        result = collection.delete_many({"_id": {"$in": ids_to_delete}})
        
        logging.info(f"Cleaned up {result.deleted_count} old versions for {pdf_hash}")
        return result.deleted_count
        
    except Exception as e:
        logging.error(f"Error cleaning up old versions: {e}")
        return 0
        
    finally:
        if close_client and client:
            client.close()


def get_responses_by_config(sections_used: List[str], instructions: str, prompt_template: str, client: MongoClient = None) -> List[Dict]:
    """Get all responses that match a specific configuration"""
    try:
        if client is None:
            client = get_mongodb_client()
            close_client = True
        else:
            close_client = False
        
        collection = get_mongodb_collection(client)
        
        # Find responses with matching configuration
        query = {
            "sections_used": {"$all": sections_used, "$size": len(sections_used)},
            "instructions": instructions,
            "prompt_template": prompt_template
        }
        
        records = list(collection.find(query).sort("created_at", -1))
        
        logging.info(f"Found {len(records)} responses matching the configuration")
        return records
        
    except Exception as e:
        logging.error(f"Error retrieving responses by configuration: {e}")
        return []
        
    finally:
        if close_client and client:
            client.close()


def test_mongodb_connection():
    """Test MongoDB connection and setup"""
    try:
        logging.info("Testing MongoDB connection...")
        client = get_mongodb_client()
        
        # Test database access
        db = get_mongodb_database(client)
        collections = db.list_collection_names()
        logging.info(f"Available collections: {collections}")
        
        # Setup indexes
        setup_mongodb_indexes(client)
        
        logging.info("MongoDB setup completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error testing MongoDB connection: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()



# MAIN FUNCTION
def main():
    setup_logger(verbosity=1)

    parser = argparse.ArgumentParser(description="Process bioRxiv PDFs")
    parser.add_argument("--max-pdfs", type=int, default=5, help="Maximum number of PDFs to process")
    parser.add_argument("--include-supplements", action="store_true", help="Include supplements")
    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity level")
    parser.add_argument("--bucket", type=str, default="biorxiv-unpacked", help="MinIO bucket name")
    parser.add_argument("--skip-database", action="store_true", help="Skip PostgreSQL database processing")
    parser.add_argument("--skip-mongodb", action="store_true", help="Skip MongoDB database processing")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embeddings processing")
    args = parser.parse_args()

    minio_client = get_minio_client()
    qdrant_client = get_qdrant_client()

    try:
        # 1. Test database connection
        if not args.skip_database:
            print("Testing database connection...")
            test_database_connection()
        
        if not args.skip_mongodb:
            print("Testing MongoDB connection...")
            test_mongodb_connection()

        # 2. Fetch PDFs from MinIO
        pdfs = fetch_pdfs_from_minio(
            minio_client,
            bucket=args.bucket,
            max_pdfs=args.max_pdfs,
            include_supplements=args.include_supplements
        )

        if not pdfs:
            logging.error("No PDFs found to process")
            return

        # 3. Extract text from PDFs
        logging.info(f"Starting to process {len(pdfs)} PDFs...")
        extracted_texts = batch_process_pdfs(minio_client, pdfs)
        logging.info("PDF processing completed successfully!")

        # 4. Save extracted texts to database
        logging.info("Starting Gemini processing...")
        gemini_processing_results = []
        if not args.skip_database:        
            logging.info(f"Saving {len(extracted_texts)} extracted texts to database...")
            for extracted_text in extracted_texts:
                save_extracted_text(extracted_text)
                gemini_processing_result = build_gemini_prompt(
                    extracted_text, 
                    "all", 
                    gemini_processing_instructions)
                gemini_processing_results.append(gemini_processing_result)

            for result in gemini_processing_results:
                # 6. Generate the RAG document and store save it 
                # in GeminiProcessingResult object for each of the extracted texts
                result.response = get_response_from_gemini(prompt=result)

            logging.info("Gemini processing completed successfully!")

        breakpoint() # TODO: Remove this

        # 7. Store PDF processing results in PostgreSQL database
        if not args.skip_database:
            logging.info(f"Storing {len(gemini_processing_results)} Gemini responses in PostgreSQL database...")
            batch_store_gemini_responses_in_postgres(gemini_processing_results)
        

        # 7. Store the Gemini JSON responses in MongoDB
        if not args.skip_mongodb:
            logging.info(f"Storing {len(gemini_processing_results)} Gemini JSONresponses in MongoDB...")
            batch_store_gemini_responses_in_mongodb(gemini_processing_results)


        # 8. Store the Gemini response and embeddings in Qdrant
        if not args.skip_embeddings:
            logging.info(f"Storing {len(gemini_processing_results)} Gemini responses in Qdrant...")
            # embedding_results = embed_gemini_responses(gemini_processing_results)
            # batch_store_gemini_responses_in_qdrant(embedding_results)
    
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()