#! /usr/bin/env python3

import os
import sys
import argparse
import logging
import json
import time
import tempfile
import re
import psycopg2
import minio
import google.generativeai as genai
import pymongo

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Tuple, Set
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

# =========== DATA CLASSES ===========

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
    pdf_uid: str
    s3_key: str
    pdf_name: str
    full_text: str
    sections: Dict[str, str]
    metadata: Dict[str, any]
    extracted_at: datetime
    text_length: int

@dataclass
class GeminiProcessingResult:
    """Data class to store Gemini processing result with prompt information"""
    prompt: str
    token_count: int
    sections_used: List[str]
    context_length: int
    pdf_uid: str
    s3_key: str
    pdf_name: str
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
        context_length: int, pdf_uid: str, s3_key: str, pdf_name: str, 
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
            pdf_uid=pdf_uid,
            s3_key=s3_key,
            pdf_name=pdf_name,
            instructions=instructions,
            prompt_template=prompt_template,
            response=response,
            processed_at=processed_at,
            metadata=metadata
        )

@dataclass
class ProcessingStats:
    """Data class to track processing statistics"""
    total_pdfs: int = 0
    skipped_text_extraction: int = 0
    skipped_gemini_processing: int = 0
    processed_text_extraction: int = 0
    processed_gemini: int = 0
    failed_processing: int = 0
    start_time: datetime = None
    end_time: datetime = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()

    @property
    def duration(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def finish(self):
        self.end_time = datetime.now()

# =========== UTILITY FUNCTIONS ===========

def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

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
        logging.error(f"Error getting database connection: {e}")
        return None

def get_mongodb_client() -> MongoClient:
    """Get a MongoDB client connection"""
    try:
        connection_string = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_HOST}:{MONGODB_PORT}/{MONGODB_DATABASE}?authSource=admin"
        
        client = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000
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

# =========== EFFICIENCY FUNCTIONS ===========

def check_text_extraction_needed(pdf_uid: str, current_library: str) -> Tuple[bool, Optional[Dict]]:
    """
    Check if text extraction is needed by comparing PDF UID and library used.
    Returns (needs_extraction, existing_record)
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check if PDF was already processed with the same library
            cur.execute("""
                SELECT pdf_uid, metadata, sections, text_length, extracted_at
                FROM extracted_texts 
                WHERE pdf_uid = %s
            """, (pdf_uid,))
            
            result = cur.fetchone()
            
            if result is None:
                logging.info(f"PDF {pdf_uid} not found in database - needs extraction")
                return True, None
            
            # Parse existing metadata
            existing_metadata = json.loads(result[1]) if result[1] else {}
            existing_library = existing_metadata.get('pdf_library_used', '')
            
            if existing_library != current_library:
                logging.info(f"PDF {pdf_uid} processed with different library ({existing_library} vs {current_library}) - needs re-extraction")
                return True, {
                    'pdf_uid': result[0],
                    'metadata': existing_metadata,
                    'sections': json.loads(result[2]) if result[2] else {},
                    'text_length': result[3],
                    'extracted_at': result[4]
                }
            
            # Check if sections are the same
            existing_sections = set(json.loads(result[2]).keys()) if result[2] else set()
            
            logging.info(f"PDF {pdf_uid} already processed with same library and sections - skipping extraction")
            return False, {
                'pdf_uid': result[0],
                'metadata': existing_metadata,
                'sections': json.loads(result[2]) if result[2] else {},
                'text_length': result[3],
                'extracted_at': result[4]
            }
            
    except Exception as e:
        logging.error(f"Error checking text extraction status: {e}")
        return True, None
    finally:
        conn.close()

def check_gemini_processing_needed(pdf_uid: str, sections_used: List[str], 
                                 instructions: str, prompt_template: str) -> Tuple[bool, Optional[Dict]]:
    """
    Check if Gemini processing is needed by comparing configuration.
    Returns (needs_processing, existing_response)
    """
    client = get_mongodb_client()
    try:
        collection = client[MONGODB_DATABASE][MONGODB_COLLECTION]
        
        # Generate config key for comparison
        config_key = f"{pdf_uid}_{hash(str(sorted(sections_used)) + instructions + prompt_template)}"
        
        # Find existing response with same configuration
        existing_record = collection.find_one({
            "pdf_uid": pdf_uid,
            "config_key": config_key
        })
        
        if existing_record is None:
            logging.info(f"No existing Gemini response found for PDF {pdf_uid} with this configuration - needs processing")
            return True, None
        
        logging.info(f"PDF {pdf_uid} already processed with same configuration - skipping Gemini processing")
        return False, existing_record
        
    except Exception as e:
        logging.error(f"Error checking Gemini processing status: {e}")
        return True, None
    finally:
        client.close()

def get_existing_extracted_text(pdf_uid: str) -> Optional[ExtractedText]:
    """Retrieve existing extracted text from database"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pdf_uid, s3_key, pdf_name, full_text, sections, metadata, text_length, extracted_at
                FROM extracted_texts 
                WHERE pdf_uid = %s
            """, (pdf_uid,))
            
            result = cur.fetchone()
            if result is None:
                return None
            
            return ExtractedText(
                pdf_uid=result[0],
                s3_key=result[1],
                pdf_name=result[2],
                full_text=result[3],
                sections=json.loads(result[4]) if result[4] else {},
                metadata=json.loads(result[5]) if result[5] else {},
                text_length=result[6],
                extracted_at=result[7]
            )
            
    except Exception as e:
        logging.error(f"Error retrieving existing extracted text: {e}")
        return None
    finally:
        conn.close()

def get_existing_gemini_response(pdf_uid: str, sections_used: List[str], 
                                instructions: str, prompt_template: str) -> Optional[GeminiProcessingResult]:
    """Retrieve existing Gemini response from MongoDB"""
    client = get_mongodb_client()
    try:
        collection = client[MONGODB_DATABASE][MONGODB_COLLECTION]
        
        # Generate config key for lookup
        config_key = f"{pdf_uid}_{hash(str(sorted(sections_used)) + instructions + prompt_template)}"
        
        existing_record = collection.find_one({
            "pdf_uid": pdf_uid,
            "config_key": config_key
        })
        
        if existing_record is None:
            return None
        
        return GeminiProcessingResult.create_from_prompt(
            prompt=existing_record.get('prompt', ''),
            token_count=existing_record.get('token_count', 0),
            sections_used=existing_record.get('sections_used', []),
            context_length=existing_record.get('context_length', 0),
            pdf_uid=existing_record.get('pdf_uid', ''),
            s3_key=existing_record.get('s3_key', ''),
            pdf_name=existing_record.get('pdf_name', ''),
            instructions=existing_record.get('instructions', ''),
            prompt_template=existing_record.get('prompt_template', ''),
            response=json.dumps(existing_record.get('gemini_response', {})),
            processed_at=existing_record.get('processed_at', datetime.now()),
            metadata=existing_record.get('metadata', {})
        )
        
    except Exception as e:
        logging.error(f"Error retrieving existing Gemini response: {e}")
        return None
    finally:
        client.close()

# =========== PDF PROCESSING FUNCTIONS ===========

def generate_pdf_uid(s3_key: str, pdf_name: str, is_supplement: bool = False) -> str:
    """Generate a unique PDF identifier"""
    supplement_suffix = "_supplement" if is_supplement else ""
    return f"{s3_key}_{pdf_name}{supplement_suffix}"

def download_pdf_from_minio(minio_client: minio.Minio, pdf_info: PDFInfo, temp_dir: str, bucket_name: str = "biorxiv-unpacked") -> Optional[str]:
    """Download a PDF from MinIO to a temporary directory"""
    try:
        safe_filename = pdf_info.pdf_name.replace("/", "_").replace("\\", "_")
        local_path = os.path.join(temp_dir, safe_filename)
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
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)
    
    reference_patterns = [
        r'\[\d+\]',
        r'\(\d{4}[a-z]?\)',
        r'\([A-Za-z]+\s+et\s+al\.?\s*,\s*\d{4}\)',
    ]
    
    for pattern in reference_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text)
    
    return cleaned_text.strip()

def detect_sections(text: str) -> Dict[str, str]:
    """Detect and extract main sections from the text"""
    sections = {}
    
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
    
    lines = text.split('\n')
    current_section = 'header'
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        section_found = None
        for section_name, pattern in section_patterns.items():
            if re.match(pattern, f'\n{line}\n', re.IGNORECASE):
                section_found = section_name
                break
        
        if section_found:
            if current_section != 'header' and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = section_found
            current_content = []
        else:
            current_content.append(line)
    
    if current_section != 'header' and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    if not sections:
        sections['full_text'] = text
    
    return sections

def process_pdf_text_extraction_optimized(minio_client: minio.Minio, pdf_info: PDFInfo, stats: ProcessingStats) -> Optional[ExtractedText]:
    """Optimized PDF text extraction with database checks"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_pdf_path = download_pdf_from_minio(minio_client, pdf_info, temp_dir)
            if not local_pdf_path:
                stats.failed_processing += 1
                return None

            pdf_uid = generate_pdf_uid(pdf_info.s3_key, pdf_info.pdf_name, pdf_info.is_supplement)
            if not pdf_uid:
                stats.failed_processing += 1
                return None

            needs_extraction, existing_record = check_text_extraction_needed(pdf_uid, PDF_LIBRARY)

            if not needs_extraction and existing_record:
                logging.info(f"Using existing extracted text for {pdf_info.pdf_name}")
                stats.skipped_text_extraction += 1

                return ExtractedText(
                    pdf_uid=existing_record['pdf_uid'],
                    s3_key=pdf_info.s3_key,
                    pdf_name=pdf_info.pdf_name,
                    full_text=existing_record.get('full_text', ''),
                    sections=existing_record.get('sections', {}),
                    metadata=existing_record.get('metadata', {}),
                    extracted_at=existing_record.get('extracted_at', datetime.now()),
                    text_length=existing_record.get('text_length', 0)
                )

            logging.info(f"Extracting text from {pdf_info.pdf_name}...")
            full_text = extract_text_from_pdf(local_pdf_path)

            if not full_text.strip():
                logging.warning(f"No text extracted from {pdf_info.pdf_name}")
                stats.failed_processing += 1
                return None

            cleaned_text = remove_bibliography_references(full_text)
            sections = detect_sections(cleaned_text)

            metadata = {
                'original_text_length': len(full_text),
                'cleaned_text_length': len(cleaned_text),
                'sections_found': list(sections.keys()),
                'pdf_library_used': PDF_LIBRARY,
                'processing_timestamp': datetime.now().isoformat()
            }

            extracted_text = ExtractedText(
                pdf_uid=pdf_uid,
                s3_key=pdf_info.s3_key,
                pdf_name=pdf_info.pdf_name,
                full_text=cleaned_text,
                sections=sections,
                metadata=metadata,
                extracted_at=datetime.now(),
                text_length=len(cleaned_text)
            )

            save_extracted_text(extracted_text)
            stats.processed_text_extraction += 1
            logging.info(f"Successfully processed {pdf_info.pdf_name}")
            return extracted_text

    except Exception as e:
        logging.error(f"Error processing {pdf_info.pdf_name}: {e}")
        stats.failed_processing += 1
        return None

def save_extracted_text(extracted_text: ExtractedText) -> bool:
    """Save extracted text to database"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.pdf_processing_state (pdf_uid, s3_key, pdf_name, status, processed_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (pdf_uid)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    processed_at = EXCLUDED.processed_at,
                    updated_at = CURRENT_TIMESTAMP,
                    metadata = EXCLUDED.metadata
            """, (
                extracted_text.pdf_uid, 
                extracted_text.s3_key, 
                extracted_text.pdf_name, 
                'completed', 
                extracted_text.extracted_at,
                json.dumps(extracted_text.metadata)
            ))
            cur.execute("""
                INSERT INTO extracted_texts (pdf_uid, s3_key, pdf_name, full_text, sections, metadata, text_length)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pdf_uid)
                DO UPDATE SET
                    full_text = EXCLUDED.full_text,
                    sections = EXCLUDED.sections,
                    metadata = EXCLUDED.metadata,
                    text_length = EXCLUDED.text_length
            """, (
                extracted_text.pdf_uid, 
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

# =========== GEMINI PROCESSING FUNCTIONS ===========

def count_tokens_gemini(text: str) -> int:
    """Count tokens using Gemini's official tokenizer"""
    try:        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.count_tokens(text)
        return response.total_tokens
    except Exception as e:
        logging.warning(f"Could not count tokens with Gemini API: {e}")
        return int(len(text.split()) * 1.3)

def build_gemini_prompt(extracted_text: ExtractedText, sections: List[str], instructions: str) -> GeminiProcessingResult:
    """Build a Gemini prompt for the specified sections and instructions"""
    if not extracted_text:
        logging.warning("No extracted text provided")
        return None
    
    all_context_parts = []
    available_sections = list(extracted_text.sections.keys())
    
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
    
    all_context_parts.append(f"# PAPER: {extracted_text.pdf_name}\n{paper_context}")
    context = "\n\n".join(all_context_parts)
    
    prompt = f"CONTEXT:\n{context}\n\nINSTRUCTIONS:\n{instructions}"
    token_count = count_tokens_gemini(prompt)
    logging.info(f"Number of tokens in prompt: {token_count}")

    return GeminiProcessingResult.create_from_prompt(
        prompt=prompt.strip(),
        token_count=token_count,
        sections_used=valid_sections if 'all' not in sections else ["all"],
        context_length=len(context),
        pdf_uid=extracted_text.pdf_uid,
        s3_key=extracted_text.s3_key,
        pdf_name=extracted_text.pdf_name,
        instructions=instructions,
        prompt_template=gemini_processing_instructions,
        response="",
        metadata={}
    )

def get_response_from_gemini(prompt: GeminiProcessingResult, model_name: str = 'gemini-2.0-flash-lite') -> str:
    """Get a response from Gemini"""
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel(model_name)

        logging.info(f"Sending request to {model_name} with prompt: {prompt.prompt_preview}...")

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

def process_gemini_optimized(
    extracted_text: ExtractedText,
    sections: List[str],
    instructions: str,
    prompt_template: str,
    stats: ProcessingStats,
) -> Optional[GeminiProcessingResult]:
    """Optimized Gemini processing with database checks"""
    try:
        needs_processing, existing_response = check_gemini_processing_needed(
            extracted_text.pdf_uid, sections, instructions, prompt_template
        )

        if not needs_processing and existing_response:
            logging.info(f"Using existing Gemini response for {extracted_text.pdf_name}")
            stats.skipped_gemini_processing += 1

            return GeminiProcessingResult.create_from_prompt(
                prompt=existing_response.get('prompt', ''),
                token_count=existing_response.get('token_count', 0),
                sections_used=existing_response.get('sections_used', []),
                context_length=existing_response.get('context_length', 0),
                pdf_uid=existing_response.get('pdf_uid', ''),
                s3_key=existing_response.get('s3_key', ''),
                pdf_name=existing_response.get('pdf_name', ''),
                instructions=existing_response.get('instructions', ''),
                prompt_template=existing_response.get('prompt_template', ''),
                response=json.dumps(existing_response.get('gemini_response', {})),
                processed_at=existing_response.get('processed_at', datetime.now()),
                metadata=existing_response.get('metadata', {})
            )
            
        gemini_result = build_gemini_prompt(extracted_text, sections, instructions)
        if not gemini_result:
            stats.failed_processing += 1
            return None

        gemini_result.response = get_response_from_gemini(gemini_result)

        save_gemini_response_to_mongodb(gemini_result)
        stats.processed_gemini += 1

        logging.info(f"Successfully processed {extracted_text.pdf_name}")
        return gemini_result

    except Exception as e:
        logging.error(f"Error processing {extracted_text.pdf_name}: {e}")
        stats.failed_processing += 1
        return None

def save_gemini_response_to_mongodb(gemini_result: GeminiProcessingResult) -> bool:
    """Save Gemini response to MongoDB"""
    try:
        client = get_mongodb_client()
        collection = client[MONGODB_DATABASE][MONGODB_COLLECTION]

        config_key = f"{gemini_result.pdf_uid}_{hash(str(sorted(gemini_result.sections_used)) + gemini_result.instructions + gemini_result.prompt_template)}"

        document = {
            "pdf_uid": gemini_result.pdf_uid,
            "config_key": config_key,
            "s3_key": gemini_result.s3_key,
            "pdf_name": gemini_result.pdf_name,
            "prompt": gemini_result.prompt,
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
            "version": 1
        }

        result = collection.insert_one(document)

        if result.inserted_id:
            logging.info(f"Successfully saved Gemini response for {gemini_result.pdf_name} to MongoDB")
            return True
        else:
            logging.warning(f"Failed to save Gemini response for {gemini_result.pdf_name} to MongoDB")
            return False

    except Exception as e:
        logging.error(f"Error saving Gemini response to MongoDB: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()

# =========== PDF DISCOVERY FUNCTIONS ===========

def fetch_pdfs_from_minio(minio_client: minio.Minio, bucket: str = "biorxiv-papers", 
                         max_pdfs: int = 10, include_supplements: bool = True) -> List[PDFInfo]:
    """Fetch PDFs from MinIO bucket"""
    if bucket is None:
        bucket = MINIO_BUCKET

    print(f"Fetching {max_pdfs} PDFs from MinIO bucket: {bucket}")
    pdfs = []

    try:
        if not minio_client.bucket_exists(bucket):
            raise Exception(f"Bucket {bucket} does not exist")

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

    return pdfs

# =========== MAIN PROCESSING FUNCTIONS ===========

def process_pdfs_optimized(minio_client: minio.Minio, pdf_list: List[PDFInfo], 
                          sections: List[str], instructions: str, 
                          prompt_template: str) -> Tuple[List[GeminiProcessingResult], ProcessingStats]:
    """Optimized PDF processing pipeline with efficiency checks"""
    stats = ProcessingStats(total_pdfs=len(pdf_list))
    gemini_results = []
    
    logging.info(f"Starting optimized processing of {len(pdf_list)} PDFs...")
    
    for i, pdf_info in enumerate(pdf_list, 1):
        logging.info(f"Processing PDF {i}/{len(pdf_list)}: {pdf_info.pdf_name}")
        
        # Phase 1: Text extraction (with efficiency checks)
        extracted_text = process_pdf_text_extraction_optimized(minio_client, pdf_info, stats)
        if not extracted_text:
            continue
        
        # Phase 2: Gemini processing (with efficiency checks)
        gemini_result = process_gemini_optimized(
            extracted_text, sections, instructions, prompt_template, stats
        )
        if gemini_result:
            gemini_results.append(gemini_result)
    
    stats.finish()
    return gemini_results, stats

def print_processing_stats(stats: ProcessingStats):
    """Print processing statistics"""
    print("\n" + "="*60)
    print("PROCESSING STATISTICS")
    print("="*60)
    print(f"Total PDFs: {stats.total_pdfs}")
    print(f"Text Extraction:")
    print(f"  - Processed: {stats.processed_text_extraction}")
    print(f"  - Skipped (already exists): {stats.skipped_text_extraction}")
    print(f"Gemini Processing:")
    print(f"  - Processed: {stats.processed_gemini}")
    print(f"  - Skipped (already exists): {stats.skipped_gemini_processing}")
    print(f"Failed: {stats.failed_processing}")
    print(f"Total Duration: {stats.duration:.2f} seconds")
    print(f"Efficiency: {((stats.skipped_text_extraction + stats.skipped_gemini_processing) / (stats.total_pdfs * 2) * 100):.1f}% operations skipped")
    print("="*60)

# =========== MAIN FUNCTION ===========

def main():
    setup_logger(verbosity=1)

    parser = argparse.ArgumentParser(description="Optimized PDF processing pipeline")
    parser.add_argument("--max-pdfs", type=int, default=5, help="Maximum number of PDFs to process")
    parser.add_argument("--include-supplements", action="store_true", help="Include supplements")
    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity level")
    parser.add_argument("--bucket", type=str, default="biorxiv-unpacked", help="MinIO bucket name")
    parser.add_argument("--sections", type=str, nargs="+", default=["all"], help="Sections to process")
    parser.add_argument("--instructions", type=str, default=gemini_processing_instructions, help="Processing instructions")
    args = parser.parse_args()

    try:
        # Initialize clients
        minio_client = get_minio_client()
        
        # Fetch PDFs
        pdfs = fetch_pdfs_from_minio(
            minio_client,
            bucket=args.bucket,
            max_pdfs=args.max_pdfs,
            include_supplements=args.include_supplements
        )

        if not pdfs:
            logging.error("No PDFs found to process")
            return

        # Process PDFs with optimization
        gemini_results, stats = process_pdfs_optimized(
            minio_client, pdfs, args.sections, args.instructions, gemini_processing_instructions
        )

        # Print statistics
        print_processing_stats(stats)
        
        logging.info(f"Processing completed successfully! Generated {len(gemini_results)} Gemini responses.")

    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
