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
import google.generativeai as genai

from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import minio
from dotenv import load_dotenv

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


def download_pdf_from_minio(pdf_info: PDFInfo, temp_dir: str) -> Optional[str]:
    """Download a PDF from MinIO to a temporary directory"""
    try:
        # Create safe filename
        safe_filename = pdf_info.pdf_name.replace("/", "_").replace("\\", "_")
        local_path = os.path.join(temp_dir, safe_filename)

        #Download PDF from MinIO
        minio_client.fget_object(MINIO_BUCKET, pdf_info.pdf_path, local_path)
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


def process_pdf_text_extraction(pdf_info: PDFInfo) -> Optional[ExtractedText]:
    """Main function to process PDF text extraction (Phase 1)"""
    try:
        # Create temporary directory for PDF processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download PDF from MinIO
            local_pdf_path = download_pdf_from_minio(pdf_info, temp_dir)
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


def batch_process_pdfs(pdf_list: List[PDFInfo]) -> List[ExtractedText]:
    """Process multiple PDFs in batch"""
    extracted_texts = []
    
    logging.info(f"Starting batch processing of {len(pdf_list)} PDFs...")
    
    for i, pdf_info in enumerate(pdf_list, 1):
        logging.info(f"Processing PDF {i}/{len(pdf_list)}: {pdf_info.pdf_name}")
        
        extracted_text = process_pdf_text_extraction(pdf_info)
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
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.count_tokens(text)
        return response.total_tokens
    except Exception as e:
        logging.warning(f"Could not count tokens with Gemini API: {e}")
        # Fallback to word-based approximation
        return int(len(text.split()) * 1.3)



def build_gemini_prompt(extracted_texts: List[ExtractedText], sections: List[str], instructions: str) -> str:
    """Build a Gemini prompt for the specified sections and instructions"""
    # Handle both single ExtractedText and list of ExtractedText objects
    if isinstance(extracted_texts, list):
        if not extracted_texts:
            logging.warning("No extracted texts provided")
            return instructions
        
        # Process all extracted texts
        all_context_parts = []
        for i, extracted_text in enumerate(extracted_texts):
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
            all_context_parts.append(f"# PAPER {i+1}: {extracted_text.pdf_name}\n{paper_context}")
        
        context = "\n\n" + "="*80 + "\n\n".join(all_context_parts)
    else:
        logging.warning("No extracted texts provided")
        return instructions
    
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

def fetch_pdfs_from_minio(bucket: str = "biorxiv-papers", max_pdfs: int = 10, include_supplements: bool = True) -> List[PDFInfo]:
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





def main():
    setup_logger(verbosity=1)

    parser = argparse.ArgumentParser(description="Process bioRxiv PDFs")
    parser.add_argument("--max-pdfs", type=int, default=5, help="Maximum number of PDFs to process")
    parser.add_argument("--include-supplements", action="store_true", help="Include supplements")
    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity level")
    parser.add_argument("--skip-database", action="store_true", help="Skip PostgreSQL database processing")
    args = parser.parse_args()

    try:
        # Test database connection
        if not args.skip_database:
            print("Testing database connection...")
            test_database_connection()

        # Fetch PDFs from MinIO
        pdfs = fetch_pdfs_from_minio(
            max_pdfs=args.max_pdfs,
            include_supplements=args.include_supplements
        )

        if not pdfs:
            logging.error("No PDFs found to process")
            return

        print("PDFs found:")
        print_pdfs(pdfs)

        # Process PDFs
        logging.info(f"Starting to process {len(pdfs)} PDFs...")
        extracted_texts = batch_process_pdfs(pdfs)

        # Save extracted texts to database
        if not args.skip_database:
            logging.info(f"Saving {len(extracted_texts)} extracted texts to database...")
            for extracted_text in extracted_texts:
                save_extracted_text(extracted_text)

        logging.info("PDF processing completed successfully!")
        logging.info("Starting Gemini processing...")
        
        prompt = build_gemini_prompt(extracted_texts, "all", gemini_processing_instructions)
        
        response = get_response_from_gemini(prompt=prompt)

        breakpoint()

        logging.info(f"Gemini prompt: {prompt.prompt_preview}...")
        logging.info(f"Number of tokens in prompt: {prompt.token_count}. Is too long: {prompt.is_too_long}.")
        
        logging.info("Gemini processing completed successfully!")
    
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()