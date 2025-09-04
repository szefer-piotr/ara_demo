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

load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'biorxiv_processing'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9002")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "biorxiv-papers")

# Processing status enum
class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class PDFInfo:
    filename: str
    s3_key: str
    minio_path: str
    file_size: int
    file_hash: str
    is_supplement: bool

@dataclass
class TextChunk:
    chunk_id: str
    pdf_hash: str
    chunk_index: int
    content: str
    token_count: int
    chunk_hash: str
    metadata: Dict

@dataclass
class ProcessingRecord:
    pdf_hash: str
    filename: str
    s3_key: str
    minio_path: str
    status: ProcessingStatus
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str]
    processing_time: Optional[float]
    chunk_count: Optional[int]
    metadata: Dict

class StateManager:
    """PostgreSQL-based state management for PDF processing pipeline."""
    
    def __init__(self, db_config: Dict = None):
        self.db_config = db_config or DB_CONFIG
        self.connection = None
        self._ensure_database_exists()
        self._create_tables()
    
    def _ensure_database_exists(self):
        """Ensure the database exists, create if it doesn't."""
        try:
            # Connect to default postgres database to create our database
            temp_config = self.db_config.copy()
            temp_config['database'] = 'postgres'
            
            conn = psycopg2.connect(**temp_config)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.db_config['database'],)
            )
            
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {self.db_config['database']}")
                print(f"✅ Created database: {self.db_config['database']}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error ensuring database exists: {e}")
            raise
    
    def _create_tables(self):
        """Create all necessary tables for state management."""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            cursor = self.connection.cursor()
            
            # Create processing_status enum type
            cursor.execute("""
                DO $$ BEGIN
                    CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'skipped');
                EXCEPTION
                    WHEN duplicate_object THEN null;
                END $$;
            """)
            
            # PDF processing state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pdf_processing_state (
                    pdf_hash VARCHAR(64) PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    s3_key VARCHAR(500) NOT NULL,
                    minio_path VARCHAR(500) NOT NULL,
                    file_size BIGINT NOT NULL,
                    file_hash VARCHAR(64) NOT NULL,
                    status processing_status DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT,
                    processing_time FLOAT,
                    chunk_count INTEGER,
                    metadata JSONB DEFAULT '{}',
                    UNIQUE(filename, s3_key)
                );
            """)
            
            # Text chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_chunks (
                    chunk_id VARCHAR(64) PRIMARY KEY,
                    pdf_hash VARCHAR(64) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    chunk_hash VARCHAR(64) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}',
                    FOREIGN KEY (pdf_hash) REFERENCES pdf_processing_state(pdf_hash) ON DELETE CASCADE
                );
            """)
            
            # Processing logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    log_id SERIAL PRIMARY KEY,
                    pdf_hash VARCHAR(64),
                    log_level VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}',
                    FOREIGN KEY (pdf_hash) REFERENCES pdf_processing_state(pdf_hash) ON DELETE CASCADE
                );
            """)
            
            # Processing metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_metrics (
                    metric_id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    pdf_hash VARCHAR(64),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}',
                    FOREIGN KEY (pdf_hash) REFERENCES pdf_processing_state(pdf_hash) ON DELETE CASCADE
                );
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pdf_processing_status 
                ON pdf_processing_state(status);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pdf_processing_updated_at 
                ON pdf_processing_state(updated_at);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_chunks_pdf_hash 
                ON text_chunks(pdf_hash);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_processing_logs_pdf_hash 
                ON processing_logs(pdf_hash);
            """)
            
            self.connection.commit()
            cursor.close()
            print("✅ Database tables created successfully")
            
        except Exception as e:
            if self.connection:
                self.connection.rollback()
            print(f"❌ Error creating tables: {e}")
            raise
    
    def get_connection(self):
        """Get database connection, create if needed."""
        if not self.connection or self.connection.closed:
            self.connection = psycopg2.connect(**self.db_config)
        return self.connection
    
    def calculate_file_hash(self, file_data: bytes) -> str:
        """Calculate SHA256 hash of file data."""
        return hashlib.sha256(file_data).hexdigest()
    
    def calculate_chunk_hash(self, content: str) -> str:
        """Calculate SHA256 hash of text content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def add_pdf_to_queue(self, pdf_info: PDFInfo) -> bool:
        """Add PDF to processing queue if not already processed."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check if PDF already exists
            cursor.execute("""
                SELECT status FROM pdf_processing_state 
                WHERE pdf_hash = %s
            """, (pdf_info.file_hash,))
            
            result = cursor.fetchone()
            
            if result:
                status = result[0]
                if status in ['completed', 'processing']:
                    print(f"⏭️  PDF already processed/processing: {pdf_info.filename}")
                    return False
                elif status == 'failed':
                    print(f"🔄 Retrying failed PDF: {pdf_info.filename}")
                    # Update status to pending for retry
                    cursor.execute("""
                        UPDATE pdf_processing_state 
                        SET status = 'pending', updated_at = CURRENT_TIMESTAMP, error_message = NULL
                        WHERE pdf_hash = %s
                    """, (pdf_info.file_hash,))
            else:
                # Insert new PDF
                cursor.execute("""
                    INSERT INTO pdf_processing_state 
                    (pdf_hash, filename, s3_key, minio_path, file_size, file_hash, status)
                    VALUES (%s, %s, %s, %s, %s, %s, 'pending')
                """, (
                    pdf_info.file_hash, pdf_info.filename, pdf_info.s3_key,
                    pdf_info.minio_path, pdf_info.file_size, pdf_info.file_hash
                ))
                print(f"➕ Added PDF to queue: {pdf_info.filename}")
            
            conn.commit()
            cursor.close()
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"❌ Error adding PDF to queue: {e}")
            return False
    
    def get_pending_pdfs(self, limit: int = None) -> List[ProcessingRecord]:
        """Get list of PDFs pending processing."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT * FROM pdf_processing_state 
                WHERE status = 'pending' 
                ORDER BY created_at ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            records = []
            for row in results:
                record = ProcessingRecord(
                    pdf_hash=row['pdf_hash'],
                    filename=row['filename'],
                    s3_key=row['s3_key'],
                    minio_path=row['minio_path'],
                    status=ProcessingStatus(row['status']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    error_message=row['error_message'],
                    processing_time=row['processing_time'],
                    chunk_count=row['chunk_count'],
                    metadata=row['metadata'] or {}
                )
                records.append(record)
            
            cursor.close()
            return records
            
        except Exception as e:
            print(f"❌ Error getting pending PDFs: {e}")
            return []
    
    def update_pdf_status(self, pdf_hash: str, status: ProcessingStatus, 
                         error_message: str = None, processing_time: float = None,
                         chunk_count: int = None, metadata: Dict = None):
        """Update PDF processing status."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Build dynamic update query
            update_fields = ["status = %s", "updated_at = CURRENT_TIMESTAMP"]
            values = [status.value]
            
            if error_message is not None:
                update_fields.append("error_message = %s")
                values.append(error_message)
            
            if processing_time is not None:
                update_fields.append("processing_time = %s")
                values.append(processing_time)
            
            if chunk_count is not None:
                update_fields.append("chunk_count = %s")
                values.append(chunk_count)
            
            if metadata is not None:
                update_fields.append("metadata = %s")
                values.append(json.dumps(metadata))
            
            values.append(pdf_hash)
            
            query = f"""
                UPDATE pdf_processing_state 
                SET {', '.join(update_fields)}
                WHERE pdf_hash = %s
            """
            
            cursor.execute(query, values)
            conn.commit()
            cursor.close()
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"❌ Error updating PDF status: {e}")
            raise
    
    def save_text_chunks(self, pdf_hash: str, chunks: List[TextChunk]):
        """Save text chunks for a PDF."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Delete existing chunks for this PDF
            cursor.execute("DELETE FROM text_chunks WHERE pdf_hash = %s", (pdf_hash,))
            
            # Insert new chunks
            for chunk in chunks:
                cursor.execute("""
                    INSERT INTO text_chunks 
                    (chunk_id, pdf_hash, chunk_index, content, token_count, chunk_hash, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    chunk.chunk_id, chunk.pdf_hash, chunk.chunk_index,
                    chunk.content, chunk.token_count, chunk.chunk_hash,
                    json.dumps(chunk.metadata)
                ))
            
            conn.commit()
            cursor.close()
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"❌ Error saving text chunks: {e}")
            raise
    
    def get_text_chunks(self, pdf_hash: str) -> List[TextChunk]:
        """Get text chunks for a PDF."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM text_chunks 
                WHERE pdf_hash = %s 
                ORDER BY chunk_index ASC
            """, (pdf_hash,))
            
            results = cursor.fetchall()
            chunks = []
            
            for row in results:
                chunk = TextChunk(
                    chunk_id=row['chunk_id'],
                    pdf_hash=row['pdf_hash'],
                    chunk_index=row['chunk_index'],
                    content=row['content'],
                    token_count=row['token_count'],
                    chunk_hash=row['chunk_hash'],
                    metadata=row['metadata'] or {}
                )
                chunks.append(chunk)
            
            cursor.close()
            return chunks
            
        except Exception as e:
            print(f"❌ Error getting text chunks: {e}")
            return []
    
    def log_processing_event(self, pdf_hash: str, level: str, message: str, metadata: Dict = None):
        """Log processing event."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO processing_logs (pdf_hash, log_level, message, metadata)
                VALUES (%s, %s, %s, %s)
            """, (pdf_hash, level, message, json.dumps(metadata or {})))
            
            conn.commit()
            cursor.close()
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"❌ Error logging event: {e}")
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get status counts
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM pdf_processing_state 
                GROUP BY status
            """)
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Get total chunks
            cursor.execute("SELECT COUNT(*) as total_chunks FROM text_chunks")
            total_chunks = cursor.fetchone()['total_chunks']
            
            # Get average processing time
            cursor.execute("""
                SELECT AVG(processing_time) as avg_time 
                FROM pdf_processing_state 
                WHERE processing_time IS NOT NULL
            """)
            avg_time = cursor.fetchone()['avg_time']
            
            cursor.close()
            
            return {
                'status_counts': status_counts,
                'total_chunks': total_chunks,
                'average_processing_time': avg_time
            }
            
        except Exception as e:
            print(f"❌ Error getting processing stats: {e}")
            return {}
    
    def cleanup_old_logs(self, days: int = 30):
        """Clean up old processing logs."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM processing_logs 
                WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
            """, (days,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            
            print(f"🧹 Cleaned up {deleted_count} old log entries")
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"❌ Error cleaning up logs: {e}")
    
    def close(self):
        """Close database connection."""
        if self.connection and not self.connection.closed:
            self.connection.close()

# Example usage and testing
def test_state_manager():
    """Test the state manager functionality."""
    print("🧪 Testing State Manager...")
    
    # Initialize state manager
    state_manager = StateManager()
    
    # Create test PDF info
    test_pdf = PDFInfo(
        filename="test_paper.pdf",
        s3_key="test-uuid/content/test_paper.pdf",
        minio_path="test-uuid/content/test_paper.pdf",
        file_size=1024000,
        file_hash="test_hash_123",
        is_supplement=False
    )
    
    # Test adding PDF to queue
    state_manager.add_pdf_to_queue(test_pdf)
    
    # Test getting pending PDFs
    pending = state_manager.get_pending_pdfs()
    print(f"�� Pending PDFs: {len(pending)}")
    
    # Test updating status
    state_manager.update_pdf_status(
        test_pdf.file_hash, 
        ProcessingStatus.PROCESSING,
        metadata={"test": "data"}
    )
    
    # Test creating text chunks
    test_chunks = [
        TextChunk(
            chunk_id="chunk_1",
            pdf_hash=test_pdf.file_hash,
            chunk_index=0,
            content="This is test content for chunk 1.",
            token_count=10,
            chunk_hash="chunk_hash_1",
            metadata={"section": "abstract"}
        ),
        TextChunk(
            chunk_id="chunk_2",
            pdf_hash=test_pdf.file_hash,
            chunk_index=1,
            content="This is test content for chunk 2.",
            token_count=10,
            chunk_hash="chunk_hash_2",
            metadata={"section": "introduction"}
        )
    ]
    
    state_manager.save_text_chunks(test_pdf.file_hash, test_chunks)
    
    # Test retrieving chunks
    retrieved_chunks = state_manager.get_text_chunks(test_pdf.file_hash)
    print(f"�� Retrieved {len(retrieved_chunks)} chunks")
    
    # Test logging
    state_manager.log_processing_event(
        test_pdf.file_hash, 
        "INFO", 
        "Test processing completed",
        {"processing_time": 1.5}
    )
    
    # Test getting stats
    stats = state_manager.get_processing_stats()
    print(f"📊 Processing stats: {stats}")
    
    # Clean up
    state_manager.close()
    print("✅ State Manager test completed")

if __name__ == "__main__":
    test_state_manager()