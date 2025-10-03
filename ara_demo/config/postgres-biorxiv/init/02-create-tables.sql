-- Create tables for PDF processing
-- This script runs after the database initialization

-- PDF processing state table
CREATE TABLE IF NOT EXISTS pdf_processing_state (
    id SERIAL PRIMARY KEY,
    pdf_hash VARCHAR(64) UNIQUE NOT NULL,
    s3_key VARCHAR(255) NOT NULL,
    pdf_name VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    error_message TEXT,
    metadata JSONB
);

-- Extracted text storage
CREATE TABLE IF NOT EXISTS extracted_texts (
    id SERIAL PRIMARY KEY,
    pdf_hash VARCHAR(64) UNIQUE NOT NULL,
    s3_key VARCHAR(255) NOT NULL,
    pdf_name VARCHAR(255) NOT NULL,
    full_text TEXT NOT NULL,
    sections JSONB,
    metadata JSONB,
    text_length INTEGER,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pdf_hash) REFERENCES pdf_processing_state(pdf_hash)
);

-- Text chunks for RAG processing
CREATE TABLE IF NOT EXISTS text_chunks (
    id SERIAL PRIMARY KEY,
    pdf_hash VARCHAR(64) NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_hash VARCHAR(64) NOT NULL,
    chunk_type VARCHAR(50) DEFAULT 'text',
    section_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    FOREIGN KEY (pdf_hash) REFERENCES pdf_processing_state(pdf_hash)
);

-- Processing logs
CREATE TABLE IF NOT EXISTS processing_logs (
    id SERIAL PRIMARY KEY,
    pdf_hash VARCHAR(64) NOT NULL,
    log_level VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pdf_hash) REFERENCES pdf_processing_state(pdf_hash)
);

CREATE TABLE gemini_responses (
    id SERIAL PRIMARY KEY,
    pdf_hash VARCHAR(64) REFERENCES pdf_processing_state(pdf_hash),
    s3_key VARCHAR(255),
    pdf_name VARCHAR(255),
    gemini_response JSONB NOT NULL,
    sections_used TEXT[],
    token_count INTEGER,
    context_length INTEGER,
    instructions TEXT,
    prompt_template TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add indexes for better performance
CREATE INDEX idx_gemini_responses_pdf_hash ON gemini_responses(pdf_hash);
CREATE INDEX idx_gemini_responses_s3_key ON gemini_responses(s3_key);
CREATE INDEX idx_gemini_responses_processed_at ON gemini_responses(processed_at);
CREATE INDEX idx_gemini_responses_json_gin ON gemini_responses USING GIN (gemini_response);


-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pdf_processing_status ON pdf_processing_state(status);
CREATE INDEX IF NOT EXISTS idx_pdf_processing_created_at ON pdf_processing_state(created_at);
CREATE INDEX IF NOT EXISTS idx_pdf_processing_s3_key ON pdf_processing_state(s3_key);

CREATE INDEX IF NOT EXISTS idx_text_chunks_pdf_hash ON text_chunks(pdf_hash);
CREATE INDEX IF NOT EXISTS idx_text_chunks_chunk_type ON text_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_text_chunks_section ON text_chunks(section_name);

CREATE INDEX IF NOT EXISTS idx_processing_logs_pdf_hash ON processing_logs(pdf_hash);
CREATE INDEX IF NOT EXISTS idx_processing_logs_level ON processing_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_processing_logs_created_at ON processing_logs(created_at);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_pdf_processing_state_updated_at 
    BEFORE UPDATE ON pdf_processing_state 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
