
## Summary

I've created a complete separate PostgreSQL setup for your PDF processing needs:

### What I've created:

1. **New PostgreSQL Service** (`services/biorxiv-postgres/service.yaml`):
   - Dedicated database on port 5433
   - Separate from Langfuse database
   - Includes PgAdmin for database management

2. **Database Initialization Scripts**:
   - `01-init-database.sql`: Creates database and user
   - `02-create-tables.sql`: Creates all necessary tables for PDF processing

3. **Updated Configuration**:
   - Updated `docker-compose.yaml` to include the new service
   - Updated environment variables in `misc/biorxiv_config.env.example`
   - Updated `biorxiv-gemini-processing.py` to use the new database

4. **Database Schema**:
   - `pdf_processing_state`: Tracks PDF processing status
   - `extracted_texts`: Stores full extracted text
   - `text_chunks`: Stores text chunks for RAG
   - `processing_logs`: Stores processing logs

### To use this setup:

1. **Start the services**:
   ```bash
   docker-compose up -d biorxiv-postgres
   ```

2. **Access the database**:
   ```bash
   psql -h localhost -p 5433 -U biorxiv_user -d biorxiv_processing
   ```

3. **Access PgAdmin** (optional):
   - URL: http://localhost:5051
   - Email: admin@biorxiv.local
   - Password: admin123

The database will be automatically initialized with all necessary tables when you first start the service. Your PDF processing pipeline can now use this dedicated database without interfering with Langfuse.

# BioRxiv PDF Processing Database

This PostgreSQL instance is specifically dedicated to storing PDF processing data for the BioRxiv pipeline, separate from the Langfuse database.

## Configuration

- **Port**: 5433 (to avoid conflicts with other PostgreSQL instances)
- **Database**: `biorxiv_processing`
- **User**: `biorxiv_user`
- **Password**: `biorxiv_password`

## Services

### biorxiv-postgres
The main PostgreSQL database for PDF processing.

### biorxiv-pgadmin
Web-based PostgreSQL administration tool accessible at http://localhost:5051

## Database Schema

### Tables

1. **pdf_processing_state**: Tracks the processing status of each PDF
2. **extracted_texts**: Stores the full extracted text from PDFs
3. **text_chunks**: Stores text chunks for RAG processing
4. **processing_logs**: Stores processing logs and errors

### Key Features

- Automatic timestamp updates
- JSONB support for flexible metadata storage
- Comprehensive indexing for performance
- Foreign key relationships for data integrity

## Usage

1. Start the services:
   ```bash
   docker-compose up -d biorxiv-postgres
   ```

2. Access the database:
   ```bash
   psql -h localhost -p 5433 -U biorxiv_user -d biorxiv_processing
   ```

3. Access PgAdmin:
   - URL: http://localhost:5051
   - Email: admin@biorxiv.local
   - Password: admin123

## Environment Variables

Add these to your `.env` file:

```env
# BioRxiv PDF Processing Database
BIORXIV_DB_HOST=localhost
BIORXIV_DB_PORT=5433
BIORXIV_DB_NAME=biorxiv_processing
BIORXIV_DB_USER=biorxiv_user
BIORXIV_DB_PASSWORD=biorxiv_password

# PgAdmin (optional)
BIORXIV_PGADMIN_EMAIL=admin@biorxiv.local
BIORXIV_PGADMIN_PASSWORD=admin123
```

