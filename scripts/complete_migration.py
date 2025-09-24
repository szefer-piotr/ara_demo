#!/usr/bin/env python3
"""
Complete Migration Script: pdf_hash to pdf_uid

This script handles the complete migration from pdf_hash to pdf_uid format,
including:
1. Schema analysis and validation
2. Creating new pdf_uid columns
3. Migrating data from pdf_hash to pdf_uid
4. Dropping old pdf_hash columns
5. Updating indexes and constraints

Usage:
    python scripts/complete_migration.py [--dry-run] [--confirm] [--steps STEPS]
    
Options:
    --dry-run    Show what would be done without making changes
    --confirm    Skip confirmation prompt
    --steps      Comma-separated list of steps to run (schema,data,cleanup,all)
"""

import os
import sys
import argparse
import logging
import psycopg2
import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv("BIORXIV_DB_HOST", "localhost")
DB_PORT = int(os.getenv("BIORXIV_DB_PORT", "9999"))
DB_NAME = os.getenv("BIORXIV_DB_NAME", "postgres")
DB_USER = os.getenv("BIORXIV_DB_USER", "postgres")
DB_PASSWORD = os.getenv("BIORXIV_DB_PASSWORD", "postgres")

MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGODB_PORT = int(os.getenv("MONGODB_PORT", "27017"))
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME", "admin")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD", "admin123")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "biorxiv_gemini")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "gemini_responses")

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )

def get_postgres_connection():
    """Get PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        logging.info(f"Connected to PostgreSQL at {DB_HOST}:{DB_PORT}")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to PostgreSQL: {e}")
        raise

def get_mongodb_connection():
    """Get MongoDB connection"""
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

def generate_pdf_uid(s3_key, pdf_name, is_supplement=False):
    """Generate PDF UID from existing data"""
    supplement_suffix = "_supplement" if is_supplement else ""
    return f"{s3_key}_{pdf_name}{supplement_suffix}"

def analyze_schema_status():
    """Analyze current schema status"""
    logging.info("Analyzing current schema status...")
    
    conn = get_postgres_connection()
    try:
        with conn.cursor() as cur:
            # Check for pdf_hash columns
            cur.execute("""
                SELECT table_name, column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND column_name = 'pdf_hash'
                ORDER BY table_name
            """)
            pdf_hash_columns = cur.fetchall()
            
            # Check for pdf_uid columns
            cur.execute("""
                SELECT table_name, column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND column_name = 'pdf_uid'
                ORDER BY table_name
            """)
            pdf_uid_columns = cur.fetchall()
            
            # Get table counts
            cur.execute("""
                SELECT table_name, 
                       (SELECT COUNT(*) FROM information_schema.columns 
                        WHERE table_schema = 'public' AND table_name = t.table_name) as column_count
                FROM information_schema.tables t
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            tables = cur.fetchall()
            
            logging.info(f"Tables found: {[t[0] for t in tables]}")
            logging.info(f"Tables with pdf_hash: {[t[0] for t in pdf_hash_columns]}")
            logging.info(f"Tables with pdf_uid: {[t[0] for t in pdf_uid_columns]}")
            
            return {
                'pdf_hash_tables': [t[0] for t in pdf_hash_columns],
                'pdf_uid_tables': [t[0] for t in pdf_uid_columns],
                'all_tables': [t[0] for t in tables]
            }
    
    finally:
        conn.close()

def create_pdf_uid_columns(dry_run=False):
    """Create pdf_uid columns in all tables that have pdf_hash"""
    logging.info("Creating pdf_uid columns...")
    
    conn = get_postgres_connection()
    try:
        with conn.cursor() as cur:
            # Get tables that have pdf_hash but not pdf_uid
            cur.execute("""
                SELECT DISTINCT table_name
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND column_name = 'pdf_hash'
                AND table_name NOT IN (
                    SELECT table_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND column_name = 'pdf_uid'
                )
            """)
            tables_to_update = [row[0] for row in cur.fetchall()]
            
            if not tables_to_update:
                logging.info("All tables already have pdf_uid columns")
                return
            
            logging.info(f"Tables to add pdf_uid column: {tables_to_update}")
            
            for table in tables_to_update:
                sql = f"ALTER TABLE {table} ADD COLUMN pdf_uid VARCHAR(500)"
                
                if dry_run:
                    logging.info(f"Would execute: {sql}")
                else:
                    cur.execute(sql)
                    logging.info(f"Added pdf_uid column to {table}")
            
            if not dry_run:
                conn.commit()
                logging.info("pdf_uid columns created successfully")
            else:
                logging.info("pdf_uid column creation preview completed (dry run)")
    
    except Exception as e:
        logging.error(f"Error creating pdf_uid columns: {e}")
        if not dry_run:
            conn.rollback()
        raise
    finally:
        conn.close()

def migrate_postgres_data(dry_run=False):
    """Migrate PostgreSQL data from pdf_hash to pdf_uid"""
    logging.info("Migrating PostgreSQL data...")
    
    conn = get_postgres_connection()
    try:
        with conn.cursor() as cur:
            # Get all records with pdf_hash
            cur.execute("""
                SELECT pdf_hash, s3_key, pdf_name, 
                       CASE WHEN pdf_name LIKE '%supplement%' THEN true ELSE false END as is_supplement
                FROM extracted_texts
                WHERE pdf_hash IS NOT NULL
            """)
            records = cur.fetchall()
            
            logging.info(f"Found {len(records)} records to migrate")
            
            for pdf_hash, s3_key, pdf_name, is_supplement in records:
                pdf_uid = generate_pdf_uid(s3_key, pdf_name, is_supplement)
                
                if dry_run:
                    logging.info(f"Would migrate: {pdf_hash} -> {pdf_uid}")
                else:
                    # Update the record
                    cur.execute("""
                        UPDATE extracted_texts 
                        SET pdf_uid = %s 
                        WHERE pdf_hash = %s
                    """, (pdf_uid, pdf_hash))
                    
                    # Update related tables
                    cur.execute("""
                        UPDATE pdf_processing_state 
                        SET pdf_uid = %s 
                        WHERE pdf_hash = %s
                    """, (pdf_uid, pdf_hash))
                    
                    logging.info(f"Migrated: {pdf_hash} -> {pdf_uid}")
            
            if not dry_run:
                conn.commit()
                logging.info("PostgreSQL data migration completed")
            else:
                logging.info("PostgreSQL data migration preview completed (dry run)")
    
    except Exception as e:
        logging.error(f"Error migrating PostgreSQL data: {e}")
        if not dry_run:
            conn.rollback()
        raise
    finally:
        conn.close()

def migrate_mongodb_data(dry_run=False):
    """Migrate MongoDB data from pdf_hash to pdf_uid"""
    logging.info("Migrating MongoDB data...")
    
    client = get_mongodb_connection()
    try:
        db = client[MONGODB_DATABASE]
        collection = db[MONGODB_COLLECTION]
        
        # Find all documents with pdf_hash
        documents = list(collection.find({"pdf_hash": {"$exists": True}}))
        
        logging.info(f"Found {len(documents)} MongoDB documents to migrate")
        
        for doc in documents:
            pdf_hash = doc.get('pdf_hash', '')
            s3_key = doc.get('s3_key', '')
            pdf_name = doc.get('pdf_name', '')
            
            # Determine if it's a supplement
            is_supplement = 'supplement' in pdf_name.lower()
            pdf_uid = generate_pdf_uid(s3_key, pdf_name, is_supplement)
            
            if dry_run:
                logging.info(f"Would migrate: {pdf_hash} -> {pdf_uid}")
            else:
                # Update the document
                collection.update_one(
                    {"_id": doc["_id"]},
                    {
                        "$set": {"pdf_uid": pdf_uid},
                        "$unset": {"pdf_hash": ""}
                    }
                )
                logging.info(f"Migrated: {pdf_hash} -> {pdf_uid}")
        
        if not dry_run:
            logging.info("MongoDB data migration completed")
        else:
            logging.info("MongoDB data migration preview completed (dry run)")
    
    except Exception as e:
        logging.error(f"Error migrating MongoDB data: {e}")
        raise
    finally:
        client.close()

def update_constraints_and_indexes(dry_run=False):
    """Update constraints and indexes to use pdf_uid"""
    logging.info("Updating constraints and indexes...")
    
    conn = get_postgres_connection()
    try:
        with conn.cursor() as cur:
            # Get all foreign key constraints that reference pdf_hash
            cur.execute("""
                SELECT 
                    tc.table_name, 
                    kcu.column_name, 
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name,
                    tc.constraint_name
                FROM information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' 
                AND ccu.column_name = 'pdf_hash'
                AND tc.table_schema = 'public'
            """)
            fk_constraints = cur.fetchall()
            
            # Get all unique constraints on pdf_hash
            cur.execute("""
                SELECT table_name, constraint_name
                FROM information_schema.table_constraints 
                WHERE constraint_type = 'UNIQUE'
                AND table_schema = 'public'
                AND constraint_name IN (
                    SELECT constraint_name 
                    FROM information_schema.key_column_usage 
                    WHERE column_name = 'pdf_hash'
                    AND table_schema = 'public'
                )
            """)
            unique_constraints = cur.fetchall()
            
            # Get all indexes on pdf_hash
            cur.execute("""
                SELECT schemaname, tablename, indexname
                FROM pg_indexes 
                WHERE schemaname = 'public'
                AND indexdef LIKE '%pdf_hash%'
            """)
            pdf_hash_indexes = cur.fetchall()
            
            operations = []
            
            # Drop foreign key constraints
            for table, column, f_table, f_column, constraint in fk_constraints:
                operations.append(f"ALTER TABLE {table} DROP CONSTRAINT {constraint}")
            
            # Drop unique constraints
            for table, constraint in unique_constraints:
                operations.append(f"ALTER TABLE {table} DROP CONSTRAINT {constraint}")
            
            # Drop indexes
            for schema, table, index in pdf_hash_indexes:
                operations.append(f"DROP INDEX IF EXISTS {index}")
            
            # Add new unique constraints on pdf_uid
            for table in ['extracted_texts', 'pdf_processing_state']:
                operations.append(f"ALTER TABLE {table} ADD CONSTRAINT {table}_pdf_uid_unique UNIQUE (pdf_uid)")
            
            # Add new foreign key constraints
            operations.append("ALTER TABLE extracted_texts ADD CONSTRAINT fk_extracted_texts_pdf_uid FOREIGN KEY (pdf_uid) REFERENCES pdf_processing_state(pdf_uid)")
            operations.append("ALTER TABLE text_chunks ADD CONSTRAINT fk_text_chunks_pdf_uid FOREIGN KEY (pdf_uid) REFERENCES pdf_processing_state(pdf_uid)")
            operations.append("ALTER TABLE processing_logs ADD CONSTRAINT fk_processing_logs_pdf_uid FOREIGN KEY (pdf_uid) REFERENCES pdf_processing_state(pdf_uid)")
            operations.append("ALTER TABLE gemini_responses ADD CONSTRAINT fk_gemini_responses_pdf_uid FOREIGN KEY (pdf_uid) REFERENCES pdf_processing_state(pdf_uid)")
            
            # Add new indexes
            operations.append("CREATE INDEX idx_extracted_texts_pdf_uid ON extracted_texts(pdf_uid)")
            operations.append("CREATE INDEX idx_pdf_processing_state_pdf_uid ON pdf_processing_state(pdf_uid)")
            operations.append("CREATE INDEX idx_text_chunks_pdf_uid ON text_chunks(pdf_uid)")
            operations.append("CREATE INDEX idx_processing_logs_pdf_uid ON processing_logs(pdf_uid)")
            operations.append("CREATE INDEX idx_gemini_responses_pdf_uid ON gemini_responses(pdf_uid)")
            
            logging.info(f"Found {len(operations)} constraint/index operations to perform")
            
            for operation in operations:
                if dry_run:
                    logging.info(f"Would execute: {operation}")
                else:
                    try:
                        cur.execute(operation)
                        logging.info(f"Executed: {operation}")
                    except Exception as e:
                        logging.warning(f"Failed to execute {operation}: {e}")
            
            if not dry_run:
                conn.commit()
                logging.info("Constraints and indexes updated successfully")
            else:
                logging.info("Constraint and index update preview completed (dry run)")
    
    except Exception as e:
        logging.error(f"Error updating constraints and indexes: {e}")
        if not dry_run:
            conn.rollback()
        raise
    finally:
        conn.close()

def drop_pdf_hash_columns(dry_run=False):
    """Drop pdf_hash columns from all tables"""
    logging.info("Dropping pdf_hash columns...")
    
    conn = get_postgres_connection()
    try:
        with conn.cursor() as cur:
            # Get all tables with pdf_hash columns
            cur.execute("""
                SELECT table_name 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND column_name = 'pdf_hash'
                ORDER BY table_name
            """)
            tables_with_pdf_hash = [row[0] for row in cur.fetchall()]
            
            if not tables_with_pdf_hash:
                logging.info("No pdf_hash columns found to drop")
                return
            
            logging.info(f"Tables to drop pdf_hash column: {tables_with_pdf_hash}")
            
            for table in tables_with_pdf_hash:
                sql = f"ALTER TABLE {table} DROP COLUMN pdf_hash"
                
                if dry_run:
                    logging.info(f"Would execute: {sql}")
                else:
                    cur.execute(sql)
                    logging.info(f"Dropped pdf_hash column from {table}")
            
            if not dry_run:
                conn.commit()
                logging.info("pdf_hash columns dropped successfully")
            else:
                logging.info("pdf_hash column drop preview completed (dry run)")
    
    except Exception as e:
        logging.error(f"Error dropping pdf_hash columns: {e}")
        if not dry_run:
            conn.rollback()
        raise
    finally:
        conn.close()

def confirm_migration(dry_run, steps):
    """Ask for user confirmation"""
    action = "preview" if dry_run else "execute"
    steps_str = ", ".join(steps) if steps != ["all"] else "all steps"
    
    print("\n" + "="*60)
    print(f"⚠️  COMPLETE MIGRATION {action.upper()}")
    print("="*60)
    print(f"This will {action} the following steps: {steps_str}")
    print(f"  • PostgreSQL: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"  • MongoDB: {MONGODB_HOST}:{MONGODB_PORT}/{MONGODB_DATABASE}")
    
    if not dry_run:
        print("\nThis action CANNOT be undone!")
        print("Make sure you have a backup of your data!")
    
    print("="*60)
    
    response = input(f"\nAre you sure you want to {action}? (type 'yes' to confirm): ")
    return response.lower() == 'yes'

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Complete migration from pdf_hash to pdf_uid format")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Preview migration without making changes")
    parser.add_argument("--confirm", action="store_true", 
                       help="Skip confirmation prompt")
    parser.add_argument("--steps", default="all",
                       help="Comma-separated list of steps: schema,data,cleanup,all")
    args = parser.parse_args()
    
    setup_logging()
    
    # Parse steps
    if args.steps == "all":
        steps = ["schema", "data", "cleanup"]
    else:
        steps = [s.strip() for s in args.steps.split(",")]
    
    # Analyze current schema
    schema_status = analyze_schema_status()
    
    # Check if migration is needed
    if not schema_status['pdf_hash_tables'] and schema_status['pdf_uid_tables']:
        logging.info("Database already uses pdf_uid format. No migration needed.")
        return
    
    if not schema_status['pdf_hash_tables']:
        logging.error("No pdf_hash columns found. Nothing to migrate.")
        return
    
    # Confirm action unless --confirm flag is used
    if not args.confirm:
        if not confirm_migration(args.dry_run, steps):
            print("Migration cancelled.")
            return
    
    try:
        logging.info(f"Starting complete migration {'preview' if args.dry_run else 'process'}...")
        
        # Step 1: Schema changes
        if "schema" in steps:
            logging.info("=== STEP 1: SCHEMA CHANGES ===")
            create_pdf_uid_columns(args.dry_run)
            update_constraints_and_indexes(args.dry_run)
        
        # Step 2: Data migration
        if "data" in steps:
            logging.info("=== STEP 2: DATA MIGRATION ===")
            migrate_postgres_data(args.dry_run)
            migrate_mongodb_data(args.dry_run)
        
        # Step 3: Cleanup
        if "cleanup" in steps:
            logging.info("=== STEP 3: CLEANUP ===")
            drop_pdf_hash_columns(args.dry_run)
        
        if args.dry_run:
            logging.info("✅ Complete migration preview completed successfully!")
            print("\n" + "="*60)
            print("✅ MIGRATION PREVIEW COMPLETED")
            print("="*60)
            print("Review the changes above and run without --dry-run to apply them.")
            print("="*60)
        else:
            logging.info("✅ Complete migration completed successfully!")
            print("\n" + "="*60)
            print("✅ COMPLETE MIGRATION COMPLETED")
            print("="*60)
            print("Database has been successfully migrated from pdf_hash to pdf_uid format.")
            print("All constraints, indexes, and data have been updated.")
            print("="*60)
        
    except Exception as e:
        logging.error(f"❌ Migration failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
