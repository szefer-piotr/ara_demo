#!/usr/bin/env python3
"""
Clear All Data Script

This script clears all data from both PostgreSQL and MongoDB databases
while preserving the table structure.

Usage:
    python scripts/clear_all_data.py [--confirm]
"""

import os
import sys
import argparse
import logging
import psycopg2
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv

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

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )

def clear_postgres_data():
    """Clear all data from PostgreSQL tables"""
    logging.info("Clearing PostgreSQL data...")
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        with conn.cursor() as cur:
            # Get list of all tables
            cur.execute("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            if not tables:
                logging.info("No tables found in PostgreSQL database")
                return
            
            logging.info(f"Found {len(tables)} tables: {', '.join(tables)}")
            
            # Disable foreign key checks temporarily
            cur.execute("SET session_replication_role = replica;")
            
            # Clear all tables
            for table in tables:
                cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                logging.info(f"Cleared table: {table}")
            
            # Re-enable foreign key checks
            cur.execute("SET session_replication_role = DEFAULT;")
            
            # Reset sequences
            cur.execute("""
                SELECT sequence_name FROM information_schema.sequences 
                WHERE sequence_schema = 'public'
            """)
            sequences = [row[0] for row in cur.fetchall()]
            
            for sequence in sequences:
                cur.execute(f"ALTER SEQUENCE {sequence} RESTART WITH 1")
                logging.info(f"Reset sequence: {sequence}")
            
            conn.commit()
            logging.info("PostgreSQL data cleared successfully")
            
    except Exception as e:
        logging.error(f"Error clearing PostgreSQL data: {e}")
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def clear_mongodb_data():
    """Clear all data from MongoDB collections"""
    logging.info("Clearing MongoDB data...")
    
    try:
        connection_string = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_HOST}:{MONGODB_PORT}/{MONGODB_DATABASE}?authSource=admin"
        
        client = MongoClient(connection_string)
        db = client[MONGODB_DATABASE]
        
        # Get list of collections
        collections = db.list_collection_names()
        
        if not collections:
            logging.info("No collections found in MongoDB database")
            return
        
        logging.info(f"Found {len(collections)} collections: {', '.join(collections)}")
        
        # Clear all collections
        for collection_name in collections:
            collection = db[collection_name]
            result = collection.delete_many({})
            logging.info(f"Cleared collection '{collection_name}': {result.deleted_count} documents")
        
        logging.info("MongoDB data cleared successfully")
        
    except Exception as e:
        logging.error(f"Error clearing MongoDB data: {e}")
        raise
    finally:
        if 'client' in locals():
            client.close()

def confirm_action():
    """Ask for user confirmation"""
    print("\n" + "="*60)
    print("⚠️  WARNING: CLEAR ALL DATA")
    print("="*60)
    print("This will PERMANENTLY DELETE ALL DATA from:")
    print(f"  • PostgreSQL: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"  • MongoDB: {MONGODB_HOST}:{MONGODB_PORT}/{MONGODB_DATABASE}")
    print("\nThis action CANNOT be undone!")
    print("Table structure will be preserved.")
    print("="*60)
    
    response = input("\nAre you sure you want to clear all data? (type 'yes' to confirm): ")
    return response.lower() == 'yes'

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Clear all data from PostgreSQL and MongoDB")
    parser.add_argument("--confirm", action="store_true", 
                       help="Skip confirmation prompt")
    args = parser.parse_args()
    
    setup_logging()
    
    # Confirm action unless --confirm flag is used
    if not args.confirm:
        if not confirm_action():
            print("Operation cancelled.")
            return
    
    try:
        logging.info("Starting data clearing...")
        
        # Clear PostgreSQL
        clear_postgres_data()
        
        # Clear MongoDB
        clear_mongodb_data()
        
        logging.info("✅ All data cleared successfully!")
        print("\n" + "="*60)
        print("✅ DATA CLEARING COMPLETED")
        print("="*60)
        print("All data has been cleared from both databases.")
        print("Table structures have been preserved.")
        print("="*60)
        
    except Exception as e:
        logging.error(f"❌ Data clearing failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
