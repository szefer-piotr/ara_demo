#!/usr/bin/env python3
"""
Database initialization script

This script initializes the database, creates all tables,
and optionally adds test data.

Usage:
    python scripts/init_database.py
    python scripts/init_database.py --with-test-data
    python scripts/init_database.py --drop-all
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import init_db, check_database_connection, get_database_url
from app.database.models import Base, Session, File, Hypothesis, AnalysisPlan, ApiKey
from app.config import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def drop_all_tables():
    """Drop all tables from the database"""
    from sqlalchemy.ext.asyncio import create_async_engine
    
    logger.warning("⚠️  Dropping all tables...")
    
    db_url = get_database_url()
    engine = create_async_engine(db_url)
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("✅ All tables dropped")
    except Exception as e:
        logger.error(f"❌ Failed to drop tables: {e}")
        raise
    finally:
        await engine.dispose()


async def create_test_data():
    """Create some test data in the database"""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from datetime import datetime
    import uuid
    import hashlib
    
    logger.info("📝 Creating test data...")
    
    db_url = get_database_url()
    engine = create_async_engine(db_url)
    AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    try:
        async with AsyncSessionLocal() as db:
            # Create test session
            test_session = Session(
                session_id=f"test-session-{uuid.uuid4().hex[:8]}",
                data={
                    "user": "test_user",
                    "environment": "development",
                    "created_by": "init_script"
                }
            )
            db.add(test_session)
            await db.flush()
            
            # Create test file
            test_file = File(
                file_id=f"test-file-{uuid.uuid4().hex[:8]}",
                session_id=test_session.session_id,
                filename="sample_data.csv",
                storage_path="/uploads/test/sample_data.csv",
                file_size=1024,
                file_type="text/csv",
                metadata={"rows": 100, "columns": 5}
            )
            db.add(test_file)
            
            # Create test hypothesis
            test_hypothesis = Hypothesis(
                hypothesis_id=f"test-hyp-{uuid.uuid4().hex[:8]}",
                session_id=test_session.session_id,
                title="Test Hypothesis: Data Analysis",
                description="This is a test hypothesis for system verification",
                data={
                    "prediction": "Data will show positive correlation",
                    "variables": ["var1", "var2"]
                },
                status="draft",
                confidence_level=70
            )
            db.add(test_hypothesis)
            await db.flush()
            
            # Create test analysis plan
            test_plan = AnalysisPlan(
                plan_id=f"test-plan-{uuid.uuid4().hex[:8]}",
                hypothesis_id=test_hypothesis.hypothesis_id,
                plan_data={
                    "steps": [
                        {"id": 1, "title": "Data Exploration"},
                        {"id": 2, "title": "Statistical Analysis"}
                    ]
                },
                accepted=False,
                status="draft",
                total_steps=2,
                completed_steps=0
            )
            db.add(test_plan)
            
            # Create test API key
            test_key = "test-key-12345"
            key_hash = hashlib.sha256(test_key.encode()).hexdigest()
            test_api_key = ApiKey(
                key_id=f"test-key-{uuid.uuid4().hex[:8]}",
                key_hash=key_hash,
                description="Test API key for development",
                active=True,
                usage_count=0,
                metadata={"created_by": "init_script"}
            )
            db.add(test_api_key)
            
            # Commit all
            await db.commit()
            
            logger.info("✅ Test data created successfully:")
            logger.info(f"   - Session ID: {test_session.session_id}")
            logger.info(f"   - File ID: {test_file.file_id}")
            logger.info(f"   - Hypothesis ID: {test_hypothesis.hypothesis_id}")
            logger.info(f"   - Plan ID: {test_plan.plan_id}")
            logger.info(f"   - API Key: {test_key}")
            
    except Exception as e:
        logger.error(f"❌ Failed to create test data: {e}")
        raise
    finally:
        await engine.dispose()


async def verify_tables():
    """Verify that all tables were created"""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    
    logger.info("🔍 Verifying tables...")
    
    db_url = get_database_url()
    engine = create_async_engine(db_url)
    
    try:
        async with engine.connect() as conn:
            # Query to get all table names
            result = await conn.execute(
                text("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """)
            )
            tables = [row[0] for row in result.fetchall()]
            
            expected_tables = [
                'sessions', 'files', 'hypotheses', 
                'analysis_plans', 'api_keys', 'step_runs', 'reports'
            ]
            
            logger.info(f"📊 Found {len(tables)} tables:")
            for table in tables:
                status = "✅" if table in expected_tables else "❓"
                logger.info(f"   {status} {table}")
            
            missing = set(expected_tables) - set(tables)
            if missing:
                logger.warning(f"⚠️  Missing tables: {', '.join(missing)}")
            else:
                logger.info("✅ All expected tables found!")
                
    except Exception as e:
        logger.error(f"❌ Failed to verify tables: {e}")
        raise
    finally:
        await engine.dispose()


async def main(args):
    """Main function"""
    try:
        logger.info("=" * 60)
        logger.info("🚀 ARA Demo API - Database Initialization")
        logger.info("=" * 60)
        logger.info("")
        
        # Show configuration
        logger.info("📋 Configuration:")
        logger.info(f"   Database: {settings.postgres_db}")
        logger.info(f"   Host: {settings.postgres_host}")
        logger.info(f"   Port: {settings.postgres_port}")
        logger.info(f"   User: {settings.postgres_user}")
        logger.info("")
        
        # Drop tables if requested
        if args.drop_all:
            if args.yes or input("⚠️  Drop all tables? (yes/no): ").lower() == 'yes':
                await drop_all_tables()
                logger.info("")
            else:
                logger.info("❌ Operation cancelled")
                return
        
        # Initialize database
        logger.info("🔧 Initializing database...")
        await init_db()
        logger.info("✅ Database initialized successfully!")
        logger.info("")
        
        # Check connection
        logger.info("🔌 Checking database connection...")
        if await check_database_connection():
            logger.info("✅ Database connection verified!")
        else:
            logger.error("❌ Database connection failed!")
            return
        logger.info("")
        
        # Verify tables
        await verify_tables()
        logger.info("")
        
        # Create test data if requested
        if args.with_test_data:
            await create_test_data()
            logger.info("")
        
        logger.info("=" * 60)
        logger.info("🎉 Database initialization completed successfully!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Start the API server: uvicorn app.main:app --reload")
        logger.info("  2. Visit the docs: http://localhost:8000/docs")
        logger.info("")
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 60)
        logger.error(f"❌ Initialization failed: {e}")
        logger.error("=" * 60)
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("  1. Check if PostgreSQL is running")
        logger.error("  2. Verify connection settings in .env or config")
        logger.error("  3. Check database credentials")
        logger.error("  4. Ensure database exists")
        logger.error("")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the database")
    parser.add_argument(
        "--with-test-data",
        action="store_true",
        help="Create test data after initialization"
    )
    parser.add_argument(
        "--drop-all",
        action="store_true",
        help="Drop all existing tables before initialization (DANGEROUS!)"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    args = parser.parse_args()
    
    # Run async main
    asyncio.run(main(args))

