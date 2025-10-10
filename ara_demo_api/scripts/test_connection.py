#!/usr/bin/env python3
"""
Quick database connection test script

This script performs a simple connection test to verify
that the database is accessible and configured correctly.

Usage:
    python scripts/test_connection.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_database_connection():
    """Test PostgreSQL database connection"""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    
    logger.info("🔍 Testing PostgreSQL connection...")
    
    try:
        # Convert to async URL
        db_url = settings.computed_database_url
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        # Create engine
        engine = create_async_engine(db_url, echo=False)
        
        # Test connection
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            
            logger.info("✅ PostgreSQL connection successful!")
            logger.info(f"   Version: {version.split(',')[0]}")
            
            # Get database info
            result = await conn.execute(text("SELECT current_database()"))
            db_name = result.scalar()
            logger.info(f"   Database: {db_name}")
            
            result = await conn.execute(text("SELECT current_user"))
            user = result.scalar()
            logger.info(f"   User: {user}")
        
        await engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return False


async def test_redis_connection():
    """Test Redis connection"""
    logger.info("🔍 Testing Redis connection...")
    
    try:
        import redis.asyncio as redis
        
        client = redis.from_url(settings.computed_redis_url)
        
        # Test ping
        await client.ping()
        logger.info("✅ Redis connection successful!")
        
        # Get info
        info = await client.info()
        logger.info(f"   Version: {info.get('redis_version', 'unknown')}")
        
        await client.close()
        return True
        
    except ImportError:
        logger.warning("⚠️  Redis library not installed, skipping test")
        return None
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False


async def test_minio_connection():
    """Test MinIO connection"""
    logger.info("🔍 Testing MinIO connection...")
    
    try:
        from minio import Minio
        
        client = Minio(
            f"{settings.minio_host}:{settings.minio_port}",
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_use_ssl
        )
        
        # Test by listing buckets
        buckets = client.list_buckets()
        logger.info("✅ MinIO connection successful!")
        logger.info(f"   Buckets: {len(buckets)}")
        
        # Check if our bucket exists
        bucket_name = settings.minio_bucket_name
        if client.bucket_exists(bucket_name):
            logger.info(f"   Bucket '{bucket_name}' exists")
        else:
            logger.info(f"   Bucket '{bucket_name}' does not exist yet")
        
        return True
        
    except ImportError:
        logger.warning("⚠️  MinIO library not installed, skipping test")
        return None
    except Exception as e:
        logger.error(f"❌ MinIO connection failed: {e}")
        return False


async def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("🚀 ARA Demo API - Connection Test")
    logger.info("=" * 60)
    logger.info("")
    
    logger.info("📋 Configuration:")
    logger.info(f"   Database: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    logger.info(f"   Redis: {settings.redis_host}:{settings.redis_port}")
    logger.info(f"   MinIO: {settings.minio_host}:{settings.minio_port}")
    logger.info("")
    
    results = {}
    
    # Test each service
    results['database'] = await test_database_connection()
    logger.info("")
    
    results['redis'] = await test_redis_connection()
    logger.info("")
    
    results['minio'] = await test_minio_connection()
    logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 Connection Test Summary")
    logger.info("=" * 60)
    
    for service, status in results.items():
        if status is True:
            logger.info(f"✅ {service.title()}: Connected")
        elif status is False:
            logger.info(f"❌ {service.title()}: Failed")
        elif status is None:
            logger.info(f"⏭️  {service.title()}: Skipped")
    
    logger.info("")
    
    # Overall status
    failed = [k for k, v in results.items() if v is False]
    if failed:
        logger.error(f"❌ Some connections failed: {', '.join(failed)}")
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("  1. Ensure all services are running (docker-compose up -d)")
        logger.error("  2. Check .env configuration")
        logger.error("  3. Verify network connectivity")
        logger.error("  4. Check service logs")
        sys.exit(1)
    else:
        logger.info("🎉 All connections successful!")
        logger.info("")
        logger.info("You can now run:")
        logger.info("  python scripts/init_database.py")


if __name__ == "__main__":
    asyncio.run(main())

