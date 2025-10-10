"""Database connection and session management"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
import logging

from app.config import settings
from app.database.models import Base

logger = logging.getLogger(__name__)

# Database engine and session factory
engine = None
AsyncSessionLocal = None


def get_database_url() -> str:
    """Get the database URL from settings"""
    # Convert postgresql:// to postgresql+asyncpg://
    db_url = settings.computed_database_url
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return db_url


async def init_db():
    """Initialize database engine and session factory"""
    global engine, AsyncSessionLocal
    
    try:
        database_url = get_database_url()
        logger.info(f"Initializing async database connection to {database_url.split('@')[1] if '@' in database_url else 'database'}")
        
        # Create async engine with connection pool
        engine = create_async_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
            echo=settings.debug,  # Log SQL queries in debug mode
        )
        
        # Create async session factory
        AsyncSessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def get_db():
    """
    Async dependency function to get database session.
    Use with FastAPI Depends.
    
    Example:
        @app.get("/items/")
        async def read_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    if AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_db():
    """Close database connections and dispose engine"""
    global engine
    
    try:
        if engine:
            logger.info("Disposing database engine...")
            await engine.dispose()
            logger.info("Database engine disposed")
    except Exception as e:
        logger.error(f"Error disposing database engine: {e}")


async def check_database_connection() -> bool:
    """Check if database connection is working"""
    try:
        if engine is None:
            return False
        
        # Try to execute a simple query
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        
        return True
        
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


# Keep sync version for compatibility during transition
def init_database():
    """
    Synchronous wrapper for init_db().
    For use in non-async contexts.
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(init_db())


async def close_database_connections():
    """Alias for close_db() for backward compatibility"""
    await close_db()
