"""Database connection and session management - Synchronous"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import logging

from app.config import settings
from app.database.models import Base

logger = logging.getLogger(__name__)

# Database engine and session factory
engine = None
SessionLocal = None


def get_database_url() -> str:
    """Get the database URL from settings"""
    return settings.computed_database_url


def init_db():
    """Initialize database engine and session factory"""
    global engine, SessionLocal
    
    try:
        database_url = get_database_url()
        logger.info(f"Initializing database connection to {database_url.split('@')[1] if '@' in database_url else 'database'}")
        
        # Create synchronous engine with connection pool
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
            echo=settings.debug,  # Log SQL queries in debug mode
        )
        
        # Create session factory
        SessionLocal = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_db():
    """
    Dependency function to get database session.
    Use with FastAPI Depends.
    
    Example:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def close_db():
    """Close database connections and dispose engine"""
    global engine
    
    try:
        if engine:
            logger.info("Disposing database engine...")
            engine.dispose()
            logger.info("Database engine disposed")
    except Exception as e:
        logger.error(f"Error disposing database engine: {e}")


def check_database_connection() -> bool:
    """Check if database connection is working"""
    try:
        if engine is None:
            return False
        
        # Try to execute a simple query
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        
        return True
        
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
