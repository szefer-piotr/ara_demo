from sre_parse import SUCCESS
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import logging
from contextlib import asynccontextmanager
import redis.asyncio as redis
from minio import Minio
from minio.error import S3Error
import asyncpg
from datetime import datetime



from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import logging

from app.database import get_db
from app.services.storage_service import storage_service
from app.services.file_service import FileService
from app.utils.file_utils import robust_read_csv
from app.models.api_models import UploadResponse, AnalyzeResponse
from app.config import settings
from app.models.api_models import ApiResponse, UploadResponse
from app.database import init_db, check_database_connection, close_db
from app.utils.file_utils import robust_read_csv

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

router = APIRouter()
logger = logging.getLogger(__name__)

redis_client = None
minio_client = None
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""

    # Startup
    logger.info("Starting ARA Demo API...")

    try:
        # Initialize database
        logger.info(" Initializing database connection...")
        await init_db()
        if await check_database_connection():
            logger.info(" Database connection established")
        else:
            logger.warning(" Database connection check failed")
        
        logger.info(" Initializing Redis connection...")
        global redis_client
        redis_client = redis.from_url(settings.computed_redis_url)
        await redis_client.ping()
        logger.info(" Redis connection established")

        logger.info("Initializing MinIO connection...")
        global minio_client
        minio_client = Minio(
            settings.minio_host + ":" + str(settings.minio_port),
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_use_ssl
        )

        bucket_name = settings.minio_bucket_name
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logger.info(f" Created a MinIO bucket: {bucket_name}")
        else:
            logger.info(f" MinIO bucket exists: {bucket_name}")

        logger.info("Initializing PostgreSQL connection..")
        global db_pool
        db_pool = await asyncpg.create_pool(
            settings.computed_database_url,
            min_size=1,
            max_size=10,
            command_timeout=60
        )

        async with db_pool.acquire() as conn:
            await conn.fetchval('SELECT 1')
        logger.info(" PostgreSQL connection established")
        logger.info(" ARA Demo API startup complete!")

    except Exception as e:
        logger.error(f" Startup failed: {e}")
        raise

    yield

    logger.info("Shutting down ARA Demo API...")

    try:
        # Close database connections
        logger.info(" Closing database connections...")
        await close_db()
        logger.info(" Database connections closed")
        
        if redis_client:
            await redis_client.close()
            logger.info(" Redis connection closed")
        if db_pool:
            await db_pool.close()
            logger.info(" PostgreSQL connection pool closed")

        logger.info(" Shutdown complete")

    except Exception as e:
        logger.error(f" Shutdown error: {e}")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    debug=settings.debug,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Root endpoint
@app.get("/", response_model=ApiResponse)
async def root():
    """Root endpoint with API information"""
    return ApiResponse(
        success=True,
        data={
            "message": "ARA Demo API",
            "version": settings.api_version,
            "title": settings.api_title
        }
    )

# Health check endpoint
@app.get("/health", response_model=ApiResponse)
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "version": settings.api_version,
        "services": {}
    }
    
    try:
        # Check Redis
        if redis_client:
            await redis_client.ping()
            health_status["services"]["redis"] = "healthy"
        else:
            health_status["services"]["redis"] = "unavailable"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
    
    try:
        # Check MinIO
        if minio_client:
            minio_client.list_buckets()
            health_status["services"]["minio"] = "healthy"
        else:
            health_status["services"]["minio"] = "unavailable"
    except Exception as e:
        health_status["services"]["minio"] = f"unhealthy: {str(e)}"
    
    try:
        # Check PostgreSQL
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            health_status["services"]["postgresql"] = "healthy"
        else:
            health_status["services"]["postgresql"] = "unavailable"
    except Exception as e:
        health_status["services"]["postgresql"] = f"unhealthy: {str(e)}"
    
    # Determine overall health
    all_healthy = all(
        status == "healthy" 
        for status in health_status["services"].values()
    )
    
    if not all_healthy:
        return JSONResponse(
            status_code=503,
            content=ApiResponse(
                success=False,
                data=health_status,
                error="Some services are unhealthy"
            ).model_dump()
        )
    
    return ApiResponse(
        success=True,
        data=health_status
    )


@router.post("/files", response_model=UploadResponse, tags=["files"])
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Header(..., alias="X-Session-ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a file to storage
    
    Headers:
    - X-Session-ID: Session identifier
    """
    try:
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(400, "Empty file")
        
        if len(content) > settings.max_file_size:
            raise HTTPException(
                400,
                f"File too large. Max size: {settings.max_file_size} bytes"
            )
        
        # Save to MinIO
        file_id, storage_path = await storage_service.save_file(
            file_content=content,
            session_id=session_id,
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream"
        )
        
        # Save metadata to database
        file_record = await FileService.create_file(
            db=db,
            file_id=file_id,
            session_id=session_id,
            filename=file.filename,
            storage_path=storage_path,
            file_size=len(content),
            file_type=file.content_type or "application/octet-stream"
        )
        
        logger.info(f"File uploaded: {file_id} ({file.filename})")
        
        return UploadResponse(
            file_id=file_id,
            message="File uploaded successfully",
            file_name=file.filename,
            file_size=len(content),
            file_type=file.content_type,
            upload_timestamp=file_record.created_at
        )
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(500, f"File upload failed: {str(e)}")


@router.post("/files/analyze", response_model=AnalyzeResponse, tags=["files"])
async def upload_and_analyze_csv(
    file: UploadFile = File(...),
    session_id: str = Header(..., alias="X-Session-ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and analyze a CSV file
    
    Headers:
    - X-Session-ID: Session identifier
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(400, "Only CSV files are supported")
        
        # Read file content
        content = await file.read()
        
        # Parse CSV
        loop = asyncio.get_event_loop()
        df, encoding, delimiter = await loop.run_in_executor(
            None,
            robust_read_csv,
            content,
            file.filename
        )
        
        # Save to MinIO
        file_id, storage_path = await storage_service.save_file(
            file_content=content,
            session_id=session_id,
            filename=file.filename,
            content_type="text/csv"
        )
        
        # Prepare column information
        columns = []
        for col in df.columns:
            columns.append({
                "name": col,
                "type": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isna().sum()),
                "unique_values": int(df[col].nunique())
            })
        
        # Save metadata to database
        await FileService.create_file(
            db=db,
            file_id=file_id,
            session_id=session_id,
            filename=file.filename,
            storage_path=storage_path,
            file_size=len(content),
            file_type="text/csv",
            extra_metadata={
                "encoding": encoding,
                "delimiter": delimiter,
                "rows": len(df),
                "columns": len(df.columns)
            }
        )
        
        logger.info(f"CSV analyzed: {file_id} ({len(df)} rows, {len(df.columns)} cols)")
        
        return AnalyzeResponse(
            file_id=file_id,
            columns=columns,
            summary={
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "encoding": encoding,
                "delimiter": delimiter
            },
            row_count=len(df),
            column_count=len(df.columns),
            data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            missing_values={col: int(df[col].isna().sum()) for col in df.columns},
            sample_data=df.head(5).to_dict('records')
        )
        
    except UnicodeDecodeError as e:
        raise HTTPException(400, f"Failed to decode CSV file: {str(e)}")
    except Exception as e:
        logger.error(f"CSV analysis failed: {e}")
        raise HTTPException(500, f"CSV analysis failed: {str(e)}")


@router.get("/files/{file_id}", tags=["files"])
async def get_file_info(
    file_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get file metadata"""
    file_record = await FileService.get_file(db, file_id)
    
    if not file_record:
        raise HTTPException(404, "File not found")
    
    return {
        "file_id": file_record.file_id,
        "filename": file_record.filename,
        "file_size": file_record.file_size,
        "file_type": file_record.file_type,
        "created_at": file_record.created_at,
        "extra_metadata": file_record.extra_metadata
    }


@router.delete("/files/{file_id}", tags=["files"])
async def delete_file(
    file_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a file"""
    # Get file metadata
    file_record = await FileService.get_file(db, file_id)
    
    if not file_record:
        raise HTTPException(404, "File not found")
    
    # Delete from MinIO
    storage_service.delete_file(file_record.storage_path)
    
    # Delete from database
    await FileService.delete_file(db, file_id)
    
    return {"message": "File deleted successfully"}
# Dependency to get Redis client
async def get_redis():
    """Dependency to get Redis client"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    return redis_client

# Dependency to get MinIO client
def get_minio():
    """Dependency to get MinIO client"""
    if not minio_client:
        raise HTTPException(status_code=503, detail="MinIO not available")
    return minio_client

# Dependency to get database pool
async def get_db():
    """Dependency to get database connection"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    return db_pool