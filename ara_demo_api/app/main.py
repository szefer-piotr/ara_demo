from sre_parse import SUCCESS
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import logging
from contextlib import asynccontextmanager
import redis.asyncio as redis
from minio import Minio
from minio.error import S3Error
import asyncpg

from app.config import settings
from app.models.api_models import ApiResponse

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
        logger.info(" Initializing Redis connection...")
        global redis_client
        redis_client = redis.from_url(settings.computed_redis_url)
        await redis_client.ping()
        logger.info(" Redis connection established.")

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
        if redis_client:
            await redis_client.close()
            logger.info(" Redis connection closed")
        if db_pool:
            await db_pool.close()
            logger.info(" PostgreSQL connection pool closed")

        logger.info(" Shutsown complete")

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