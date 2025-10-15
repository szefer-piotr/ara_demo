"""FastAPI Main Application - Synchronous"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.config import settings
from app.database import init_db, check_database_connection, close_db
from app.services.storage_service import storage_service
from app.services.llm_service import get_llm_service
from app.services.session_service import get_session_service
from app.api.routes import health_routes, session_routes, file_routes

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

session_service = None
llm_service = None

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    description=settings.api_description,
    debug=settings.debug,
    contact={
        "name": settings.api_contact_name,
        "email": settings.api_contact_email,
    },
    openapi_tags=[
        {
            "name": "files",
            "description": "File upload and management operations",
        },
        {
            "name": "analysis",
            "description": "Data analysis operations",
        },
        {
            "name": "health",
            "description": "Health check and system status",
        },
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(health_routes.router)
app.include_router(session_routes.router, prefix="/api")
app.include_router(file_routes.router, prefix="/api")


@app.on_event("startup")
def startup_event():
    """Application startup event"""
    global session_service, llm_service

    logger.info("Starting ARA Demo API...")
    
    try:
        # Initialize database
        logger.info(" Initializing database connection...")
        init_db()
        if check_database_connection():
            logger.info(" Database connection established")
        else:
            logger.warning(" Database connection check failed")

        # Initialize Session Service
        logger.info(" Initializing Session Service")
        session_service = get_session_service()
        session_health = session_service.health_check()
        if session_health.get('redis'):
            logger.info(" Redis cache connected")
        else:
            logger.warning(" Redis unavailable - using database only")

        logger.info(" Initializing LLM Service...")
        llm_service = get_llm_service(tier='lower')
        logger.info(" LLM Service ready")

        logger.info(" Testing MinIO storage...")
        try:
            storage_service.health_check()
            logger.info(" MinIO storage connected")
        except Exception as e:
            logger.warning(f" MinIO connection issue: {e}")

        logger.info(" ARA Demo API startup complete!")
        
    except Exception as e:
        logger.error(f" Startup failed: {e}")
        raise


@app.on_event("shutdown")
def shutdown_event():
    """Application shutdown event"""
    logger.info("Shutting down ARA Demo API...")
    
    try:
        # Close database connections
        logger.info(" Closing database connections...")
        close_db()
        logger.info(" Database connections closed")
        logger.info(" Shutdown complete")
        
    except Exception as e:
        logger.error(f" Shutdown error: {e}")


# Exception handlers
@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc) if settings.debug else "Internal server error"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
