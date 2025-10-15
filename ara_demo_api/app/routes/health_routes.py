"""
Health check and system info routes
"""
from fastapi import APIRouter
from datetime import datetime, timezone

from app.models.api_models import ApiResponse
from app.config import settings
from app.database.connection import check_database_connection
from app.services.storage_service import storage_service

router = APIRouter(tags=["health"])

@router.get("/", response_model=ApiResponse)
def root():
    """Root endpoint with API information"""
    return ApiResponse(
        success=True,
        message="ARA Demo API is running",
        data={
            "name": settings.app_name,
            "version": settings.api_version,
            "description": settings.api_description,
            "debug": settings.debug,
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json"
            }
        }
    )


@router.get("/health", response_model=ApiResponse)
def health_check():
    """Health check endpoint"""
    from app.main import session_service, llm_service
    db_status = "connected" if check_database_connection() else "disconnected"

    redis_status = "disconnected"
    cache_enabled = False
    if session_service:
        session_health = session_service.health_check()
        redis_status = "connected" if session_health.get("redis") else "disconnected"
        cache_enabled = session_health.get("using_cache", False)
    
    storage_status = "unknown"
    try:
        storage_service.health_check()
        storage_status = "connected"
    except Exception:
        storage_status = "disconnected"

    llm_status = "ready" if llm_service else "not initialized"

    return ApiResponse(
        success=True,
        message="System is healthy",
        data={
            "status": "healthy",
            "services": {
                "database": db_status,
                "redis": redis_status,
                "cache_enabled": cache_enabled,
                "storage": storage_status,
                "llm_service": llm_status
            }
        }
    )


@router.get("/info", response_model=ApiResponse)
def api_info():
    """API information endpoint"""
    return ApiResponse(
        success=True,
        message="API information",
        data={
            "app_name": settings.app_name,
            "version": settings.api_version,
            "description": settings.api_description,
            "contact": {
                "name": settings.api_contact_name,
                "email": settings.api_contact_email,
            },
            "debug_mode": settings.debug,
            "allowed_origins": settings.allowed_origins,
            "max_file_size": settings.max_file_size,
            "upload_path": settings.upload_path,
            "log_level": settings.log_level
        }
    )