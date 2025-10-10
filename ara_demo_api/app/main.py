"""FastAPI Main Application - Synchronous"""

from ara_demo_api.app.services import llm_service
from ara_demo_api.app.services.llm_service import get_llm_service
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from contextlib import contextmanager
import logging
from datetime import datetime

from app.config import settings
from app.models.api_models import ApiResponse, UploadResponse, AnalyzeResponse
from app.database import init_db, get_db, check_database_connection, close_db
from app.database.models import Session as SessionModel
from app.services.storage_service import storage_service
from app.services.file_service import FileService
from app.utils.file_utils import robust_read_csv

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


@app.on_event("startup")
def startup_event():
    """Application startup event"""
    logger.info("Starting ARA Demo API...")
    
    try:
        # Initialize database
        logger.info(" Initializing database connection...")
        init_db()
        if check_database_connection():
            logger.info(" Database connection established")
        else:
            logger.warning(" Database connection check failed")
        
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


@app.get("/", response_model=ApiResponse)
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


@app.get("/health", response_model=ApiResponse, tags=["health"])
def health_check():
    """Health check endpoint"""
    db_status = "connected" if check_database_connection() else "disconnected"
    
    return ApiResponse(
        success=True,
        message="System is healthy",
        data={
            "status": "healthy",
            "services": {
                "database": db_status,
            }
        }
    )


@app.get("/info", response_model=ApiResponse, tags=["health"])
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


@app.post("/sessions", tags=["sessions"])
def create_session(
    session_id: str = Header(..., alias="X-Session-ID"),
    db: Session = Depends(get_db)
):
    """Create a new session"""
    existing = db.query(SessionModel).filter(
        SessionModel.session_id == session_id
    ).first()

    if existing:
        return {
            "message": "Session already exists",
            "session_id": session_id,
            "created_at": existing.created_at
        }

    new_session = SessionModel(
        session_id=session_id,
        data={}
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    return {
        "message": "Session created",
        "session_id": session_id,
        "created_at": new_session.created_at
    }


@app.post("/files", response_model=UploadResponse, tags=["files"])
def upload_csv_file(
    file: UploadFile = File(...),
    session_id: str = Header(..., alias="X-Session-ID"),
    db: Session = Depends(get_db)
):
    """
    Upload and process a CSV file
    
    Headers:
    - X-Session-ID: Session identifier
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(400, "Only CSV files allowed")
        
        # Read file content (synchronous in FastAPI)
        content = file.file.read()
        
        if len(content) == 0:
            raise HTTPException(400, "Empty file")
        
        if len(content) > settings.max_file_size:
            raise HTTPException(
                400,
                f"File too large. Max size: {settings.max_file_size} bytes"
            )
        
        # Parse CSV with encoding detection (synchronous)
        df, encoding, delimiter = robust_read_csv(content, file.filename)

        llm_service = get_llm_service()
        
        # Save to MinIO storage
        file_id, storage_path = storage_service.save_file(
            file_content=content,
            session_id=session_id,
            filename=file.filename,
            content_type="text/csv"
        )
        
        # Save metadata to PostgreSQL database
        file_record = FileService.create_file(
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
                "columns": len(df.columns),
                "column_names": list(df.columns)
            }
        )
        
        logger.info(f"CSV uploaded: {file_id} ({len(df)} rows, {len(df.columns)} cols)")
        
        # Return response
        return UploadResponse(
            file_id=file_id,
            file_name=file.filename,
            file_size=len(content),
            file_type=file.content_type or "text/csv",
            upload_timestamp=datetime.utcnow(),
            metadata={
                "encoding": encoding,
                "delimiter": delimiter,
                "rows": len(df),
                "columns": len(df.columns)
            }
        )
        
    except UnicodeDecodeError as e:
        raise HTTPException(400, f"Failed to decode CSV file: {str(e)}")
    except Exception as e:
        logger.error(f"CSV upload failed: {e}")
        raise HTTPException(500, f"CSV upload failed: {str(e)}")


@app.post("/files/analyze", response_model=AnalyzeResponse, tags=["files"])
def analyze_csv_file(
    file: UploadFile = File(...),
    session_id: str = Header(..., alias="X-Session-ID"),
    db: Session = Depends(get_db)
):
    """
    Analyze a CSV file without saving
    
    Headers:
    - X-Session-ID: Session identifier
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(400, "Only CSV files are supported")
        
        # Read file content
        content = file.file.read()
        
        # Parse CSV
        df, encoding, delimiter = robust_read_csv(content, file.filename)
        
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
        
        logger.info(f"CSV analyzed: {file.filename} ({len(df)} rows, {len(df.columns)} cols)")
        
        return AnalyzeResponse(
            file_id="",  # Not saved, so no file_id
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


@app.get("/files/{file_id}", tags=["files"])
def get_file_info(
    file_id: str,
    db: Session = Depends(get_db)
):
    """Get file metadata"""
    file_record = FileService.get_file(db, file_id)
    
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


@app.delete("/files/{file_id}", tags=["files"])
def delete_file(
    file_id: str,
    db: Session = Depends(get_db)
):
    """Delete a file"""
    # Get file metadata
    file_record = FileService.get_file(db, file_id)
    
    if not file_record:
        raise HTTPException(404, "File not found")
    
    # Delete from MinIO
    storage_service.delete_file(file_record.storage_path)
    
    # Delete from database
    FileService.delete_file(db, file_id)
    
    return {"message": "File deleted successfully"}


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
