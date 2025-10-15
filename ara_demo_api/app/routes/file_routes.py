"""
File upload and management routes
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Header
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import logging

from app.database.connection import get_db
from app.models.api_models import ApiResponse, UploadResponse, AnalyzeResponse
from app.services.storage_service import storage_service
from app.services.file_service import FileService
from app.services.llm_service import get_llm_service
from app.utils.file_utils import robust_read_csv
from app.config import settings
from app.api.dependencies import verify_session_id, get_or_create_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


@router.post("", response_model=UploadResponse)
def upload_csv_file(
    file: UploadFile = File(...),
    session_id: str = Depends(verify_session_id),
    db: Session = Depends(get_db)
):
    """
    Upload and process a CSV file

    Headers:
    - X-Session-ID: Session identifier

    Returns:
    - file_id: Use this to get summary later
    - Basic metadata: encoding, delimiter, row/column counts
    """
    try:
        _ = get_or_create_session(session_id, db)

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
                "columns": len(df.columns),
                "column_names": list(df.columns)
            }
        )
        
    except UnicodeDecodeError as e:
        raise HTTPException(400, f"Failed to decode CSV file: {str(e)}")
    except Exception as e:
        logger.error(f"CSV upload failed: {e}")
        raise HTTPException(500, f"CSV upload failed: {str(e)}")


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_csv_file(
    file: UploadFile = File(...),
    session_id: str = Depends(verify_session_id),
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


@router.get("/{file_id}", response_model=ApiResponse)
def get_file_info(
    file_id: str,
    db: Session = Depends(get_db)
):
    """Get file metadata"""
    file_record = FileService.get_file(db, file_id)
    
    if not file_record:
        raise HTTPException(404, "File not found")
    
    return ApiResponse(
        success=True,
        message="File retrieved",
        data={
        "file_id": file_record.file_id,
        "filename": file_record.filename,
        "file_size": file_record.file_size,
        "file_type": file_record.file_type,
        "created_at": file_record.created_at.isoformat(),
        "extra_metadata": file_record.extra_metadata
        }
    )


@router.get("/{file_id}/summary", response_model=ApiResponse)
def get_file_summary(
    file_id: str,
    infer_descriptions: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get data summary for a file with optional LLM-inferred column descriptions
    
    Query Parameters:
    - infer_descriptions: Whether to use LLM to infer column descriptions (default: True)
    """
    try:
        # Get file metadata
        file_record = FileService.get_file(db, file_id)
        
        if not file_record:
            raise HTTPException(404, "File not found")
        
        # Load file content from MinIO
        content = storage_service.get_file(file_record.storage_path)
        
        # Parse CSV
        df, encoding, delimiter = robust_read_csv(content, file_record.filename)
        
        # Generate summary with LLM service
        llm_service = get_llm_service(tier="lower")
        summary = llm_service.summarize_dataframe(
            df=df,
            infer_descriptions=infer_descriptions
        )
        
        # Add file metadata to response
        summary["file_metadata"] = {
            "file_id": file_record.file_id,
            "filename": file_record.filename,
            "file_size": file_record.file_size,
            "encoding": encoding,
            "delimiter": delimiter
        }
        
        logger.info(f"Generated summary for file {file_id}")
        return ApiResponse(
            success=True,
            message="File summary generated",
            data=summary
        )
        
    except Exception as e:
        logger.error(f"Failed to generate file summary: {e}")
        raise HTTPException(500, f"Failed to generate summary: {str(e)}")


@router.delete("/{file_id}", response_model=ApiResponse)
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

    logger.info(f"Deleted file: {file_id}")
    
    return ApiResponse(
        success=True,
        message= "File deleted successfully"
    )