# app/services/file_service.py

from sqlalchemy.orm import Session
from typing import Optional
import logging

from app.database.models import File

logger = logging.getLogger(__name__)


class FileService:
    """Service for file database operations - Synchronous"""
    
    @staticmethod
    def create_file(
        db: Session,
        file_id: str,
        session_id: str,
        filename: str,
        storage_path: str,
        file_size: int,
        file_type: str,
        extra_metadata: dict = None
    ) -> File:
        """Create file record in database"""
        
        file_record = File(
            file_id=file_id,
            session_id=session_id,
            filename=filename,
            storage_path=storage_path,
            file_size=file_size,
            file_type=file_type,
            extra_metadata=extra_metadata or {}
        )
        
        db.add(file_record)
        db.commit()
        db.refresh(file_record)
        
        logger.info(f"Created file record: {file_id}")
        return file_record
    
    @staticmethod
    def get_file(db: Session, file_id: str) -> Optional[File]:
        """Get file by ID"""
        return db.query(File).filter(File.file_id == file_id).first()
    
    @staticmethod
    def get_session_files(db: Session, session_id: str) -> list[File]:
        """Get all files for a session"""
        return db.query(File).filter(File.session_id == session_id).all()
    
    @staticmethod
    def delete_file(db: Session, file_id: str) -> bool:
        """Delete file record from database"""
        file_record = FileService.get_file(db, file_id)
        if file_record:
            db.delete(file_record)
            db.commit()
            logger.info(f"Deleted file record: {file_id}")
            return True
        return False