# app/services/file_service.py

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from typing import Optional

from app.database.models import File, Session
import logging

logger = logging.getLogger(__name__)


class FileService:
    """Service for file database operations"""
    
    @staticmethod
    async def create_file(
        db: AsyncSession,
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
        await db.commit()
        await db.refresh(file_record)
        
        logger.info(f"Created file record: {file_id}")
        return file_record
    
    @staticmethod
    async def get_file(db: AsyncSession, file_id: str) -> Optional[File]:
        """Get file by ID"""
        result = await db.execute(
            select(File).where(File.file_id == file_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_session_files(db: AsyncSession, session_id: str) -> list[File]:
        """Get all files for a session"""
        result = await db.execute(
            select(File).where(File.session_id == session_id)
        )
        return result.scalars().all()
    
    @staticmethod
    async def delete_file(db: AsyncSession, file_id: str) -> bool:
        """Delete file record from database"""
        file_record = await FileService.get_file(db, file_id)
        if file_record:
            await db.delete(file_record)
            await db.commit()
            logger.info(f"Deleted file record: {file_id}")
            return True
        return False