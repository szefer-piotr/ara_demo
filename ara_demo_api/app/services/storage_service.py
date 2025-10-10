import uuid
from datetime import datetime
from typing import Optional, Tuple
from minio import Minio
from minio.error import S3Error
import logging

from app.config import settings

logger = logging.getLogger(__name__)

class StorageService:
    """Service for handling file storage in MinIO"""

    def __init__(self):
        self.client = Minio(
            f"{settings.minio_host}:{settings.minio_port}",
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_use_ssl
        )
        self.bucket_name = settings.minio_bucket_name
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if not"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Error checking/creating bucket: {e}")
            raise

    def generate_file_id(self) -> str:
        return str(uuid.uuid4())

    
    def generate_storage_path(self, session_id: str, file_id: str, filename: str) -> str:
        safe_filename = filename.replace('/', '_').replace('\\', '_')
        return f"sessions/{session_id}/files/{file_id}/{safe_filename}"

    def save_file(
        self,
        file_content: bytes,
        session_id: str,
        filename: str,
        content_type: str = "application/octet-stream"
    ) -> Tuple[str, str]:
        """Save file to MinIO storage - Synchronous"""
        from io import BytesIO
        file_id = self.generate_file_id()
        storage_path = self.generate_storage_path(session_id, file_id, filename)
        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=storage_path,
                data=BytesIO(file_content),
                length=len(file_content),
                content_type=content_type
            )
            logger.info(f"Saved file {filename} to {storage_path}")
            return file_id, storage_path

        except S3Error as e:
            logger.error(f"Failed to save file to MinIO: {e}")
            raise

    def get_file(self, storage_path: str) -> bytes:
        try:
            response = self.client.get_object(self.bucket_name, storage_path)
            return response.read()
        except S3Error as e:
            logger.error(f"Failed to retrieve file from MinIO: {e}")
            raise

    def delete_file(self, storage_path: str) -> bool:
        """Delete file from MinIO"""
        try:
            self.client.remove_object(self.bucket_name, storage_path)
            logger.info(f"Deleted file: {storage_path}")
            return True
        except S3Error as e:
            logger.error(f"Failed to delete file: {e}")
            return False
    
    def file_exists(self, storage_path: str) -> bool:
        """Check if file exists in MinIO"""
        try:
            self.client.stat_object(self.bucket_name, storage_path)
            return True
        except S3Error:
            return False


# Global instance
storage_service = StorageService()