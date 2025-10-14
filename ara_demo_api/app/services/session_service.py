from asyncio.subprocess import create_subprocess_shell
import json
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import logging

import redis
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import select

from app.config import settings
from app.database.models import Session as SessionModel
from app.models.schemas import SessionData, SessionStatus

logger = logging.getLogger(__name__)


class SessionService:
    """
    Session management with Redis caching and PostgreSQL persistence.
    Synchronous implementation to replace st.session_state.
    """
    def __init__(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(
                settings.computed_redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("Redis connection established succesfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            logger.warning("Running without Redis cache - using database only")

    def _get_redis_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"session:{session_id}"

    def _get_expiry_seconds(self) -> int:
        """Get session expiry time in seconds"""
        return settings.session_expire_minutes * 60

    def create_session(
        self,
        db: DBSession,
        session_name: str = "New Session",
        user_id: Optional[str] = None
    ) -> str:
        """
        Create new session in both Redis and PostgreSQL.

        Args:
            db: SQLAlchemy database session
            session_name: Name for the session
            user_id: Optional user identifier

        Returns:
            session_id: UUID of created session
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=settings.session_expire_minutes)

        session_data = SessionData(
            id=session_id,
            user_id=user_id,
            session_name=session_name,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            last_activity=now,
            expires_at=expires_at,
            hypotheses=[],
            steps=[],
            settings={},
            metadata={}
        )

        try:
            db_session = SessionModel(
                session_id=session_id,
                data=session_data.model_dump(mode='json'),
                created_at=now,
                updated_at=now
            )
            db.add(db_session)
            db.commit()
            db.refresh(db_session)
            logger.info(f"Session {session_id} created in PostgreSQL")
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create session in database: {e}")

        if self.redis_client:
            try:
                redis_key = self._get_redis_key(session_id)
                self.redis_client.setex(
                    redis_key,
                    self._get_expiry_seconds(),
                    json.dumps(session_data.model_dump(mode='json'), default=str)
                )
                logger.info(f"Session {session_id} cached in Redis")
            except Exception as e:
                logger.warning(f"Failed to cache session in Redis: {e}")

        return session_id

    def get_session(
        self, 
        session_id: str, 
        db: DBSession
    ) -> Optional[SessionData]:
        """
        Get session from Redis (fast) or PostgreSQL (fallback).
        
        Args:
            session_id: UUID of the session
            db: SQLAlchemy database session
        
        Returns:
            SessionData object or None if not found
        """
        # Try Redis first (fast path)
        if self.redis_client:
            try:
                redis_key = self._get_redis_key(session_id)
                cached_data = self.redis_client.get(redis_key)
                
                if cached_data:
                    session_dict = json.loads(cached_data)
                    # Update last activity
                    session_dict['last_activity'] = datetime.utcnow().isoformat()
                    
                    # Refresh TTL in Redis
                    self.redis_client.expire(redis_key, self._get_expiry_seconds())
                    
                    logger.debug(f"Session {session_id} retrieved from Redis cache")
                    return SessionData(**session_dict)
            except Exception as e:
                logger.warning(f"Failed to retrieve session from Redis: {e}")
        
        # Fallback to PostgreSQL
        try:
            db_session = db.query(SessionModel).filter(
                SessionModel.session_id == session_id
            ).first()
            
            if db_session:
                session_data = SessionData(**db_session.data)
                
                # Restore to Redis cache if available
                if self.redis_client:
                    try:
                        redis_key = self._get_redis_key(session_id)
                        self.redis_client.setex(
                            redis_key,
                            self._get_expiry_seconds(),
                            json.dumps(session_data.model_dump(mode='json'), default=str)
                        )
                        logger.debug(f"Session {session_id} restored to Redis cache")
                    except Exception as e:
                        logger.warning(f"Failed to restore session to Redis: {e}")
                
                logger.debug(f"Session {session_id} retrieved from PostgreSQL")
                return session_data
            else:
                logger.warning(f"Session {session_id} not found in database")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve session from database: {e}")
            return None

    def update_session(
        self, 
        session_id: str, 
        session_data: SessionData, 
        db: DBSession
    ) -> bool:
        """
        Update session in both Redis and PostgreSQL.
        
        Args:
            session_id: UUID of the session
            session_data: Updated SessionData object
            db: SQLAlchemy database session
        
        Returns:
            True if successful, False otherwise
        """
        # Update timestamps
        session_data.updated_at = datetime.utcnow()
        session_data.last_activity = datetime.utcnow()
        
        # Update in PostgreSQL
        try:
            db_session = db.query(SessionModel).filter(
                SessionModel.session_id == session_id
            ).first()
            
            if db_session:
                db_session.data = session_data.model_dump(mode='json')
                db_session.updated_at = session_data.updated_at
                db.commit()
                logger.debug(f"Session {session_id} updated in PostgreSQL")
            else:
                logger.warning(f"Session {session_id} not found for update")
                return False
                
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update session in database: {e}")
            return False
        
        # Update in Redis if available
        if self.redis_client:
            try:
                redis_key = self._get_redis_key(session_id)
                self.redis_client.setex(
                    redis_key,
                    self._get_expiry_seconds(),
                    json.dumps(session_data.model_dump(mode='json'), default=str)
                )
                logger.debug(f"Session {session_id} updated in Redis")
            except Exception as e:
                logger.warning(f"Failed to update session in Redis: {e}")
        
        return True
    
    def delete_session(
        self, 
        session_id: str, 
        db: DBSession
    ) -> bool:
        """
        Delete session from both Redis and PostgreSQL.
        
        Args:
            session_id: UUID of the session
            db: SQLAlchemy database session
        
        Returns:
            True if successful, False otherwise
        """
        # Delete from Redis if available
        if self.redis_client:
            try:
                redis_key = self._get_redis_key(session_id)
                self.redis_client.delete(redis_key)
                logger.debug(f"Session {session_id} deleted from Redis")
            except Exception as e:
                logger.warning(f"Failed to delete session from Redis: {e}")
        
        # Delete from PostgreSQL
        try:
            db_session = db.query(SessionModel).filter(
                SessionModel.session_id == session_id
            ).first()
            
            if db_session:
                db.delete(db_session)
                db.commit()
                logger.info(f"Session {session_id} deleted from PostgreSQL")
                return True
            else:
                logger.warning(f"Session {session_id} not found for deletion")
                return False
                
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete session from database: {e}")
            return False
    
    def session_exists(
        self, 
        session_id: str, 
        db: DBSession
    ) -> bool:
        """
        Check if session exists in Redis or PostgreSQL.
        
        Args:
            session_id: UUID of the session
            db: SQLAlchemy database session
        
        Returns:
            True if session exists, False otherwise
        """
        # Check Redis first (fast)
        if self.redis_client:
            try:
                redis_key = self._get_redis_key(session_id)
                if self.redis_client.exists(redis_key):
                    return True
            except Exception as e:
                logger.warning(f"Failed to check Redis for session: {e}")
        
        # Check PostgreSQL
        try:
            exists = db.query(SessionModel).filter(
                SessionModel.session_id == session_id
            ).first() is not None
            return exists
        except Exception as e:
            logger.error(f"Failed to check database for session: {e}")
            return False
    
    def get_session_value(
        self, 
        session_id: str, 
        key: str, 
        db: DBSession,
        default: Any = None
    ) -> Any:
        """
        Get a specific value from session (like st.session_state.get()).
        
        Args:
            session_id: UUID of the session
            key: Key to retrieve from session metadata/settings
            db: SQLAlchemy database session
            default: Default value if key not found
        
        Returns:
            Value from session or default
        """
        session_data = self.get_session(session_id, db)
        if not session_data:
            return default
        
        # Check in settings first, then metadata
        if key in session_data.settings:
            return session_data.settings[key]
        elif key in session_data.metadata:
            return session_data.metadata[key]
        else:
            return default
    
    def set_session_value(
        self, 
        session_id: str, 
        key: str, 
        value: Any, 
        db: DBSession,
        in_settings: bool = True
    ) -> bool:
        """
        Set a specific value in session (like st.session_state[key] = value).
        
        Args:
            session_id: UUID of the session
            key: Key to set in session
            value: Value to store
            db: SQLAlchemy database session
            in_settings: If True, store in settings; otherwise in metadata
        
        Returns:
            True if successful, False otherwise
        """
        session_data = self.get_session(session_id, db)
        if not session_data:
            return False
        
        # Store in appropriate location
        if in_settings:
            session_data.settings[key] = value
        else:
            session_data.metadata[key] = value
        
        # Update session
        return self.update_session(session_id, session_data, db)
    
    def cleanup_expired_sessions(self, db: DBSession) -> int:
        """
        Clean up expired sessions from PostgreSQL.
        Redis automatically handles expiry via TTL.
        
        Args:
            db: SQLAlchemy database session
        
        Returns:
            Number of sessions deleted
        """
        try:
            now = datetime.utcnow()
            
            # Find and delete expired sessions
            expired_sessions = db.query(SessionModel).filter(
                SessionModel.updated_at < now - timedelta(minutes=settings.session_expire_minutes)
            ).all()
            
            count = len(expired_sessions)
            
            for session in expired_sessions:
                db.delete(session)
            
            db.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired sessions")
            
            return count
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def get_all_active_sessions(self, db: DBSession) -> list[str]:
        """
        Get list of all active session IDs.
        
        Args:
            db: SQLAlchemy database session
        
        Returns:
            List of session IDs
        """
        try:
            sessions = db.query(SessionModel.session_id).all()
            return [s.session_id for s in sessions]
        except Exception as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check health of Redis and verify it's accessible.
        
        Returns:
            Dict with redis and database status
        """
        status = {
            "redis": False,
            "using_cache": False
        }
        
        # Check Redis
        if self.redis_client:
            try:
                self.redis_client.ping()
                status["redis"] = True
                status["using_cache"] = True
            except Exception as e:
                logger.error(f"Redis health check failed: {e}")
        
        return status


# Singleton instance
_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    """
    Get or create the singleton SessionService instance.
    
    Returns:
        SessionService instance
    """
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service