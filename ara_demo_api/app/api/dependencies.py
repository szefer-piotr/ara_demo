"""
Shared dependencies for API routes
"""
from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.database.models import Session as SessionModel
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def verify_session_id(
    x_session_id: str = Header(..., alias="X-Session-ID")
) -> str:
    """Extract and validate session ID from header"""
    if not x_session_id or len(x_session_id) < 8:
        raise HTTPException(400, "Invalid session ID format")
    return x_session_id


def get_or_create_session(
    session_id: str = Depends(verify_session_id),
    db: Session = Depends(get_db)
) -> SessionModel:
    """
    Get existing session or auto-create if doesn't exist.
    Useful for file upload endpoints that should auto-create sessions.
    """
    existing = db.query(SessionModel).filter(
        SessionModel.session_id == session_id
    ).first()

    if not existing:
        logger.info(f"Auto-creating session: {session_id}")
        new_session = SessionModel(
            session_id=session_id,
            data={"auto_created": True}
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        return new_session, True

    return existing, False


def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """
    Verify API key (optional for now, can be enforced later)
    """
    if x_api_key and x_api_key in settings.api_keys:
        return x_api_key
    
    # Allow requests without API keys for now, uncomment when done.
    # raise HTTPException(401, "Invalid or missing API key")

    return ""