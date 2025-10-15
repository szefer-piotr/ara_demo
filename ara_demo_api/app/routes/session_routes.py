from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import logging
from typing import Tuple

from app.api.dependencies import get_or_create_session

from app.database.connection import get_db
from app.database.models import Session as SessionModel
from app.models.api_models import ApiResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("", response_model=ApiResponse)
def create_session(
    session_data: Tuple[SessionModel, bool] = Depends(get_or_create_session)
):
    """
    Create a new session

    Headers:
    - X-Session-ID: Unique session identifier
    """

    session, is_new = session_data
    session_id = session.session_id

    if is_new:
        logger.info(f"Created new session: {session_id}")
        message = "Session created successfully"
        status = "created"

    else:
        message = "Session already exists"
        status = "existing"
    
    return ApiResponse(
        success=True,
        message=message,
        data={
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "status": status
        }
    )


@router.get("/{session_id}", response_model=ApiResponse)
def get_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get session information"""
    session = db.query(SessionModel).filter(
        SessionModel.session_id == session_id
    ).first()

    if not session:
        raise HTTPException(404, "Session not found")

    return ApiResponse(
        success=True,
        message="Session retrieved",
        data={
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "data": session.data
        }
    )


@router.delete("/{session_id}", response_model=ApiResponse)
def delete_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Delete a session"""
    session = db.query(SessionModel).filter(
        SessionModel.session_id == session_id
    ).first()

    if not session:
        raise HTTPException(404, "Session not found")

    db.delete(session)
    db.commit()

    logger.info(f"Delete session: {session_id}")

    return ApiResponse(
        success=True,
        message="Session deleted successfully"
    )