"""
Example usage of async database connection with FastAPI

This file demonstrates how to use the async database session
in your FastAPI endpoints.
"""

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.database.models import Session, File, Hypothesis


# Example 1: Simple query
async def get_all_sessions(db: AsyncSession = Depends(get_db)):
    """Get all sessions from database"""
    result = await db.execute(select(Session))
    sessions = result.scalars().all()
    return sessions


# Example 2: Query with filter
async def get_session_by_id(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get a specific session by ID"""
    result = await db.execute(
        select(Session).where(Session.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    return session


# Example 3: Create new record
async def create_session(session_id: str, data: dict, db: AsyncSession = Depends(get_db)):
    """Create a new session"""
    new_session = Session(
        session_id=session_id,
        data=data
    )
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)
    return new_session


# Example 4: Update record
async def update_session(session_id: str, new_data: dict, db: AsyncSession = Depends(get_db)):
    """Update an existing session"""
    result = await db.execute(
        select(Session).where(Session.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if session:
        session.data = new_data
        await db.commit()
        await db.refresh(session)
    
    return session


# Example 5: Delete record
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a session"""
    result = await db.execute(
        select(Session).where(Session.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if session:
        await db.delete(session)
        await db.commit()
        return True
    
    return False


# Example 6: Query with relationships
async def get_session_with_files(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get session with all related files"""
    result = await db.execute(
        select(Session)
        .where(Session.session_id == session_id)
        .options(selectinload(Session.files))
    )
    session = result.scalar_one_or_none()
    return session


# Example 7: Complex query with joins
async def get_hypotheses_with_plans(db: AsyncSession = Depends(get_db)):
    """Get all hypotheses with their analysis plans"""
    result = await db.execute(
        select(Hypothesis)
        .options(selectinload(Hypothesis.analysis_plans))
    )
    hypotheses = result.scalars().all()
    return hypotheses


# Example 8: Bulk operations
async def create_multiple_files(files_data: list, db: AsyncSession = Depends(get_db)):
    """Create multiple files at once"""
    files = [File(**data) for data in files_data]
    db.add_all(files)
    await db.commit()
    return files


# Example 9: Transaction with error handling
async def transfer_data_with_transaction(
    source_id: str,
    target_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Example of transaction with rollback on error"""
    try:
        # Get source session
        result = await db.execute(
            select(Session).where(Session.session_id == source_id)
        )
        source = result.scalar_one_or_none()
        
        # Get target session
        result = await db.execute(
            select(Session).where(Session.session_id == target_id)
        )
        target = result.scalar_one_or_none()
        
        if not source or not target:
            raise ValueError("Source or target session not found")
        
        # Perform operations
        target.data.update(source.data)
        
        # Commit will be automatic if no exception
        await db.commit()
        
        return True
        
    except Exception as e:
        # Rollback is automatic in get_db() on exception
        raise


# Example 10: Raw SQL query
async def execute_raw_query(db: AsyncSession = Depends(get_db)):
    """Execute a raw SQL query"""
    from sqlalchemy import text
    
    result = await db.execute(
        text("SELECT session_id, created_at FROM sessions WHERE data IS NOT NULL")
    )
    rows = result.fetchall()
    return rows

