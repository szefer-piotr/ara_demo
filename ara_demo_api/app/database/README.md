# Database Module

This module provides async database connectivity using SQLAlchemy and AsyncPG for the ARA Demo API.

## Overview

The database module consists of:
- **models.py** - SQLAlchemy ORM models
- **connection.py** - Async database connection and session management
- **example_usage.py** - Usage examples for reference

## Setup

### Database URL

The database connection is configured via settings in `app/config.py`:

```python
DATABASE_URL = "postgresql://user:password@host:port/database"
```

The connection module automatically converts this to use AsyncPG:
```python
postgresql+asyncpg://user:password@host:port/database
```

### Initialization

The database is initialized automatically on application startup in `app/main.py`:

```python
from app.database import init_db, check_database_connection, close_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    if await check_database_connection():
        logger.info("Database connected")
    
    yield
    
    # Shutdown
    await close_db()
```

## Usage in Endpoints

### Basic Query

```python
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.database.models import Session

@app.get("/sessions/")
async def get_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Session))
    sessions = result.scalars().all()
    return sessions
```

### Create Record

```python
@app.post("/sessions/")
async def create_session(
    session_data: dict,
    db: AsyncSession = Depends(get_db)
):
    new_session = Session(
        session_id=str(uuid.uuid4()),
        data=session_data
    )
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)
    return new_session
```

### Update Record

```python
@app.put("/sessions/{session_id}")
async def update_session(
    session_id: str,
    new_data: dict,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Session).where(Session.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.data = new_data
    await db.commit()
    await db.refresh(session)
    return session
```

### Delete Record

```python
@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Session).where(Session.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    await db.delete(session)
    await db.commit()
    return {"message": "Session deleted"}
```

## Database Models

### Session
Stores user session data.

Fields:
- `id` (int) - Primary key
- `session_id` (str) - Unique identifier
- `data` (JSON) - Session data
- `created_at`, `updated_at` (datetime)

### File
Stores uploaded file metadata.

Fields:
- `id` (int) - Primary key
- `file_id` (str) - Unique identifier
- `session_id` (str) - Foreign key to sessions
- `filename` (str) - Original filename
- `storage_path` (str) - Path to stored file
- `file_size`, `file_type` (int, str)
- `metadata` (JSON)
- `created_at` (datetime)

### Hypothesis
Stores research hypotheses.

Fields:
- `id` (int) - Primary key
- `hypothesis_id` (str) - Unique identifier
- `session_id` (str) - Foreign key to sessions
- `title` (str)
- `description` (text)
- `data` (JSON)
- `status` (str)
- `confidence_level` (int)
- `created_at`, `updated_at` (datetime)

### AnalysisPlan
Stores generated analysis plans.

Fields:
- `id` (int) - Primary key
- `plan_id` (str) - Unique identifier
- `hypothesis_id` (str) - Foreign key to hypotheses
- `plan_data` (JSON)
- `accepted` (bool)
- `status` (str)
- `total_steps`, `completed_steps` (int)
- `created_at`, `updated_at` (datetime)

### ApiKey
Stores API authentication keys.

Fields:
- `id` (int) - Primary key
- `key_id` (str) - Unique identifier
- `key_hash` (str) - Hashed key
- `description` (text)
- `active` (bool)
- `created_at`, `last_used_at`, `expires_at` (datetime)
- `usage_count` (int)
- `metadata` (JSON)

### StepRun
Stores execution runs for analysis steps.

Fields:
- `id` (int) - Primary key
- `run_id` (str) - Unique identifier
- `plan_id` (str) - Foreign key to analysis_plans
- `step_id` (str)
- `step_number` (int)
- `status` (str)
- `started_at`, `completed_at` (datetime)
- `duration_seconds` (int)
- `input_data`, `output_data`, `results` (JSON)
- `error_message` (text)
- `created_at` (datetime)

### Report
Stores generated reports.

Fields:
- `id` (int) - Primary key
- `report_id` (str) - Unique identifier
- `session_id` (str) - Foreign key to sessions
- `report_type` (str)
- `title` (str)
- `content` (text)
- `format` (str)
- `file_path` (text)
- `citation_count` (int)
- `metadata` (JSON)
- `created_at` (datetime)

## Advanced Usage

### Relationships

```python
from sqlalchemy.orm import selectinload

# Load session with related files
result = await db.execute(
    select(Session)
    .where(Session.session_id == session_id)
    .options(selectinload(Session.files))
)
session = result.scalar_one_or_none()

# Access related data
for file in session.files:
    print(file.filename)
```

### Raw SQL

```python
from sqlalchemy import text

result = await db.execute(
    text("SELECT * FROM sessions WHERE created_at > :date"),
    {"date": "2024-01-01"}
)
rows = result.fetchall()
```

### Transaction Management

The `get_db()` dependency automatically handles transactions:
- Commits on success
- Rolls back on exception
- Closes session when done

For manual control:

```python
async with db.begin():
    # Your operations here
    # Auto-commit at end of block
    pass
```

## Connection Pool

The async engine uses a connection pool with these settings:
- `pool_size=5` - Number of connections to maintain
- `max_overflow=10` - Additional connections allowed
- `pool_pre_ping=True` - Verify connections before use

## Best Practices

1. **Always use `Depends(get_db)`** in your endpoints
2. **Don't call `db.commit()` multiple times** - once per request is sufficient
3. **Use `scalar_one_or_none()`** when expecting 0 or 1 result
4. **Use `scalars().all()`** when expecting multiple results
5. **Eager load relationships** when needed to avoid N+1 queries
6. **Handle exceptions** - the dependency will auto-rollback
7. **Use indexes** on frequently queried columns
8. **Keep JSON fields reasonable** in size

## Troubleshooting

### Connection Issues

If you see connection errors:
1. Check DATABASE_URL in settings
2. Verify PostgreSQL is running
3. Check firewall/network settings
4. Verify credentials

### AsyncPG Import Error

Make sure you have installed:
```bash
pip install asyncpg
```

### Table Creation

Tables are created automatically on startup. To recreate:
```python
async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.drop_all)
    await conn.run_sync(Base.metadata.create_all)
```

## Migration

For production, use Alembic for database migrations:

```bash
alembic init alembic
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

## Testing

For testing, you can create a separate test database:

```python
TEST_DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/test_db"

test_engine = create_async_engine(TEST_DATABASE_URL)
TestSessionLocal = async_sessionmaker(test_engine, class_=AsyncSession)

async def override_get_db():
    async with TestSessionLocal() as session:
        yield session

app.dependency_overrides[get_db] = override_get_db
```

