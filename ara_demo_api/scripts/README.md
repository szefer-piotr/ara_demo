# Database Setup Scripts

This directory contains utility scripts for database initialization and testing.

## Scripts

### 1. test_connection.py

**Purpose**: Quick connection test for all services (PostgreSQL, Redis, MinIO)

**Usage**:
```bash
# Test all connections
python scripts/test_connection.py

# Or make it executable and run directly
./scripts/test_connection.py
```

**What it does**:
- Tests PostgreSQL connection and displays version
- Tests Redis connection and displays version
- Tests MinIO connection and lists buckets
- Provides a summary of all connection statuses

**When to use**:
- Before initializing the database
- To verify your configuration
- To troubleshoot connection issues
- After starting docker-compose services

---

### 2. init_database.py

**Purpose**: Initialize the database schema and optionally create test data

**Usage**:
```bash
# Basic initialization (creates tables)
python scripts/init_database.py

# Initialize with test data
python scripts/init_database.py --with-test-data

# Drop all tables and reinitialize (DANGEROUS!)
python scripts/init_database.py --drop-all

# Drop and reinitialize with test data (skip confirmation)
python scripts/init_database.py --drop-all --with-test-data -y
```

**Options**:
- `--with-test-data`: Create sample data after initialization
- `--drop-all`: Drop all existing tables before initialization
- `-y, --yes`: Skip confirmation prompts

**What it does**:
- Creates database engine with connection pool
- Creates all tables defined in models
- Verifies table creation
- Optionally creates test data:
  - Test session
  - Test file
  - Test hypothesis
  - Test analysis plan
  - Test API key

**Test Data Created**:
When using `--with-test-data`, the script creates:
- 1 Session with ID like `test-session-abc12345`
- 1 File with ID like `test-file-abc12345`
- 1 Hypothesis with ID like `test-hyp-abc12345`
- 1 Analysis Plan with ID like `test-plan-abc12345`
- 1 API Key: `test-key-12345`

---

## Quick Start

### Step 1: Start Services

Make sure Docker services are running:

```bash
cd ara_demo_api
docker-compose up -d
```

Verify services are up:
```bash
docker-compose ps
```

### Step 2: Test Connections

```bash
python scripts/test_connection.py
```

Expected output:
```
============================================================
🚀 ARA Demo API - Connection Test
============================================================

📋 Configuration:
   Database: localhost:5433/ara_demo
   Redis: localhost:6380
   MinIO: localhost:9002

🔍 Testing PostgreSQL connection...
✅ PostgreSQL connection successful!
   Version: PostgreSQL 15.x
   Database: ara_demo
   User: ara_user

🔍 Testing Redis connection...
✅ Redis connection successful!
   Version: 7.x.x

🔍 Testing MinIO connection...
✅ MinIO connection successful!
   Buckets: 0
   Bucket 'ara_demo' does not exist yet

============================================================
📊 Connection Test Summary
============================================================
✅ Database: Connected
✅ Redis: Connected
✅ Minio: Connected

🎉 All connections successful!
```

### Step 3: Initialize Database

```bash
# Create tables only
python scripts/init_database.py

# Or create tables with test data
python scripts/init_database.py --with-test-data
```

Expected output:
```
============================================================
🚀 ARA Demo API - Database Initialization
============================================================

📋 Configuration:
   Database: ara_demo
   Host: localhost
   Port: 5433
   User: ara_user

🔧 Initializing database...
✅ Database initialized successfully!

🔌 Checking database connection...
✅ Database connection verified!

🔍 Verifying tables...
📊 Found 7 tables:
   ✅ analysis_plans
   ✅ api_keys
   ✅ files
   ✅ hypotheses
   ✅ reports
   ✅ sessions
   ✅ step_runs
✅ All expected tables found!

📝 Creating test data...
✅ Test data created successfully:
   - Session ID: test-session-abc12345
   - File ID: test-file-abc12345
   - Hypothesis ID: test-hyp-abc12345
   - Plan ID: test-plan-abc12345
   - API Key: test-key-12345

============================================================
🎉 Database initialization completed successfully!
============================================================

Next steps:
  1. Start the API server: uvicorn app.main:app --reload
  2. Visit the docs: http://localhost:8000/docs
```

### Step 4: Start API Server

```bash
cd ara_demo_api
uvicorn app.main:app --reload
```

Or using the main module:
```bash
python -m app.main
```

---

## Troubleshooting

### Connection Refused

**Error**: `Connection refused` or `could not connect to server`

**Solution**:
1. Check if PostgreSQL is running:
   ```bash
   docker-compose ps postgres
   ```
2. Check if the port matches your configuration (default: 5433)
3. Verify database credentials in `.env` or `config.py`

### Authentication Failed

**Error**: `password authentication failed`

**Solution**:
1. Check `POSTGRES_PASSWORD` in docker-compose and config match
2. Restart PostgreSQL container:
   ```bash
   docker-compose restart postgres
   ```

### Database Does Not Exist

**Error**: `database "ara_demo" does not exist`

**Solution**:
1. Check `POSTGRES_DB` in docker-compose.yaml
2. Recreate the database:
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

### Tables Already Exist

**Error**: `table "sessions" already exists`

**Solution**:
Use the `--drop-all` flag to recreate tables:
```bash
python scripts/init_database.py --drop-all -y
```

**Warning**: This will delete all data!

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'app'`

**Solution**:
1. Make sure you're running from the `ara_demo_api` directory
2. Or run with Python module syntax:
   ```bash
   python -m scripts.init_database
   ```

---

## Development Workflow

### Clean Start

```bash
# Stop all services
docker-compose down -v

# Start services
docker-compose up -d

# Wait for services to be ready (5-10 seconds)
sleep 10

# Test connections
python scripts/test_connection.py

# Initialize database with test data
python scripts/init_database.py --with-test-data

# Start API server
uvicorn app.main:app --reload
```

### Reset Database

```bash
# Drop all tables and recreate with test data
python scripts/init_database.py --drop-all --with-test-data -y
```

### Check Current State

```bash
# Test if services are accessible
python scripts/test_connection.py

# Connect to database and check tables
docker-compose exec postgres psql -U ara_user -d ara_demo -c "\dt"
```

---

## Environment Variables

Make sure these are set in your `.env` file or environment:

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=ara_user
POSTGRES_PASSWORD=changeme123
POSTGRES_DB=ara_demo

# Redis
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_PASSWORD=myredissecret

# MinIO
MINIO_HOST=localhost
MINIO_PORT=9002
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET_NAME=ara_demo
```

---

## Next Steps

After successful initialization:

1. **Start the API server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **View API documentation**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
   - OpenAPI JSON: http://localhost:8000/openapi.json

3. **Test the API**:
   ```bash
   curl http://localhost:8000/
   curl http://localhost:8000/health
   ```

4. **Query test data** (if created):
   ```bash
   # Via psql
   docker-compose exec postgres psql -U ara_user -d ara_demo
   SELECT * FROM sessions;
   ```

---

## Additional Resources

- [Database Models Documentation](../app/database/README.md)
- [API Documentation](../README.md)
- [Configuration Guide](../app/config.py)

