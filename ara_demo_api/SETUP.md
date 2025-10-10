# ARA Demo API - Setup Guide

Complete setup guide for the ARA Demo API backend.

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

## Quick Start

### 1. Install Dependencies

```bash
cd ara_demo_api
pip install -r requirements.txt
```

Or use the Makefile:
```bash
make install
```

### 2. Start Docker Services

```bash
# Start PostgreSQL, Redis, and MinIO
docker-compose up -d

# Check services are running
docker-compose ps
```

Or use the Makefile:
```bash
make start-services
```

### 3. Test Connections

```bash
# Test all service connections
python3 scripts/test_connection.py
```

Or use the Makefile:
```bash
make test-connection
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

============================================================
📊 Connection Test Summary
============================================================
✅ Database: Connected
✅ Redis: Connected
✅ Minio: Connected

🎉 All connections successful!
```

### 4. Initialize Database

```bash
# Create tables only
python3 scripts/init_database.py

# Or create tables WITH test data
python3 scripts/init_database.py --with-test-data
```

Or use the Makefile:
```bash
make init-db          # Without test data
make init-db-test     # With test data
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

============================================================
🎉 Database initialization completed successfully!
============================================================
```

### 5. Start API Server

```bash
# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Development mode (with auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or use the Makefile:
```bash
make run    # Production
make dev    # Development (auto-reload)
```

### 6. Access API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

Test the API:
```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

## Complete Setup (One Command)

```bash
make setup
```

This will:
1. Start Docker services
2. Test connections
3. Initialize database with test data
4. Show next steps

## Makefile Commands

All available commands:

```bash
# Setup
make install          # Install Python dependencies
make start-services   # Start Docker services
make stop-services    # Stop Docker services
make test-connection  # Test service connections
make init-db          # Initialize database
make init-db-test     # Initialize with test data
make reset-db         # Drop all tables and recreate
make setup            # Complete setup workflow

# Run
make run              # Start API server
make dev              # Start API server (development mode)

# Maintenance
make clean            # Remove __pycache__ and temp files
make clean-all        # Stop services and remove volumes
```

## Project Structure

```
ara_demo_api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── api/                 # API routes
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── dependencies.py
│   ├── database/            # Database layer
│   │   ├── __init__.py
│   │   ├── models.py        # SQLAlchemy models
│   │   ├── connection.py    # Async connection management
│   │   ├── example_usage.py
│   │   └── README.md
│   ├── models/              # Pydantic models
│   │   ├── __init__.py
│   │   ├── schemas.py       # Business logic schemas
│   │   └── api_models.py    # API request/response models
│   ├── services/            # Business logic
│   │   ├── __init__.py
│   │   ├── llm_service.py
│   │   └── analysis_service.py
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── llm_roles.py
│       └── prompt_templates.py
├── scripts/                 # Setup and utility scripts
│   ├── __init__.py
│   ├── test_connection.py   # Connection test script
│   ├── init_database.py     # Database initialization script
│   └── README.md
├── docker-compose.yaml      # Docker services configuration
├── requirements.txt         # Python dependencies
├── Makefile                 # Makefile commands
├── SETUP.md                 # This file
└── README.md                # Project documentation
```

## Configuration

### Environment Variables

Create a `.env` file in the `ara_demo_api` directory:

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
MINIO_CONSOLE_PORT=9003
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET_NAME=ara-demo-bucket
MINIO_USE_SSL=false

# API
APP_NAME=ARA Demo API
API_VERSION=1.0.0
DEBUG=true

# Security
SECRET_KEY=your-secret-key-change-in-production
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# OpenAI (optional)
OPENAI_API_KEY=your-openai-api-key
```

### Docker Services Ports

Default ports used:
- **PostgreSQL**: 5433 (mapped from 5432)
- **Redis**: 6380 (mapped from 6379)
- **MinIO API**: 9002 (mapped from 9000)
- **MinIO Console**: 9003 (mapped from 9001)
- **API Server**: 8000

## Troubleshooting

### Services Won't Start

```bash
# Check what's using the ports
netstat -tuln | grep -E '5433|6380|9002|9003'

# Stop existing containers
docker-compose down

# Remove volumes and restart
docker-compose down -v
docker-compose up -d
```

### Connection Test Fails

1. **Database connection failed**:
   ```bash
   # Check PostgreSQL logs
   docker-compose logs postgres
   
   # Verify PostgreSQL is running
   docker-compose ps postgres
   
   # Test connection manually
   docker-compose exec postgres psql -U ara_user -d ara_demo
   ```

2. **Redis connection failed**:
   ```bash
   # Check Redis logs
   docker-compose logs redis
   
   # Test connection manually
   docker-compose exec redis redis-cli -a myredissecret ping
   ```

3. **MinIO connection failed**:
   ```bash
   # Check MinIO logs
   docker-compose logs minio
   
   # Visit MinIO console
   # http://localhost:9003
   ```

### Database Initialization Fails

```bash
# Reset everything
make reset-db

# Or manually
python3 scripts/init_database.py --drop-all --with-test-data -y
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or
make install
```

### Permission Denied

```bash
# Make scripts executable
chmod +x scripts/*.py
```

## Development Workflow

### Daily Development

```bash
# Start services
make start-services

# Start API in development mode
make dev
```

### Reset Everything

```bash
# Clean restart
make clean-all
make setup
make dev
```

### Database Changes

```bash
# After changing models, reset database
make reset-db
```

## Testing

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/health

# Get API info
curl http://localhost:8000/info

# Use HTTPie (if installed)
http localhost:8000/health
```

### Database Testing

```bash
# Connect to database
docker-compose exec postgres psql -U ara_user -d ara_demo

# List tables
\dt

# Query sessions
SELECT * FROM sessions;

# Exit
\q
```

## Next Steps

After successful setup:

1. **Read the documentation**:
   - [Database Documentation](app/database/README.md)
   - [Scripts Documentation](scripts/README.md)

2. **Explore the API**:
   - Visit http://localhost:8000/docs
   - Try the example endpoints

3. **Start development**:
   - Create new routes in `app/api/routes.py`
   - Add business logic in `app/services/`
   - Update models as needed

4. **Frontend integration**:
   - API runs on http://localhost:8000
   - CORS configured for http://localhost:3000

## Support

For issues and questions:
- Check logs: `docker-compose logs`
- Review documentation in `app/database/README.md`
- Check scripts documentation in `scripts/README.md`

Happy coding! 🚀

