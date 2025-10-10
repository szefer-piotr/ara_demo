.PHONY: help test-connection init-db init-db-test reset-db start-services stop-services clean run dev install

# Default target
help:
	@echo "ARA Demo API - Makefile Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make start-services   - Start Docker services (PostgreSQL, Redis, MinIO)"
	@echo "  make stop-services    - Stop Docker services"
	@echo "  make test-connection  - Test connections to all services"
	@echo "  make init-db          - Initialize database (create tables)"
	@echo "  make init-db-test     - Initialize database with test data"
	@echo "  make reset-db         - Drop all tables and recreate with test data"
	@echo ""
	@echo "Run Commands:"
	@echo "  make run              - Run the API server"
	@echo "  make dev              - Run the API server in development mode (reload)"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  make clean            - Remove pycache and temporary files"
	@echo "  make clean-all        - Stop services and remove all data volumes"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Start Docker services
start-services:
	@echo "Starting Docker services..."
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@echo "Services started!"
	@echo "Run 'make test-connection' to verify"

# Stop Docker services
stop-services:
	@echo "Stopping Docker services..."
	docker-compose down
	@echo "Services stopped!"

# Test connections
test-connection:
	@echo "Testing service connections..."
	python scripts/test_connection.py

# Initialize database
init-db:
	@echo "Initializing database..."
	python scripts/init_database.py

# Initialize database with test data
init-db-test:
	@echo "Initializing database with test data..."
	python scripts/init_database.py --with-test-data

# Reset database (drop and recreate with test data)
reset-db:
	@echo "Resetting database..."
	python scripts/init_database.py --drop-all --with-test-data -y

# Run API server
run:
	@echo "Starting API server..."
	uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run API server in development mode
dev:
	@echo "Starting API server in development mode..."
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"

# Clean everything including Docker volumes
clean-all:
	@echo "Stopping services and removing volumes..."
	docker-compose down -v
	@make clean
	@echo "Complete cleanup done!"

# Complete setup workflow
setup: start-services test-connection init-db-test
	@echo ""
	@echo "=========================================="
	@echo "Setup complete!"
	@echo "=========================================="
	@echo ""
	@echo "Run 'make dev' to start the API server"
	@echo "Then visit: http://localhost:8000/docs"
	@echo ""

