.PHONY: minio stop-minio qdrant stop-qdrant

minio:
	cd minio && docker compose up -d
	@echo "Waiting for MinIO HTTP endpoint..."
	@until curl -s -o /dev/null http://localhost:9002; do \
		sleep 1; \
		echo "Still waiting..."; \
	done
	@echo "✅ MinIO is up!"

stop-minio:
	cd minio && docker compose down

upload-chunks:
	python rag_agent/file_uploader.py

clean-minio:
	sudo rm -rf minio/minio_data/*

qdrant:
	cd qdrant && docker compose up -d
	@echo "Waiting for Qdrant to be ready on port 6333..."
	@until curl -s -o /dev/null http://localhost:6333/healthz; do \
		sleep 1; \
		echo "Still waiting for Qdrant..."; \
	done
	@echo "✅ Qdrant is ready at http://localhost:6333"

stop-qdrant:
	cd qdrant && docker compose down

download_arxiv_papers:
	python3 download_arxiv_papers.py \
	--keywords ecology \
	--bucket research-papers \
	--minio-url http://localhost:9002 \
	--minio-access-key piotrminio \
	--minio-secret-key piotrminio \
	--max-results 50 \
	--batch-size 10 \
	--user-agent "piotr-arxiv-minio/0.1 (mailto:szefer85@gmail.com)" \
	-vv

biorxiv:
	python3 src/biorxiv-upload-papers.py \

chunk_arxiv_papers:
	python3 chunk_arxiv_papers.py \
    --minio-endpoint localhost:9002 \
	--minio-bucket research-papers \
	--minio-secure false \
    --prefix papers/ \
    --qdrant-url http://localhost:6333 \
	--collection arxiv_chunks \
    --openai-model text-embedding-3-large \
    --chunk-size 1200 \
	--chunk-overlap 200 \
	--force

# Testing targets
.PHONY: test test-unit test-integration test-coverage test-html test-clean install-test-deps

# Install testing dependencies
install-test-deps:
	pip install -r tests/requirements-test.txt

# Run all tests
test: install-test-deps
	python -m pytest tests/ -v

# Run unit tests only
test-unit: install-test-deps
	python -m pytest tests/ -m "unit" -v

# Run integration tests only
test-integration: install-test-deps
	python -m pytest tests/ -m "integration" -v

# Run tests with coverage
test-coverage: install-test-deps
	python -m pytest tests/ --cov=. --cov-report=html:htmlcov --cov-report=term-missing

# Run tests and generate HTML report
test-html: install-test-deps
	python -m pytest tests/ --html=test-report.html --self-contained-html

# Run tests in parallel
test-parallel: install-test-deps
	python -m pytest tests/ -n auto

# Run specific test categories
test-download: install-test-deps
	python -m pytest tests/ -m "download" -v

test-chunking: install-test-deps
	python -m pytest tests/ -m "chunking" -v

test-evaluation: install-test-deps
	python -m pytest tests/ -m "evaluation" -v

test-search: install-test-deps
	python -m pytest tests/ -m "search" -v

# Run example user query test
test-example: install-test-deps
	python -m pytest tests/test_example_user_query.py -v

# Clean test artifacts
test-clean:
	rm -rf htmlcov/
	rm -f .coverage
	rm -f test-report.html
	rm -f junit.xml
	rm -rf __pycache__/
	rm -rf tests/__pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Quick test run (fastest)
test-quick: install-test-deps
	python -m pytest tests/ -x --tb=short

# Debug test run
test-debug: install-test-deps
	python -m pytest tests/ -v -s --tb=long

# Performance test run
test-performance: install-test-deps
	python -m pytest tests/ -m "slow" --durations=10