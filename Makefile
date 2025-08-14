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
	--max-results 10 \
	--batch-size 10 \
	--user-agent "piotr-arxiv-minio/0.1 (mailto:szefer85@gmail.com)" \
	-vv

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