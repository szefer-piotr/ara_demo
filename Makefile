.PHONY: minio stop-minio qdrant stop-qdrant

minio:
	cd minio && docker compose up -d

stop-minio:
	cd minio && docker compose down

upload-chunks:
	python rag_agent/file_uploader.py

clean-minio:
	sudo rm -rf minio/minio_data/*

qdrant:
	cd qdrant && docker compose up -d

stop-qdrant:
	cd qdrant && docker compose down