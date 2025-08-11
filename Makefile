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

download_arxiv_papers:
	python download_arxiv_papers.py \
	--keywords ecology \
	--bucket research-papers \
	--minio-url http://localhost:9002 \
	--minio-access-key piotrminio \
	--minio-secret-key piotrminio \
	--max-results 10 \
	--batch-size 10 \
	--user-agent "piotr-arxiv-minio/0.1 (mailto:szefer85@gmail.com)" \
	-vv