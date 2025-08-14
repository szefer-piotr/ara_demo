#!/usr/bin/env python3
"""
CLI tool to run semantic similarity search over arXiv chunks stored in Qdrant.

A Qdrant collection already has embedded chunk vectors based on recusive chraracter text splitting
Payload for each point include keys:
- "paper_id": paper_id,
- "chunk_id": i,
- source": source,
- "bucket": args.minio_bucket,
- "object": key,
- "arxiv_id": arxiv_id,
- "etag": etag,
- "bytes": size,
- "pages": pages,
- "ingested_at": now,
- "text": txt,     

Features
- Reads config from CLI and/or environment
- Uses OpenAI embeddings by default
- Pretty output (table) or JSON
- Optional metadata filters
- Safe defaults & robust errorn handling.

Usage
$ python3 similarity_search_arxiv_papers.py \
    --query "population ecology" \
    --qdrant-url http://localhost:6333 \
    --collection  arxiv_chunks \
    --top-k 5

Eith environment:
    OPENAI_API_KEY QDRANT_URL QDRANT_API_KEY

Advanced:
$ python3 similarity_search_arxiv_papers.py \

"""