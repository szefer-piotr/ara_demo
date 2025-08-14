#! env/bin/activate python3

"""
Description:
Convert papers to LangChain Document objects and split into chunks using RecursiveCharacterTextSplitter.

Accept CLI parameters (e.g., Qdrant connection). Store chunks in Qdrant. 
Assign a unique paper_id (UUID) to each paper and sequential chunk IDs (0..N) per paper. 
Check if a paper already exists in Qdrant before processing.
Add --force option to reprocess and overwrite chunks (useful when chunking method changes).

Usage examples:
    # Process all PDFs under a prefix
    python chunk_arxiv_papers.py \
      --minio-endpoint localhost:9000 --minio-bucket research-papers --minio-secure false \
      --minio-access-key minioadmin --minio-secret-key minioadmin \
      --prefix arxiv/2025-08-10/ \
      --qdrant-url \
      --chunk-size 1200 --chunk-overlap 200

    # Process specific objects (comma-separated)
    python chunk_arxiv_papers.py \
      --objects arxiv/2508.01234.pdf,arxiv/2508.04567v2.pdf --force

"""

from __future__ import annotations
import argparse, os, re, uuid, tempfile, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Iterable, Optional
from dotenv import load_dotenv

from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    Filter, 
    FieldCondition, 
    MatchValue,
    PointStruct
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

try:
    import fitz
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    HAVE_PDFMINER = True
except:
    HAVE_PDFMINER = False


NAMESPACE_URL = uuid.NAMESPACE_URL
OPENAI_EMBED_MODEL = os.getenv('OPENAI_EMBED_MODEL')


def log_setup(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser("Minio PDFs to Qdrant embeddings")
    # MinIO
    p.add_argument("--minio-endpoint", required=True)
    p.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", ""))
    p.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", ""))
    p.add_argument("--minio-bucket", required=True)
    p.add_argument("--minio-secure", choices=["true", "false"], default=os.getenv("MINIO_SECURE", "false"))
    p.add_argument("--prefix", default="")
    p.add_argument("--objects", default="", help="Comma-separated object keys")
    p.add_argument("--limit", type=int, default=0)

    # Qdrant
    p.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", ""))
    p.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "arxiv_chunks"))
    
    # Embeddings
    p.add_argument("--openai-model", default=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    p.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY",  ""))

    # Chunking
    p.add_argument("--chunk-size", type=int, default=1200)
    p.add_argument("--chunk-overlap", type=int, default=200)

    # Behavior
    p.add_argument("--force", action="store_true", help="Re-ingest even if exists (deletes by filter first)")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return p.parse_args()


def stable_paper_id(bucket: str, object_name: str) -> str:
    return str(uuid.uuid5(NAMESPACE_URL, f"s3://{bucket}/{object_name}"))


def arxiv_id_from_name(name: str) -> Optional[str]:
    m = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", Path(name).name)
    return m.group(1) if m else None


def list_pdf_objects(m: Minio, bucket: str, prefix: str):
    """
    Yield ONLY real PDF object keys under prefix. Skip directory markers.
    """
    for obj in m.list_objects(bucket, prefix=prefix, recursive=True):
        name = obj.object_name
        # skip "folders" like 'arxiv/' and any non-pdf
        if name.endswith("/"):
            continue
        if not name.lower().endswith(".pdf"):
            continue
        yield name


def get_pdf_bytes(m: Minio, bucket: str, key: str) -> Tuple[bytes, str, int]:
    resp = m.get_object(bucket, key)
    try:
        pdf = resp.read()
        etag = (resp.headers.get("ETag") or "").strip('"')
        size = int(resp.headers.get("Content-Length") or len(pdf))
        return pdf, etag, size
    finally:
        resp.close(); resp.release_conn()


def extract_text(pdf_bytes: bytes) -> Tuple[str, int]:
    if HAVE_PYMUPDF:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [p.get_text("text") for p in doc]
        doc.close()
        return "\n".join(pages).strip(), len(pages)
    if HAVE_PDFMINER:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_bytes); tmp.flush()
            txt = pdfminer_extract_text(tmp.name) or ""
        return txt.strip(), -1
    raise RuntimeError("Install PyMuPDF of pdfminer.six to extract PDF text")


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def semantic_chunk_text(
        text: str,
        model: str = OPENAI_EMBED_MODEL if OPENAI_EMBED_MODEL is not None else "text-embedding-3-small",
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold: float = 95,
        buffer_size: int = 1,
        embeddings = None,
) -> List[str]:
    """
    Split text into semantically coherent chunks using embeddings.
    """
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model = model)
    
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold,
    )

    chunks = splitter.split_text(text)

    if chunks and hasattr(chunks[0], "page_content"):
        return [doc.page_content for doc in chunks]
    
    return chunks


def ensure_collection(client: QdrantClient, name: str, vector_size: int):
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            on_disk_payload=True,
        )
    else:
        pass


def embed_texts_openai(
        texts: List[str], 
        model: str, 
        api_key: str
) -> List[List[float]]:
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def upsert_points(
        q: QdrantClient,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[dict],
):
    points = [
        PointStruct(id=pid, vector=v, payload=pl) #type: ignore
        for pid, v, pl in zip(ids, vectors, payloads)
    ]
    q.upsert(collection_name=collection, points=points)


def delete_existing(q: QdrantClient, collection: str, paper_id: str):
    # Delete all previous chunks for this paper (by payload filter)
    q.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        ),
    )


def chunk_uuid(paper_uuid: str, i: int) -> str:
    return str(uuid.uuid5(uuid.UUID(paper_uuid), str(i)))


def safe_tokenizer_len(text: str, model: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)
    

# def enforce


def main():
    args = parse_args()
    log_setup(args.log_level)

    minio = Minio(
        endpoint=args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        secure=(args.minio_secure.lower() == "true")
    )

    qdrant = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)

    if args.objects:
        objects = [s.strip() for s in args.objects.split(",") if s.strip()]
    else:
        objects = list(list_pdf_objects(minio, args.minio_bucket, args.prefix))
    if args.limit > 0:
        objects = objects[: args.limit]

    logging.info("Found %d PDF object(s) to process", len(objects))

    first_vector_size: Optional[int] = None
    processed = skipped = failed = 0

    for key in objects:
        try:
            paper_id = stable_paper_id(args.minio_bucket, key)
            source = f"s3://{args.minio_bucket}/{key}"
            arxiv_id = arxiv_id_from_name(key)
            logging.info("-> %s (paper_id=%s, arxiv=%s)", source, paper_id, arxiv_id or "-")

            pdf_bytes, etag, size = get_pdf_bytes(minio, args.minio_bucket, key)
            text, pages = extract_text(pdf_bytes)
            if not text:
                logging.warning("No text extracted, skipping %s", key)
                skipped += 1
                continue

            chunks = chunk_text(
                text,
                size = args.chunk_size,
                overlap= args.chunk_overlap,
            )

            # chunks = semantic_chunk_text(
            #     text, 
            #     model= os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            #     breakpoint_threshold=95,
            #     buffer_size=1,
            # )

            if not chunks:
                logging.warning("No chunks produced, skipping %s", key)
                skipped += 1
                continue

            if args.force:
                delete_existing(qdrant, args.collection, paper_id)

            vectors = embed_texts_openai(chunks, model=args.openai_model, api_key=args.openai_api_key)

            if first_vector_size is None:
                first_vector_size = len(vectors[0])
                ensure_collection(qdrant, args.collection, first_vector_size)

            ids = [chunk_uuid(paper_id, i) for i in range(len(chunks))]
            payloads = []
            now = datetime.now(timezone.utc).isoformat()
            for i, txt in enumerate(chunks):
                payloads.append({
                    "paper_id": paper_id,
                    "chunk_id": i,
                    "source": source,
                    "bucket": args.minio_bucket,
                    "object": key,
                    "arxiv_id": arxiv_id,
                    "etag": etag,
                    "bytes": size,
                    "pages": pages,
                    "ingested_at": now,
                    "text": txt,                    
                })

            upsert_points(qdrant, args.collection, ids, vectors, payloads)
            processed += 1
            logging.info("+ Upserted %d chunks for %s", len(chunks), key)

        except Exception as e:
            failed += 1
            logging.exception("- Failed %s: %s", key, e)

    logging.info("Done. processed=%d, skipped=%d, failed=%d", processed, skipped, failed)


if __name__ == "__main__":
    main()

