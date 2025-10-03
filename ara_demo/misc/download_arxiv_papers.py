#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import feedparser
import requests
from minio import Minio
from minio.error import S3Error


ARXIV_API = "https://export.arxiv.org/api/query"
DEFAULT_PREFIX = "papers/"


def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )


@dataclass
class ArxivEntry:
    paper_id: str
    title: str
    pdf_url: str
    updated: Optional[str] = None


def build_query(keywords: List[str]) -> str:
    category_filter = "cat:q-bio.PE"
    keyword_terms = " AND ".join(
        f'all:"{k}"' if " " in k else f"all:{k}" for k in keywords
    )
    return f"{category_filter}" + (f" AND {keyword_terms}" if keyword_terms else "")


def parse_paper_id(entry_id: str) -> str:
    # entry ids are like: http://
    part = entry_id.rsplit("/", 1)[-1]
    return part


def extract_pdf_url(entry) -> str:
    # arXiv ATOM often has a link with rel='related' type ='application/pdf'
    pdf_link = None
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf":
            pdf_link = link.get("href")
            break
    if not pdf_link:
        entry_id = entry.get("id", "")
        pdf_link = entry_id.replace("/abs/", "/pdf/") + "pdf"
    return pdf_link


def _debug_preview_request(params: dict, headers: dict) -> None:
    req = requests.Request("GET", ARXIV_API, params=params, headers=headers).prepare()
    logging.info("User-Agent: %s", headers.get("User-Agent"))
    logging.info("URL: %s", req.url)


def arxiv_search(keywords: List[str], start: int, max_results: int, user_agent: str) -> Tuple[List[ArxivEntry], int]:
    query = build_query(keywords)
    # arXiv recommends setting a UA
    headers = {"User-Agent": user_agent}

    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    logging.debug("Querying arXiv: %s", params)
    _debug_preview_request(params, headers)
    resp = requests.get(ARXIV_API, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    total_results = int(feed.feed.get("opensearch_totalresults", 0))
    logging.info("Fetched %d entries (total ~%d)", len(feed.entries), total_results)

    entries: List[ArxivEntry] = []
    for e in feed.entries:
        paper_id = parse_paper_id(e.get("id", ""))
        pdf_url = extract_pdf_url(e)
        entries.append(
            ArxivEntry(
                paper_id=paper_id,
                title=e.get("title", "").strip().replace("\n", " "),
                pdf_url=pdf_url,
                updated=e.get("updated", None),
            )
        )
    return entries, total_results

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_exists(minio: Minio, bucket: str, key: str) -> bool:
    try:
        minio.stat_object(bucket, key)
        return True
    except S3Error as e:
        if e.code in ("NoSuchKey", "NoSuchObject", "NoSuchBucket"):
            return False
        raise


def ensure_bucket(minio: Minio, bucket: str) -> None:
    try:
        minio.make_bucket(bucket)
        logging.info("Created bucket: %s", bucket)
    except S3Error as e:
        if e.code == "BucketAlreadyOwnedByYou" or e.code == "BucketAlreadyExists":
            pass
        else:
            raise


def download_pdf(pdf_url: str, retries: int = 3, backoff: float = 2.0) -> bytes:
    for attempt in range(1, retries+1):
        try:
            with requests.get(pdf_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                buf = io.BytesIO()
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        buf.write(chunk)
                return buf.getvalue()
        except Exception as e:
            logging.warning("Download failed (attempt %d/%d): %s", attempt, retries, e)
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)
    raise RuntimeError("unreachable")


def upload_pdf(minio: Minio, bucket: str, key: str, content: bytes, sha256: str | None = None) -> None:
    data = io.BytesIO(content)
    length = len(content)
    metadata = {"sha256": sha256} if sha256 else None
    minio.put_object(bucket, key, data, length, content_type="application/pdf", metadata=metadata)


def run(
        keywords: List[str],
        bucket: str, 
        minio_url: str,
        minio_access_key: str,
        minio_secret_key: str,
        minio_secure: bool,
        prefix: str,
        max_results: int,
        batch_size: int,
        user_agent: str,
        force: bool,
        delay_between_batches: float,
) -> None:
    setup_logger(verbosity=1)
    client = Minio(
        endpoint=minio_url.replace("http://", "").replace("https://", ""),
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=minio_secure,
    )
    
    ensure_bucket(client, bucket)

    processed = 0
    start = 0
    total = None
    
    while total is None or processed < min(total, max_results):
        to_fetch = min(batch_size, max_results - processed)
        entries, total_found = arxiv_search(
            keywords, 
            start=start, 
            max_results=to_fetch,
            user_agent=user_agent)
        total = total_found if total is None else total
        if not entries:
            logging.info("No more entries returned by arXiv.")
            break

        for entry in entries:
            key = f"{prefix}{entry.paper_id}.pdf"
            if not force and object_exists(client, bucket, key):
                logging.info("Skip existing: %s", key)
                continue

            logging.info("Downloading: %s (%s)", entry.title, entry.paper_id)
            pdf_bytes = download_pdf(entry.pdf_url)
            digest = sha256_bytes(pdf_bytes)

            logging.info("Uploading: %s (%s)", bucket, key)
            upload_pdf(client, bucket, key, pdf_bytes, sha256=digest)

        processed += len(entries)
        start += len(entries)
        if len(entries) < to_fetch:
            # likely no more results
            break
        if delay_between_batches > 0:
            time.sleep(delay_between_batches)

    logging.info("Done. Processed ~%d papers (max_results=%d).", processed, max_results)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download arXiv PDFs and upload to MinIO.")
    p.add_argument(
        "--keywords",
        nargs="+",
        required=True,
        help="Search kewords (space means AND). Example: --keywords plant population ecology",
    )
    p.add_argument("--bucket", required=True, help="MinIO bucket name.")
    p.add_argument("--prefix", default=DEFAULT_PREFIX, help="Object key prefix in the bucket (default: papers/).")
    p.add_argument("--minio-url", required=True, help="MinIO URL, e.g., http://localhost:9002")
    p.add_argument("--minio-access-key", required=True, help="Minio access key")
    p.add_argument("--minio-secret-key", required=True, help="MinIO secret key")
    p.add_argument("--minio-secure", action="store_true", help="Use HTTPS to connect to MinIO (default: HTTP)")
    p.add_argument("--max-results", type=int, default=100, help="Max number of results to process (default: 100)")
    p.add_argument("--batch-size", type=int, default=50, help="arXiv page size per request (default: 50)")
    p.add_argument("--force", action="store_true", help="Overwrite PDFs i fthey already exist in MinIO")
    p.add_argument("--delay-between-batches", type=float, default=1.0, help="Delay (seconds) to be nice to arXiv API")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase logging verbosity (-v, -vv)")
    p.add_argument(
        "--user-agent",
        default="piotr-arxiv-minio/0.1 (mailto:szefer85@gmail.com)",
        help="Custom User-Agent string for arXiv API requests"
            '(format: "AppName/Version (mailto:email@example.com)")'
    )
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logger(args.verbose)
    try:
        run(
            keywords=args.keywords,
            bucket=args.bucket,
            minio_url=args.minio_url,
            minio_access_key=args.minio_access_key,
            minio_secret_key=args.minio_secret_key,
            minio_secure=args.minio_secure,
            prefix=args.prefix,
            max_results=args.max_results,
            batch_size=args.batch_size,
            force=args.force,
            delay_between_batches=args.delay_between_batches,
            user_agent=args.user_agent,
        )
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130
    except Exception as e:
        logging.exception("Failed: %s", e)
        return 1
    

if __name__ == "__main__":
    sys.exit(main())


