#!/usr/bin/env python3
"""
Download or list bioRxiv/medRxiv PDFs.

- API: https://api.biorxiv.org/details/<server>/<from>/<to>/<cursor>
- Supports: --list-only (no MinIO needed) to print first N matches
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import logging
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import requests
from minio import Minio
from minio.error import S3Error
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_PREFIX = "papers/"
BIO_API = "https://api.biorxiv.org"
BIO_PDF_BASE = "https://www.biorxiv.org/content"
MED_PDF_BASE = "https://www.medrxiv.org/content"

# -------------------- logging --------------------
def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


# -------------------- model --------------------
@dataclass
class RxivEntry:
    doi: str
    title: str
    version: int
    server: str  # "biorxiv" or "medrxiv"
    updated: Optional[str] = None

    @property
    def paper_id(self) -> str:
        return self.doi.replace("/", "_")

    def pdf_url(self) -> str:
        base = BIO_PDF_BASE if self.server == "biorxiv" else MED_PDF_BASE
        return f"{base}/{self.doi}v{self.version}.full.pdf"


# -------------------- helpers --------------------
def build_query(keywords: List[str]) -> List[str]:
    return [k.strip() for k in keywords if k and k.strip()]

def match_all_terms(text: str, terms: List[str]) -> bool:
    t = text.lower()
    for term in terms:
        tt = term.strip().strip('"').lower()
        if tt not in t:
            return False
    return True

def date_range_windows(date_from: dt.date, date_to: dt.date, window_days: int) -> Iterable[Tuple[dt.date, dt.date]]:
    cur_start = date_from
    one_day = dt.timedelta(days=1)
    wdelta = dt.timedelta(days=window_days - 1)
    while cur_start <= date_to:
        cur_end = min(cur_start + wdelta, date_to)
        yield cur_start, cur_end
        cur_start = cur_end + one_day

def ensure_bucket(minio: Minio, bucket: str) -> None:
    try:
        minio.make_bucket(bucket)
        logging.info("Created bucket: %s", bucket)
    except S3Error as e:
        if e.code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            pass
        else:
            raise

def object_exists(minio: Minio, bucket: str, key: str) -> bool:
    try:
        minio.stat_object(bucket, key)
        return True
    except S3Error as e:
        if e.code in ("NoSuchKey", "NoSuchObject", "NoSuchBucket"):
            return False
        raise

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def download_pdf(pdf_url: str, retries: int = 3, backoff: float = 2.0) -> bytes:
    for attempt in range(1, retries + 1):
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

def make_session(user_agent: str, connect_timeout: float, read_timeout: float) -> tuple[requests.Session, tuple]:
    sess = requests.Session()
    sess.headers.update({"User-Agent": user_agent})
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504), respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess, (connect_timeout, read_timeout)

# -------------------- API --------------------
def rxiv_search(
    server: str,
    query_terms: Optional[List[str]],
    start_date: dt.date,
    end_date: dt.date,
    max_results: int,
    delay_between_calls: float,
    user_agent: str,
    window_days: int,
    max_pages_per_window: int,
    connect_timeout: float,
    read_timeout: float,
) -> List[RxivEntry]:
    sess, timeouts = make_session(user_agent, connect_timeout, read_timeout)
    results: List[RxivEntry] = []

    for win_from, win_to in date_range_windows(start_date, end_date, window_days):
        if len(results) >= max_results:
            break

        cursor = 0
        pages = 0
        consecutive_failures = 0

        while len(results) < max_results and pages < max_pages_per_window:
            url = f"{BIO_API}/details/{server}/{win_from.isoformat()}/{win_to.isoformat()}/{cursor}"
            logging.debug("Querying %s", url)
            try:
                resp = sess.get(url, timeout=timeouts)
                resp.raise_for_status()
                data = resp.json()
                consecutive_failures = 0
            except Exception as e:
                logging.warning("Details failed %s..%s cursor=%d: %s", win_from, win_to, cursor, e)
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    break
                time.sleep(2 * consecutive_failures)
                continue

            collection = data.get("collection", []) or []
            if not collection:
                break

            for it in collection:
                if len(results) >= max_results:
                    break

                doi = (it.get("doi") or "").strip()
                title = (it.get("title") or "").strip().replace("\n", " ")
                abstr = (it.get("abstract") or "").strip()
                version_str = str(it.get("version") or "1").strip()
                try:
                    version = int(version_str)
                except Exception:
                    version = 1
                updated = it.get("date") or it.get("updated") or None

                if not doi or not title:
                    continue

                if query_terms:
                    hay = f"{title}\n{abstr}".lower()
                    if not match_all_terms(hay, query_terms):
                        continue

                results.append(RxivEntry(doi=doi, title=title, version=version, server=server, updated=updated))

            pages += 1
            if len(collection) < 100:
                break
            cursor += 100
            if delay_between_calls > 0:
                time.sleep(delay_between_calls)

    return results[:max_results]

# -------------------- printers --------------------
def print_entries(entries: List[RxivEntry], as_json: bool) -> None:
    if as_json:
        import json
        payload = [
            {
                "doi": e.doi,
                "title": e.title,
                "version": e.version,
                "server": e.server,
                "updated": e.updated,
                "pdf_url": e.pdf_url(),
                "paper_id": e.paper_id,
            }
            for e in entries
        ]
        print(json.dumps({"count": len(entries), "items": payload}, indent=2))
        return

    print(f"Showing {len(entries)} result(s):\n")
    for i, e in enumerate(entries, 1):
        print(f"{i:02d}. {e.title}")
        print(f"    doi:   {e.doi}  (v{e.version}, {e.server})")
        if e.updated:
            print(f"    date:  {e.updated}")
        print(f"    pdf:   {e.pdf_url()}")
        print(f"    key:   {e.paper_id}")
        print()

# -------------------- pipeline --------------------
def run(
    server: str,
    keywords: List[str],
    date_from: Optional[str],
    date_to: Optional[str],
    bucket: Optional[str],
    minio_url: Optional[str],
    minio_access_key: Optional[str],
    minio_secret_key: Optional[str],
    minio_secure: bool,
    prefix: str,
    max_results: int,
    user_agent: str,
    force: bool,
    delay_between_batches: float,
    window_days: int,
    max_pages_per_window: int,
    connect_timeout: float,
    read_timeout: float,
    list_only: bool,
    print_json: bool,
    verbose: int,
) -> None:
    setup_logger(verbose)

    # Dates
    today = dt.date.today()
    default_from = today - dt.timedelta(days=365)
    start_date = dt.date.fromisoformat(date_from) if date_from else default_from
    end_date = dt.date.fromisoformat(date_to) if date_to else today
    if end_date < start_date:
        raise ValueError("--date-to must be on/after --date-from")

    query_terms = build_query(keywords) if keywords else None
    logging.info(
        "Server=%s | Query=%r | Date window: %s..%s | max_results=%d",
        server, query_terms, start_date, end_date, max_results
    )

    # Short-circuit: list only (no MinIO required)
    if list_only:
        entries = rxiv_search(
            server=server,
            query_terms=query_terms,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            delay_between_calls=delay_between_batches,
            user_agent=user_agent,
            window_days=window_days,
            max_pages_per_window=max_pages_per_window,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )
        print_entries(entries, as_json=print_json)
        return

    # Validate MinIO args when not listing
    required = {"--bucket": bucket, "--minio-url": minio_url, "--minio-access-key": minio_access_key, "--minio-secret-key": minio_secret_key}
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise SystemExit(f"Missing required args for download mode: {', '.join(missing)}")

    # MinIO client
    client = Minio(
        endpoint=minio_url.replace("http://", "").replace("https://", ""),
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=minio_secure,
    )
    ensure_bucket(client, bucket)

    # Fetch metadata
    entries = rxiv_search(
        server=server,
        query_terms=query_terms,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
        delay_between_calls=delay_between_batches,
        user_agent=user_agent,
        window_days=window_days,
        max_pages_per_window=max_pages_per_window,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )
    if not entries:
        logging.info("No results from %s for the given query/date range.", server)
        return

    # Download + upload PDFs
    processed = 0
    for entry in entries:
        key = f"{prefix}{entry.paper_id}.pdf"
        if not force and object_exists(client, bucket, key):
            logging.info("Skip existing: %s", key)
            continue

        pdf_url = entry.pdf_url()
        logging.info("Downloading: %s (%s v%d)", entry.title, entry.doi, entry.version)
        pdf_bytes = download_pdf(pdf_url)
        digest = sha256_bytes(pdf_bytes)

        logging.info("Uploading: %s (%s)", bucket, key)
        upload_pdf(client, bucket, key, pdf_bytes, sha256=digest)
        processed += 1

        if delay_between_batches > 0:
            time.sleep(delay_between_batches)

    logging.info("Done. Processed %d papers (requested max_results=%d).", processed, max_results)


# -------------------- CLI --------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List or download bioRxiv/medRxiv PDFs; optional MinIO upload.")
    p.add_argument("--server", choices=["biorxiv", "medrxiv"], default="biorxiv")
    p.add_argument("--keywords", nargs="*", default=[])
    p.add_argument("--date-from", help="Start date (YYYY-MM-DD). Default: 1 year ago.")
    p.add_argument("--date-to", help="End date (YYYY-MM-DD). Default: today.")
    # MinIO args are OPTIONAL here (only required if not --list-only)
    p.add_argument("--bucket", help="MinIO bucket name.")
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--minio-url", help="MinIO URL, e.g., http://localhost:9002")
    p.add_argument("--minio-access-key", help="MinIO access key")
    p.add_argument("--minio-secret-key", help="MinIO secret key")
    p.add_argument("--minio-secure", action="store_true", help="Use HTTPS to connect to MinIO")
    p.add_argument("--max-results", type=int, default=100)
    # compatibility with your arXiv CLI (not used here, but accepted)
    p.add_argument("--batch-size", type=int, default=100, help="Ignored for bioRxiv; accepted for CLI compatibility.")
    p.add_argument("--force", action="store_true", help="Overwrite PDFs if they already exist in MinIO")
    p.add_argument("--delay-between-batches", type=float, default=1.0)
    p.add_argument("-v", "--verbose", action="count", default=1)
    p.add_argument("--user-agent", default="piotr-rxiv-minio/0.1 (mailto:szefep85@gmail.com)")
    p.add_argument("--window-days", type=int, default=30, help="Days per date window (default: 30).")
    p.add_argument("--max-pages-per-window", type=int, default=5, help="Cap pages per window (100 items each).")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--read-timeout", type=float, default=20.0)
    # listing flags
    p.add_argument("--list-only", action="store_true", help="Only list results and exit (no MinIO needed).")
    p.add_argument("--print-json", "--print_json", dest="print_json", action="store_true",
                   help="When listing, print JSON instead of text.")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(
            server=args.server,
            keywords=args.keywords,
            date_from=args.date_from,
            date_to=args.date_to,
            bucket=args.bucket,
            minio_url=args.minio_url,
            minio_access_key=args.minio_access_key,
            minio_secret_key=args.minio_secret_key,
            minio_secure=args.minio_secure,
            prefix=args.prefix,
            max_results=args.max_results,
            user_agent=args.user_agent,
            force=args.force,
            delay_between_batches=args.delay_between_batches,
            window_days=args.window_days,
            max_pages_per_window=args.max_pages_per_window,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            list_only=args.list_only,
            print_json=args.print_json,
            verbose=args.verbose,
        )
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130
    except SystemExit as e:
        # bubble up our manual validation message
        raise
    except Exception as e:
        logging.exception("Failed: %s", e)
        return 1

if __name__ == "__main__":
    sys.exit(main())
