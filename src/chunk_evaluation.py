#!/usr/bin/env python3
"""
Dump all chunks from a Qdrant collection with simple quality metrics.

Examples:
# Prosess everything to CSV
python3 chunk_evaluation.py -c arxiv_cunks --out chunks_quality.csv

# Only first 2000 points, larger batch size, preview to stdout
python chunk_evaluation.py -c arxiv_chunks --max 2000 --batch 2048 --preview 20

# Custom text field(s) to read from payload
python3 chunk_evaluation.py -c arxiv_chunks --fields page_content text
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

def extract_text(payload: Dict[str, Any], fields: List[str]) -> str:
    """Return first non-empty text across candidate payload fields."""
    for f in fields:
        if f in payload and payload[f]:
            val = payload[f]
            if isinstance(val, str):
                return val
            if isinstance(val, list):
                parts = [str(x) for x in val if isinstance(x, (str, int, float))]
                if parts:
                    return "\n".join(parts)
            try:
                s = str(val)
                if s.strip():
                    return s
            except Exception:
                pass
    return ""


def longest_run_len(text: str) -> int:
    """Length of the longest run of the same character."""
    if not text:
        return 0
    best, cur, prev = 1, 1, text[0]
    for ch in text[1:]:
        if ch == prev:
            cur += 1
            best = max(best, cur)
        else:
            prev, cur = ch, 1
    return best


def quality_metrics(text: str) -> Dict[str, Any]:
    n = len(text)
    if n == 0:
        return dict(
            char_len=0, token_est=0, lines=0, avg_line_len=0.0,
            pct_ascii=0.0, pct_alnum=0.0, pct_ws=0.0, pct_punct=0.0,
            uniq_char_ratio=0.0, longest_run=0, has_refs=False, looks_like_table=False
        )
    
    lines = text.splitlines() or [text]
    tokens = len(text.split())

    ascii_count = sum(1 for c in text if ord(c) < 128)
    alnum_count = sum(1 for c in text if c.isalnum())
    ws_count = sum(1 for c in text if c.isspace())
    punct_count = sum(1 for c in text if unicodedata.category(c).startswith("P"))

    uniq_ratio = len(set(text)) / n
    llrun = longest_run_len(text)

    lower = text.lower()
    has_refs = bool(re.search(r"\b(references|bibliography|doi:)\b", lower))
    looks_like_table = bool(re.search(r"(^|\n)\s*(table\s+\d+|^\s*\|.+\|\s*$)", text, flags=re.IGNORECASE | re.MULTILINE))

    return dict(
        char_len=n,
        token_est=tokens,
        lines=len(lines),
        avg_line_len=(n / max(len(lines), 1)),
        pct_ascii=ascii_count / n,
        pct_alnum=alnum_count / n,
        pct_ws=ws_count / n,
        pct_punct=punct_count / n,
        uniq_char_ratio=uniq_ratio,
        longest_run=llrun,
        has_refs=has_refs,
        looks_like_table=looks_like_table,
    )


def iter_points(client: QdrantClient, collection: str, batch: int, with_payload: bool = True) -> Iterable[qm.Record]:
    """Yield all points via scroll."""
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            with_payload=with_payload,
            with_vectors=False,
            limit=batch,
            offset=offset,
        )
        if not points:
            break
        for p in points:
            yield p
        if not next_offset:
            break
        offset = next_offset


def build_row(p: qm.Record, fields: List[str], snippet_len: int) -> Dict[str, Any]:
    payload = p.payload or {}
    text = extract_text(payload, fields)
    metrics = quality_metrics(text)
    snippet = text[:snippet_len].replace("\n", " ").replace("\r", " ")
    row = {
        "id": p.id,
        "paper_id": payload.get("paper_id"),
        "arxiv_id": payload.get("arxiv_id") or payload.get("arxiv"),
        "path": payload.get("path") or payload.get("source") or payload.get("file_path"),
        "page": payload.get("page") or payload.get("page_number"),
        "snippet": snippet,
        **metrics,
    }
    return row
    


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dump Qdrant chunks with simple quality metrics.")
    ap.add_argument("-c", "--collection", required=True, help="Qdrant collection name")
    ap.add_argument("--url", default=os.getenv("QDRANT_URL", "http://localhost:6333"),
                    help="Qdrant URL (default %(default)s or $QDRANT_URL)")
    ap.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY"),
                    help="Qdrant API key if enabled (default $QDRANT_API_KEY)")
    ap.add_argument("--batch", type=int, default=1024, help="Scroll page size (default %(default)s)")
    ap.add_argument("--max", type=int, default=None, help="Process at most N points")
    ap.add_argument("--fields", nargs="*", default=["page_content", "text"],
                    help="Payload fields to pull text from, in order (default: %(default)s)")
    ap.add_argument("--snippet-len", type=int, default=160, help="Chars for preview snippet (default %(default)s)")
    ap.add_argument("--out", default="-",
                    help="Output file (.csv or .jsonl). Use '-' for stdout (default)")
    ap.add_argument("--preview", type=int, default=0,
                    help="Also print the first N rows as a quick preview to stderr")
    return ap.parse_args(argv)


class _SafeCSVDialect(csv.Dialect):
    delimiter = ","
    quotechar = '"'
    doublequote = True          # allow " inside fields -> becomes ""
    escapechar = "\\"           # available if ever needed
    lineterminator = "\n"
    quoting = csv.QUOTE_ALL     # safest: quote every field
    skipinitialspace = False


def _norm(v):
    """Make any value CSV-safe and serializable."""
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    # lists/dicts/other -> JSON string (single cell)
    return json.dumps(v, ensure_ascii=False)


def write_csv(rows, fp):
    writer = None
    for r in rows:
        clean = {k: _norm(v) for k, v in r.items()}
        if writer is None:
            fieldnames = list(clean.keys())
            writer = csv.DictWriter(
                fp,
                fieldnames=fieldnames,
                extrasaction="ignore",
                dialect=_SafeCSVDialect,   # <— force our safe dialect
            )
            writer.writeheader()
        writer.writerow(clean)


def write_jsonl(rows: Iterable[Dict[str, Any]], fp) -> None:
    import json
    for r in rows:
        fp.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    client = QdrantClient(url=args.url, api_key=args.api_key, timeout=60)

    rows: List[Dict[str, Any]] = []
    total = 0

    for p in iter_points(client, args.collection, args.batch, with_payload=True):
        row = build_row(p, args.fields, args.snippet_len)
        rows.append(row)
        total += 1
        if args.preview and total <= args.preview:
            print(f"[preview] {row['id']} | len={row['char_len']} | {row['snippet']}")
        if args.max and total >= args.max:
            break

    # decide writer based on extension
    out = args.out
    if out == "-" or out == "/dev/stdout":
        # default to JSONL on stdout
        write_jsonl(rows, sys.stdout)
    else:
        ext = os.path.splitext(out)[1].lower()
        with open(out, "w", encoding="utf-8", newline="") as f:
            if ext == ".csv":
                write_csv(rows, f)
            else:
                write_jsonl(rows, f)

    print(f"[done] wrote {len(rows)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
