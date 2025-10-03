#!/usr/bin/env python3
"""
Scrape bioRxiv with paperscraper:
  1) ensure a local bioRxiv dump (optionally for a date range),
  2) run a keyword query (AND of OR-groups),
  3) optionally limit results,
  4) optionally download PDFs/XMLs.

Examples
--------
# 1 OR-group (comma-separated synonyms), ANDed with another OR-group:
python scrape_biorxiv.py \
  --group "population ecology,wild bees,pollination" \
  --group "network,community" \
  --start-date 2025-01-01 --end-date 2025-08-27 \
  --limit 20 \
  --out-json biorxiv_query.jsonl \
  --download --pdf-dir papers --api-keys .env \
  -vv

# Quick single-group search, print top 10, no downloads:
python scrape_biorxiv.py \
  --group "plant community" \
  --limit 10 \
  --out-json biorxiv_query.jsonl \
  -vv

# Reuse existing dump (skip downloading), just search & download PDFs:
python scrape_biorxiv.py \
  --skip-dump \
  --group "population ecology,wild bees,pollination" \
  --limit 25 --download --pdf-dir papers -vv
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

# paperscraper imports
from paperscraper.get_dumps import biorxiv as fetch_biorxiv_dump
from paperscraper.xrxiv.xrxiv_query import XRXivQuery
from paperscraper.pdf import save_pdf_from_dump


DEFAULT_DUMPS_DIR = Path("server_dumps")          # where paperscraper writes dumps by default
DEFAULT_OUT_JSON = Path("biorxiv_query.jsonl")


def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def find_latest_biorxiv_dump(dumps_dir: Path) -> Optional[Path]:
    """Return path to newest server_dumps/biorxiv_*.jsonl (or None)."""
    candidates = sorted(dumps_dir.glob("biorxiv_*.jsonl"))
    return candidates[-1] if candidates else None


def ensure_dump(start: Optional[str], end: Optional[str], max_retries: int, skip_dump: bool, dumps_dir: Path) -> Path:
    """
    Ensure a bioRxiv dump exists. If skip_dump==False, fetch a fresh dump
    (optionally bounded by start/end dates). Return the dump path.
    """
    dumps_dir.mkdir(parents=True, exist_ok=True)

    if not skip_dump:
        kwargs = {}
        if start:
            kwargs["start_date"] = start
        if end:
            kwargs["end_date"] = end
        if max_retries:
            kwargs["max_retries"] = max_retries

        logging.info("Fetching bioRxiv dump with args: %s", kwargs or "(defaults)")
        # paperscraper writes a file under server_dumps/ and (currently) returns None.
        fetch_biorxiv_dump(**kwargs)  # may take a while if wide date range

    latest = find_latest_biorxiv_dump(dumps_dir)
    if not latest:
        raise SystemExit(f"No bioRxiv dump found in {dumps_dir}. (Did the fetch fail?)")
    logging.info("Using dump: %s", latest)
    return latest


def parse_groups(groups: List[str]) -> List[List[str]]:
    """
    Convert --group strings into paperscraper's nested query:
      top-level list = AND, inner lists = OR
    Example:
      --group "population ecology,wild bees,pollination"
      --group "network,community"
    becomes:
      [
        ["population ecology", "wild bees", "pollination"],
        ["network", "community"]
      ]
    """
    parsed: List[List[str]] = []
    for g in groups:
        ors = [t.strip() for t in g.split(",") if t.strip()]
        if ors:
            parsed.append(ors)
    if not parsed:
        raise SystemExit("At least one --group is required.")
    return parsed


def head_jsonl(in_path: Path, out_path: Path, n: int) -> int:
    """Write the first n lines of a JSONL file to out_path. Return count written."""
    if n <= 0:
        # copy path (no limit)
        if in_path != out_path:
            out_path.write_bytes(in_path.read_bytes())
        with out_path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    written = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if written >= n:
                break
            fout.write(line)
            written += 1
    return written


def maybe_print_preview(jsonl_path: Path, limit: int = 10) -> None:
    """Print a small preview of results (title + doi)."""
    shown = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            title = (rec.get("title") or "").strip().replace("\n", " ")
            doi = rec.get("doi") or rec.get("DOI") or ""
            print(f"- {title}  [{doi}]")
            shown += 1
            if shown >= limit:
                break


def run(args: argparse.Namespace) -> None:
    setup_logger(args.verbose)

    # 1) Ensure local dump
    dump_path = ensure_dump(
        start=args.start_date, end=args.end_date,
        max_retries=args.max_retries, skip_dump=args.skip_dump,
        dumps_dir=args.dumps_dir
    )

    # 2) Build nested query: AND of OR groups
    query = parse_groups(args.group)

    # 3) Search the dump
    out_json = Path(args.out_json).resolve()
    logging.info("Searching dump with query=%s", query)
    querier = XRXivQuery(str(dump_path))
    querier.search_keywords(query, output_filepath=str(out_json))
    logging.info("Wrote matches -> %s", out_json)

    # 4) Apply --limit (optional)
    effective_json = out_json
    if args.limit is not None:
        limited_json = out_json.with_name(out_json.stem + f"_head{args.limit}" + out_json.suffix)
        count = head_jsonl(out_json, limited_json, args.limit)
        effective_json = limited_json
        logging.info("Trimmed to first %d result(s) -> %s", count, effective_json)

    # 5) Preview to stdout (optional, handy in CI)
    if args.print_preview:
        print("\nPreview:")
        maybe_print_preview(effective_json, limit=min(args.limit or 10, 10))
        print()

    # 6) Download PDFs/XMLs (optional)
    if args.download:
        pdf_path = Path(args.pdf_dir).resolve()
        pdf_path.mkdir(parents=True, exist_ok=True)

        kwargs = dict(
            pdf_path=str(pdf_path),
            key_to_save="doi",
        )
        if args.api_keys:
            kwargs["api_keys"] = args.api_keys
        # Some versions of paperscraper support xml_path; pass it if user asked.
        if args.xml_dir:
            xml_path = Path(args.xml_dir).resolve()
            xml_path.mkdir(parents=True, exist_ok=True)
            kwargs["xml_path"] = str(xml_path)

        logging.info("Downloading full-text using save_pdf_from_dump(..., %s)", kwargs)
        # This function handles fallbacks (PMC/eLife) and bioRxiv TDM via AWS if keys are provided.
        save_pdf_from_dump(str(effective_json), **kwargs)
        logging.info("Full-text download complete.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape bioRxiv using paperscraper (dump -> query -> optional full-text).")

    # Query: AND of OR-groups
    p.add_argument("--group", action="append", required=True,
                   help='Comma-separated OR terms. Repeat for AND. Example: --group "population ecology,wild bees,pollination" --group "network,community"')

    # Dump control
    p.add_argument("--start-date", help="YYYY-MM-DD (optional; reduces dump time/size).")
    p.add_argument("--end-date", help="YYYY-MM-DD (optional).")
    p.add_argument("--skip-dump", action="store_true", help="Do not fetch dump; reuse latest in server_dumps/.")
    p.add_argument("--max-retries", type=int, default=10, help="Retries for dump fetching (paperscraper default ~10).")
    p.add_argument("--dumps-dir", type=Path, default=DEFAULT_DUMPS_DIR, help="Where dumps are stored (default: server_dumps/).")

    # Output & limiting
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Output JSONL file for matched metadata.")
    p.add_argument("--limit", type=int, help="Keep only the first N matches (affects preview & downloads).")
    p.add_argument("--print-preview", action="store_true", help="Print first results (title + DOI) to stdout.")

    # Full-text download
    p.add_argument("--download", action="store_true", help="Download PDFs (and XMLs if --xml-dir provided).")
    p.add_argument("--pdf-dir", default="papers", help="Where to save PDFs (default: ./papers).")
    p.add_argument("--xml-dir", help="Where to save XMLs (optional; if omitted, only PDFs are requested).")
    p.add_argument("--api-keys", help="Path to a file with API keys (.env style) for TDM (WILEY/ELSEVIER/AWS).")

    # Logging
    p.add_argument("-v", "--verbose", action="count", default=1)

    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
