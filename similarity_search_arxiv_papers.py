#!/usr/bin/env python3
"""
CLI tool to run semantic similarity search over arXiv chunks stored in Qdrant.

Qdrant collection payload for each point include keys:
- "paper_id": paper_id,
- "chunk_id": i,
- "source": source,
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

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None  # type: ignore

@dataclass
class EmbeddingResults:
    vector: List[float]
    model: str


class EmbeddingProvider:
    """Strategy interface for embedding text queries."""
    def embed(self, text: str) -> EmbeddingResults:
        raise NotImplementedError
    

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Uses OpenAI Embeddings API
    Default model: text-embedding-3-small (1536 dims, cost-effective)
    """
    def __init__(self, api_key: Optional[str], model: str = "text-embedding-3-small", timeout: float = 30):
        if not api_key:
            raise ValueError("OPENAI_API_KEY not provided. Set env or pass --openai-api-key.")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(
                "Failed to import/use OpenAI client."
            ) from e
        
    def embed(self, text: str) -> EmbeddingResults:
        clean = " ".join(text.split())
        resp = self._client.embeddings.create(
            model=self.model,
            input=clean,
            timeout=self.timeout,
        )
        vec = resp.data[0].embedding
        return EmbeddingResults(vector=vec, model=self.model)
    

@dataclass
class SearchConfig:
    qdrant_url: str
    qdrant_api_key: Optional[str]
    collection: str
    top_k: int
    with_payload: bool
    with_vectors: bool
    score_threshold: Optional[float]
    query_filter: Optional[dict]
    using_vector_name: Optional[str]
    timeout: Optional[int]


def make_qdrant_client(conf: SearchConfig) -> QdrantClient:
    return QdrantClient(
        url=conf.qdrant_url,
        api_key=conf.qdrant_api_key or None,
        timeout=conf.timeout,
    )


def run_search(
    client: QdrantClient,
    conf: SearchConfig,
    query_vector: Sequence[float],
) -> list[types.ScoredPoint]:
    resp = client.query_points(
        collection_name=conf.collection,
        query=list(query_vector),
        using=conf.using_vector_name,
        limit=conf.top_k,
        with_payload=conf.with_payload,
        with_vectors=conf.with_vectors,
        score_threshold=conf.score_threshold,
        query_filter=conf.query_filter,  # <— pass dict as-is
        timeout=conf.timeout,
    )
    return list(resp.points or [])


def print_results_table(points: List[types.ScoredPoint], fields_to_show: Optional[List[str]] = None) -> None:
    if not RICH_AVAILABLE:
        print(json.dumps([_point_to_dict(p) for p in points], ensure_ascii=False, indent=2))
        return
    
    if not points:
        console.print("[bold yellow]No results.[/bold yellow]")
        return

    default_fields = ["paper_id", "chunk_id", "source"]
    fields = fields_to_show or default_fields

    table = Table(show_header=True, header_style="bold")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    for f in fields:
        table.add_column(f)

    for idx, p in enumerate(points, start=1):
        payload = p.payload or {}
        row = [str(idx), f"{p.score:.4f}" if p.score is not None else "-"]
        for f in fields:
            val = payload.get(f, "")
            if isinstance(val, (dict, list)):
                val = json.dumps(val, ensure_ascii=False)
            if isinstance(val, str) and len(val) > 140:
                val = textwrap.shorten(val, width=140, placeholder=" ...")
            row.append(str(val))
        table.add_row(*row)

    console.print(table)

    for idx, p in enumerate(points, start=1):
        payload = p.payload or {}
        text = payload.get("text")
        if text:
            preview = textwrap.shorten(" ".join(str(text).split()), width=300, placeholder=" ...")
            console.print(f"[dim]#{idx} text: [/dim] {preview}")
    

def _point_to_dict(p: types.ScoredPoint) -> Dict[str, Any]:
    return {
        "id": p.id,
        "score": p.score,
        "version": p.version,
        "payload": p.payload,
        "vector": p.vector if isinstance(p.vector, (list, dict)) else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic similarity search over arXiv chunks in Qdrant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Query
    parser.add_argument("--query", type=str, required=True, help="Search phrase.")

    # Output
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Output format."
    )
    parser.add_argument(
        "--show-fields",
        type=str,
        default="paper_id,chunk_id",
        help="Comma-separated payload fields to display in table output.",
    )

    # Qdrant
    parser.add_argument("--collection", type=str, default=os.getenv("QDRANT_COLLECTION", "arxiv_chunks"))
    parser.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", type=str, default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--using", type=str, default=None, help="Named vector to use (if your collection has multiple).")
    parser.add_argument("--score-threshold", type=float, default=None, help="Minimum score (cosine ~ higher is better).")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout (seconds).")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Qdrant filter JSON, e.g. '{\"must\": [{\"key\":\"source\",\"match\":{\"value\":\"arxiv\"}}]}'",
    )

    # Embeddings (OpenAI by default)
    parser.add_argument("--embed-provider", choices=["openai"], default=os.getenv("EMBED_PROVIDER", "openai"))
    parser.add_argument("--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--openai-model", type=str, default=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Parse optional filter JSON
    query_filter = None
    if args.filter:
        try:
            query_filter = json.loads(args.filter)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in --filter: {e}", file=sys.stderr)
            return 2

    # Choose embedding provider
    if args.embed_provider == "openai":
        try:
            emb = OpenAIEmbeddingProvider(api_key=args.openai_api_key, model=args.openai_model)
        except Exception as e:
            print(f"Failed to initialize OpenAI embedding provider: {e}", file=sys.stderr)
            return 2
    else:
        print("Unsupported embedding provider selected.", file=sys.stderr)
        return 2

    # Embed query
    try:
        emb_res = emb.embed(args.query)
    except Exception as e:
        print(f"Embedding failed: {e}", file=sys.stderr)
        return 3

    # Build search config
    conf = SearchConfig(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection=args.collection,
        top_k=int(args.top_k),
        with_payload=True,
        with_vectors=False,
        score_threshold=args.score_threshold,
        query_filter=query_filter,
        using_vector_name=args.using,
        timeout=args.timeout,
    )

    # Qdrant search
    try:
        client = make_qdrant_client(conf)
        points = run_search(client, conf, emb_res.vector)
    except Exception as e:
        print(f"Search failed: {e}", file=sys.stderr)
        return 4

    # Output
    if args.output == "json":
        print(json.dumps([_point_to_dict(p) for p in points], ensure_ascii=False, indent=2))
    else:
        fields = [f.strip() for f in (args.show_fields or "").split(",") if f.strip()]
        print_results_table(points, fields_to_show=fields)

    return 0


if __name__ == "__main__":
    sys.exit(main())