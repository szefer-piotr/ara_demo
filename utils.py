# utils.py
import uuid
from typing import List, Dict, Optional

Chunk = Dict[str, str]        # {"type": "text" | "code" | "image", "content": str}

# ────────────────────────────────────────────────────────────────────────────────
# The single mock LLM entry-point
# ────────────────────────────────────────────────────────────────────────────────
def mock_llm(
    prompt: str,
    history: Optional[List[Chunk]] = None,
    tools: Optional[List[str]] = None,
) -> List[Chunk]:
    """
    A *single* mock gateway that fakes different behaviours depending on `tools`.
    Returns a list of structured chunks:
        [{"type": "text",  "content": ...},
         {"type": "code",  "content": ...},
         {"type": "image", "content": ...}, ...]
    """
    history = history or []
    tools   = tools or []

    prompt_lc = prompt.lower()
    chunks: List[Chunk] = []

    # ── 1. PLAN GENERATION ───────────────────────────────────────
    if "generate analysis plan" in prompt_lc or "analysis plan" in prompt_lc:
        steps = [
            "Explore data quality & descriptive stats",
            "Visualise key relationships",
            "Statistical model / test for the hypothesis",
        ]
        plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        chunks.append({"type": "text", "content": plan_text})
        return chunks                             # nothing else needed

    # ── 2. CODE-INTERPRETER TOOL ─────────────────────────────────
    if "code interpreter" in tools:
        # Pretend we executed the supplied code
        chunks.append({"type": "text", "content": "Execution finished. See artefacts below."})
        chunks.append({"type": "code", "content": prompt + "\n# (executed)"})
        img_html = '<img src="https://placehold.co/600x300?text=Mock+Plot">'
        chunks.append({"type": "image", "content": img_html})
        return chunks

    # ── 3. WEB SEARCH TOOL ──────────────────────────────────────
    if "web search" in tools:
        chunks.append({
            "type": "text",
            "content": f"[MOCK web-search] Top result for: “{prompt}”."
        })
        return chunks

    # ── 4. VECTOR SEARCH TOOL ───────────────────────────────────
    if "vector search" in tools:
        chunks.append({
            "type": "text",
            "content": f"[MOCK vector-search] Retrieved context for: “{prompt}”."
        })
        return chunks

    # ── 5. DEFAULT CHAT TURN ────────────────────────────────────
    chunks.append({"type": "text", "content": f"[MOCK-LLM] {prompt[:60]}…" })
    return chunks
