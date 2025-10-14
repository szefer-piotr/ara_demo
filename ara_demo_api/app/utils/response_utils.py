"""
Response utilities for processing LLM responses and managing analysis state.
Adapted from Streamlit utils - stateless versions for API use.
"""

import re
import uuid
from typing import List, Dict, Optional
from datetime import datetime


def first_chunk(
    chunks: List[Dict[str, str]], _type: str, default: str = ""
) -> str:
    """Return first chunk of given type (e.g. 'text', 'code')."""
    return next((c["content"] for c in chunks if c["type"] == _type), default)


def ordered_to_bullets(md: str) -> str:
    """
    Turn leading '1.', '2.' … into '-'.
    Works line-by-line, keeps the rest of the text untouched.
    """
    return re.sub(r"^\s*\d+\.", "", md, flags=re.MULTILINE)


def record_run(step: Dict, chunks: List[Dict[str, str]]) -> Dict:
    """
    Create run record from LLM chunks.
    
    Parameters
    ----------
    step : dict
        The step dictionary (for context, not modified)
    chunks : list of dict
        LLM response chunks with 'type' and 'content' keys
    
    Returns
    -------
    dict
        Run record with run_id, code, images, tables, summary, timestamp
    """
    run = {
        "run_id": str(uuid.uuid4())[:8],
        "code_input": [c["content"] for c in chunks if c["type"] == "code"],
        "images": [c["content"] for c in chunks if c["type"] == "image"],
        "tables": [c["content"] for c in chunks if c["type"] == "table"],
        "summary": [c["content"] for c in chunks if c["type"] == "text"],
        "created_at": datetime.utcnow().isoformat()
    }
    
    return run


def serialize_step(step: dict) -> str:
    """
    Turn one item from hypo['analysis_plan'] into a single prompt string.
    Keeps the fields: title, text, runs, chat_history.
    
    Parameters
    ----------
    step : dict
        Analysis step with keys: title, text, step_id, runs, chat_history
    
    Returns
    -------
    str
        Formatted string representation of the step
    """
    # — 1. Title
    parts = [f"## Title\n{step['title'].strip()}"]

    # — 2. Description / free-text
    if step.get("text"):
        parts.append(f"## Description\n{step['text'].rstrip()}")

    # — 3. Runs
    for run in step.get("runs", []):
        parts.append(
            f"Step {step['step_id']}.\n\n"
            f"Code: {run['code_input']}.\n\n"
            f"summary: {run['summary']}"
        )
        
    # — 4. Prior dialogue
    if step.get("chat_history"):
        chat = "\n".join(
            f"{turn['role']}: {turn['content']}"
            for turn in step["chat_history"]
        )
        parts.append(f"## Chat history\n{chat}")

    return "\n\n".join(parts)


def serialize_previous_steps(
    analysis_plan: List[Dict],
    current_hypothesis: str,
    current_hypothesis_plan: List[Dict],
    current_step_id: Optional[str] = None,
    include_current: bool = False,
) -> str:
    """
    Build a prompt that contains:
    - The current hypothesis
    - An overview of the analysis plan (step titles and texts)
    - All finished steps (or up to the specified step) in execution order.

    Parameters
    ----------
    analysis_plan : list[dict]
        The list stored at `hypo['analysis_plan']`.
    current_hypothesis : str
        The hypothesis to include at the beginning.
    current_hypothesis_plan : list[dict]
        The list of analysis plan steps to summarize.
    current_step_id : str | None
        If provided, only steps **before** this one are included
        (unless `include_current=True`).
    include_current : bool
        If True and `current_step_id` is given, the current step is included.

    Returns
    -------
    str
        A prompt string ready to be sent to the LLM.
    """
    # 1. Hypothesis section
    prompt_sections = [f"## Hypothesis\n{current_hypothesis}"]

    # 2. Analysis plan overview section
    plan_lines = []
    for idx, step in enumerate(current_hypothesis_plan, 1):
        # Show just the title and the first line of text for clarity
        first_line = step['text'].strip().split('\n')[0]
        plan_lines.append(f"{idx}. {step.get('title', f'Step {idx}')}: {first_line}")
    plan_overview = "## Analysis Plan\n" + "\n".join(plan_lines)
    prompt_sections.append(plan_overview)

    # 3. Finished steps section
    finished_sections = []
    for step in analysis_plan:
        if not step.get("finished", False):
            continue
        if current_step_id and step["step_id"] == current_step_id:
            if include_current:
                finished_sections.append(serialize_step(step))
            break
        finished_sections.append(serialize_step(step))
    
    if finished_sections:
        prompt_sections.append("## Finished Steps\n" + "\n\n---\n\n".join(finished_sections))

    return "\n\n".join(prompt_sections)


def plan_to_string(plan: List[Dict]) -> str:
    """
    Convert analysis plan to a readable string.
    
    Parameters
    ----------
    plan : list of dict
        Analysis plan with steps containing 'title' and 'text'
    
    Returns
    -------
    str
        Formatted plan string
    """
    out = []
    for i, step in enumerate(plan, 1):
        out.append(f"{i}. {step['title']}\n{step['text']}\n")
    return "\n".join(out)


def history_to_string(history: List[Dict]) -> str:
    """
    Convert chat history to a readable string.
    
    Parameters
    ----------
    history : list of dict
        Chat messages with 'content' key
    
    Returns
    -------
    str
        Formatted history string
    """
    out = []
    for i, msg in enumerate(history, 1):
        out.append(f"User: {msg['content']}")
    return "\n".join(out)


FILE_ID = re.compile(r"cfile_[A-Za-z0-9]+")


def explode_text_and_images(chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Split every text-chunk on each cfile_… identifier so that
    the returned list alternates between 'text' and 'image' items.

    Parameters
    ----------
    chunks : list of {'type': str, 'content': str}
        Original chunks from LLM response

    Returns
    -------
    list of {'type': 'text'|'image', 'content': str}
        Expanded chunks with images separated
    """
    output: List[Dict[str, str]] = []

    for chunk in chunks:
        if chunk.get("type") != "text":
            # keep non-text chunks exactly as they are
            output.append(chunk)
            continue

        text = chunk["content"]
        last = 0

        for match in FILE_ID.finditer(text):
            # 1️⃣ text before the file-ID
            if match.start() > last:
                output.append({"type": "text", "content": text[last : match.start()]})

            # 2️⃣ the file-ID itself
            output.append({"type": "image", "content": match.group(0)})

            last = match.end()

        # 3️⃣ trailing text after the final ID (or the whole string if none)
        if last < len(text):
            output.append({"type": "text", "content": text[last:]})

    return output