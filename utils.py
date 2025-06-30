# utils.py

from typing import Any, List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st
from PIL import Image
import io
import requests
import openai
from openai import OpenAI
import uuid
import re
import os

Chunk = Dict[str, str]


from typing import List, Dict
from openai.types.responses import (
    Response,
    ResponseCodeInterpreterToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from openai.types.responses.response_output_text import AnnotationContainerFileCitation

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def to_mock_chunks(resp: Response) -> List[Chunk]:
    """
    Convert an `openai.Response` into the mock-LLM chunk list:

        [{"type": "text",  "content": ...},
         {"type": "code",  "content": ...},
         {"type": "image", "content": <file_id>}, ...]
    """
    chunks: List[Chunk] = []

    for item in resp.output:

        # ── Code interpreter blocks ────────────────────────────────────────
        if isinstance(item, ResponseCodeInterpreterToolCall):
            if item.code:
                chunks.append({"type": "code", "content": item.code})

        # ── Regular assistant messages ─────────────────────────────────────
        elif isinstance(item, ResponseOutputMessage):
            for block in item.content:
                if isinstance(block, ResponseOutputText):
                    chunks.append({"type": "text", "content": block.text})

                # Scan for image annotations on this text block
                for anno in getattr(block, "annotations", []):
                    if isinstance(anno, AnnotationContainerFileCitation):
                        chunks.append({
                            "type": "image",
                            "content": anno.file_id          # <── ONLY the ID
                        })

    return chunks


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

    if "generate analysis plan" in prompt_lc or "analysis plan" in prompt_lc:
        steps = [
            "Explore data quality & descriptive stats",
            "Visualise key relationships",
            "Statistical model / test for the hypothesis",
        ]
        plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        chunks.append({"type": "text", "content": plan_text})
        return chunks                             # nothing else needed

    if "code interpreter" in tools:
        # Pretend we executed the supplied code
        chunks.append({"type": "text", "content": "Execution finished. See artefacts below."})
        chunks.append({"type": "code", "content": prompt + "\n# (executed)"})
        img_html = '<img src="https://placehold.co/600x300?text=Mock+Plot">'
        chunks.append({"type": "image", "content": img_html})
        return chunks

    if "web search" in tools:
        chunks.append({
            "type": "text",
            "content": f"[MOCK web-search] Top result for: “{prompt}”."
        })
        return chunks

    if "vector search" in tools:
        chunks.append({
            "type": "text",
            "content": f"[MOCK vector-search] Retrieved context for: “{prompt}”."
        })
        return chunks

    # ── 5. DEFAULT CHAT TURN ────────────────────────────────────
    chunks.append({"type": "text", "content": f"[MOCK-LLM] {prompt[:60]}…" })
    return chunks


def create_container(client: OpenAI, file_ids: List[str], name: str = "user-container"):
    container = client.containers.create(
        name=name, 
        file_ids=file_ids, 
        expires_after={
        "anchor": "last_active_at",
        "minutes": 20
    })
    print(f"Created container {container.id} for code interpreter runs.")
    return container


def create_code_interpreter_tool(container):
    return {"type": "code_interpreter", "container": container.id if container else "auto"}


def create_web_search_tool():
    return {"type": "web_search_preview"}


def upload_csv_and_get_file_id(client: OpenAI, uploaded_file: UploadedFile):
    if uploaded_file.type != "text/csv":
        raise ValueError("Uploaded file is not a CSV file.")
    df = pd.read_csv(uploaded_file)
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    openai_file = client.files.create(file=csv_buffer, purpose="user_data")
    return openai_file.id


def load_image_from_openai_container(api_key: str | None, container_id: str, file_id: str) -> Image.Image:
    url = f"https://api.openai.com/v1/containers/{container_id}/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    raise Exception(f"Failed to retrieve file: {response.status_code}, {response.text}")


def render_llm_response(response):
    elements = []
    for item in response.output:
        if item.type == "code_interpreter_call":
            elements.append({"type": "code", "content": item.code})
        elif item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    elements.append({"type": "text", "content": block.text})
                    if hasattr(block, "annotations"):
                        for ann in block.annotations:
                            if ann.type == "container_file_citation":
                                elements.append({"type": "image", "filename": ann.filename, "content": ann.file_id})
                                container = st.session_state.container
                                image = load_image_from_openai_container(OPENAI_API_KEY, container.id, ann.file_id) # type: ignore
                                img_bytes = io.BytesIO()
                                image.save(img_bytes, format='PNG')
                                st.session_state.images[ann.file_id] = img_bytes.getvalue()
    return elements



def render_chat_elements(elements, role="assistant"):
    with st.chat_message(role):
        for el in elements:
            if el["type"] == "text":
                st.markdown(el["content"])
            elif el["type"] == "code":
                st.code(el["content"], language="python")
            elif el["type"] == "image":
                image_id = el["content"]
                if image_id in st.session_state.images:
                    image = Image.open(io.BytesIO(st.session_state.images[image_id]))
                    st.image(image)
                else:
                    st.warning("Image not found.")


from typing import Any, Dict, List, Optional
import openai
from openai import OpenAI

def get_llm_response(
        client: OpenAI,
        model: str,
        prompt: str,
        instructions: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        context: str = "",
        text_format: Optional[Dict[str, str]] = None,  # ← NEW
        stream: bool = False,
        temperature: float = 0
    ):
    """
    Call OpenAI with optional structured-output control.

    Parameters
    ----------
    text_format : dict | None
        • Pass {'type': 'json_object'} to force JSON-only replies  
        • Pass None (default) for normal free-text behaviour
    """

    try:
        # Build the argument set dynamically so we only include
        # keys that are actually requested.
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": [
                {"role": "system", "content": context},
                {"role": "user",   "content": prompt}
            ],
            "temperature": temperature,
            "stream": stream,
        }

        if tools:
            create_kwargs["tools"] = tools
        if text_format:
            create_kwargs["text_format"] = text_format


        response = client.responses.parse(**create_kwargs)

        print(response.output_parsed)


        # if text_format is None:
        #     response = client.responses.create(**create_kwargs)
        # elif text_format is not None:
        #     response = client.responses.parse(**create_kwargs)

        return response

        # If the reply is a tool call, return the structured payload.
        # for item in response.output:
        #     if item.type == "tool_use":
        #         return item.outputs[0]          # structured dict
            
        # Otherwise fall back to your existing renderer
        # return render_llm_response(response)

    except openai.BadRequestError as e:
        if "Container is expired" in str(e):
            print("Container expired! Re-create or refresh the container before retrying.")
        else:
            print(f"BadRequestError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")



# ───────────────────────── helpers ──────────────────────────
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
    """Create a run dict from LLM chunks and append to step['runs']."""    
    run = {
        "run_id": uuid.uuid4().hex[:8],
        "code_input": [c["content"] for c in chunks if c["type"] == "code"],
        "images":  [c["content"] for c in chunks if c["type"] == "image"],
        "tables":  [c["content"] for c in chunks if c["type"] == "table"],
        "summary": [c["content"] for c in chunks if c["type"] == "text"],
    }
    
    step["runs"].append(run)

    container = st.session_state.get("container")

    if not container:
        if "container" not in st.session_state:
            st.session_state.container = None
        print("No container available. Images won't be loaded.")
    else:
        for img in run["images"]:
            # Load the image from OpenAI container and store it in session state
            st.session_state.images[img] = load_image_from_openai_container(
                OPENAI_API_KEY, container.id, img
            )

            print(
                f"Image {img} loaded from container {container.id} and stored in session state."
            )
    
    return run


def serialize_step(step: dict) -> str:
    """
    Turn one item from hypo['analysis_plan'] into a single prompt string.
    Keeps the fields: title, text, code, chat_history, images.
    """

    # — 1.   Title -----------------------------------------------------------
    parts = [f"## Title\n{step['title'].strip()}"]

    # — 2.   Description / free-text ----------------------------------------
    if step.get("text"):
        parts.append(f"## Description\n{step['text'].rstrip()}")

    for run in step.get("runs", []):
        parts.append(f"Step {step['step_id']}.\n\nCode: {run['code_input']}.\n\nsummary: {run['summary']}")
        
    # — 4.   Prior dialogue --------------------------------------------------
    if step.get("chat_history"):
        # print(f"\nIn serialize_step this is the step's chat history {step['chat_history']}")
        chat = "\n".join(
            f"{turn['role']}: {turn['content']}"
            for turn in step["chat_history"]
        )
        parts.append(f"## Chat history\n{chat}")

    # — 5.   Images ----------------------------------------------------------
    # TODO Attach only the image IDs, not the full URLs.
    # if step.get("images"):
    #     parts.append(
    #         "## Images (one per line)\n" + "\n".join(step["images"])
    #     )

    return "\n\n".join(parts)


def serialize_previous_steps(
    analysis_plan: List[Dict],
    current_step_id: Optional[str] = None,   # or use an int for index
    include_current: bool = False,
) -> str:
    """
    Build a prompt that contains **all finished steps** (or up to the specified
    step) in execution order.

    Parameters
    ----------
    analysis_plan : list[dict]
        The list stored at `hypo['analysis_plan']`.
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
    prompt_sections = []

    for step in analysis_plan:
        # Skip unfinished steps
        if not step.get("finished", False):
            continue

        # Stop once we reach the current step (unless we also want it)
        if current_step_id and step["step_id"] == current_step_id:
            if include_current:
                prompt_sections.append(serialize_step(step))
            break

        prompt_sections.append(serialize_step(step))

    # Join individual step prompts with a visible separator
    return "\n\n---\n\n".join(prompt_sections)



def edit_column_summaries() -> None:
    """
    Show editable description + type for every ColumnSummary in
    st.session_state.column_summaries.
    """
    st.subheader("Edit column metadata")

    # ---- collect the selectable python types --------------------------------
    defaults = {
        "int",
        "float",
        "str",
        "bool",
        "datetime",
        "list",
        "dict",
        "NoneType",
        "category",
    }
    existing = {cs.type for cs in st.session_state.column_summaries}
    py_types = tuple(sorted(defaults | existing))

    # ---- form UI ------------------------------------------------------------
    with st.form("edit_metadata", clear_on_submit=False):
        for cs in st.session_state.column_summaries:
            desc_col, type_col = st.columns([3, 1])

            with desc_col:
                st.text_area(
                    f"{cs.column_name} description",
                    value=cs.description,
                    key=f"desc_{cs.column_name}",
                    height=80,
                )

            with type_col:
                current = cs.type or "str"
                st.selectbox(
                    f"{cs.column_name} type",
                    options=py_types,
                    index=py_types.index(current)
                    if current in py_types
                    else py_types.index("str"),
                    key=f"type_{cs.column_name}",
                )

        save_btn, cancel_btn = st.columns(2)
        save_clicked   = save_btn.form_submit_button("💾 Save",   type="primary")
        cancel_clicked = cancel_btn.form_submit_button("❌ Cancel")

    # ---- handle Save / Cancel ----------------------------------------------
    if save_clicked:
        # Write the edits back into the ColumnSummary objects
        for cs in st.session_state.column_summaries:
            cs.description = st.session_state[f"desc_{cs.column_name}"]
            cs.type        = st.session_state[f"type_{cs.column_name}"]

            # clean up the temporary widgets’ values
            st.session_state.pop(f"desc_{cs.column_name}", None)
            st.session_state.pop(f"type_{cs.column_name}", None)

        st.session_state.edit_mode = False
        st.session_state.need_refinement = True
        st.success("Column metadata updated!")
        st.rerun()

    elif cancel_clicked:
        # discard widget values
        for cs in st.session_state.column_summaries:
            st.session_state.pop(f"desc_{cs.column_name}", None)
            st.session_state.pop(f"type_{cs.column_name}", None)

        st.session_state.edit_mode = False
        st.info("No changes saved.")
        st.rerun()


def plan_to_string(plan):
    out = []
    for i, step in enumerate(plan, 1):
        out.append(f"{i}. {step['title']}\n{step['text']}\n")
    return "\n".join(out)

def history_to_string(history):
    out = []
    for i, msg in enumerate(history, 1):
        out.append(f"User: {msg['content']}")
    return "\n".join(out)

def render_assistant_message(elements):
    with st.chat_message("assistant"):
        for item in elements:
            if item["type"] == "code":
                st.code(item["content"], language="python")
            elif item["type"] == "text":
                st.markdown(item["content"])
            elif item["type"] == "image":
                print(f"Image ID: {item['content']}")
                image_to_display = st.session_state.images.get(item["content"])
                if image_to_display:
                    st.image(image_to_display, caption=item.get("filename", "Image"))
                else:
                    st.warning("Image not found in session state.")
                # st.write(f"Image:{item['content']}")
                # img = item["content"]
                # if img in st.session_state.images:
                #     st.image(img, caption=item.get("filename", "Image"))
                # else:
                #     st.warning("Image not found in session state.")
            else:
                st.warning(f"Unknown element type: {item['type']}")



import re
from typing import List, Dict

# just the “cfile_…” token – no surrounding punctuation
FILE_ID = re.compile(r"cfile_[A-Za-z0-9]+")

def explode_text_and_images(chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Split every text-chunk on each cfile_… identifier so that
    the returned list alternates between 'text' and 'image' items.

    Parameters
    ----------
    chunks : list of {'type': str, 'content': str}

    Returns
    -------
    list of {'type': 'text'|'image', 'content': str}
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

        print("\nFUNCTION CALL!!!\n")

    return output



