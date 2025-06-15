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


Chunk = Dict[str, str]


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
    container = client.containers.create(name=name, file_ids=file_ids)
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


def load_image_from_openai_container(api_key: str, container_id: str, file_id: str) -> Image.Image:
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
                                image = load_image_from_openai_container(OPENAI_API_KEY, container.id, ann.file_id)
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

        print(response)


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




data_summary_tool = {
    "name": "return_dataset_summary",
    "type": "function",
    "function": {
        "description": "Return a structured summary of the dataset",
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "object",
                    "description": "Dictionary of column summaries",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "column_name": {"type": "string"},
                            "description": {"type": "string"},
                            "type": {"type": "string"},
                            "unique_value_count": {"type": "integer"},
                        },
                        "required": ["column_name", "description", "type", "unique_value_count"]
                    },
                }
            },
            "required": ["columns"]
        }
    }
}



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