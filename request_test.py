from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import io
import os
from PIL import Image
import requests


load_dotenv()

client = OpenAI()

# File DONE
# Code interpreter DONE
# Web search DONE

# Structured response


instructions = "You are a data analyst proficient in Python."
context = ""
prompt = "Describe every column of the dataset."
model = "gpt-4o-mini"



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



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



from typing import List, Tuple, Any
import json
import pandas as pd
import streamlit as st

def split_response(response) -> Tuple[List[str], List[str], List[Any], List[Any]]:
    """
    Extract code, plain text, parsed Pydantic objects and tool results
    from an OpenAI Responses API object.
    """
    code_blocks, texts, parsed_objs, tool_results = [], [], [], []

    # response.output is a list of messages/tool-calls
    for item in response.output:
        # 1) Code-Interpreter call
        if item.type == "code_interpreter_call":
            code_blocks.append(item.code)
            if item.results is not None:
                tool_results.append(item.results)

        # 2) Ordinary assistant message
        elif item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    texts.append(block.text)

                    # If the SDK parsed the text into a Pydantic model
                    if getattr(block, "parsed", None) is not None:
                        parsed_objs.append(block.parsed)

    return code_blocks, texts, parsed_objs, tool_results


def render_to_streamlit(
    code_blocks: List[str],
    texts: List[str],
    parsed_objs: List[Any],
    tool_results: List[Any],
    role: str = "assistant",
) -> None:
    """
    Pretty-print each part in a Streamlit chat message.
    """
    with st.chat_message(role):
        # Code
        for code in code_blocks:
            st.code(code, language="python")

        # Plain text / JSON strings
        for txt in texts:
            # If it looks like JSON show the tree, else markdown
            try:
                st.json(json.loads(txt))
            except Exception:
                st.markdown(txt)

        # Parsed Pydantic objects
        for obj in parsed_objs:
            # Show as JSON
            st.json(obj.model_dump())

            # Special case: your DatasetSummary with .columns → dataframe
            if hasattr(obj, "columns"):
                df = pd.DataFrame([c.model_dump() for c in obj.columns])
                st.dataframe(df, use_container_width=True)

        # Optional: any printed results from the Code-Interpreter
        for res in tool_results:
            st.markdown("**Interpreter output:**")
            st.write(res)

# schema

from pydantic import BaseModel
from typing import List

class ColumnSummary(BaseModel):
    column_name: str
    description: str
    type: str
    unique_value_count: int


class DatasetSummary(BaseModel):
    columns: List[ColumnSummary]



## UI

uploaded_file = st.file_uploader("Upload your CSV file...")

if "messages" not in st.session_state:
    st.session_state.messages = [{"type": "text", "content": "Ask me about your data!"}]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    openai_file = client.files.create(file=csv_buffer, purpose="user_data")
    container = client.containers.create(name="test-container", file_ids=[openai_file.id])

    # Contaier has the file uploaded.
    tools = [{"type": "code_interpreter", "container": container.id if container else "auto"},
            {"type": "web_search_preview"}]
    
    if st.button("JSON summary...", key="json_summary"):
        response = client.responses.parse(
            model=model,
            tools=tools,
            instructions=instructions,
            input=[
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            stream=False,
            text_format=DatasetSummary,   # or whatever root Pydantic model you use
        )

        # Split → render
        code_blocks, texts, parsed_objs, tool_results = split_response(response)
        render_to_streamlit(code_blocks, texts, parsed_objs, tool_results)