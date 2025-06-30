import streamlit as st
import pandas as pd
import uuid
from io import StringIO

import utils

from utils import mock_llm, create_container, create_web_search_tool, create_code_interpreter_tool, edit_column_summaries

from sidebar import render_sidebar
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Any, List
from utils import mock_llm, create_web_search_tool, create_code_interpreter_tool, get_llm_response

from schemas import ColumnSummary, DatasetSummary
from instructions import data_summary_instructions

st.set_page_config(layout="wide")

utils.inject_global_css()

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# ── Initialise session containers ──────────────────────────────────────────────
if "analyses" not in st.session_state:
    st.session_state["analyses"] = []      # list of hypothesis-level dictionaries

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI()

if "file_ids" not in st.session_state:
    st.session_state["file_ids"] = []

if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

# ── STEP 1 ─ Upload data ───────────────────────────────────────────────────────
st.markdown("#### Upload data (CSV)")

st.markdown("""
<div style="background-color:#fff3cd; padding: 1em; border-radius: 10px; border: 1px solid #ffeeba;">
<h4>📊 Step 1: Upload Data & Add Hypotheses</h4>
<ul>
<li>Upload your <strong>CSV file</strong>. The app will automatically summarize your dataset.</li>
<li>Edit column descriptions if needed.</li>
<li>Add one or more <strong>hypotheses</strong> — type them in or import from a <code>.txt</code> file.</li>
</ul>
<p>✅ Once your data is uploaded <strong>and</strong> at least one hypothesis is added, click <em>“Go to analysis plan”</em> to continue.</p>
</div>
""", unsafe_allow_html=True)


# Put the CSV uploader in a centred, narrower container
with st.container():
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        if "current_data" not in st.session_state:
            st.session_state["current_data"] = None

        if st.session_state["current_data"] is None:
            st.write("Choose a CSV file")
            data_file = st.file_uploader(label="Upload data", type="csv")
            if data_file:
                df = pd.read_csv(data_file)
                st.session_state["current_data"] = df
                data_file.seek(0)

                # Upload CSV to OpenAI files & create container
                file_id = utils.upload_csv_and_get_file_id(st.session_state.openai_client, data_file)
                st.session_state["file_ids"].append(file_id)
                container = create_container(st.session_state.openai_client, [file_id])
                st.session_state.container = container

                # TODO Get the structured data summary.
                with st.spinner("Summarizing your data..."):
                    client = st.session_state.openai_client
                    
                    response = client.responses.parse(
                        model="gpt-4o",
                        tools=[
                            {"type": "code_interpreter", "container": container.id if container else "auto"}
                        ],
                        input=[
                            {"role": "user",   "content": "Read the data and return the dataset summary using the structured tool."}
                        ],
                        instructions=data_summary_instructions,
                        text_format=DatasetSummary,
                    )

                    print(response.output_parsed)

                if "column_summaries" not in st.session_state:
                    # response.output_parsed.columns is already a List[ColumnSummary]
                    st.session_state.column_summaries: List[ColumnSummary] = ( # type: ignore[attr-defined]
                        response.output_parsed.columns  # type: ignore[attr-defined]
                    )

                st.dataframe(df.head(), use_container_width=True)
                st.rerun()

        else:
            st.markdown("#### Your dataset preview.")
            st.dataframe(st.session_state["current_data"].head(), use_container_width=True)
            # st.json(st.session_state.column_summaries)
            
            # ---------- Column list / Edit button ----------
            st.markdown("#### Dataset summary")
            st.write("Edit individual column descriptions if necessary.")
            for col in st.session_state.column_summaries:
                st.markdown(
                    f"**{col.column_name}**  "
                    f"({col.type}, {col.unique_value_count} unique) – "
                    f"{col.description}"
                )

            if st.session_state.edit_mode:
                edit_column_summaries()             # ← always render the form when editing
            else:
                if st.button("Edit column metadata", key="edit_metadata"):
                    st.session_state.edit_mode = True
                    st.rerun()


st.divider()


# ── List current hypotheses ────────────────────────────────────────────────────
st.markdown("#### Existing hypotheses")
st.write("""Your hypotheses will appear here. You can remove them and add new ones below. When you are ready click Refine Hypotheses and the Assistant will refine you rhypotheses using your data, web search and knowledge""")
if st.session_state["analyses"]:

    for idx, a in enumerate(st.session_state["analyses"]):
        col1, col2 = st.columns([9, 1])          # wide text · narrow icon
        with col1:
            st.write(f"**{a['hypothesis_id']}** — {a['title']}")
        with col2:
            pressed = st.button(
                "🗑️",                           # trash-can emoji
                key=f"del_{a['hypothesis_id']}",
                help="Delete this hypothesis",
            )
            if pressed:
                # remove the hypothesis
                st.session_state["analyses"].pop(idx)

                # keep selection sensible
                if st.session_state.get("selected_hypothesis_id") == a["hypothesis_id"]:
                    if st.session_state["analyses"]:
                        st.session_state["selected_hypothesis_id"] = (
                            st.session_state["analyses"][-1]["hypothesis_id"]
                        )
                    else:
                        st.session_state["selected_hypothesis_id"] = None

                st.rerun()          # refresh the page

else:
    st.info("None yet – add at least one before proceeding.")

# ── STEP 2 ─ Add hypotheses ────────────────────────────────────────────────────
st.divider()

st.markdown("#### Add hypotheses")

# Put the hypothesis inputs in the same centred, narrower container
with st.container():
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        st.write("Type a hypothesis then click Add")
        # Forms allow us to avoid partial reruns while typing
        with st.form(key="typed_hypothesis", clear_on_submit=True):
            text = st.text_input(label="Type a new hypothesis", key="hypothesis_input", label_visibility="collapsed")
            submitted = st.form_submit_button("Add")
            if submitted and text.strip():
                new_hypothesis = {
                    "hypothesis_id": uuid.uuid4().hex[:8],
                    "title": text.strip(),
                    "data": st.session_state["current_data"],
                    "analysis_plan": [],
                }
                st.session_state["analyses"].append(new_hypothesis)
                st.session_state["selected_hypothesis_id"] = new_hypothesis["hypothesis_id"]
                st.success("Hypothesis added.")
                st.rerun()

        st.write("…or upload a .txt file (one hypothesis per line)")

        with st.form(key="txt_hypothesis"):
            txt_file = st.file_uploader("Upload a TXT file with one hypothesis per line", 
                                        type="txt",
                                        label_visibility="collapsed")
            uploaded = st.form_submit_button("Import")
            if uploaded and txt_file:
                raw = StringIO(txt_file.getvalue().decode()).read().splitlines()
                lines = [l.strip() for l in raw if l.strip()]
                last_id = None
                for line in lines:
                    new_hypothesis = {
                        "hypothesis_id": uuid.uuid4().hex[:8],
                        "title": line,
                        "data": st.session_state["current_data"],
                        "analysis_plan": [],
                    }
                    st.session_state["analyses"].append(new_hypothesis)
                    last_id = new_hypothesis["hypothesis_id"]
                if last_id:
                    st.session_state["selected_hypothesis_id"] = last_id
                st.success(f"Imported {len(lines)} hypotheses.")
                st.rerun()



# ── Sidebar navigation (must come after at least one hypothesis exists) ────────
render_sidebar(show_steps=False)

# Ready to move on?
ready = ("current_data" in st.session_state
         and st.session_state["current_data"] is not None
         and st.session_state["analyses"])

st.divider()

if ready:
    st.success("Data and at least one hypothesis are ready.")
    if st.button("Go to analysis plan"):
        st.switch_page("pages/3_Analysis_Plan.py")
else:
    st.warning("Upload data and add at least one hypothesis to continue.")
