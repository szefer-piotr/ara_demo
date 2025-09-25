import streamlit as st
import pandas as pd
import uuid
from io import StringIO
import csv
# Change from: import utils
# To:
from utils import (
    robust_read_csv, 
    create_container, 
    create_web_search_tool, 
    create_code_interpreter_tool, 
    edit_column_summaries,
    inject_global_css,
    mock_llm,
    get_llm_response
)

from sidebar import render_sidebar
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Any, List
from schemas import ColumnSummary, DatasetSummary
from utils.prompt_templates import data_summary_instructions

st.set_page_config(layout="wide")

inject_global_css()

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# ── Initialise session containers ──────────────────────────────────────────────
if "analyses" not in st.session_state:
    st.session_state["analyses"] = []      # list of hypothesis-level dictionaries

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(base_url="http://localhost:4000/v1")

if "file_ids" not in st.session_state:
    st.session_state["file_ids"] = []

if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

# ── STEP 1 ─ Upload data ───────────────────────────────────────────────────────

st.markdown("""
<div style="background-color:#fff3cd; padding: 1em; border-radius: 10px; border: 1px solid #ffeeba;">
<h4>📊 Step 1: Upload Data & Add Hypotheses</h4>
<ul>
<li>Upload your <strong>CSV file</strong>. The app will automatically summarize your dataset.</li>
<li>Edit column descriptions if needed. **Be sure to provide accurate variable type**. For example, is it an integer, continuos or a categorical variable.</li>
<li>Add one or more <strong>hypotheses</strong> — type them in or import from a <code>.txt</code> file.</li>
</ul>
<p>✅ Once your data is uploaded <strong>and</strong> at least one hypothesis is added, click <em>“Go to analysis plan”</em> to continue.</p>
</div>
""", unsafe_allow_html=True)


# Put the CSV uploader in a centred, narrower container
st.markdown("""\n""")
st.markdown("##### Upload your data")

with st.container():
    col_left, col_mid, col_right = st.columns([1, 28, 1])
    with col_mid:
        
        if "current_data" not in st.session_state:
            st.session_state["current_data"] = None

        if st.session_state["current_data"] is None:

            data_file = st.file_uploader(label="Simply drag and drop or use the Browse files button.", type="csv",label_visibility="collapsed")
            
            # if data_file:
            #     if data_file.type != "text/csv":
            #         raise ValueError("Uploaded file is not a CSV file.")

            #     # Read a sample for delimiter sniffing & encoding detection
            #     raw_bytes = data_file.read(4096)
            #     data_file.seek(0)

            #     # --- Try to detect encoding
            #     detected = chardet.detect(raw_bytes)
            #     encoding = detected["encoding"] or "utf-8"

            #     # Try to decode for delimiter sniffing
            #     try:
            #         sample = raw_bytes.decode(encoding, errors="replace")
            #     except Exception:
            #         # Fallback to UTF-8 or Latin1 if detection fails
            #         encoding = "utf-8"
            #         try:
            #             sample = raw_bytes.decode(encoding)
            #         except Exception:
            #             for enc in ("utf-8", "latin1", "cp1250"):
            #                 try:
            #                     sample = raw_bytes.decode(enc, errors="replace")
            #                     encoding = enc
            #                     break
            #                 except Exception:
            #                     continue

            #     # --- Detect delimiter
            #     try:
            #         sample = raw_bytes.decode(encoding, errors="replace")
            #         sniffer = csv.Sniffer()
            #         dialect = sniffer.sniff(sample, delimiters=";,|\t")
            #         delimiter = dialect.delimiter
            #     except Exception:
            #         delimiter = ","

            #     # Now read the file with detected encoding and delimiter
            #     data_file.seek(0)
                
            #     try:
            #         df = pd.read_csv(data_file, delimiter=delimiter, encoding=encoding)
            #         st.session_state["current_data"] = df
            #     except Exception as e:
            #         # 👉 Friendly message to the user
            #         st.error(
            #             "❌ **Could not read your file.** "
            #             "Please make sure it’s saved as a CSV (not Excel) and encoded in UTF-8 if possible."
            #         )
            #         # (Optional) log the real error for debugging:
            #         st.write(f"Details for developer: {e}")
                
            #     data_file.seek(0)

            #     if "current_data" in st.session_state:
            #         file_id = utils.upload_csv_and_get_file_id(st.session_state.openai_client, data_file)
            #         st.session_state["file_ids"].append(file_id)
            #         container = create_container(st.session_state.openai_client, [file_id])
            #         st.session_state.container = container

            if data_file:
                if data_file.type != "text/csv":
                    st.error("❌ Uploaded file is not a CSV.")
                    st.stop()

                try:
                    df, enc_used, delim_used = robust_read_csv(data_file)
                    st.session_state["current_data"] = df
                    st.success(f"Loaded {len(df):,} rows "
                            f"(encoding = **{enc_used}**, delimiter = **'{delim_used}'**).")
                except UnicodeDecodeError:
                    st.error(
                        "❌ **Could not read your file.** "
                        "Please ensure it’s saved as a CSV (not XLS/XLSX) and, if possible, "
                        "use UTF-8 or CP1250 encoding."
                    )
                    st.stop()

                # ― continue with OpenAI upload here ―
                data_file.seek(0)
                file_id = utils.upload_csv_and_get_file_id(
                    st.session_state.openai_client, data_file
                )
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
            st.markdown("##### Your dataset preview.")
            st.dataframe(st.session_state["current_data"].head(), use_container_width=True)
            # st.json(st.session_state.column_summaries)
            
            # ---------- Column list / Edit button ----------
            st.markdown("##### Dataset summary")
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


st.markdown("##### Add hypotheses")

# Put the hypothesis inputs in the same centred, narrower container
with st.container():
    col_left, col_mid, col_right = st.columns([1, 28, 1])
    with col_mid:
        # Forms allow us to avoid partial reruns while typing
        with st.form(key="typed_hypothesis", clear_on_submit=True):
            text = st.text_input(label="Type a new hypothesis", key="hypothesis_input", label_visibility="collapsed")
            submitted = st.form_submit_button("Add")
            if submitted and text.strip():
                new_hypothesis = {
                    "hypothesis_id": uuid.uuid4().hex[:8],
                    "title": text.strip(),
                    "data": st.session_state.column_summaries,
                    "analysis_plan": [],
                }
                st.session_state["analyses"].append(new_hypothesis)
                st.session_state["selected_hypothesis_id"] = new_hypothesis["hypothesis_id"]
                st.success("Hypothesis added.")
                st.rerun()

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
                        "data_summary": st.session_state["current_data"],
                        "analysis_plan": [],
                    }
                    st.session_state["analyses"].append(new_hypothesis)
                    last_id = new_hypothesis["hypothesis_id"]
                if last_id:
                    st.session_state["selected_hypothesis_id"] = last_id
                st.success(f"Imported {len(lines)} hypotheses.")
                st.rerun()


# ── List current hypotheses ────────────────────────────────────────────────────
st.markdown("##### Your hypotheses")
st.write("""
Your hypotheses will appear here. You can remove them or add new ones below.
When you're ready, click Refine Hypotheses — the Assistant will improve your hypotheses using your data, web search, and its knowledge.
""")
if st.session_state["analyses"]:

    for idx, a in enumerate(st.session_state["analyses"]):
        col1, col2 = st.columns([9, 1])          # wide text · narrow icon
        with col1:
            st.write(f"📝 **{a['title']}** — (ID: {a['hypothesis_id']})")
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

# ── Sidebar navigation (must come after at least one hypothesis exists) ────────
render_sidebar(show_steps=False)

# Ready to move on?
ready = ("current_data" in st.session_state
         and st.session_state["current_data"] is not None
         and st.session_state["analyses"])

if ready:
    st.success("Data and at least one hypothesis are ready.")
    if st.button("Go to analysis plan"):
        st.switch_page("pages/3_Analysis_Plan.py")
else:
    st.warning("Upload data and add at least one hypothesis to continue.")
