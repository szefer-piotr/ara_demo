import streamlit as st
import pandas as pd
import uuid
from io import StringIO

from utils import mock_llm
from sidebar import render_sidebar

# ── Initialise session containers ──────────────────────────────────────────────
if "analyses" not in st.session_state:
    st.session_state["analyses"] = []      # list of hypothesis-level dictionaries

# ── STEP 1 ─ Upload data ───────────────────────────────────────────────────────
st.markdown("#### Upload data (CSV)")

if "current_data" not in st.session_state:
    st.session_state["current_data"] = None

if st.session_state["current_data"] is None:
    st.write("Choose a CSV file")
    data_file = st.file_uploader(label="Upload data", type="csv")
    if data_file:
        df = pd.read_csv(data_file)
        st.session_state["current_data"] = df
        st.success("Dataset loaded and ready.")
        st.dataframe(df.head(), use_container_width=True)
        # st.stop()            # stop execution until data is present
else:
    st.write("Dataset already loaded.")
    st.dataframe(st.session_state["current_data"].head(), use_container_width=True)

st.divider()

# ── STEP 2 ─ Add hypotheses ────────────────────────────────────────────────────
st.markdown("#### Add hypotheses")

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

st.write("…or upload a .txt file (one hypothesis per line)")

with st.form(key="txt_hypothesis"):    
    txt_file = st.file_uploader("Uploas a TXT file with one hypothesis per line", 
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

# ── List current hypotheses ────────────────────────────────────────────────────
if st.session_state["analyses"]:
    st.subheader("Existing hypotheses")

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
    st.stop()

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