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
    data_file = st.file_uploader(label="", type="csv")
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
with st.form(key="typed_hypothesis"):
    text = st.text_input("")
    submitted = st.form_submit_button("Add")
    if submitted and text.strip():
        st.session_state["analyses"].append(
            {
                "hypothesis_id": uuid.uuid4().hex[:8],
                "title": text.strip(),
                "data": st.session_state["current_data"],
                "analysis_plan": [],
            }
        )
        st.success("Hypothesis added.")

st.write("…or upload a .txt file (one hypothesis per line)")

with st.form(key="txt_hypothesis"):    
    txt_file = st.file_uploader("",
                                type="txt")
    uploaded = st.form_submit_button("Import")
    if uploaded and txt_file:
        raw = StringIO(txt_file.getvalue().decode()).read().splitlines()
        lines = [l.strip() for l in raw if l.strip()]
        for line in lines:
            st.session_state["analyses"].append(
                {
                    "hypothesis_id": uuid.uuid4().hex[:8],
                    "title": line,
                    "data": st.session_state["current_data"],
                    "analysis_plan": [],
                }
            )
        st.success(f"Imported {len(lines)} hypotheses.")

# ── List current hypotheses ────────────────────────────────────────────────────
# if st.session_state["analyses"]:
#     st.subheader("Existing hypotheses")
#     for a in st.session_state["analyses"]:
#         st.write(f"{a['hypothesis_id']}: {a['title']}")
# else:
#     st.info("None yet – add at least one before proceeding.")
#     st.stop()

# ── Sidebar navigation (must come after at least one hypothesis exists) ────────
render_sidebar()

# ── STEP 3 ─ Build / edit analysis plan for the selected hypothesis ────────────
chosen = next(
    h for h in st.session_state["analyses"]
    if h["hypothesis_id"] == st.session_state["selected_hypothesis_id"]
)
# st.header(f"3. Analysis plan – {chosen['title']}")

# Add new step
# if st.button("Add analysis step"):
#     step = {
#         "step_id": uuid.uuid4().hex[:8],
#         "code": "",
#         "images": [],     # list of {"img_id": str, "html": str}
#         "text": "",
#     }
#     chosen["analysis_plan"].append(step)
#     st.session_state["selected_step_id"] = step["step_id"]

# If there is at least one step, show the editor
if chosen["analysis_plan"]:
    step = next(s for s in chosen["analysis_plan"]
                if s["step_id"] == st.session_state["selected_step_id"])

    st.subheader(f"Editing step {step['step_id']}")

    # ——— Editable text description ———
    step["text"] = st.text_area("Narrative / observations", value=step["text"])

    # ——— Editable code ———
    step["code"] = st.text_area(
        "Python code to execute",
        value=step["code"],
        height=200,
        help="This is a stub – execution is mocked."
    )

    # ——— Images list ———
    st.write("Images (HTML snippets)")
    col1, col2 = st.columns([3, 1])
    with col1:
        for idx, img in enumerate(step["images"]):
            st.markdown(f"{idx+1}. {img['html']}", unsafe_allow_html=True)
    with col2:
        new_html = st.text_input("Add image (HTML)")
        if st.button("Add"):
            if new_html.strip():
                step["images"].append(
                    {"img_id": uuid.uuid4().hex[:8], "html": new_html.strip()}
                )
                st.rerun()

    st.divider()

    # ── Chat to refine this step (mocked) ───────────────────
    if "chat_history" not in step:
        step["chat_history"] = []   # [(role, message)]

    for role, msg in step["chat_history"]:
        st.chat_message(role).write(msg)

    prompt = st.chat_input("Ask the LLM about this step")
    
    if prompt:
        # Record the user turn
        step["chat_history"].append(("user", prompt))

        # One unified mock-LLM call – show a spinner while it “thinks”
        with st.spinner("LLM is thinking…"):
            chunks = mock_llm(
                prompt,
                history=[{"type": "text", "content": m} for _, m in step["chat_history"]],
            )

    # Extract the first text chunk to display
    reply_text = next(c["content"] for c in chunks if c["type"] == "text")
    step["chat_history"].append(("assistant", reply_text))
    st.chat_message("assistant").write(reply_text)


    

    # ── Run / save button (mock execution) ──────────────────
    if st.button("Run & save"):
        # In a real app this would execute `step['code']` and
        # attach outputs to `step['images']` / `step['text']`.
        st.success("Mock execution complete – results saved.")
# else:
    # st.info("Add at least one analysis step to begin editing.")

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