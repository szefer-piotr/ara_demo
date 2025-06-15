# pages/3_Analysis_Plan.py
"""
Analysis-Plan workflow
────────────────────────────────────────────────────────────────────────────
1. Generate a draft plan → chat with LLM → Accept plan
2. Step-by-step execution with Finish / Edit gating
3. Overview mode lists every step plus all runs
4. Chat that returns "code" chunks automatically records a new run
"""
from __future__ import annotations

import uuid
import os
import streamlit as st

from typing import Dict, List
from utils import mock_llm, create_web_search_tool, create_code_interpreter_tool, get_llm_response
from sidebar import render_sidebar
from openai import OpenAI
from dotenv import load_dotenv
from instructions import analysis_steps_generation_instructions

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# if "openai_client" not in st.session_state:
#     st.session_state.openai_client = OpenAI()

# ───────────────────────── helpers ──────────────────────────
def first_chunk(
    chunks: List[Dict[str, str]], _type: str, default: str = ""
) -> str:
    """Return first chunk of given type (e.g. 'text', 'code')."""
    return next((c["content"] for c in chunks if c["type"] == _type), default)


def record_run(step: Dict, chunks: List[Dict[str, str]], code_input: str) -> Dict:
    """Create a run dict from LLM chunks and append to step['runs']."""    
    run = {
        "run_id": uuid.uuid4().hex[:8],
        "code_input": code_input,
        "images":  [c["content"] for c in chunks if c["type"] == "image"],
        "tables":  [c["content"] for c in chunks if c["type"] == "table"],
        "summary": first_chunk(chunks, "text"),
    }
    step["runs"].append(run)
    return run


# ───────────────────────── guards ───────────────────────────
if "analyses" not in st.session_state or not st.session_state["analyses"]:
    st.error("You need to create a hypothesis first.")
    st.stop()

if "images" not in st.session_state:
    st.session_state.images = {}

if "container" not in st.session_state:
    st.session_state.container = None

# left-hand navigation (steps visible on this page)
render_sidebar()

hid  = st.session_state["selected_hypothesis_id"]
hypo = next(a for a in st.session_state["analyses"] if a["hypothesis_id"] == hid)

# ensure extra fields exist
hypo.setdefault("plan_accepted", False)
plan_chat = hypo.setdefault("plan_chat", [])
plan      = hypo.setdefault("analysis_plan", [])

st.title("Analysis plan")
st.header(hypo["title"])

tools = [create_code_interpreter_tool(st.session_state.container), create_web_search_tool()]

# ─────────────────────── DRAFT-PLAN FLOW ─────────────────────
if not plan:
    if st.button("Generate draft plan"):
        with st.spinner("LLM drafting…"):
            lines = first_chunk(
                get_llm_response(
                    client=st.session_state.openai_client,
                    model="gpt-4o-mini",
                    prompt=f"Generate analysis plan for: {hypo['title']}",
                    instructions=analysis_steps_generation_instructions,
                    tools=[],
                    context=st.session_state.data_summary,
                ), _type = "text"
            )
                # mock_llm(f"Generate analysis plan for: {hypo['title']}"), "text").splitlines()

        hypo["analysis_plan"] = lines


        # hypo["analysis_plan"] = [
        #     {
        #         "step_id": uuid.uuid4().hex[:8],
        #         "title":  ln.strip(),
        #         "code":   "# write code here\n",
        #         "runs":   [],
        #         "chat_history": [],
        #         "finished": False,
        #         "text":    "",      # ensure keys exist
        #         "images":  [],
        #     }
        #     for ln in filter(None, lines)
        # ]

        st.success("Draft created. Review and accept.")
        st.rerun()

    st.stop()

# ─────────────────────── REVIEW-AND-ACCEPT ───────────────────
if not hypo["plan_accepted"]:
    st.subheader("Draft plan (not yet accepted)")
    for i, s in enumerate(plan, 1):
        st.markdown(f"{i}. **{s['title']}**")

    st.divider()
    st.markdown("#### Discuss plan")
    for role, msg in plan_chat:
        st.chat_message(role).write(msg)

    prompt = st.chat_input("Talk about the plan")
    if prompt:
        plan_chat.append(("user", prompt))
        chunks = mock_llm(
            prompt,
            history=[{"type": "text", "content": m} for _, m in plan_chat],
        )
        reply = first_chunk(chunks, "text", "[No response]")
        plan_chat.append(("assistant", reply))
        st.chat_message("assistant").write(reply)

        if any(c["type"] == "code" for c in chunks):
            st.toast("Code chunk detected – stored for reference.", icon="💾")

    col_regen, col_accept = st.columns(2)
    if col_regen.button("Regenerate plan"):
        hypo["analysis_plan"].clear()
        st.rerun()
    if col_accept.button("Accept plan", type="primary"):
        hypo["plan_accepted"] = True
        st.session_state["selected_step_id"] = hypo["analysis_plan"][0]["step_id"]
        st.rerun()

    st.stop()

# ─────────────────────── ACCEPTED-PLAN UI ────────────────────
if st.button("Edit plan (re-draft)"):
    hypo["plan_accepted"] = False
    st.session_state["selected_step_id"] = None
    st.rerun()

sid = st.session_state.get("selected_step_id")   # may be None (Overview)

# ───────────────────────── OVERVIEW ──────────────────────────
if sid is None:
    st.subheader("Overview – every step & run")
    for idx, s in enumerate(plan, 1):
        st.markdown(f"### {idx}. {s['title']}")
        if not s["runs"]:
            st.info("No runs yet for this step.")
            st.divider()
            continue
        for r in s["runs"]:
            st.markdown(f"**Run `{r['run_id']}`**")
            if r["summary"]:
                st.write(r["summary"])
            for img in r["images"]:
                st.markdown(img, unsafe_allow_html=True)
            for tbl in r["tables"]:
                st.markdown(tbl, unsafe_allow_html=True)
            with st.expander("Code"):
                st.code(r["code_input"], language="python")
            st.markdown("---")
        st.divider()
    st.stop()

# ─────────────────────── SINGLE-STEP MODE ───────────────────
step_idx = next(i for i, s in enumerate(plan) if s["step_id"] == sid)
step     = plan[step_idx]

# ensure legacy steps have the keys
step.setdefault("text", "")
step.setdefault("images", [])

# sequential gating
prev_done = step_idx == 0 or plan[step_idx - 1]["finished"]
if not prev_done:
    st.warning("Finish the previous step first.")
    st.stop()

main, arte = st.columns([3, 1], gap="medium")

# ---------------- MAIN PANEL ----------------
with main:
    st.subheader(step["title"])

    # action buttons
    if step["finished"]:
        if st.button("Edit step"):
            step["finished"] = False
            st.rerun()
    else:
        col_run, col_finish = st.columns(2)
        if col_run.button("Run step"):
            chunks = mock_llm(step["code"], tools=["code interpreter"])
            record_run(step, chunks, step["code"])
            st.success("Run stored.")
            st.rerun()
        if step["runs"] and col_finish.button("Finish step", type="primary"):
            step["finished"] = True
            st.success("Step marked as finished.")
            st.rerun()

    # latest run display
    if step["runs"]:
        latest = step["runs"][-1]
        st.markdown(f"##### Latest run `{latest['run_id']}`")
        if latest["summary"]:
            st.write(latest["summary"])
        for img in latest["images"]:
            st.markdown(img, unsafe_allow_html=True)
        for tbl in latest["tables"]:
            st.markdown(tbl, unsafe_allow_html=True)
        with st.expander("Code"):
            st.code(latest["code_input"], language="python")
    else:
        st.info("No runs yet – click **Run step** or chat.")

    # step-level chat (only if unlocked)
    if not step["finished"]:
        for role, msg in step["chat_history"]:
            st.chat_message(role).write(msg)
        prompt = st.chat_input("Discuss this step")
        if prompt:
            step["chat_history"].append(("user", prompt))
            chunks = mock_llm(
                prompt,
                history=[{"type": "text", "content": m} for _, m in step["chat_history"]],
            )
            assistant_text = first_chunk(chunks, "text", "[No response]")
            step["chat_history"].append(("assistant", assistant_text))
            st.chat_message("assistant").write(assistant_text)

            # auto-run code chunks
            if any(c["type"] == "code" for c in chunks):
                code_chunk = first_chunk(chunks, "code")
                record_run(step, chunks, code_input=code_chunk)
                st.success("Run stored from chat.")
                st.rerun()

# ---------------- ARTIFACT SIDEBAR ----------------
with arte:
    st.subheader("Artifacts")
    if not step["runs"]:
        st.info("No runs yet.")
    else:
        for r in reversed(step["runs"]):
            with st.expander(f'Run {r["run_id"]}'):
                col_rm, col_ct = st.columns([1, 4])
                if col_rm.button("🗑️", key=f"del_{r['run_id']}"):
                    step["runs"] = [
                        x for x in step["runs"] if x["run_id"] != r["run_id"]
                    ]
                    st.rerun()
                col_ct.code(r["code_input"], language="python")
                for html in r["images"]:
                    col_ct.markdown(html, unsafe_allow_html=True)
                for tbl in r["tables"]:
                    col_ct.markdown(tbl, unsafe_allow_html=True)
                if r["summary"]:
                    col_ct.markdown(r["summary"])
