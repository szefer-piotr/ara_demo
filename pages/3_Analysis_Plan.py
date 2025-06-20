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
import openai

from typing import Dict, List

import utils
from utils import (
    create_container,
    upload_csv_and_get_file_id,
    create_web_search_tool, 
    create_code_interpreter_tool, 
    get_llm_response,
    to_mock_chunks,
    ordered_to_bullets,
    first_chunk,
    record_run,
    serialize_previous_steps,
    plan_to_string,
    history_to_string,

)

from sidebar import render_sidebar
from openai import OpenAI
from dotenv import load_dotenv
from instructions import (
    analysis_steps_generation_instructions,
    analysis_step_execution_instructions,
    run_execution_prompt,
    run_execution_chat_instructions
)
from schemas import AnalysisStep, AnalysisPlan
import re

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


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

# Plan is assigned to hypo['analysis_plan']
plan      = hypo.setdefault("analysis_plan", [])

st.title("Analysis plan")
st.header(hypo["title"])

tools = [create_code_interpreter_tool(st.session_state.container), create_web_search_tool()]

client = st.session_state.openai_client

# ─────────────────────── DRAFT-PLAN FLOW ─────────────────────
if not plan:
    if st.button("Generate draft plan"):
        
        context = str(st.session_state.column_summaries)

        with st.spinner("LLM drafting…"):
            
            response = client.responses.parse(
                    model="gpt-4o-mini",
                    tools=[create_web_search_tool()],
                    input=[
                        {"role": "system", "content": context},
                        {"role": "user", "content": f"Considerig the attached data summary generate analysis plan for: {hypo['title']}"}
                    ],
                    instructions=analysis_steps_generation_instructions,
                    text_format=AnalysisPlan,
                )
            
                # mock_llm(f"Generate analysis plan for: {hypo['title']}"), "text").splitlines()

        plan_schema: AnalysisPlan = response.output_parsed      # pick one
        
        hypo["analysis_plan"] = [
            {
                "step_id": uuid.uuid4().hex[:8],
                "title":   step.step_title.strip(),
                "text":    step.step_text.rstrip(),
                "code":    "# write code here\n",
                "runs":    [],
                "chat_history": [],
                "finished": False,
                "images":  [],
            }
            for step in plan_schema.steps
        ]

        st.success("Draft created. Review and accept.")
        st.rerun()

    st.stop()

# ─────────────────────── REVIEW-AND-ACCEPT ───────────────────
if not hypo["plan_accepted"]:
    # Plan is not yet accepted, so we display the draft.
    st.subheader("Draft plan (not yet accepted)")
    
    # Display the plan
    for i, s in enumerate(plan, 1):
        st.markdown(f"{i}. **{s['title']}**")
        st.markdown(ordered_to_bullets(s['text']))

    st.divider()

    st.markdown("#### Discuss plan")
    
    # Display the chat history
    for role, msg in plan_chat:
        if role == "user":
            st.chat_message(role).write(msg)
        elif role == "assistant":
            # print(msg)
            for i, s in enumerate(msg.steps):
                # print(f"\n\nSINGLE STEP:\n\n{s}")
                st.markdown(f"{i+1}. **{s.step_title}**")
                st.markdown(ordered_to_bullets(s.step_text))

    # When prompt is sent, then save the parsed output in teh chat_history, and display it in the chat history.
    prompt = st.chat_input("Talk about the plan")

    if prompt:
        
        plan_chat.append(("user", prompt))

        with st.spinner("Thinking..."):
            # Original plan
            original_plan = hypo["analysis_plan"]
            # Available chat history
            plan_chat_history=[{"type": "input_text", "content": m} for _, m in plan_chat]
            
            # Convert plan an history to strings for context.
            plan_str = plan_to_string(original_plan)
            history_str = history_to_string(plan_chat_history)

            final_string = f"Original Plan:\n\n{plan_str}\n\nChat History:\n\n{history_str}"
            
            # print(f"\n\nFinal string:\n\n{final_string}")
            # print(f"\n\nOriginal plan:\n\n{original_plan}.\n\nThe history:\n\n{plan_chat_history}")
            
            response = client.responses.parse(
                model="gpt-4o-mini",
                tools=[create_web_search_tool()],
                instructions=run_execution_chat_instructions,
                input=[
                    {"role": "system", "content": final_string},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                text_format=AnalysisPlan,
            )

        # Assistant response parsed as AnalysisPlan
        plan_schema: AnalysisPlan = response.output_parsed

        plan_chat.append(("assistant", plan_schema))

        # print(f"Response generated:\n\n{response.output_parsed}")

        st.rerun()

    col_regen, col_accept = st.columns(2)
    
    if col_regen.button("Regenerate plan"):
        hypo["analysis_plan"].clear()
        st.rerun()
    
    if col_accept.button("Accept plan", type="primary"):

        print(f"\n\nACCEPTED PLAN:\n\n{plan_chat[-1][1]}")


        plan_schema: AnalysisPlan = plan_chat[-1][1]


        hypo["analysis_plan"] = [
            {
                "step_id": uuid.uuid4().hex[:8],
                "title":   step.step_title.strip(),
                "text":    step.step_text.rstrip(),
                "code":    "# write code here\n",
                "runs":    [],
                "chat_history": [],
                "finished": False,
                "images":  [],
            }
            for step in plan_schema.steps
        ]
        
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
        st.markdown(f"{s['text']}")
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
    st.write(step["text"])

    # action buttons
    if step["finished"]:
        if st.button("Edit step"):
            step["finished"] = False
            st.rerun()
    else:
        col_run, col_finish = st.columns(2)
        if col_run.button("Run step"):
            
            # TODO: reactivate the container if expired            
            current = step["step_id"]

            context_for_llm = serialize_previous_steps(
                hypo["analysis_plan"],
                current_step_id=current,
                include_current=False  # usually you exclude the one you’re about to run
            )

            with st.spinner("Runing the step..."):
                try:
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        tools=tools,
                        instructions=analysis_step_execution_instructions,
                        input=[
                            {"role": "system", "content": context_for_llm},
                            {"role": "user", 
                            "content": run_execution_prompt}
                        ],
                        temperature=0,
                        stream=False,
                    )
                    chunks = to_mock_chunks(response)
                    record_run(step, chunks)
                    print(f"\n\nRun for a {step['step_id']} RECORDED:\n\n{step}")
                    st.success("Run stored.")
                    st.rerun()

                except openai.BadRequestError as e:
                    if 'expired' in str(e).lower():
                        # Container expired, create a new one
                        new_container = create_container(st.session_state.openai_client, st.session_state["file_ids"])
                        print((f"Container expired, created a new one: {new_container.id}"))
                        st.session_state.container = new_container
                        
                        # Re-run the step with the new container
                        tools = [create_code_interpreter_tool(new_container), create_web_search_tool()]
                        
                        response = client.responses.create(
                        model="gpt-4o-mini",
                        tools=tools,
                        instructions=analysis_step_execution_instructions,
                        input=[
                                {"role": "system", "content": context_for_llm},
                                {"role": "user", 
                                "content": run_execution_prompt}
                            ],
                            temperature=0,
                            stream=False,
                        )
                        chunks = to_mock_chunks(response)
                        record_run(step, chunks)
                        print(f"\n\nRun for a {step['step_id']} RECORDED:\n\n{step}")
                        st.success("Run stored.")
                        st.rerun()
                    else:
                        raise
        
        if step["runs"] and col_finish.button("Finish step", type="primary"):
            step["finished"] = True
            st.success("Step marked as finished.")
            st.rerun()

    # latest run display
    if step["runs"]:
        latest = step["runs"][-1]
        st.markdown(f"##### Latest run `{latest['run_id']}`")
        if latest["summary"]:
            st.write("".join(latest["summary"]))
        for img in latest["images"]:
            st.markdown(img, unsafe_allow_html=True)
        for tbl in latest["tables"]:
            st.markdown(tbl, unsafe_allow_html=True)
        with st.expander("Code"):
            st.code("".join(latest["code_input"]), language="python")
    else:
        st.info("No runs yet – click **Run step** or chat.")

    # step-level chat (only if unlocked)
    if not step["finished"]:
        # Display the 
        for role, msg in step["chat_history"]:
            st.chat_message(role).write(msg)
        
        prompt = st.chat_input("Discuss this step")
        if prompt:
            step["chat_history"].append(("user", prompt))

            # TODO - discuss the step allow to run the run and register it, 
            # display chat history but display overview for the run results.

            print(f"\n\nRAW History{step['chat_history']}")

            history=[{"type": "text", "content": m} for _, m in step["chat_history"]]

            print(f"\n\nPARSED History{history}")

            response = client.responses.create(
                model="gpt-4o-mini",
                tools=tools,
                instructions=run_execution_chat_instructions,
                input=[
                    {"role": "system", "content": history},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                stream=False,
            )

            chunks = to_mock_chunks(response)

            print(f"\n\nCHUNKS: {chunks}")

            step["chat_history"].append(("assistant", chunks))
            
            st.chat_message("assistant").write(chunks)

            # auto-run code chunks
            if any(c["type"] == "code" for c in chunks):
                code_chunk = first_chunk(chunks, "code")
                record_run(step, chunks)
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
