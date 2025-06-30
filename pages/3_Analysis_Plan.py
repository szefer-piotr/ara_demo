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
import copy

from typing import Dict, List

import utils
from utils import (
    create_container,
    load_image_from_openai_container,
    upload_csv_and_get_file_id,
    create_web_search_tool, 
    create_code_interpreter_tool, 
    render_assistant_message,
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
# keep history of all previous plans with runs
hypo.setdefault("plan_history", [])

# Plan is assigned to hypo['analysis_plan']
plan      = hypo.setdefault("analysis_plan", [])

st.title("Analysis plan")

st.markdown("""Here the Assistant will heklp you develop a detailed analysis plan. After an initial plan is generated, if you have some specific analyses in mind that youy would like to perform ask the Assistant in the chat. You can always come back to refine the plan further.""")

st.divider()


st.markdown(f"##### Current hypothesis:\n ###### {hypo['title']}")

tools = [create_code_interpreter_tool(st.session_state.container), create_web_search_tool()]

client = st.session_state.openai_client

# ─────────────────────── DRAFT-PLAN FLOW ─────────────────────
if not plan:
    if st.button("Generate draft plan"):
        
        context = str(st.session_state.column_summaries)

        with st.spinner("LLM drafting…"):
            
            response = client.responses.parse(
                    model="gpt-4o",
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
    
    # Display chat history
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
        
        # TODO this uses tuple as a chat message, but it should be a dict with role and content.
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
                model="gpt-4o",
                tools=[create_web_search_tool()],
                instructions=run_execution_chat_instructions,
                input=[
                    {"role": "system", "content": final_string},
                    {"role": "user", "content": f"Update the original plan considering this request: {prompt}"}
                ],
                temperature=0,
                text_format=AnalysisPlan,
            )

        # Assistant response parsed as AnalysisPlan
        plan_schema: AnalysisPlan = response.output_parsed

        plan_chat.append(("assistant", plan_schema))

        # print(f"Response generated:\n\n{response.output_parsed}")

        st.rerun()

    # show confirmation warning if needed
    confirm_needed = bool(hypo.get("plan_history"))
    if st.session_state.get("confirm_plan_accept") and confirm_needed:
        st.warning("Accepting this new plan will discard runs from the previous plan.")
        if st.button("Confirm plan acceptance", key="confirm_accept_btn", type="primary"):
            st.session_state.confirm_plan_accept = False
            if not plan_chat:
                hypo["plan_accepted"] = True
                st.session_state["selected_step_id"] = hypo["analysis_plan"][0]["step_id"]
                st.rerun()
            else:
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

    col_regen, col_accept = st.columns(2)

    if col_regen.button("Regenerate plan"):
        # archive current plan before clearing it
        hypo.setdefault("plan_history", []).append(copy.deepcopy(hypo["analysis_plan"]))
        hypo["analysis_plan"] = []
        st.warning("Previous plan stored. Generate a new draft.")
        st.rerun()

    if col_accept.button("Accept plan", type="primary"):
        if confirm_needed:
            st.session_state.confirm_plan_accept = True
            st.rerun()

        if not plan_chat:
            hypo["plan_accepted"] = True
            st.session_state["selected_step_id"] = hypo["analysis_plan"][0]["step_id"]
            st.rerun()
        else:
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
            print(f"Run:, {r}")
            if r["summary"]:
                for i in r["summary"]:
                    st.write(i)
            for img in r["images"]:
                st.markdown(st.session_state.images[img], unsafe_allow_html=True)
            # for tbl in r["tables"]:
            #     st.markdown(tbl, unsafe_allow_html=True)
            for cd in r["code_input"]:
                st.code(cd, language="python")
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
    st.subheader(step["title"])
    st.write(step["text"])
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

            print(f"\n\n{'***' * 10}\n\nRunning step {step['step_id']}:\n\n{step}")

            # It losts the connection to the container file
            container = st.session_state.get("container")
            
            if not container:
                container = create_container(st.session_state.openai_client, st.session_state["file_ids"])
                st.session_state.container = container
                tools = [create_code_interpreter_tool(st.session_state.container), create_web_search_tool()]
                print(f"Created a new container: {container.id}")
            
            else:
                tools = [create_code_interpreter_tool(container), create_web_search_tool()]
                print(f"CONTAINER: {container.id}")

            current = step["step_id"]

            context_for_llm = serialize_previous_steps(
                hypo["analysis_plan"],
                # current_step_id=current,
                include_current=False
            )

            print(f"\n\n{'---'*10}\n\nContext for the current run {current}:\n\n{context_for_llm}.\n\n{'---'*10} End of context\n\n")

            run_execution_prompt = f"""
            Run the following step in Python, using the provided context and instructions:\n
            step_id: `{current}`\n
            step_title: `{step['title']}`\n
            step_text: `{step['text']}`\n
            """

            print(f"\n\nRUN EXECUTION PROMPT:\n\n{run_execution_prompt}")

            with st.spinner("Runing the step..."):

                print(f"\n\nHERE\n\n")
                
                try:
                    response = client.responses.create(
                        model="gpt-4o",
                        tools=tools,
                        instructions=analysis_step_execution_instructions,
                        input=[
                            {"role": "system", "content": context_for_llm},
                            {"role": "user", "content": run_execution_prompt}
                        ],
                        temperature=0,
                        stream=False,
                    )
                    
                    chunks = to_mock_chunks(response)
                    record_run(step, chunks)
                    print(f"\n\nRun for a {step['step_id']} RECORDED:\n\n{step}")
                    st.success("Run stored.")
                    st.rerun()

                except openai.APIStatusError as e:
                    print(f"\n\n ENCOUNTERED ERROR: {e}")
                    if 'expired' in str(e).lower():
                        # Container expired, create a new one
                        new_container = create_container(st.session_state.openai_client, st.session_state["file_ids"])
                        print((f"Container expired, created a new one: {new_container.id}"))
                        st.session_state.container = new_container
                        
                        # Re-run the step with the new container
                        tools = [create_code_interpreter_tool(new_container), create_web_search_tool()]
                        
                        response = client.responses.create(
                        model="gpt-4o",
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
                        print(f"\n\nSOMETHIG WENT WRONG...\n\n")
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
            st.write("".join(latest["summary"])) # we only have this...
        for img in latest["images"]:
            st.image(st.session_state.images[img])
        for tbl in latest["tables"]:
            st.markdown(tbl, unsafe_allow_html=True)
        with st.expander("Code"):
            st.code("".join(latest["code_input"]), language="python") # ... and this
    else:
        st.info("No runs yet – click **Run step** or chat.")

    # step-level chat (only if unlocked)
    if not step["finished"]:
        
        # Display the previous chat history
        for msg in step["chat_history"]:
            # Display the chat history
            if msg['role'] == 'user':
                st.chat_message(msg['role']).write(msg['content'])
            elif msg['role'] == 'assistant':
                render_assistant_message(msg['content'])
        
        prompt = st.chat_input("Discuss this step")
        
        # Also parsed previous steps for context
        previous_steps = serialize_previous_steps(
            hypo["analysis_plan"],
            current_step_id=step["step_id"],
            include_current=False
        )
        
        if prompt:
            # step["chat_history"].append(("user", prompt))

            st.chat_message('user').write(prompt)
            
            step["chat_history"].append({"role": "user", "content": prompt})

            # TODO - discuss the step allow to run the run and register it, 
            # display chat history but display overview for the run results.

            print(f"\n\nRAW History{step['chat_history']}")

            history= "".join(f"{m['role']}: {m['content']}" for m in step["chat_history"])

            

            print(f"\n\nPREVIOUS STEPS:\n\n{previous_steps}")
            print(f"\n\nPARSED History{history}")

            context_for_a_run = f"""
                Chat history for the current step:\n
                {history}\n\n
                Previous Steps:\n
                {previous_steps}
            """

            with st.spinner("Thinking..."):
                try:
                    response = client.responses.create(
                        model="gpt-4o",
                        tools=tools,
                        instructions=run_execution_chat_instructions,
                        input=[
                            {"role": "system", "content": context_for_a_run},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        stream=False,
                    )


                except openai.BadRequestError as e:
                        if 'expired' in str(e).lower():
                            # Container expired, create a new one
                            new_container = create_container(st.session_state.openai_client, st.session_state["file_ids"])
                            print((f"Container expired, created a new one: {new_container.id}"))
                            st.session_state.container = new_container
                            
                            # Re-run the step with the new container
                            tools = [create_code_interpreter_tool(new_container), create_web_search_tool()]
                            
                            response = client.responses.create(
                            model="gpt-4o",
                            tools=tools,
                            instructions=analysis_step_execution_instructions,
                            input=[
                                {"role": "system", "content": history},
                                {"role": "user", "content": prompt}
                            ],
                                temperature=0,
                                stream=False,
                            )
                            
                            # This has to b ethe same as outside the try block
                            print(f"\n\nLLM RESPONSE:\n\n{response}")
                            chunks = to_mock_chunks(response)
                            print(f"\n\nCHUNKS: {chunks}")
                            # input("Press Enter to continue...")
                            
                            # Add the response to the chat history
                            step["chat_history"].append({'role': 'assistant', 'content': chunks})
                            
                            # record a run ate every interaction
                            record_run(step, chunks)
                            st.success("Run stored from chat.")
                            st.rerun()
                        
                        else:
                            raise

            print(f"\n\nLLM RESPONSE:\n\n{response}")
            chunks = to_mock_chunks(response)
            print(f"\n\nCHUNKS: {chunks}")
            # Add the response to the chat history
            step["chat_history"].append({'role': 'assistant', 'content': chunks})
            # record a run at every interaction
            record_run(step, chunks)
            st.success("Run stored from chat.")
            st.rerun()

# ---------------- ARTIFACT SIDEBAR ----------------
# small_css = "font-size:0.8rem;"
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
                
                for code in r["code_input"]:
                    print(f"\n\nCODE:\n\n{code}")
                    st.code(code, language="python")

                for img in r["images"]:
                    image_to_display = st.session_state.images.get(img, None)
                    if image_to_display:
                        col_ct.image(st.session_state.images[img])
                    else:
                        print(f"\n\nIMAGE ID:\n\n{img}")
                        # container = st.session_state.get("container")
                        # if not container:
                        #     st.error("No container available. Please run the step first.")
                        #     continue
                        # # Load image from OpenAI container if not in session state
                        # image_to_display = load_image_from_openai_container(OPENAI_API_KEY, container.id, file_id=img)
                        # st.session_state.images[img] = image_to_display
                        # col_ct.image(image_to_display)
                
                for text in r['summary']:
                    col_ct.markdown(f"<p style='font-size:0.8rem;'>{text}</p>",
                    unsafe_allow_html=True)
