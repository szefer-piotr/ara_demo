# pages/4_Final_Report.py
import streamlit as st
import openai

from utils import (
    create_code_interpreter_tool, 
    create_web_search_tool, 
    create_container, 
    to_mock_chunks, 
    render_assistant_message
)
from instructions import report_generation_instructions, report_chat_instructions



# ───────────────────────── guards ────────────────────────────────
if "analyses" not in st.session_state or not st.session_state["analyses"]:
    st.error("No hypotheses or runs available.")
    st.stop()

if "report_chat" not in st.session_state:
    st.session_state["report_chat"] = []  # Initialize chat history for report

# persistent selections
selected_runs: set[str] = st.session_state.setdefault("report_selected_runs", set())
preview_run_id: str | None = st.session_state.setdefault("preview_run_id", None)

st.title("📑 Final report builder")

if st.button("← Back to plan execution"):
    st.switch_page("pages/3_Analysis_Plan.py")



# layout: sidebar • main • preview
run_picker = st.sidebar
main_col, preview_col = st.columns([3, 1], gap="medium")



# ───────────────────────── 1. Run picker (left sidebar) ──────────
with run_picker:
    st.header("Select runs")

    run_clicked: str | None = None  # track which checkbox toggled to ON
    all_runs: list[dict] = []       # {"id","run"}

    for hypo in st.session_state["analyses"]:
        st.markdown(f"### {hypo['title']}")
        for step in hypo["analysis_plan"]:
            if not step["runs"]:
                continue
            st.markdown(f"**{step['title']}**")
            for run in step["runs"]:
                run_id = run["run_id"]
                label = f"Run {run_id}"
                chk_key = f"inc_{run_id}"

                prev_checked = run_id in selected_runs
                curr_checked = st.checkbox(label, value=prev_checked, key=chk_key)

                # detect toggle ON → set preview
                if curr_checked and not prev_checked:
                    run_clicked = run_id

                # update selection set
                if curr_checked:
                    selected_runs.add(run_id)
                else:
                    selected_runs.discard(run_id)

                all_runs.append({"id": run_id, "run": run})

    # after loop, update preview id
    if run_clicked:
        preview_run_id = run_clicked
    elif preview_run_id not in selected_runs:
        # current preview was un-checked; choose another or None
        preview_run_id = next(iter(selected_runs), None)

    st.session_state["report_selected_runs"] = selected_runs
    st.session_state["preview_run_id"] = preview_run_id

# ───────────────────────── 2. Preview pane (right) ───────────────
with preview_col:
    title = "Run preview" if preview_run_id is None else f"Run preview – {preview_run_id}"
    st.subheader(title)

    run_obj = next((r["run"] for r in all_runs if r["id"] == preview_run_id), None)

    if run_obj:
        if run_obj["summary"]:
            st.markdown(run_obj["summary"])
        for html in run_obj["images"]:
            st.markdown(html, unsafe_allow_html=True)
        for tbl in run_obj["tables"]:
            st.markdown(tbl, unsafe_allow_html=True)
        with st.expander("Code"):
            st.code(run_obj["code_input"], language="python")
    else:
        st.info("Tick a run to preview it here.")

# ───────────────────────── 3. Main panel – report & chat ─────────
with main_col:
    st.subheader("Generate report")

    client = st.session_state.openai_client

    if st.button("Generate final report"):
        #Clean the chat history before generating a new report
        st.session_state["report_chat"] = []

        if not selected_runs:
            st.warning("Select at least one run first.")
        else:
            ctx_lines: list[str] = []
            for hypo in st.session_state["analyses"]:
                runs_in_hypo = [
                    (step, run)
                    for step in hypo["analysis_plan"]
                    for run in step["runs"]
                    if run["run_id"] in selected_runs
                ]
                if not runs_in_hypo:
                    continue
                ctx_lines.append(f"### Hypothesis: {hypo['title']}")
                for step, run in runs_in_hypo:
                    ctx_lines.append(
                        f"- Step: {step['title']} | Run {run['run_id']} | "
                        f"Summary: {run['summary'] or 'N/A'}"
                    )

            context = "Draft a final report from this context:\n" + "\n".join(ctx_lines)
            
            print(f"\n\nSELECTED RUNS for context:\n\n{selected_runs}")
            
            with st.spinner("LLM composing report…"):

                tools = [# Create tools for code execution and web search
                    create_code_interpreter_tool(st.session_state.container),
                    create_web_search_tool()
                ]
                
                
                # -------------------------------------

                try:
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        tools=tools,
                        instructions=report_generation_instructions,
                        input=[
                            {"role": "system", "content": context},
                            {"role": "user", "content": "Draft a final report from the provided context."}
                        ],
                        temperature=0,
                        stream=False,
                    )
                    
                    chunks = to_mock_chunks(response)
                    report_txt = next(c["content"] for c in chunks if c["type"] == "text")
                    print(f"\n\nFINAL REPORT:\n\n{report_txt}")
                    st.session_state["final_report"] = report_txt
                    # st.session_state["report_chat"].append({'role': 'assistant', 'content': report_txt})
                    # st.success("Report generated!")
                    # st.rerun()

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
                        instructions=report_generation_instructions,
                        input=[
                            {"role": "system", "content": context},
                            {"role": "user", "content": "Draft a final report from the provided context."}
                        ],
                            temperature=0,
                            stream=False,
                        )
                        chunks = to_mock_chunks(response)
                        # TODO handle chunks properly
                        report_txt = next(c["content"] for c in chunks if c["type"] == "text")
                        st.session_state["final_report"] = report_txt
                        # st.session_state["report_chat"].append({'role': 'assistant', 'content': report_txt})
                        
                        # st.rerun()

                    else:
                        raise

                # -------------------------------------

            # What with the chunks?
            st.success("Report generated!")
            st.rerun()

    # ---------- show report & chat ----------
    if "final_report" in st.session_state:
        st.markdown("## Final report")
        st.write(st.session_state["final_report"])
        st.divider()
        
        st.markdown("### Discuss report")

        for message in st.session_state["report_chat"]:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            elif message['role'] == 'assistant':
                # Use the custom render function to handle the assistant's message
                render_assistant_message(message['content'])
            # st.chat_message(message["role"]).write(message["content"])

        prompt = st.chat_input("Ask about this report")

        if prompt:
            
            st.session_state["report_chat"].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            context = f""" Here is the final report":\n{st.session_state["final_report"]}.\n\n Here is the chat history:\n{st.session_state["report_chat"]} """

            print(f"\n\n{'*****' * 10}\nREPORT CHAT CONTEXT:\n\n{context}\n\n{'*****' * 10}")

            with st.spinner("LLM generating response…"):
                response = client.responses.create(
                    model="gpt-4o-mini",
                    instructions=report_chat_instructions,
                    tools=[
                        create_code_interpreter_tool(st.session_state.container),
                        create_web_search_tool()
                    ],
                    input=[
                        {"role": "system", "content": context},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    stream=False,
                )

                chunks = to_mock_chunks(response)
                
                reply = next(c["content"] for c in chunks if c["type"] == "text")
                print(f"\n\n{'*****' * 10}\nREPORT CHAT RESPONSE:\n\n{reply}\n\n{'*****' * 10}")
                
                st.session_state["report_chat"].append({'role': 'assistant', 'content': chunks})

                # st.chat_message("assistant").write(reply)

                if st.button("Update report"):

                    context = f""" Here is the final report":\n{st.session_state["final_report"]}.\n\n Here is the chat history:\n{st.session_state["report_chat"]} """

                    with st.spinner("LLM generating response…"):
                        response = client.responses.create(
                            model="gpt-4o-mini",
                            instructions=report_chat_instructions,
                            tools=[
                                create_code_interpreter_tool(st.session_state.container),
                                create_web_search_tool()
                            ],
                            input=[
                                {"role": "system", "content": context},
                                {"role": "user", "content": "Update the final report based on the chat discussion."}
                            ],
                            temperature=0,
                            stream=False,
                        )

                        chunks = to_mock_chunks(response)
                        reply = next(c["content"] for c in chunks if c["type"] == "text")
                        print(f"\n\n{'*****' * 10}\nUPDATED REPORT RESPONSE:\n\n{reply}\n\n{'*****' * 10}")

                    st.session_state["final_report"] = reply

                    st.session_state["report_chat"].append({'role': 'assistant', 'content': reply})
                    
                    st.success("Report updated!")

                st.rerun()
