# pages/4_Final_Report.py
import streamlit as st
import openai
import utils

from utils import (
    create_code_interpreter_tool, 
    create_web_search_tool, 
    create_container, 
    to_mock_chunks,
    explode_text_and_images,
    render_assistant_message
)
from instructions import report_generation_instructions, report_chat_instructions

from st_copy import copy_button

utils.inject_global_css()

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

st.markdown("""
<div style="background-color:#fff3cd; padding: 1em; border-radius: 10px; border: 1px solid #ffeeba;">
<h4>📄 Step 4: Generate Final Report</h4>
<ul>
<li>Select the <strong>runs (code and outputs)</strong> you want to include in the report.</li>
<li>Click <strong>“Generate final report”</strong> to let the AI summarize your analysis.</li>
<li>You can preview, copy, and refine the report through chat.</li>
<li>When you are happy with the refinements click **Update report** button.</li>
</ul>
<p>🧾 The report is built from your completed analysis steps — reproducible and ready to publish!</p>
</div>
""", unsafe_allow_html=True)
st.markdown("""\n\n""")

if st.button("Back to plan execution"):
    st.switch_page("pages/3_Analysis_Plan.py")



# layout: sidebar • main • preview
run_picker = st.sidebar
main_col, preview_col = st.columns([3, 1], gap="medium")



# ───────────────────────── 1. Run picker (left sidebar) ──────────
with run_picker:

    st.markdown("""
        <div style="background-color:#fff3cd; padding: 1em; border-radius: 10px; border: 1px solid #ffeeba;">
        <h4>🧭 Run Selection Tips</h4>
        <ul style="margin-bottom:0;">
            <li><b>Browse all runs:</b> Expand each hypothesis and step to see the runs you've executed.</li>
            <li><b>Select runs to include:</b> Tick checkboxes to select the runs you want in your final report or for deeper review.</li>
            <li><b>Preview details:</b> Click a run's checkbox to instantly preview its results, code, and images on the right.</li>
            <li><b>Multi-select:</b> You can select any combination of runs across all hypotheses and steps.</li>
            <li><b>Live sync:</b> Your current selection is always reflected in the preview and saved for report generation.</li>
            <li><b>Unselect to remove:</b> Untick a run to remove it from your selection and preview.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    
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
        for img in run_obj["images"]:
            image_to_display = st.session_state.images.get(img, None)
            if image_to_display:
                st.image(image_to_display)

        for text in run_obj["summary"]:
            st.markdown(text)
        
        for code in run_obj["code_input"]:
            st.code(code, language="python")
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
                    # print(f"\n\nProcessing run {run['run_id']} for step {step['title']}")
                    # print(f"Run OBJ: {run}")
                    print(run['images'])
                    ctx_lines.append(
                        f"- Step: {step['title']} | Run {run['run_id']} | "
                        f" Image ids: {', '.join(run['images']) if run['images'] else 'N/A'} | "
                        f"Code: {', '.join(run['code_input']) if run['code_input'] else 'N/A'} | "
                        f"Summary: {run['summary'] or 'N/A'}"
                    )

            context = "Draft a final report from this context:\n" + "\n".join(ctx_lines)
            
            print(f"\n\nSELECTED RUNS for context:{selected_runs}")
            
            with st.spinner("Composing report…"):

                tools = [# Create tools for code execution and web search
                    # create_code_interpreter_tool(st.session_state.container),
                    create_web_search_tool()
                ]
                
                
                # -------------------------------------

                # try:
                response = client.responses.create(
                    model="gpt-4o",
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
                st.session_state["final_report"] = chunks

            # What with the chunks?
            st.success("Report generated!")
            st.rerun()

    # ---------- show report & chat ----------
    if "final_report" in st.session_state:
        st.markdown("## Final report")
        print(f"\n\nFINAL REPORT: {st.session_state['final_report']}")

        # Display the report inside a light grey container
        st.markdown(
            "<div style='background-color:#f0f0f0; padding:1em; border-radius:10px;'>",
            unsafe_allow_html=True,
        )

        new_chunks = explode_text_and_images(st.session_state["final_report"])

        for chunk in new_chunks:
            if chunk["type"] == "text":
                st.markdown(chunk["content"])
            elif chunk["type"] == "image":
                image_id = chunk["content"]           # ← your requested snippet
                image_to_display = st.session_state.images.get(image_id, None)
                if image_to_display:
                    st.image(image_to_display)
                else:
                    st.warning(f"Image with {image_id} not found. Available images are: ")

        st.markdown("</div>", unsafe_allow_html=True)

        wide, narrow = st.columns([95,5])
        
        with wide:
            st.markdown(
                """
                <div style="
                    text-align: right;
                    font-size: 0.8rem;     /* smaller than the normal body font   */
                    font-style: italic;    /* italicise the whole sentence        */
                ">
                    Copy the report to clipboard <span style="opacity:.6">[experimental]</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with narrow: 
            copy_button(
                    f"{st.session_state['final_report']}",
                    tooltip="Copy",
                    copied_label="✔",
                    icon="st",
                    )
        
        st.divider()
        
        st.markdown("### Discuss report")

        for message in st.session_state["report_chat"]:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            elif message['role'] == 'assistant':
                
                
                new_chunks = explode_text_and_images(message['content'])

                for chunk in new_chunks:
                    if chunk["type"] == "text":
                        st.markdown(chunk["content"])
                    elif chunk["type"] == "image":
                        image_id = chunk["content"]           # ← your requested snippet
                        image_to_display = st.session_state.images.get(image_id, None)
                        if image_to_display:
                            st.image(image_to_display)
                        else:
                            st.warning(f"Image with {image_id} not found. Available images are: ")

        prompt = st.chat_input("Ask about this report")

        if prompt:
            
            st.session_state["report_chat"].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # TODO make each run available in the context
            context = f""" Here is the final report":\n{st.session_state["final_report"]}.\n\n Here is the chat history:\n{st.session_state["report_chat"]} """

            print(f"\n\n{'*****' * 10}\nREPORT CHAT CONTEXT:\n\n{context}\n\n{'*****' * 10}")

            with st.spinner("LLM generating response…"):
                response = client.responses.create(
                    model="gpt-4o",
                    instructions=report_chat_instructions,
                    tools=[
                        # create_code_interpreter_tool(st.session_state.container),
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
                # print(f"\n\n{'*****' * 10}\nREPORT CHAT RESPONSE:\n\n{reply}\n\n{'*****' * 10}")
                
                st.session_state["report_chat"].append({'role': 'assistant', 'content': chunks})

                st.rerun()

                # st.chat_message("assistant").write(reply)

        if st.button("Update report"):

            print(f"\n\nUpdated report button pressed {'***' * 10}")

            context = f""" Here is the final report":\n{st.session_state["final_report"]}.\n\n Here is the chat history:\n{st.session_state["report_chat"]} """

            print(f"\n\nHere is the collected context:\n{context}")

            with st.spinner("Updating report…"):
                
                response = client.responses.create(
                    model="gpt-4o",
                    instructions=report_chat_instructions,
                    tools=[
                        # create_code_interpreter_tool(st.session_state.container),
                        create_web_search_tool()
                    ],
                    input=[
                        {"role": "system", "content": context},
                        {"role": "user", "content": "Return an updated the final report based on the provided context."}
                    ],
                    temperature=0,
                    stream=False,
                )

                chunks = to_mock_chunks(response)

                print(f"\n\nChunks created: {chunks}")

                print(f"\n\n{'*****' * 10}\n")

            st.session_state["final_report"] = chunks

            st.session_state["report_chat"].append({'role': 'assistant', 'content': chunks})
            
            st.success("Report updated!")

            st.rerun()
