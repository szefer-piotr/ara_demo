# sidebar.py
import streamlit as st

def render_sidebar(*, show_steps: bool = True) -> None:   # ← new arg
    with st.sidebar:

        st.markdown("""
            <div style="background-color:#fff3cd; padding: 1em; border-radius: 10px; border: 1px solid #ffeeba;">
            <h4>🧭 Navigation Tips</h4>
            <ul>
            <li>Switch between analysis steps</li>
            <li>Select different hypotheses</li>
            <li>Generate or revisit your report</li>
            <li>Start a new analysis anytime</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("\n")

        # ───── Legend of status icons ─────
        with st.expander("Legend of icons", expanded=False):
            st.markdown(
                """
                    | Icon | Meaning |
                    |------|---------|
                    | ⚪ | Plan **not** accepted yet |
                    | 🟡 | Plan accepted, some runs still pending |
                    | 🔵 | Step exists, run not finished |
                    | 🟢 | Everything finished |
                """,
                unsafe_allow_html=True,
            )

        st.header("Your items")

        # ────────────────── Data preview ──────────────────
        df = st.session_state.get("current_data")
        st.subheader("Data")
        if df is not None:
            
            st.write(f"{df.shape[0]:,} rows × {df.shape[1]} cols")
        else:
            st.info("No dataset uploaded yet.")

        # ────────────────── Hypotheses list & status ──────
        analyses   = st.session_state.get("analyses", [])
        st.subheader("Hypotheses")

        if not analyses:
            st.info("No hypotheses yet")
            return

        labels, ids = [], []
        for a in analyses:
            plan_ok   = a.get("plan_accepted", False)
            runs_tot  = len(a["analysis_plan"])
            runs_done = sum(1 for s in a["analysis_plan"] if s.get("finished", False))
            hypo_done = plan_ok and runs_tot and runs_tot == runs_done

            if   hypo_done: icon = "🟢"
            elif plan_ok:   icon = "🟡"
            else:           icon = "⚪"

            labels.append(f"{icon} {a['title'][:40]} - (ID: {a['hypothesis_id']})")
            ids.append(a["hypothesis_id"])

        # choose hypothesis
        if "selected_hypothesis_id" not in st.session_state:
            st.session_state["selected_hypothesis_id"] = ids[-1]

        hid_current = st.session_state["selected_hypothesis_id"]
        choice_label = st.selectbox(
            "Hypothesis",
            options=labels,
            index=ids.index(hid_current),
            key="hypothesis_select",
            label_visibility='collapsed'
        )
        hid = ids[labels.index(choice_label)]
        st.session_state["selected_hypothesis_id"] = hid
        st.markdown(f"**Total added:** {len(analyses)}")

        # ────────────────── Step radio (only if allowed) ───
        if show_steps:
            hypo = next(a for a in analyses if a["hypothesis_id"] == hid)

            radio_options, label_to_step = ["Overview"], {}
            if hypo.get("plan_accepted", False):
                
                for s in hypo["analysis_plan"]:
                    icon = "🟢" if s.get("finished", False) else "🔵"
                    lbl  = f"{icon} {s['title']}" 
                    radio_options.append(lbl)
                    label_to_step[lbl] = s["step_id"]

            sid_current = st.session_state.get("selected_step_id")
            current_lbl = next(
                (lbl for lbl, sid in label_to_step.items() if sid == sid_current),
                "Overview"
            )

            step_choice = st.radio(
                "Step",
                options=radio_options,
                index=radio_options.index(current_lbl),
                key="step_radio",
            )

            st.session_state["selected_step_id"] = (
                None if step_choice == "Overview" else label_to_step[step_choice]
            )

            # ── Final‐report gate only on pages that show steps
            all_complete = all(
                h.get("plan_accepted")
                and h["analysis_plan"]
                and all(s.get("finished", False) for s in h["analysis_plan"])
                for h in analyses
            )

            if all_complete:
                if st.session_state["selected_step_id"] is not None:
                    st.session_state["selected_step_id"] = None
                    st.rerun()
                st.success("🎉 All hypotheses have completed runs!")
                # Jump to overview when all steps are ready
                if st.button("➡️ Build Final Report"):
                    st.switch_page("pages/4_Final_Report.py")
