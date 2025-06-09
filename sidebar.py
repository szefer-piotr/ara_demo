# sidebar.py
import streamlit as st


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Context")

        # ────────────────── Data preview ──────────────────
        df = st.session_state.get("current_data")
        if df is not None:
            st.subheader("Data")
            st.write(f"{df.shape[0]:,} rows × {df.shape[1]} cols")
        else:
            st.info("No dataset")

        # ────────────────── Hypotheses list ───────────────
        analyses = st.session_state.get("analyses", [])
        if not analyses:
            st.info("No hypotheses yet")
            return

        # Build label ↔ id maps for the selectbox
        label_for = lambda a: f"{a['hypothesis_id']} – {a['title'][:40]}"
        labels = [label_for(a) for a in analyses]
        ids = [a["hypothesis_id"] for a in analyses]

        # Current selection (fallback to first)
        hid = st.session_state.get("selected_hypothesis_id", ids[0])
        idx_default = ids.index(hid)

        choice_label = st.selectbox(
            label="Hypothesis",
            options=labels,
            index=idx_default,
            key="hypothesis_select",
        )
        hid = ids[labels.index(choice_label)]
        st.session_state["selected_hypothesis_id"] = hid

        # Obtain the chosen hypothesis object
        hypo = next(a for a in analyses if a["hypothesis_id"] == hid)

        # ────────────────── Step radio (only if plan accepted) ─────
        radio_options = ["Overview"]

        if hypo.get("plan_accepted", False):
            # Append each step title
            radio_options += [s["title"] for s in hypo["analysis_plan"]]

        # Determine which radio item should be pre-selected
        sid_current = st.session_state.get("selected_step_id")
        if sid_current and hypo.get("plan_accepted"):
            try:
                current_title = next(
                    s["title"]
                    for s in hypo["analysis_plan"]
                    if s["step_id"] == sid_current
                )
            except StopIteration:
                current_title = "Overview"
        else:
            current_title = "Overview"

        step_choice = st.radio(
            label="Step",
            options=radio_options,
            index=radio_options.index(current_title),
            key="step_radio",
        )

        # Update session-state according to the radio selection
        if step_choice == "Overview":
            st.session_state["selected_step_id"] = None
        else:
            if hypo.get("plan_accepted", False):
                step = next(
                    s for s in hypo["analysis_plan"] if s["title"] == step_choice
                )
                st.session_state["selected_step_id"] = step["step_id"]
            else:
                # Plan not accepted ⇒ ignore any step selections
                st.session_state["selected_step_id"] = None
