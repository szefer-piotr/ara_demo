import streamlit as st
# from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="LLM Analysis Assistant",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.switch_page("pages/1_Welcome.py")
# st.write("⬅️ Use the navigation pane to start. The real action is on “Data & Hypotheses”.")
