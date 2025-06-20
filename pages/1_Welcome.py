import streamlit as st

st.set_page_config(layout="wide")

st.title("Welcome")
st.write(
    """
    This minimal prototype lets you upload a dataset, create hypotheses,
    and explore mock LLM-generated analyses. Click “Start” to begin.
    """
)

if st.button("Start", use_container_width=True):
    st.switch_page("pages/2_Data_and_Hypotheses.py")
