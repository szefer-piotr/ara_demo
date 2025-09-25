import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

st.set_page_config(layout="wide")

utils.inject_global_css()

st.title("Welcome")

st.markdown("""
<div style="background-color:#fff3cd; padding: 1em; border-radius: 10px; border: 1px solid #ffeeba;">
<h4>👋 Welcome to Your Research Assistant</h4>
<p>This app helps you analyze your dataset, test hypotheses, and generate a final report — all with AI assistance.</p>
<p>👉 Click <strong>Start</strong> to begin your analysis journey.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""\n\n""")

if st.button("Start", use_container_width=True):
    st.switch_page("pages/2_Data_and_Hypotheses.py")
