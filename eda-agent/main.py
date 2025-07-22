import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

st.title("AI EDA Module")

data_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
hypotheses = st.text_area("Paste your hypotheses (one per line or Markdown/JSON)", height=200)
uploaded_files = st.file_uploader("Upload research papers/books (PDF)", type=["pdf"], accept_multiple_files=True)

if "cleaning_plan" not in st.session_state:
    st.session_state["cleaning_plan"] = None

if st.session_state["cleaning_plan"] is not None:
    st.write(st.session_state["cleaning_plan"])

if data_file:
    df = pd.read_csv(data_file)
    st.session_state["df"] = df
    st.write("Basic data info:")
    st.write(df.info())
    st.write(df.describe())
    st.write("Null counts:", df.isnull().sum())

    # Build a basic KG
    kg = {"entities": [], "relations": []}
    for col in df.columns:
        kg["entities"].append({"name": col, "type": str(df[col].dtype)})
    st.session_state["kg"] = kg

    # Visualize data
    st.write("Column Distributions")
    for col in df.select_dtypes(include=['number']).columns:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        st.pyplot(fig)

    # Data Cleaning Suggestions
    llm = OpenAI()
    prompt = f"""
    You are a data scientist. Here is the summary of my data:
    {df.describe()}
    Nulls: {df.isnull().sum().to_dict()}
    Suggest a cleaning plan and visualize any issues.
    """
    response = llm.responses.create(
        model="gpt-4o-mini",
        input = prompt
    )
    
    if st.session_state['cleaning_plan'] is None:
        st.session_state["cleaning_plan"] = response.output[0].content[0].text
        st.rerun()
    
else:
    st.info("Please upload a CSV dataset to begin.")


# 4. RAG FOR RESEARCH CONTEXT