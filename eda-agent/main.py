import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tempfile
import streamlit.components.v1 as components

from openai import OpenAI
from dotenv import load_dotenv
from pyvis.network import Network

import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

st.title("EDA + Knowledge Graph Prototype")

data_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
hypotheses = st.text_area("Paste your hypotheses (one per line or Markdown/JSON)", height=200)

# uploaded_files = st.file_uploader("Upload research papers/books (PDF)", type=["pdf"], accept_multiple_files=True)

df = None

if data_file:
    df = pd.read_csv(data_file)
    st.session_state["df"] = df
    st.success(f"Uploaded data: {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head())


if df is not None:
    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("Column Types")
    st.write(pd.DataFrame({
        "type": df.dtypes.astype(str),
        "nulls": df.isnull().sum(),
        "unique": df.nunique()
    }))
    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))

    st.write("Column Distributions")
    for col in df.select_dtypes(include=['number']).columns:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        st.pyplot(fig)

# Build a minimal KG
def build_kg(df: pd.DataFrame, hypotheses: str) -> nx.DiGraph:
    
    kg = nx.DiGraph()
    
    for col in df.columns:
        kg.add_node(col, type="column")

    for i, hyp in enumerate(hypotheses.splitlines()):
        hyp_node = f"Hypothesis {i+1}"
        kg.add_node(hyp_node, type="hypothesis", text=hyp)
        # SIMPLE HEURISTIC - LINK IF COLUMN NAME APPEARS IN HYPOTHESIS
        for col in df.columns:
            if col.lower() in hyp.lower():
                kg.add_edge(hyp_node, col, relation="mentions")
    return kg

if df is not None and hypotheses.strip():
    st.header("Knowledge Graph")
    
    kg = build_kg(df, hypotheses)

    net = Network(height="400px", width="100%", notebook=False, directed=True)

    for node, attrs in kg.nodes(data=True):
        color = "#ADD8E6" if attrs.get("type") == "column" else "#90EE90"
        label = node if attrs.get("type") == "column" else f"{node}: {attrs.get('text', '')[:50]}"
        net.add_node(node, label=label, color=color)
    for source, target, attrs in kg.edges(data=True):
        label = attrs.get("relation", "")
        net.add_edge(source, target, label=label)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.write_html(tmp.name)
        components.html(open(tmp.name, "r").read(), height=420, scrolling=True)

    st.caption("Columns = blue, Hypotheses = green. Edges: hypothesis mentions column.")

# --- Ready for extension ---
st.info("This is a prototype. Next: add agentic EDA insights, KG enrichment, or RAG integration!")


#     # Data Cleaning Suggestions
#     llm = OpenAI()
#     prompt = f"""
#     You are a data scientist. Here is the summary of my data:
#     {df.describe()}
#     Nulls: {df.isnull().sum().to_dict()}
#     Suggest a cleaning plan and visualize any issues.
#     """
#     response = llm.responses.create(
#         model="gpt-4o-mini",
#         input = prompt
#     )
    
#     if st.session_state['cleaning_plan'] is None:
#         st.session_state["cleaning_plan"] = response.output[0].content[0].text
#         st.rerun()
    
# # else:
# #     st.info("Please upload a CSV dataset to begin.")


# # 4. RAG FOR RESEARCH CONTEXT

# # resesa

# # 5. DIRECTION GENERATOR

# directions_prompt = f"""
# You have the following:
# - Data knowledge graph: {kg}
# - User hypotheses: {hypotheses}
# - EDA summary: {}

# Suggest next steps for the research, referencing any relevant uploaded literature.
# """