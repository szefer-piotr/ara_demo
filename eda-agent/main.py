import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tempfile
import streamlit.components.v1 as components

from pandas.api.types import CategoricalDtype
from openai import OpenAI
from dotenv import load_dotenv
from pyvis.network import Network

import os

import openai
import json
import re



load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_eda_insights(df: pd.DataFrame, model="gpt-4o") -> list:
    prompt = f"""You are an expert in data analysid. Here is the summary of a dataset"
    Columns: {list(df.columns)}
    Types: {df.dtypes.to_dict()}
    Null counts: {df.isnull().sum().to_dict()}
    Unique counts: {df.nunique().to_dict()}
    Descriptions: {df.describe(include="all").to_string()}

    Generate 5-10 insightfull observations about the data.
    for each, specify which column(s) it refers to.
    Return as a JSON list of objects with 'text' and 'columns' (list of column names).
    """

    llm = OpenAI()

    response = llm.responses.create(
        model=model,
        input = prompt
    )
    raw_text = response.output[0].content[0].text
    json_strs = re.findall(f"\[.*\]", raw_text, flags=re.DOTALL)
    if json_strs:
        try:
            insights = json.loads(json_strs[0])
            return insights
        except Exception:
            pass
    
    return[{"text": raw_text, "columns": []}]


def add_eda_insights_to_kg(kg: nx.DiGraph, insights: list):
    for i, insight in enumerate(insights):
        node_id = f"EDA_Insight_{i+1}"
        kg.add_node(node_id, type="eda_insight", text=insight["text"])
        for col in insight.get("columns", []):
            if col in kg.nodes and kg.nodes[col]["type"] == "column":
                kg.add_edge(node_id, col, relation="about")




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
def build_kg(df: pd.DataFrame, hypotheses: str, corr_threshold=0.5, max_cat_levels=10) -> nx.DiGraph:
    kg = nx.DiGraph()
    type_nodes = set()
    # 1. Add columns and their properties
    for col in df.columns:
        # Node for column
        kg.add_node(col, type="column")
        # Type node
        col_type = str(df[col].dtype)
        type_nodes.add(col_type)
        kg.add_node(col_type, type="type")
        kg.add_edge(col, col_type, relation="has_type")
        # Nulls
        has_nulls = df[col].isnull().any()
        kg.add_node(f"{col}_has_nulls", type="null_flag", value=str(has_nulls))
        kg.add_edge(col, f"{col}_has_nulls", relation="has_nulls")
        # Uniqueness
        is_unique = df[col].is_unique
        kg.add_node(f"{col}_is_unique", type="unique_flag", value=str(is_unique))
        kg.add_edge(col, f"{col}_is_unique", relation="is_unique")
        # For categoricals: levels
        if isinstance(df[col].dtype, CategoricalDtype) or \
            (df[col].dtype == 'object' and df[col].nunique() <= max_cat_levels):
            for val in df[col].dropna().unique():
                level_node = f"{col}_level_{val}"
                kg.add_node(level_node, type="category_level", value=str(val))
                kg.add_edge(col, level_node, relation="has_level")
    # 2. Add pairwise correlations for numerics
    # Only float/int columns
    num_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    corr = df[num_cols].corr().abs()
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i+1:]:
            corr_value = corr.loc[c1, c2]
            if pd.notnull(corr_value) and isinstance(corr_value, (float, int)) and corr_value >= corr_threshold:
                kg.add_edge(c1, c2, relation=f"correlated_{corr_value:.2f}")
                kg.add_edge(c2, c1, relation=f"correlated_{corr_value:.2f}")
    # 3. Add hypotheses and connect
    for i, hyp in enumerate(hypotheses.splitlines()):
        hyp_node = f"Hypothesis {i+1}"
        kg.add_node(hyp_node, type="hypothesis", text=hyp)
        for col in df.columns:
            if col.lower() in hyp.lower():
                kg.add_edge(hyp_node, col, relation="mentions")
    return kg


if df is not None and hypotheses.strip():
    st.header("Knowledge Graph")
    
    kg = build_kg(df, hypotheses)

    if st.button("Run Agentic EDA"):
        with st.spinner("LLM analyzing data..."):
            insights = get_eda_insights(df)
        st.success(f"Got {len(insights)} insights from the LLM.")
        st.session_state["eda_insights"] = insights
        add_eda_insights_to_kg(kg, insights)

    net = Network(height="400px", width="100%", notebook=False, directed=True)

    node_colors = {
    "column": "#ADD8E6",
    "type": "#FFD580",
    "hypothesis": "#90EE90",
    "null_flag": "#FFB6C1",
    "unique_flag": "#C1FFC1",
    "category_level": "#E0E0E0",
    "eda_insight": "#FCDFFF"
    }

    for node, attrs in kg.nodes(data=True):
        ntype = attrs.get("type", "other")
        color = node_colors.get(ntype, "#CCCCCC")
        if ntype == "eda_insight":
            label = attrs.get("text", "")[:60]
        elif ntype == "hypothesis":
            label = f"{node}: {attrs.get('text', '')[:40]}"
        elif ntype in ["null_flag", "unique_flag", "category_level"]:
            label = f"{node}: {attrs.get('value', '')}"
        else:
            label = node
        net.add_node(node, label=label, color=color)

    for source, target, attrs in kg.edges(data=True):
        label = attrs.get("relation", "")
        net.add_edge(source, target, label=label)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.write_html(tmp.name)
        components.html(open(tmp.name, "r").read(), height=420, scrolling=True)

    # Optional: simple color names (can be more elaborate if needed)
    color_names = {
        "#ADD8E6": "light blue",
        "#FFD580": "light orange",
        "#90EE90": "light green",
        "#FFB6C1": "pink",
        "#C1FFC1": "mint green",
        "#E0E0E0": "gray",
    }

    def node_caption(colors: dict) -> str:
        parts = []
        for key, hex_color in colors.items():
            color_label = color_names.get(hex_color, hex_color)
            parts.append(f"{key.replace('_', ' ').title()} = {color_label}")
        return "Nodes: " + "; ".join(parts) + "."

    caption = node_caption(node_colors)
    st.caption(caption)

# --- Ready for extension ---
st.info("This is a prototype. Next: add agentic EDA insights, KG enrichment, or RAG integration!")




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