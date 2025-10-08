from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

db = FAISS.load_local(
    "vectorstore/ecobooks",
    embeddings=OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

query = "How to analyze time series?"
docs = db.similarity_search(query, k=3)

for d in docs:
    print("---")
    print(d.page_content[:500])  # tylko kawałek