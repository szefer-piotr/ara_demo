def run_rag_agent(query: str, vector_store, llm) -> str:
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    return llm(prompt)