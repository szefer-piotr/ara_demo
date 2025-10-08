from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("rag_agent/vectorstore/books/book1.pdf")
docs = loader.load()

print("PDF loaded.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

print("Text splitted.")

chunks = splitter.split_documents(docs)

print("Chunks created.")

embedding = OpenAIEmbeddings()

db = FAISS.from_documents(chunks, embedding)

print("DB created.")

db.save_local("vectorstore/ecobooks")

print("DB saved.")

