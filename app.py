import os
import hashlib
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv()

st.set_page_config(page_title="News Research Tool", layout="wide")

st.title("📰 News Research Tool (RAG)")

# Sidebar
st.sidebar.header("News URLs")

default_urls = """https://www.cnbc.com/quotes/TSLA
https://finance.yahoo.com/"""

urls_text = st.sidebar.text_area(
    "Enter one URL per line",
    value=default_urls,
    height=200
)

urls = [u.strip() for u in urls_text.splitlines() if u.strip()]

INDEX_DIR = "faiss_index"

def urls_hash(urls):
    return hashlib.md5("".join(sorted(urls)).encode()).hexdigest()

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

def build_vectorstore(urls):
    loader = WebBaseLoader(urls)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    embeddings = load_embeddings()

    vectordb = FAISS.from_documents(
        split_docs,
        embeddings
    )

    vectordb.save_local(INDEX_DIR)

    return vectordb

if st.sidebar.button("Build / Rebuild Knowledge Base"):
    with st.spinner("Loading URLs and creating embeddings..."):
        build_vectorstore(urls)
    st.sidebar.success("Vector database created successfully")

if not os.path.exists(INDEX_DIR):
    st.info("Build the knowledge base from the sidebar first.")
    st.stop()

embeddings = load_embeddings()

vectordb = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=groq_key,
    streaming=True
)

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

question = st.text_input(
    "Ask a question about the loaded news articles"
)

if question:
    with st.spinner("Searching articles and generating answer..."):

        result = qa_chain.invoke(
            {"question": question}
        )

        st.subheader("Answer")
        st.write(result.get("answer", "No answer generated"))

        st.subheader("Sources")
        st.write(result.get("sources", "No sources returned"))

        st.subheader("Retrieved Documents")

        docs = retriever.invoke(question)

        for i, doc in enumerate(docs, start=1):
            with st.expander(f"Chunk {i}"):
                st.write(doc.page_content[:3000])
                st.json(doc.metadata)
