# 📰 News Research Tool using RAG

A Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, Hugging Face Embeddings, FAISS, and Groq LLMs.

The application allows users to provide multiple news article URLs, build a vector database from the content, and ask questions about the articles using an AI-powered chatbot.

---

## Features

* Multiple News URL Input
* Web Article Loading
* Hugging Face Embeddings (BGE Small)
* FAISS Vector Database
* Groq LLM Integration
* Retrieval-Augmented Generation (RAG)
* Source Attribution
* Interactive Streamlit Interface
* Knowledge Base Rebuilding
* Local Vector Store Persistence

---

## Tech Stack

### Frontend

* Streamlit

### LLM

* Groq
* Llama 3.3 70B Versatile

### Embeddings

* BAAI/bge-small-en-v1.5

### Vector Database

* FAISS

### Framework

* LangChain

---

## Project Architecture

URLs
↓
WebBaseLoader
↓
Documents
↓
RecursiveCharacterTextSplitter
↓
HuggingFace Embeddings
↓
FAISS Vector Store
↓
Retriever
↓
Groq LLM
↓
Answer + Sources

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/news-research-tool.git

cd news-research-tool
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Usage

1. Open the Streamlit application.
2. Enter one or more news article URLs in the sidebar.
3. Click **Build / Rebuild Knowledge Base**.
4. Wait for vector database creation.
5. Ask questions related to the loaded articles.
6. View generated answers along with source references.

---

## Example Questions

* What are analysts saying about Tesla stock?
* Should investors be bullish on Tesla?
* What risks are highlighted in the article?
* What are the latest developments regarding Tesla?

---

## Future Improvements

* Streaming Token Output
* Chat Memory
* Source Citation Formatting
* URL Validation
* PDF Upload Support
* Multi-Document Knowledge Base
* Hybrid Search (BM25 + Vector Search)
* Conversation History Export
* Authentication System

---

## Project Structure

```text
news-research-tool/
│
├── app.py
├── .env
├── .gitignore
├── README.md
├── requirements.txt
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
└── assets/
```

---

## Author

Aarya Butolia

AI & Machine Learning Student | Web Developer | GenAI Enthusiast

---

## License

This project is licensed under the MIT License.

