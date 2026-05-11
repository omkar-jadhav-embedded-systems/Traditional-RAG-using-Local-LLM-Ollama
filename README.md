# 🧠 Traditional RAG System (100% Local & Private)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)](https://langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)](https://ollama.com/)

> A secure, fully offline Retrieval-Augmented Generation (RAG) pipeline to chat with your documents. Built with Python, LangChain, ChromaDB, SentenceTransformers, and Ollama, this system ensures your data never leaves your machine while providing accurate, context-aware AI answers.

## ✨ Key Features

- **🔐 Zero Data Leakage:** Powered entirely by local models via Ollama. No API keys, no cloud dependencies, and complete data privacy.
- **📂 Universal Document Support:** Seamlessly ingests `PDF`, `DOCX`, `CSV`, `TXT`, `Markdown`, `HTML`, `PPTX`, and `Excel` files.
- **⚡ Persistent Vector Storage:** Utilizes ChromaDB to save document embeddings to disk, avoiding the need to re-embed documents on every run.
- **🧱 Clean OOP Architecture:** Code is heavily modularized into distinct classes (`DocumentLoader`, `Chunker`, `EmbeddingManager`, `VectorStore`, `LocalLLM`) demonstrating enterprise-level software design principles.

## 🛠️ Tech Stack

*   **Orchestration:** LangChain
*   **Vector Database:** ChromaDB
*   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
*   **Local LLM:** Ollama (`qwen2.5-coder:14b` - customizable)
*   **Text Processing:** RecursiveCharacterTextSplitter

## 🏗️ System Architecture

1. **Ingestion:** Documents dropped in the `/Data` folder are parsed by format-specific loaders.
2. **Chunking:** Text is intelligently split into chunks (1200 characters / 250 overlap) preserving context.
3. **Embedding:** Chunks are converted into high-dimensional vectors via `all-MiniLM-L6-v2`.
4. **Storage:** Vectors and metadata are stored persistently in ChromaDB.
5. **Retrieval & Generation:** User queries undergo similarity search. Top-K results are injected into the Local LLM prompt for a highly accurate, hallucination-free response.

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.10+
*   [Ollama](https://ollama.com/) installed on your machine.
*   Pull the default LLM model via terminal:
    ```bash
    ollama pull qwen2.5-coder:14b
    ```

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/omkar-jadhav-embedded-systems/Traditional-RAG-using-Local-LLM-Ollama.git
cd Traditional-RAG-using-Local-LLM-Ollama
pip install -r requirements.txt
```

## 🚀 Application Preview

Here is a preview of the chatbot interface. The application provides a clean, user-friendly way to interact with your local documents while ensuring 100% privacy.

![Chatbot Preview](assests/Local%20RAG.jpg)
