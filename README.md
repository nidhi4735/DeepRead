# DeepRead
# Document Question-Answering System with LangChain & HuggingFace

A Python-based interactive system that allows users to ask questions over uploaded PDF or text documents. The system leverages **LangChain**, **Chroma vector database**, and **HuggingFace LLMs** to provide precise, context-aware answers.

---

## Features

- **Supports PDF and Text files**: Upload any PDF or text document for analysis.
- **Document Chunking**: Splits large documents into manageable chunks for efficient retrieval.
- **Vector Search with MMR**: Uses **Chroma** with Maximal Marginal Relevance (MMR) to retrieve the most relevant chunks.
- **Contextual Question Answering**: Answers questions using only the provided document context.
- **HuggingFace LLM Integration**: For high-quality text generation.
- **Interactive Query Loop**: Users can ask multiple questions in a single session.

---

## Installation

This project is designed for **Google Colab**. Install dependencies with:

```bash
!pip install -q langchain langchain-community langchain-huggingface chromadb sentence-transformers pypdf
