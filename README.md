# DeepRead
### Document Question-Answering System with LangChain & HuggingFace

A Python-based interactive system that allows users to ask questions over uploaded PDF or text documents.  
The system leverages **LangChain**, **Chroma Vector Database**, and **HuggingFace LLMs** to deliver precise, context-aware answers.

---

## 🚀 Features

- **Supports PDF and Text files** – Upload and process PDF or text documents.
- **Document Chunking** – Splits large documents into smaller, retrievable chunks.
- **Vector Search (MMR)** – Uses **Chroma** with *Maximal Marginal Relevance* for relevant chunk retrieval.
- **Contextual QA** – Answers based strictly on document context.
- **HuggingFace LLM Integration** – Ensures natural, high-quality text generation.
- **Interactive Query Loop** – Enables continuous question–answering in one session.

---

## ⚙️ Installation

Designed for **Google Colab** or local Python 3.10+ environments.

```bash
# Install dependencies
pip install -q langchain langchain-community langchain-huggingface chromadb sentence-transformers pypdf
```
## ✅Usage

- 1️⃣ Run DeepRead in a Python environment or Colab cell
- 2️⃣ Upload your document (PDF or TXT)
- 3️⃣ Ask context-based questions interactively

# Example Interaction:
- Upload a document: sample.pdf
- Question: What is the key finding of section 3?
- Answer: The model achieved state-of-the-art accuracy using transfer learning.

## 🧠 How It Works

- Document Upload – User uploads a .pdf or .txt file.
- Text Extraction & Chunking – The file is split into semantically meaningful chunks.
- Vector Embedding – Each chunk is embedded using a Sentence Transformer model.
- Storage – Embeddings are stored in ChromaDB for fast retrieval.
- Query Handling – User questions are embedded and compared to stored chunks.
- Answer Generation – The top results are passed to a HuggingFace LLM via LangChain to produce a contextual answer
