# ---------------------------------------------------------
# 📘 PDF Loader - user se path le kar PDF ko load karta hai
# ---------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path: str):
    """
    Load a PDF file and return the list of LangChain Documents.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages from: {pdf_path}")
    return docs
