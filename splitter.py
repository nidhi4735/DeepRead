# ---------------------------------------------------------
# ✂️ Text Splitter - bada text tod kar chunks me karta hai
# ---------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_docs(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"📑 Split into {len(chunks)} chunks.")
    return [chunk.page_content for chunk in chunks]
