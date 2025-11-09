# ---------------------------------------------------------
# 🤖 MainBot - Mera simple RAG chatbot system
# ---------------------------------------------------------
from openwrapper import OpenRouter
from splitter import split_docs
from prompt import make_prompt
from embeddings import get_embeddings, find_similar
from memory import ChatMemory
from loader import load_pdf
from setup.config import OPENROUTER_API_KEY
import os

def run_bot():
    print("👋 Welcome to my simple RAG Chatbot!")
    print("You can ask questions based on your uploaded PDF.")
    print("Type 'exit' anytime to quit.\n")

    # Step 1: Model setup
    model = OpenRouter(api_key=OPENROUTER_API_KEY)
    memory = ChatMemory()

    # Step 2: Load PDF
    pdf_path = input("📄 Enter PDF file path: ").strip()
    if not os.path.exists(pdf_path):
        print("❌ File not found! Please check the path.")
        return
    docs = load_pdf(pdf_path)

    # Step 3: Split text into chunks
    chunks = split_docs(docs)

    # Step 4: Generate embeddings
    embeddings = get_embeddings(chunks)

    print("\n✅ Setup done! You can start chatting below:\n")

    # Step 5: Chat loop
    while True:
        query = input("🧠 You: ").strip()
        if query.lower() == "exit":
            print("👋 Exiting... bye!")
            break

        top_chunks = find_similar(query, chunks, embeddings, top_k=3)
        memory_context = memory.get_context()
        prompt = make_prompt(query, top_chunks, memory_context)
        response = model(prompt)
        print(f"\n🤖 Bot: {response.content}\n")
        memory.add(query, response.content)

if __name__ == "__main__":
    run_bot()
