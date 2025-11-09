# ---------------------------------------------------------
# 🧭 Agent - LangGraph workflow runner for my RAG Bot
# ---------------------------------------------------------
# Start → MainBot → End
# ---------------------------------------------------------

from langgraph.graph import StateGraph
from typing import TypedDict
from mainbot import run_bot  # your RAG bot logic

# ---------------------------------------------------------
# 🧩 Step 1: Define state schema
# ---------------------------------------------------------
class GraphState(TypedDict):
    status: str
    init: bool

# ---------------------------------------------------------
# 🧠 Step 2: Define nodes
# ---------------------------------------------------------
def start_node(state: GraphState):
    print("🚀 Starting the RAG Agent Graph...")
    state["status"] = "started"
    state["init"] = True
    return state

def bot_node(state: GraphState):
    print("🤖 Running the chatbot system...")
    run_bot()  # call your main RAG bot
    state["status"] = "completed"
    state["init"] = False
    return state

def end_node(state: GraphState):
    print("🏁 Graph finished successfully!")
    return state

# ---------------------------------------------------------
# 🧱 Step 3: Build and link nodes
# ---------------------------------------------------------
def build_agent_graph():
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("start", start_node)
    graph.add_node("bot", bot_node)
    graph.add_node("end", end_node)

    # Define edges for automatic traversal
    graph.set_entry_point("start")
    graph.add_edge("start", "bot")
    graph.add_edge("bot", "end")
    graph.set_finish_point("end")

    return graph

# ---------------------------------------------------------
# 🚀 Step 4: Compile and invoke
# ---------------------------------------------------------
if __name__ == "__main__":
    rag_graph = build_agent_graph()

    # Initial state
    initial_state: GraphState = {"init": True, "status": "pending"}

    # Automatic execution via LangGraph compiler
    compiled_graph = rag_graph.compile()
    final_state = compiled_graph.invoke(initial_state)

    print("✅ Workflow Result:", final_state)
