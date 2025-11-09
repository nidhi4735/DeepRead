# ---------------------------------------------------------
# 💭 Memory System - last 5-10 chats yaad rakhta hai
# ---------------------------------------------------------
from collections import deque

class ChatMemory:
    def __init__(self, limit=10):
        self.history = deque(maxlen=limit)

    def add(self, user_msg, bot_msg):
        self.history.append((user_msg, bot_msg))

    def get_context(self):
        context = ""
        for u, b in self.history:
            context += f"User: {u}\nBot: {b}\n"
        return context.strip()

    def clear(self):
        self.history.clear()
