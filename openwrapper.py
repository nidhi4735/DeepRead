# ---------------------------------------------------------
# 💬 My OpenRouter AI Wrapper (Made for this UG Project)
# ---------------------------------------------------------
# So in my project, I wanted to connect directly with advanced AI models
# like ChatGPT or other LLMs available through the OpenRouter platform.
# But instead of writing long and repetitive API calls again and again,
# I built this small and simple **wrapper class** that handles all that internally.
#
# Basically, this class acts like a **bridge** between my Python app
# and any large language model hosted on OpenRouter.
#
# 👇 In simple terms, here’s what I did:
# ---------------------------------------------------------
# 1️⃣ I load my API key securely from the `.env` file (so I don’t hardcode it).
# 2️⃣ I connect to the OpenRouter endpoint using either:
#      - direct HTTP calls (my custom request method), or
#      - the built-in OpenAI client (if use_native=True).
# 3️⃣ I can send prompts or message lists (user/system/assistant) easily.
# 4️⃣ I can even stream model responses live (token by token).
# 5️⃣ I can bind tools (Python functions) so that the model can “call” them.
# 6️⃣ Finally, I can request structured JSON output for automation tasks.
#
# ⚙️ So overall, this makes it super easy for me to plug AI into any app or bot
# — whether it's a RAG system, a chatbot, or a content generator.
#
# ---------------------------------------------------------

import os, json, requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from openai import OpenAI
from pydantic import BaseModel

# ---------------------------------------------------------
# 🌱 Step 1: Load API key securely from .env file
# ---------------------------------------------------------
load_dotenv()  # .env se key load karte hain


class OpenRouter:
    def __init__(self,
                 api_key: str = None,
                 model: str = "nvidia/nemotron-nano-9b-v2:free",
                 temperature: float = 0.7,
                 max_tokens: int = 512,
                 stream: bool = False,
                 use_native: bool = False):
        """
        Initialize my OpenRouter connection.
        """

        # ✅ Securely load API key
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("⚠️ OPENROUTER_API_KEY missing! Add it in .env or pass manually.")

        # 🔧 Store settings
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.use_native = use_native

        # 🌍 API setup
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000"
        }

        # 🧠 Connect to OpenRouter client
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")

        # 🧩 store custom tools (Python functions)
        self.tools = {}

    # ---------------------------------------------------------
    # 🧰 Bind tools (custom functions) that model can call
    # ---------------------------------------------------------
    def bind_tools(self, tools: List[Any]):
        # tools list ko ek dict me store karte hain
        self.tools = {tool.name: tool for tool in tools}
        return self

    # ---------------------------------------------------------
    # 🧩 Convert LangChain messages → roles for OpenRouter
    # ---------------------------------------------------------
    def _role(self, msg):
        msg_type = msg.__class__.__name__
        if msg_type == "HumanMessage": return "user"
        if msg_type == "AIMessage": return "assistant"
        if msg_type == "SystemMessage": return "system"
        if msg_type == "ToolMessage": return "assistant"
        raise ValueError(f"❌ Unknown message type: {msg_type}")

    # ---------------------------------------------------------
    # 🧱 Format messages properly before sending
    # ---------------------------------------------------------
    def _format_messages(self, messages: List[Any]):
        formatted = []
        for m in messages:
            if hasattr(m, "content") and m.content:
                formatted.append({"role": self._role(m), "content": m.content})
        return formatted

    # ---------------------------------------------------------
    # ✅ Handle normal API responses (non-streamed)
    # ---------------------------------------------------------
    def _safe_response(self, resp):
        # Response ko safely parse karte hain
        try:
            data = resp.json()
        except Exception:
            return AIMessage(content=f"[ERROR] Invalid JSON from OpenRouter: {resp.text}")

        if "choices" not in data:
            err = data.get("error", {}).get("message", resp.text)
            return AIMessage(content=f"[OpenRouter Error] {err}")

        return AIMessage(content=data["choices"][0]["message"]["content"])

    # ---------------------------------------------------------
    # 🌐 Raw HTTP request (used for streaming or manual mode)
    # ---------------------------------------------------------
    def _raw_request(self, body: dict):
        # agar stream mode on hai toh live printing karega
        if self.stream:
            print("\n🧠 Streaming response:\n")
            resp = requests.post(self.base_url, headers=self.headers, data=json.dumps(body), stream=True)
            content = ""
            for line in resp.iter_lines():
                if not line or not line.decode().startswith("data: "):
                    continue
                chunk = line.decode().replace("data: ", "")
                if chunk.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        content += delta
                except Exception:
                    pass
            print("\n")
            return AIMessage(content=content)
        else:
            # agar stream off hai toh pura reply ek saath lo
            resp = requests.post(self.base_url, headers=self.headers, data=json.dumps(body))
            return self._safe_response(resp)

    # ---------------------------------------------------------
    # 🧠 Use OpenAI’s native client directly (simpler alternate)
    # ---------------------------------------------------------
    def _native_request(self, body: dict):
        try:
            r = self.client.chat.completions.create(**body)
            return AIMessage(content=r.choices[0].message.content)
        except Exception as e:
            return AIMessage(content=f"[Native Request Error] {e}")

    # ---------------------------------------------------------
    # 💬 Main function to send prompts and get model replies
    # ---------------------------------------------------------
    def invoke(self, messages: Union[str, List[Any]]):
        # agar user ne sirf ek string bheji hai toh usko HumanMessage bana do
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        msg_list = self._format_messages(messages)

        body = {
            "model": self.model,
            "messages": msg_list,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream
        }

        # native ya custom dono me se ek mode choose karo
        if self.use_native:
            return self._native_request(body)
        else:
            return self._raw_request(body)

    # ---------------------------------------------------------
    # 🧩 If I’ve bound any tools, I can call them easily here
    # ---------------------------------------------------------
    def call_tool(self, name: str, *args, **kwargs):
        # simple function call pattern
        if name not in self.tools:
            raise ValueError(f"❌ Tool '{name}' not found. Bind it first!")
        return self.tools[name](*args, **kwargs)

    # ---------------------------------------------------------
    # 🪄 Shortcut: allows direct call like model("Hello!")
    # ---------------------------------------------------------
    def __call__(self, prompt: Union[str, List[Any]]):
        return self.invoke(prompt)

    # ---------------------------------------------------------
    # 📦 Get structured (JSON) outputs using a Pydantic model
    # ---------------------------------------------------------
    def invoke_structured(self, messages, output_model: BaseModel):
        # agar user sirf ek string bhejta hai toh usme JSON schema embed karo
        if isinstance(messages, str):
            schema = output_model.model_json_schema()
            messages = [HumanMessage(content=f"""
                Please output only valid JSON matching this schema:
                {schema}
                Respond strictly in JSON format.
                Input: {messages}
            """)]

        result = self.invoke(messages)
        content = result.content.strip()

        try:
            # JSON parse karo aur Pydantic model me daal do
            data = json.loads(content)
            return output_model(**data)
        except Exception as e:
            print(f"[⚠️ JSON Parse Failed] {e}\nGot content:\n{content}\n")
            # agar parse fail hua toh empty model return karo
            empty_fields = {f: None for f in output_model.model_fields}
            return output_model(**empty_fields)
