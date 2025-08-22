import os
import json
import re
import time
import streamlit as st
from typing import List, Dict, Any, Iterable, Tuple, Optional
from openai import OpenAI
from utils import init_chat_history, export_chat_bytes, CHAT_KEY

# Optional token estimation
try:
    import tiktoken
except ImportError:
    tiktoken = None

# -----------------------------
# Constants / Model Settings
# -----------------------------
MODEL_NAME = "openai/gpt-5-2025-08-07"
MODEL_CONTEXT = 128_000  # Adjust if provider differs
SAFETY_BUFFER = 200       # Reserved tokens to avoid overflow
SUMMARY_EVERY = 6         # Summarize after this many total turns
MEMORY_WINDOW = 8         # Last N messages fed into summarizer

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Coding Teacher",
    page_icon="👨‍🏫",
    layout="wide"
)

# -----------------------------
# Client
# -----------------------------
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=st.secrets["AIML_API_KEY"]
    # os.getenv("OPENAI_API_KEY")
)
#st.secrets["AIML_API_KEY"] use this if the API key is stored in Streamlit secrets.

# -----------------------------
# Helpers
# -----------------------------
CODE_PATTERN = re.compile(r"```|;\s*$|\bclass\b|\bdef\b|\bfunction\b")

def detect_code(text: str) -> bool:
    return bool(CODE_PATTERN.search(text))

def build_system_prompt(level: str) -> str:
    base = (
        "You are an expert coding teacher and mentor.\n"
        "Personality: friendly, patient, Socratic (ask guiding questions).\n"
        "Always:\n"
        "1. Start with a high-level overview.\n"
        "2. Provide step-by-step clarity.\n"
        "3. Include code examples in ```language blocks```.\n"
        "4. Suggest improvements and practice exercises.\n"
        "5. Encourage deeper exploration.\n"
    )
    if level == "Beginner":
        base += "Simplify concepts. Use analogies. Explain any jargon."
    elif level == "Intermediate":
        base += "Balance clarity with best practices and some depth."
    else:
        base += "Use advanced terminology, performance insights, design patterns."
    return base

def make_teaching_instruction(user_text: str) -> str:
    if detect_code(user_text):
        return (
            "The user provided code. Respond with:\n"
            "1. Plain-language explanation of what it does.\n"
            "2. Bugs or inefficiencies.\n"
            "3. Improvements/refactoring suggestions.\n"
            "4. A practice exercise."
        )
    return (
        "The user asked a question. Respond with:\n"
        "1. A simple analogy.\n"
        "2. Step-by-step explanation.\n"
        "3. A concise working code example.\n"
        "4. A practice exercise.\n"
        "Keep it focused and pedagogical."
    )

def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    if tiktoken:
        try:
            enc = tiktoken.encoding_for_model(MODEL_NAME)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for m in messages:
            total += len(enc.encode(m["content"])) + 4  # crude overhead
        return total + 8
    # Fallback heuristic ~4 chars / token
    return sum(len(m["content"]) for m in messages) // 4

def load_progress() -> Dict[str, Any]:
    if os.path.exists("progress.json"):
        try:
            with open("progress.json", "r") as f:
                return json.load(f)
        except Exception:
            return {"topics": []}
    return {"topics": []}

def save_progress(progress: Dict[str, Any]):
    try:
        with open("progress.json", "w") as f:
            json.dump(progress, f, indent=2)
    except Exception:
        pass

def update_progress(user_text: str):
    parts = user_text.strip().split()
    if not parts:
        return
    topic = parts[0].capitalize()
    progress = load_progress()
    if topic not in progress["topics"]:
        progress["topics"].append(topic)
        save_progress(progress)

def summarize_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    tail = history[-MEMORY_WINDOW:]
    convo = "\n".join(
        "User: " + m["content"] if m["role"] == "user" else "Teacher: " + m["content"]
        for m in tail
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Summarize the conversation into <=5 concise bullet points focusing on learning progress and user goals."},
                {"role": "user", "content": convo}
            ]
        )
        return resp.choices[0].message.content or ""
    except Exception:
        return ""

# Streaming generator (single pass)
def stream_chunks(messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float):
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    collected = ""
    finish_reason = None
    for event in stream:
        choices = getattr(event, "choices", None)
        if not choices:
            continue
        choice = choices[0]
        fr = getattr(choice, "finish_reason", None)
        if fr:
            finish_reason = fr
        delta = getattr(choice, "delta", None)
        delta_text = getattr(delta, "content", None) if delta else None
        if not delta_text:
            continue
        collected += delta_text
        yield ("delta", delta_text, None)
    yield ("done", None, finish_reason)

# Auto continuation loop
def generate_with_auto_continue(base_messages: List[Dict[str, str]],
                                first_max: int,
                                temperature: float,
                                top_p: float,
                                auto_continue: bool,
                                max_continuations: int) -> Tuple[str, Optional[str], int]:
    assembled = ""
    finish_reason = None
    segments = 0
    working_messages = list(base_messages)
    while True:
        partial = ""
        for kind, delta_text, fr in stream_chunks(
            working_messages,
            first_max,
            temperature,
            top_p
        ):
            if kind == "delta" and delta_text:
                partial += delta_text
                yield ("delta", delta_text, None)
            elif kind == "done":
                finish_reason = fr
        assembled += partial
        segments += 1
        if not (auto_continue and finish_reason == "length" and segments <= max_continuations):
            break
        # Prepare continuation
        working_messages = working_messages + [
            {"role": "assistant", "content": partial},
            {"role": "user", "content": "Continue exactly where you stopped without repeating previous content."}
        ]
    yield ("final", assembled, finish_reason if finish_reason else "stop")

# -----------------------------
# State Initialization
# -----------------------------
if "level" not in st.session_state:
    st.session_state.level = "Beginner"
if "messages" not in st.session_state:
    # Store ONLY user/assistant turns here (no system)
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "last_finish_reason" not in st.session_state:
    st.session_state.last_finish_reason = ""
if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = 0.0
init_chat_history()
# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")

level = st.sidebar.radio("Teaching Level", ["Beginner", "Intermediate", "Advanced"],
                         index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.level))
if level != st.session_state.level:
    st.session_state.level = level

temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.7, 0.05)
top_p = st.sidebar.slider("Top‑p", 0.1, 1.0, 0.9, 0.05)
output_cap = st.sidebar.slider("Max output tokens (per segment)", 1000, 10000, 2000, 500)
auto_continue = st.sidebar.checkbox("Auto‑continue if truncated", True)
max_continuations = st.sidebar.slider("Max continuations", 0, 5, 2)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.summary = ""
    st.session_state.last_finish_reason = ""
    st.sidebar.success("Cleared.")
    st.session_state[CHAT_KEY] = []

with st.sidebar.expander("📈 Learning Progress", expanded=True):
    prog = load_progress()
    if prog["topics"]:
        st.write(", ".join(prog["topics"]))
    else:
        st.write("No topics tracked yet.")

with st.sidebar.expander("ℹ️ Session Info", expanded=False):
    st.write(f"Turns: {len(st.session_state.messages)}")
    if st.session_state.last_finish_reason:
        st.write(f"Last finish: {st.session_state.last_finish_reason}")
    if st.session_state.last_response_time:
        st.write(f"Last response time: {st.session_state.last_response_time:.2f}s")

# -----------------------------
# Main Header
# -----------------------------
st.title("👨‍🏫 AI Coding Teacher (powered by GPT-5)")
st.caption("Learn coding step by step with an AI mentor.")
st.caption("""It has 3 modes: Beginner, Intermediate, and Advanced.\n
            Beginner: Learn the basics of coding with simple explanations and examples.\n
            Intermediate: Deepen your understanding with practical coding challenges and best practices.\n
            Advanced: Master complex algorithms and design patterns with expert insights.
            """)
st.caption("""Other details:\n
            Temperature controls creativity (higher = more creative, lower = more focused),\n 
            Top-p filters randomness (lower = safer, higher = more diverse),\n
            Max tokens limits response length,\n
            Auto continuation allows the model to continue if it runs out of tokens,\n
            and Max continuation decides how long the model continues writing.""")

# -----------------------------
# Render Conversation
# -----------------------------
for msg in st.session_state.messages:
    avatar = "👨‍🎓" if msg["role"] == "user" else "👩‍🏫"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# -----------------------------
# User Input & Processing
# -----------------------------
user_input = st.chat_input("Ask a coding question, request help, or paste code...")

if user_input:
    # Append user message

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state[CHAT_KEY].append({"role": "user", "content": user_input})
    update_progress(user_input)

    # Build model messages
    system_prompt = build_system_prompt(st.session_state.level)
    instruction = make_teaching_instruction(user_input)

    model_messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    if st.session_state.summary:
        model_messages.append({"role": "system", "content": f"Conversation summary memory:\n{st.session_state.summary}"})

    # Insert an instruction as a user directive before conversation so far
    model_messages.append({"role": "user", "content": instruction})
    # Add conversation turns
    model_messages.extend(st.session_state.messages)

    # Token management
    prompt_tokens = estimate_tokens(model_messages)
    available = MODEL_CONTEXT - prompt_tokens - SAFETY_BUFFER
    safe_max = max(64, min(output_cap, available))
    if available <= 64:
        st.warning("Context almost full. Consider clearing chat.")
        safe_max = 64

    # Streaming assistant
    with st.chat_message("assistant", avatar="👩‍🏫"):
        placeholder = st.empty()
        collected = ""
        start_time = time.time()
        try:
            for kind, payload, fr in generate_with_auto_continue(
                base_messages=model_messages,
                first_max=safe_max,
                temperature=temperature,
                top_p=top_p,
                auto_continue=auto_continue,
                max_continuations=max_continuations
            ):
                if kind == "delta":
                    collected += payload
                    placeholder.markdown(collected + "▌")
                elif kind == "final":
                    collected = payload
                    st.session_state.last_finish_reason = fr
        except Exception as e:
            placeholder.error(f"Error: {e}")
        else:
            placeholder.markdown(collected)

        elapsed = time.time() - start_time
        st.session_state.last_response_time = elapsed
        st.caption(f"Finish: {st.session_state.last_finish_reason} | Time: {elapsed:.2f}s | Tokens (est. prompt): {prompt_tokens} | Output cap used: {safe_max}")

    if collected:
        st.session_state.messages.append({"role": "assistant", "content": collected})
        st.session_state[CHAT_KEY].append({"role": "assistant", "content": collected})

    # Summarize periodically
    if len(st.session_state.messages) % SUMMARY_EVERY == 0:
        st.session_state.summary = summarize_history(st.session_state.messages)

# -----------------------------
# Footer / Debug (optional)
# -----------------------------
with st.expander("🧠 Memory Summary (internal)", expanded=False):
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.write("No summary yet.")

st.download_button(
    "Download chat JSON",
    data=export_chat_bytes(),
    file_name="chat_history.json",
    mime="application/json"
)