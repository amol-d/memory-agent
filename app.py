"""Gradio demo UI for the Memory Agent, mounted on FastAPI.

Memory lives in per-session gr.State — never shared across visitors, never persisted.
"""

from __future__ import annotations

import gradio as gr

from memory_agent.agent import MemoryAgent
from memory_agent.config import CONFIG
from memory_agent.security import RateLimitError, ValidationError, sanitize_text
from memory_agent.web import LIMITER, caller_id, make_app, run

_agent: MemoryAgent | None = None


def _get_agent() -> MemoryAgent:
    global _agent
    if _agent is None:
        _agent = MemoryAgent()
    return _agent


def _render_memories(memories: list[str]) -> str:
    if not memories:
        return "*No memories yet. Tell me about yourself — your name, what you like, what you're working on.*"
    return "**What I remember**\n\n" + "\n".join(f"- {m}" for m in memories)


def respond(message: str, history: list[dict], memories: list[str], request: gr.Request):
    memories = memories or []
    try:
        clean = sanitize_text(message, field="a message", min_chars=1)
    except ValidationError as exc:
        history = history + [
            {"role": "user", "content": message or ""},
            {"role": "assistant", "content": f"⚠️ {exc}"},
        ]
        return "", history, memories, _render_memories(memories)

    try:
        LIMITER.check(caller_id(request))
    except RateLimitError as exc:
        history = history + [{"role": "user", "content": clean}, {"role": "assistant", "content": f"⏳ {exc}"}]
        return "", history, memories, _render_memories(memories)

    if not CONFIG.api_key_present:
        history = history + [{"role": "user", "content": clean}, {"role": "assistant", "content": "⚠️ The demo is not configured (missing API key)."}]
        return "", history, memories, _render_memories(memories)

    try:
        reply = _get_agent().reply(history, memories, clean)
        new_mems = _get_agent().extract_memories(clean)
    except Exception:  # noqa: BLE001
        history = history + [{"role": "user", "content": clean}, {"role": "assistant", "content": "⚠️ Something went wrong. Please try again."}]
        return "", history, memories, _render_memories(memories)

    for m in new_mems:
        if m not in memories:
            memories.append(m)

    history = history + [{"role": "user", "content": clean}, {"role": "assistant", "content": reply}]
    return "", history, memories, _render_memories(memories)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Memory Agent — Day 03", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## 🧠 Memory Agent\n"
            "Chat with an assistant that **remembers durable facts about you** and uses "
            "them to personalize replies. The panel shows exactly what it has stored.\n\n"
            "*Day 03 of 14 AI Agents in 14 Days — short- + long-term memory, extraction.*"
        )
        memories_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=380, label="Chat")
                msg = gr.Textbox(label="Message", placeholder="Hi! My name is …", lines=1)
            with gr.Column(scale=2):
                memory_view = gr.Markdown(_render_memories([]))

        msg.submit(respond, [msg, chatbot, memories_state], [msg, chatbot, memories_state, memory_view])

    demo.queue(default_concurrency_limit=2, max_size=20)
    return demo


app = make_app(build_demo(), title="Memory Agent")

if __name__ == "__main__":
    run(app)
