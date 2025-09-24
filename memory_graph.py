# memory_graph.py

from __future__ import annotations
import os
import uuid
import re
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore


# --------------------- LLM CONFIG ---------------------

MODEL_NAME = "gpt-oss:20b"
BASE_URL = "http://localhost:11434"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
    base_url=BASE_URL,
)

# --------------------- MEMORY HELPERS ---------------------

MEMORY_ROOT = ("memories",)
INDEX_CONFIG = None  # set to a dict to enable semantic search in PostgresStore.search


def extract_memory(text: str) -> Optional[str]:
    """
    Very simple heuristic: store the text after "remember:" or common facts.

    Examples captured:
    - "remember: my name is Jane" -> "my name is Jane"
    - "my name is Bob" -> "name: Bob"
    - "I live in Berlin" -> "location: Berlin"
    - "my favorite color is blue" -> "favorite color: blue"
    """
    m = re.search(r"remember\s*:\s*(.+)", text, flags=re.I)
    if m:
        return m.group(1).strip()

    m = re.search(r"\bmy name is\s+([\w\-'. ]{2,60})\b", text, flags=re.I)
    if m:
        return f"name: {m.group(1).strip()}"

    m = re.search(r"\bi live in\s+([\w\-'. ]{2,60})\b", text, flags=re.I)
    if m:
        return f"location: {m.group(1).strip()}"

    m = re.search(
        r"\bmy favourite color is\s+([\w\-']+)\b|\bmy favorite color is\s+([\w\-']+)\b",
        text,
        flags=re.I,
    )
    if m:
        color = m.group(1) or m.group(2)
        return f"favorite color: {color.strip()}"

    return None


# --------------------- AGENT NODE ---------------------

def _normalize_messages(msgs: list[BaseMessage | dict]) -> list[BaseMessage]:
    """
    If any dicts sneak in, convert them to LC message objects.
    """
    norm: list[BaseMessage] = []
    for m in msgs:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                norm.append(HumanMessage(content=content))
            else:
                # default: treat non-user as assistant messages
                from langchain_core.messages import AIMessage
                norm.append(AIMessage(content=content))
        else:
            norm.append(m)
    return norm


def agent_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """Reads relevant memories, optionally writes new ones, and calls the LLM."""
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = MEMORY_ROOT + (user_id,)

    # Ensure we have LC message objects in state
    state_messages = _normalize_messages(state["messages"])

    # Find last human message
    last_user_msg = next((m for m in reversed(state_messages) if isinstance(m, HumanMessage)), None)
    last_text = last_user_msg.content if last_user_msg else ""

    # Read memories relevant to this turn (semantic search if index configured)
    try:
        if INDEX_CONFIG is not None:
            hits = store.search(namespace, query=last_text, limit=5)
        else:
            # Fallback lexical-ish search
            hits = store.search(namespace, query=last_text or "*", limit=5)
    except Exception:
        hits = []

    memory_lines = [h.value.get("text") or h.value.get("data") or str(h.value) for h in hits]
    memory_blob = "\n".join(memory_lines).strip()

    system_msg = "You are a helpful assistant."
    if memory_blob:
        system_msg += f"\nKnown user info (long-term memory):\n{memory_blob}"

    # Write new memory if we detect a fact to remember
    new_memory = extract_memory(last_text or "")
    if new_memory:
        store.put(namespace, str(uuid.uuid4()), {"text": new_memory})

    # Call the model with LC messages
    response = llm.invoke([
        SystemMessage(content=system_msg),
        *state_messages,
    ])
    return {"messages": response}


# --------------------- GRAPH BUILD ---------------------

def build_graph(store: PostgresStore, checkpointer: PostgresSaver):
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(store=store, checkpointer=checkpointer)


# --------------------- MAIN ---------------------

if __name__ == "__main__":
    # IMPORTANT: match this port to how you run the container, e.g.:
    # docker run --name postgresql_memory ^
    #   -e POSTGRES_PASSWORD=testpassword ^
    #   -e POSTGRES_USER=testuser ^
    #   -e POSTGRES_DB=testdb ^
    #   -p 5455:5432 -d postgres
    #
    # Then the URI below will work.
    DB_URI = os.environ.get(
        "DATABASE_URL",
        "postgresql://testuser:testpassword@localhost:5455/testdb",
    )

    # Build store & checkpointer, run migrations once on first run
    with (
        PostgresStore.from_conn_string(DB_URI, index=INDEX_CONFIG) as store,
        PostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        # Initialize tables/indexes (safe to call repeatedly)
        store.setup()
        checkpointer.setup()

        graph = build_graph(store, checkpointer)

        cfg = {"configurable": {"thread_id": "t-1", "user_id": "alice"}}

        print("\n--- Turn 1 (teach a fact) ---")
        for chunk in graph.stream(
            {"messages": [HumanMessage(content="Hi! remember: my name is Alice and I live in Bucharest.")]},
            cfg,
            stream_mode="values",
        ):
            last = chunk["messages"][-1]
            print(getattr(last, "content", last))

        print("\n--- Turn 2 (ask about the fact) ---")
        for chunk in graph.stream(
            {"messages": [HumanMessage(content="What is my name and where do I live?")]},
            cfg,
            stream_mode="values",
        ):
            last = chunk["messages"][-1]
            print(getattr(last, "content", last))

        # New thread: short-term history resets, but long-term memory still helps
        cfg2 = {"configurable": {"thread_id": "t-2", "user_id": "alice"}}
        print("\n--- Turn 3 (new thread, should still use long-term memory) ---")
        for chunk in graph.stream(
            {"messages": [HumanMessage(content="Remind me who I am and my city.")]},
            cfg2,
            stream_mode="values",
        ):
            last = chunk["messages"][-1]
            print(getattr(last, "content", last))
