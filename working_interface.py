import gradio as gr
import requests
import json
import uuid
import re
import shutil # fiile and directory operations

from pathlib import Path
from datetime import datetime

from gradio import ChatMessage
from react_general_workflow import get_app
from langgraph.types import Command
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage



# TODO:
# - file management system
# - give file to LLM 
# - create graph and subgraphs
# - create autofiller agent


# Ollama local model
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gpt-oss:20b"

# CSS for chat interface

# CSS = """
# #input-row { gap: 8px; align-items: end; }
# #send-btn, #clear-btn {
#   min-width: 110px;      /* stop stretching */
#   height: 40px;          /* consistent height */
#   padding: 0 14px;       /* normal padding */
#   border-radius: 10px;   /* optional: softer look */
# }
# """

CSS = """
/* 1) full-height app scaffolding */
html, body, #root, .gradio-container, #app { height: 100%; margin: 0; padding: 0; }
.gradio-container .grid { height: 100%; }

/* 2) main column must be a flex column that can host a fixed-height chat box */
#main { display: flex; flex-direction: column; flex: 1; min-height: 0; }

/* 3) FIXED-SIZE message area: do not let it grow with content */
#chat-wrap {
  flex: 0 0 calc(100vh - 170px); /* <-- fixed height; adjust 170px for your header+input row */
  min-height: 0;                 /* IMPORTANT: allow scrolling child instead of growing parent */
}

/* 4) Force the chatbot and its internal wrapper to occupy that fixed area */
#chatbot,
#chatbot > div,
#chatbot .wrap {
  height: 100% !important;
}

/* 5) Scroll INSIDE the messages container, not the page */
#chatbot .wrap {
  overflow-y: auto !important;
}

/* (optional) keep the input row separate; your current sticky row is fine */

"""
# sessions_state will be:
# {
#   "<session_id>": {
#       "history": [ {"role":"user","content":"..."}, {"role":"assistant","content":"..."} ],
#       "thread_id": "<uuid>",
#       "pending_interrupt": None | dict,
#       "excel_path": None | str
#   },
#   ...
# }

# persistent storage for user uploads (one subfolder per chat session)
CONVS_STORAGE = Path("conversations").resolve()
CONVS_STORAGE.mkdir(parents=True, exist_ok=True)


# NOTE
# workflow_config is how we specify threads for graphss
# thread_id can be anything so we can use uuid
# we will map 1:1 the session id of the chatbot with the graph's execution thread ( for now )
WORKFLOW_CONFIG = {"configurable": {"thread_id":uuid.uuid4()}}

def generate_thread_id():
    """ 
    Utility funciton that generates a random ID.
    Used everywhere
    Turn to string if necessary (APIs, DBs)
    """
    return uuid.uuid4()
    

def _ensure_session(sessions, sid: str):
    if sid not in sessions:
        sessions[sid] = {
            "history": [],
            "thread_id": str(generate_thread_id()),
            "pending_interrupt": None,
            "excel_path": None,
            "last_seen_index": 0, # used for rendering just the last messages from the AI Assistant
        }
    return sessions[sid]

def _get_thread_id():
    pass


# -----------------------
def format_history(history,message):
    """Utility function for appending history to sessions"""
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role":"user","content":user_msg})
        messages.append({"role":"assistant","content":bot_msg})
    
    messages.append({"role":"user","content":message})
    return messages


def chat_with_ollama_json(message, history):
    """Chatting with ollama's models and the response come back as an enitre message"""
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "PetrosStav/gemma3-tools:12b", 
            "messages": [{"role": "user", "content": message}],
            "stream": False
        }
    )
    data = response.json()
    return data["message"]["content"]



def chat_with_ollama_streaming_history(history,sessions,session_id):
    """
    Chatting with Chatbot.
    Choosing local-deployed ollama model.
    Streaming and history functionalities per session.
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL, 
            "messages": history,
            "stream": True
        },
        stream=True
    )

    partial  = ""
    work_history = history + [{"role": "assistant", "content": ""}]
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line.decode("utf-8"))
        if "message" in data and "content" in data["message"]:
            # yielding the whole history instead of just the message
            # "data incomatible with tuples format" problem solved
            token = data["message"]["content"]
            partial += token
            work_history[-1]["content"] = partial
            if session_id:
                sessions[session_id] = work_history
            yield work_history, sessions
        if data.get("done", False):
            break


    if session_id:
        sessions[session_id] = work_history
    yield work_history, sessions


# NOTE: this session id might be correlated with the thread id from langgraph
def new_session(sessions,current_session_id):
    """
    Utility function for creating a new session ( chatting with the chatbot from another tab ).
    Each session will be distinguished based on an uuid.
    Each session will have its own chat history which can be persisted on further updates.
    """
    new_id = str(generate_thread_id())[:8]
    sessions[new_id] = {
        "history":[],
        "thread_id":str(uuid.uuid4()),
        "pending_interrupt":None,
        "excel_path": None
    }
    # ensure folder for new session
    new_session_folder_path = CONVS_STORAGE / f"conv_{new_id}"
    new_session_folder_path.mkdir(parents=True, exist_ok=True)
    (new_session_folder_path / "uploads").mkdir(parents=True,exist_ok=True) # file uploads
    # file index = registry of uploaded files per chat session
    return (
        gr.update(choices=list(sessions.keys()), value=new_id), # dropdown
        sessions, # sessions_state
        new_id, # current_session_id
        [], # chatbot history for this UI only <--> sessions[new_id]["history"]
    )
 
def switch_session(session_id,sessions):
    """
    Switch to a selected session based on the session id.`
    """
    sid = session_id or "default"
    _ensure_session(sessions,sid)
    return (
        sid,
        sessions[sid]["history"],
    )

def _user_submit(user_message,history,sessions,session_id):
        """
        Utility function for sending the user messages and handling file uploads.
        Provides chat history management and streaming.
        Handles both MultimodalTextbox ( user_message: dict ) 
        and simple Textbox ( user_message: str )
        """
        if not session_id:
            session_id="default"
        sess = _ensure_session(sessions,session_id)

        session_folder = CONVS_STORAGE / f"conv_{session_id}"
        uploads_folder = session_folder / "uploads"
        uploads_folder.mkdir(parents=True, exist_ok=True)

        if isinstance(user_message,dict):
            text = user_message.get("text", "")
            files = user_message.get("files", []) or []
        else:
            text = user_message
            files = []


       # saving uploaded files
        uploaded_file_names = []
        excel_file = None
        for f in files:
            destination = uploads_folder / Path(f).name
            shutil.copy(f,uploads_folder / Path(f).name)
            uploaded_file_names.append(destination.name)
            if destination.suffix.lower() == ".xlsx" and excel_file is None:
                excel_file = str(destination.resolve())
        
        if excel_file:
            sess["excel_path"] = excel_file # passing to graph

        if not text and uploaded_file_names:
            text = "📎 Uploaded:"+", ".join(uploaded_file_names)

        # NOTE: check if we have to append files to history or it's not necessarys
        history = history + [{"role": "user", "content": text}]
        sess["history"]= history
        return "", history, sessions

# --- Replacing Ollama steraming with LangGraph turn runner ----

def new_gradio_msgs(all_lc_messages, start_idx: int):
    # slice the LangGraph message list
    subset = (all_lc_messages or [])[start_idx:]
    return to_gradio_msgs(subset)



def _extract_assistant_text(messages):
    """ 
    Pick the latest AI/tool text to show as a single assistant message. 
    AIMessage, ToolMessage are LangChain message object, different from dicts.
    """
    last = None
    for m in messages or []:
        # type
        mtype = getattr(m, "type", None)
        if mtype is None and isinstance(m, dict):
            mtype = m.get("type")

        # content
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content")

        if not content:
            continue

        is_ai_or_tool = (
            mtype in ("ai", "tool")
            or isinstance(m, (AIMessage, ToolMessage))
        )
        if is_ai_or_tool:
            last = content if isinstance(content, str) else str(content)

    return last

def run_langgraph_turn(history,sessions,session_id):
    """ Execute one LangGraph step based on the latest user turn + session state """
    if not session_id:
        session_id = "default"
    sess = _ensure_session(sessions,session_id)
    app = get_app() # imported singleton graph with persistency
    config = {"configurable":{"thread_id":sess["thread_id"]}}

    # pull the latest user message from the UI history ( send to graph if needed )
    user_text = ""
    if history and history[-1]["role"] == "user":
        user_text = history[-1]["content"]

    # pause on interrupt, resume
    if sess["pending_interrupt"]:
        payload = sess["pending_interrupt"]
        action = payload.get("action")

        # if excel path is asked
        if action == "provide_excel_path":
            path = sess["excel_path"] or (user_text or "").strip()
            resume_value = {"excel_file_path": path} if path else ""
            result = app.invoke(Command(resume = resume_value),config)
            sess["pending_interrupt"] = None
    
        # next user message 
        else:
            resume_value = {"message":user_text} if user_text else ""
            result = app.invoke(Command(resume=resume_value), config)
            sess["pending_interrupt"] = None
    
    # normal turn without interrupt
    else:
        inputs = {}
        if sess["excel_path"]:
            inputs["excel_file_path"] = sess["excel_path"]
            inputs["uploaded_excel"] = True
        if user_text:
            inputs["messages"] = [("user",user_text)]
        result = app.invoke(inputs,config)

     # Handle new interrupt (human-in-the-loop)
    interrupts = result.get("__interrupt__")
    all_msgs= result.get("messages") or []
    start = sess.get("last_seen_index",0)

    if interrupts:
        # 1) append only new assistant/tool outputs produced just before the pause
        for gmsg in new_gradio_msgs(all_msgs,start):
            history = history + [gmsg]   # due to the formatter, gmsg is already {"role":"assistant", "content":..., "metadata":...}

        # 2) also show the AI response
        payload = getattr(interrupts[0], "value", {}) or {}
        prompt = payload.get("prompt") or "Please provide the requested input."
        history = history + [ChatMessage(role="assistant", content=prompt).__dict__]

        sess["last_seen_index"] = len(all_msgs)
        sess["pending_interrupt"] = payload
        sess["history"] = history
        sessions[session_id] = sess
        return history, sessions

    # otherwise, show the latest agent/tool message ( + tool calling metadata )
    # No interrupt → just show new assistant/tool outputs
    for gmsg in new_gradio_msgs(all_msgs,start):
        history = history + [gmsg]
    
    sess["last_seen_index"] = len(all_msgs)
    sess["history"] = history
    sessions[session_id] = sess
    return history, sessions


# --- pretty print tools and reasoning ---
def to_gradio_msgs(lc_messages: list[BaseMessage]) -> list[dict]:
    """
    Convert LangChain messages to ChatMessage dicts.
    For ToolMessage, attach a 'Used tool ...' accordion.
    Used in formatting UI in 'run_langgraph_turn'
    """
    out = []
    for m in lc_messages or []:
        # AI (normal assistant text)
        if isinstance(m, AIMessage) or getattr(m, "type", None) == "ai":
            content = m.content if isinstance(m.content, str) else str(m.content)
            out.append(ChatMessage(role="assistant", content=content).__dict__)
        # Tool usage → show in an accordion
        elif isinstance(m, ToolMessage) or getattr(m, "type", None) == "tool":
            content = m.content if isinstance(m.content, str) else str(m.content)
            title = f"🛠 Used tool {getattr(m, 'name', None) or 'tool'}"
            # optional detail line:
            log = f"tool_call_id={getattr(m, 'tool_call_id', '')}"
            meta = {"title": title, "log": log, "status": "done"}
            out.append(ChatMessage(role="assistant", content=content, metadata=meta).__dict__)
    return out


# --- gradio blocks rendering ---
with gr.Blocks(css=CSS,elem_id ="app",fill_height=True) as demo:
    gr.Markdown("Multi-Session Xena")

    # Default States
    sessions_state = gr.State({})
    current_session_id = gr.State("")
    #files_state = gr.State({})     # session_id -> list[metadata]

    with gr.Row(): 
        # Sidebar area
        with gr.Sidebar():

            # chats area
            gr.Markdown("### 📂Chats")
            session_dropdown = gr.Dropdown(label="Sessions", choices =[], value=None, allow_custom_value=True)
            new_button = gr.Button("➕ New Chat")
        
        # main chatbot interface area
        with gr.Column(elem_id="main", scale=1):
            with gr.Column(elem_id="chat-wrap"):
                chatbot = gr.Chatbot(
                    type="messages",
                    elem_id="chatbot",      
                    show_copy_button=True
                )
            with gr.Row(elem_id="input-row"):
                msg = gr.MultimodalTextbox(
                    interactive=True,
                    placeholder="Ask anything",
                    show_label=False,
                    file_count="multiple",
                    lines=1,
                    scale=8,
                )
                clear = gr.Button("Clear", variant="secondary", scale=0, min_width=110, elem_id="clear-btn")
                send  = gr.Button("Send",  variant="primary",   scale=0, min_width=110, elem_id="send-btn")

    # Main chat area
    # chatbot = gr.Chatbot(type="messages", elem_id="chatbot", show_copy_button=True)
    # with gr.Row(elem_id="input-row"):
    #     msg = gr.MultimodalTextbox(
    #         interactive=True,
    #         placeholder="Ask anything",
    #         show_label=False,   # hide “MultiModelTextbox label”
    #         file_count="multiple", # single | directory | multiple
    #         lines=1,
    #         scale=8,             # let the textbox take the width
    #     )

    #     clear = gr.Button("Clear", variant="secondary",
    #                       scale=0, min_width=110, elem_id="clear-btn")
    #     send  = gr.Button("Send",  variant="primary",
    #                       scale=0, min_width=110, elem_id="send-btn")

    # Wire events after components / states already exists

    # Auto-creating first session when user enters on app
    demo.load(
        new_session,
        [sessions_state, current_session_id],
        [
            session_dropdown,
            sessions_state,
            current_session_id,
            chatbot,
        ],
    )

    # New Chat button
    new_button.click(
        new_session,
        [sessions_state, current_session_id],
        [
            session_dropdown,
            sessions_state,
            current_session_id,
            chatbot,
        ],
    )

    # Submit message
    msg.submit(
        _user_submit, 
        [msg, chatbot, sessions_state, current_session_id], # inputs 
        [msg, chatbot, sessions_state] # outputs
    ).then(
        # old : chat_with_ollama_streaming_history,
        run_langgraph_turn,
        [chatbot, sessions_state, current_session_id], # inputs
        [chatbot, sessions_state], # outputs
    )

    # send file
    send.click(
        _user_submit,
        [msg, chatbot, sessions_state, current_session_id],
        [msg, chatbot, sessions_state],
    ).then(
        # old : chat_with_ollama_streaming_history,
        run_langgraph_turn,
        [chatbot, sessions_state, current_session_id],
        [chatbot, sessions_state],
    )

    # Session switching (chat + files)
    def _switch_session(sid, sessions):
        return switch_session(sid, sessions)

    session_dropdown.change(
        _switch_session,
        [session_dropdown, sessions_state ],
        [current_session_id, chatbot],
    )

    # clear current session ( files are kept )
    def clear_session(session_id,sessions):
        if session_id:
            sess = _ensure_session(sessions,session_id)
            sess["history"] = []
            sess["last_seen_index"] = 0
        return [],sessions
    
    clear.click(clear_session, [current_session_id, sessions_state], [chatbot, sessions_state])
    
    # inject CSS  
    gr.HTML(f"<style>{CSS}</style>")

if __name__ == "__main__":
    demo.launch(
        allowed_paths=[str(CONVS_STORAGE)],
        max_file_size="20mb",
    )
