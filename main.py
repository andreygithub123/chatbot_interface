import gradio as gr
import requests
import json
import uuid
import re
import shutil # fiile and directory operations
import base64
import mimetypes

from pathlib import Path
from datetime import datetime

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("PyMuPDF not found. Install with: pip install PyMuPDF")


# TODO:
# - file management system
# - give file to LLM 
# - create graph and subgraphs
# - create autofiller agent


# Ollama local model
OLLAMA_URL = "http://localhost:11434/api/chat"
#MODEL = "gpt-oss:20b"
MODEL = "qwen2.5vl:32b"

# CSS for chat interface

CSS = """
#input-row { gap: 8px; align-items: end; }
#send-btn, #clear-btn {
  min-width: 110px;      /* stop stretching */
  height: 40px;          /* consistent height */
  padding: 0 14px;       /* normal padding */
  border-radius: 10px;   /* optional: softer look */
}
"""

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
    Supports multimodal content (text + images).
    """
    # Get the multimodal history for LLM processing
    llm_history = sessions.get('_llm_history', {}).get(session_id, [])

    # Use LLM history if available, otherwise fall back to display history
    messages_to_process = llm_history if llm_history else history

    # Prepare messages for Ollama API
    processed_messages = []
    for msg in messages_to_process:
        if isinstance(msg["content"], list):
            # Multimodal content - combine text and images
            processed_msg = {
                "role": msg["role"],
                "content": ""
            }
            images = []

            for content_item in msg["content"]:
                if content_item["type"] == "text":
                    processed_msg["content"] += content_item["text"]
                elif content_item["type"] == "image_url":
                    # Extract base64 data from data URL
                    image_url = content_item["image_url"]["url"]
                    if image_url.startswith("data:"):
                        # Format: data:image/jpeg;base64,/9j/4AAQ...
                        base64_data = image_url.split(",", 1)[1]
                        images.append(base64_data)

            if images:
                processed_msg["images"] = images
            processed_messages.append(processed_msg)
        else:
            # Simple text content
            processed_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": processed_messages,
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
                # Also update LLM history with assistant response
                if '_llm_history' in sessions and session_id in sessions['_llm_history']:
                    # Update or add assistant message in LLM history
                    if len(sessions['_llm_history'][session_id]) > 0:
                        # Check if last message is assistant, if so update it, otherwise add new
                        if (len(sessions['_llm_history'][session_id]) > 0 and
                            sessions['_llm_history'][session_id][-1].get("role") == "assistant"):
                            sessions['_llm_history'][session_id][-1]["content"] = partial
                        else:
                            sessions['_llm_history'][session_id].append({"role": "assistant", "content": partial})
                    else:
                        sessions['_llm_history'][session_id].append({"role": "assistant", "content": partial})
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
    sessions[new_id] = []
    # ensure folder for new session
    new_session_folder_path = CONVS_STORAGE / f"conv_{new_id}"
    new_session_folder_path.mkdir(parents=True, exist_ok=True)
    (new_session_folder_path / "uploads").mkdir(parents=True,exist_ok=True) # file uploads
    # file index = registry of uploaded files per chat session
    return (
        gr.update(choices=list(sessions.keys()), value=new_id), # dropdown
        sessions, # sessions_state
        new_id, # current_session_id
        [], # chatbot history
    )
 
def switch_session(session_id,sessions):
    """
    Switch to a selected session based on the session id.`
    """
    sid = session_id or "default"
    return (
        sid,
        sessions.get(sid, []),
    )

def encode_image_to_base64(image_path):
    """Convert image file to base64 string for multimodal LLM"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_pdf_content(pdf_path, uploads_folder):
    """Extract both text and images from PDF file"""
    if not PDF_SUPPORT:
        return [], "PDF processing not available. Install PyMuPDF: pip install PyMuPDF"

    try:
        content_items = []
        text_content = ""

        # Open PDF document
        doc = fitz.open(pdf_path)

        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Extract text from page
            page_text = page.get_text()
            if page_text.strip():
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"

            # Extract images from page
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Save image to uploads folder
                    image_filename = f"pdf_page{page_num + 1}_img{img_index + 1}.{image_ext}"
                    image_path = uploads_folder / image_filename

                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    # Convert to base64 for LLM
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    mime_type = f"image/{image_ext}"

                    content_items.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })

                except Exception as e:
                    print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")

        doc.close()

        # Add text content if available
        if text_content.strip():
            content_items.insert(0, {
                "type": "text",
                "text": f"\n\n**PDF Text Content:**\n{text_content.strip()}\n\n"
            })

        return content_items, None

    except Exception as e:
        return [], f"Error processing PDF: {str(e)}"

def _user_submit(user_message,history,sessions,session_id):
        """
        Utility function for sending the user messages and handling file uploads.
        Provides chat history management and streaming.
        Handles both MultimodalTextbox ( user_message: dict )
        and simple Textbox ( user_message: str )
        """
        if not session_id:
            session_id="default"
            sessions.setdefault(session_id,[])
        session_folder = CONVS_STORAGE / f"conv_{session_id}"
        uploads_folder = session_folder / "uploads"
        uploads_folder.mkdir(parents=True, exist_ok=True)

        if isinstance(user_message,dict):
            text = user_message.get("text", "")
            files = user_message.get("files", [])
        else:
            text = user_message
            files = []

        # saving uploaded files and prepare multimodal content
        file_contents = []
        for f in files:
            file_path = Path(f)
            dest_path = uploads_folder / file_path.name
            shutil.copy(f, dest_path)

            # Check file type and process accordingly
            mime_type, _ = mimetypes.guess_type(str(file_path))

            if mime_type and mime_type.startswith('image/'):
                # Handle images
                try:
                    base64_image = encode_image_to_base64(dest_path)
                    file_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })
                except Exception as e:
                    print(f"Error encoding image {file_path.name}: {e}")

            elif mime_type == 'application/pdf' or file_path.suffix.lower() == '.pdf':
                # Handle PDFs - extract both text and images
                try:
                    pdf_content_items, error = extract_pdf_content(dest_path, uploads_folder)
                    if error:
                        file_contents.append({
                            "type": "text",
                            "text": f"\n\n**PDF Error ({file_path.name}):** {error}\n\n"
                        })
                    else:
                        # Add all extracted content (text and images)
                        file_contents.extend(pdf_content_items)
                except Exception as e:
                    print(f"Error processing PDF {file_path.name}: {e}")
                    file_contents.append({
                        "type": "text",
                        "text": f"\n\n**PDF Error ({file_path.name}):** Could not process PDF - {str(e)}\n\n"
                    })

        # Create multimodal message content
        content = []
        if text:
            content.append({"type": "text", "text": text})
        content.extend(file_contents)

        # For Gradio display: keep it simple with text representation
        # For LLM processing: store full multimodal content separately
        display_text = text or ""
        if file_contents:
            # Add file info to display text
            image_count = sum(1 for item in file_contents if item.get("type") == "image_url")
            text_files = sum(1 for item in file_contents if item.get("type") == "text")
            if image_count > 0:
                display_text += f" [📷 {image_count} image(s) attached]"
            if text_files > 0:
                display_text += f" [📄 PDF content attached]"

        # Store both formats: display format for Gradio, multimodal for LLM
        user_message_display = {"role": "user", "content": display_text}
        user_message_llm = {"role": "user", "content": content if file_contents else text}

        # Add display version to history for Gradio
        history = history + [user_message_display]

        # Store LLM version in a way we can access it later
        if not hasattr(sessions, '_llm_history'):
            sessions['_llm_history'] = {}
        if session_id not in sessions.get('_llm_history', {}):
            sessions['_llm_history'][session_id] = []

        sessions['_llm_history'][session_id].append(user_message_llm)
        sessions[session_id] = history

        return "", history, sessions

# --- gradio blocks rendering ---
with gr.Blocks(css=CSS) as demo:
    gr.Markdown("Multi-Session Xena")

    # Default States
    sessions_state = gr.State({})
    current_session_id = gr.State("")
    #files_state = gr.State({})     # session_id -> list[metadata]

    # Sidebar area
    with gr.Sidebar():

        # chats area
        gr.Markdown("### 📂Chats")
        session_dropdown = gr.Dropdown(label="Sessions", choices =[], value=None, allow_custom_value=True)
        new_button = gr.Button("➕ New Chat")

      
    # Main chat area
    chatbot = gr.Chatbot(type="messages", elem_id="chatbot", show_copy_button=True)
    with gr.Row(elem_id="input-row"):
        msg = gr.MultimodalTextbox(
            interactive=True,
            placeholder="Ask anything",
            show_label=False,   # hide “MultiModelTextbox label”
            file_count="multiple", # single | directory | multiple
            lines=1,
            scale=8,             # let the textbox take the width
        )

        clear = gr.Button("Clear", variant="secondary",
                          scale=0, min_width=110, elem_id="clear-btn")
        send  = gr.Button("Send",  variant="primary",
                          scale=0, min_width=110, elem_id="send-btn")

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
        chat_with_ollama_streaming_history,
        [chatbot, sessions_state, current_session_id], # inputs
        [chatbot, sessions_state], # outputs
    )

    # send file
    send.click(
        _user_submit,
        [msg, chatbot, sessions_state, current_session_id],
        [msg, chatbot, sessions_state],
    ).then(
        chat_with_ollama_streaming_history,
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
            sessions[session_id] = []
        return [],sessions
    
    clear.click(clear_session, [current_session_id, sessions_state], [chatbot, sessions_state])

if __name__ == "__main__":
    demo.launch(
        allowed_paths=[str(CONVS_STORAGE)],
        max_file_size="20mb",
    )
