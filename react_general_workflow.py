from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent,InjectedState
from langchain_core.runnables import RunnableConfig
from typing import Optional,TypedDict, List,Dict,Any
from typing_extensions import Annotated,NotRequired
from langchain_core.messages import AnyMessage,ToolMessage,AIMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command,interrupt
from langgraph.graph import  StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool,InjectedToolCallId
from pathlib import Path
import pandas as pd
import json

FILE_PATH = "src/data/HidroPrahova - Inventar_Generalist_Organizatie.xlsx"
#MODEL_NAME = "PetrosStav/gemma3-tools:12b"
MODEL_NAME="gpt-oss:20b"
BASE_URL =  "http://localhost:11434"
llm = ChatOllama(model=MODEL_NAME,
                 temperature=0.2,
                 base_url=BASE_URL)

# fileds class

class GenericCompanyDetailsFields(TypedDict,total=False):
    activity_overview: Optional[str]
    has_multiple_legal_entitites: Optional[str]
    is_international_subsidary: Optional[str]
    group_imposed_tech: Optional[str]
    essential_services: Optional[str]
    essential_services_dept_count: Optional[str]
    it_ics_architecture_diagram: Optional[str]  # link, upload file, "nu exista"
    business_process_map_defined: Optional[str]
    risk_mgmt_methodology: Optional[str]  # name / standard or "NU"
    org_chart_defined: Optional[str]
    it_ics_procedures_degined: Optional[str]

class GenericCompanyDetailsState(TypedDict):
    messages: Annotated[List[AnyMessage],add_messages] # chat history
    fields: GenericCompanyDetailsFields
    uploaded_excel: Optional[bool]
    excel_file_path: Optional[str]
    qa_dict: Optional[Dict[str,str]]
    #last_human_answer: Optional[str]
    remaining_steps: Optional[int]
    done: NotRequired[bool]

# helper excel functions 

def get_sheet_names(file_path = FILE_PATH):
    """ 
    Takes in the file_path for the .xlsx file
    Returns a list of strings where each string is the name
    of a sheet.
    """
    xl_file = pd.ExcelFile(file_path)
    sheet_list = xl_file.sheet_names 
    if len(sheet_list) == 0 :
        raise ValueError ("The provided excel files has no sheets.")
    return sheet_list


def get_xlsx_data_by_sheet(file_path,sheet_name):
    """
    Takes in the file path and the wanted sheet name for the .xlsx file
    Returns a DataFrame object with the wanted data.
    """
    data = pd.read_excel(io=file_path,sheet_name=sheet_name)
    return data



def format_data_from_xlsx(file_path,sheet_name):
    """
    Takes in the file path and the wanted sheet name for the .xlsx file.
    Returns a dictionary object with the wanted data.
    Example: {Question:Answer,Question_2:Answer_2,etc}
    """
    df = pd.read_excel(io=file_path,
                       sheet_name=sheet_name)
    columns = df.columns.to_list()
    q_col = columns[0]
    a_col = columns[1]
    qa_df = df[[q_col,a_col]].copy()
    qa_df[a_col] = qa_df[a_col].fillna("").astype(str).str.strip()
    qa_dict = qa_df.set_index(q_col)[a_col].to_dict()
    return qa_dict


# helper ToolMessage function
from langchain_core.messages import BaseMessage

def _json_safe(obj):
    """Recursively coerce LangChain messages and other non-serializable
    objects into JSON-serializable structures."""
    # primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # dicts
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    # lists/tuples
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    # LangChain messages
    if isinstance(obj, BaseMessage):
        # BaseMessage has .type (e.g. "human"/"ai"/"tool") and .content
        return {
            "message_type": obj.__class__.__name__,
            "role": getattr(obj, "type", None),
            "name": getattr(obj, "name", None),
            "tool_call_id": getattr(obj, "tool_call_id", None),
            "content": obj.content,
        }
    # anything else -> string fallback
    try:
        json.dumps(obj)  # will succeed for many simple types
        return obj
    except TypeError:
        return str(obj)

def _pack(payload):
    return json.dumps(_json_safe(payload), ensure_ascii=False, indent=2)
    

# tools 
@tool
def list_sheets(
    file_path: Annotated[Optional[str],InjectedState("excel_file_path")] = None,
    tool_call_id: Annotated[Optional[str],InjectedToolCallId] = "",
) -> str:
    """ List Excel sheet names for the current file_path in state."""

    if not file_path:
        msg = ToolMessage(
            content=_pack({"status":"error","error":"No excel_file_path in state."}),
            tool_call_id=tool_call_id,
            name="list_sheets"
        )

    # NOTE: replace if storing files in database    
    p = Path(file_path)
    if not p.exists():
        msg = ToolMessage(
            content=_pack({"status": "error", "error": f"Path not found: {file_path}"}),
            tool_call_id=tool_call_id,
            name="list_sheets",
        )
        return Command(update={"messages": [msg]})
    
    try:
        names = get_sheet_names(file_path=str(p))
        msg = ToolMessage(
            content=_pack({"status":"ok","sheets": names}),
            tool_call_id=tool_call_id,
            name="list_sheets"
        )
        return Command(update={"messages":[msg]})
    except Exception as e:
        msg = ToolMessage(
            content=_pack({"status": "error", "error": f"{e.__class__.__name__}: {e}"}),
            tool_call_id=tool_call_id,
            name="list_sheets",
        )
        return Command(update={"messages": [msg]})

@tool
def extract_fields(
    sheet_name: Optional[str] = None,
    file_path: Annotated[Optional[str],InjectedState("excel_file_path")] = None,
    current_fields: Annotated[GenericCompanyDetailsFields,InjectedState("fields")] = {},
    tool_call_id: Annotated[str,InjectedToolCallId] = "",
)-> Command:
    """
    Read the Excel and the sheet and update state.fields where keys match.
    If sheet_name is omitted, the second sheet is used.
    """
    if not file_path:
        return Command(update={"messages": [ToolMessage("Missing excel_file_path in state.", tool_call_id=tool_call_id)]})

    qa_dict = format_data_from_xlsx(file_path=file_path,sheet_name="Chestionar preliminar generic")

    # ToolMessage for agent 
    preview = {k:qa_dict[k] for i, k in enumerate(qa_dict.keys()) if i < 3}
    msg = ToolMessage(
        f"Loaded Q->A from sheet '{sheet_name}' "
        f"({len(qa_dict)} rows). Example entries: {json.dumps(preview, ensure_ascii=False)}",
        tool_call_id=tool_call_id,
        name="extract_fields",
    )

    return Command(update={
        "uploaded_excel":True,
        "excel_file_path":file_path,
        "qa_dict":qa_dict,
        "messages":[msg]
    })


@tool
def associate_fields(
    qa: Annotated[Optional[Dict[str,str]], InjectedState("qa_dict")] = None,
    current_fields: Annotated[GenericCompanyDetailsFields, InjectedState("fields")] = {},
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
)-> Command:
    """
    Agent that maps extracted fields with state fields.
    Retruns a partial dict of fields to update
    """
    if not qa:
        return Command(update={"messages": [ToolMessage("No last_qa in state. Run extract_fields first.", tool_call_id=tool_call_id)]})

    field_keys = list(GenericCompanyDetailsFields.__annotations__.keys())

    SYSTEM_PROMPT = (
        "You are a data-mapping assistant. You will receive a dict of Q→A pairs "
        "(questions in Romanian) and a list of canonical field keys (in English, "
        "snake_case). Map each Q→A to exactly one field key."
        "Use the answer text as the field value. If the answer is empty fill it as 'Nu este specificat'. "
        "Return a single JSON object with only the mapped fields. Do not include commentary."
    )

    USER_PROMPT = f"""
    Q_A_DICT (JSON):

    {json.dumps(qa,ensure_ascii=False)}

    FIELD_KEYS:
    {json.dumps(field_keys,ensure_ascii=False)}

    Return JSON ONLY.
    """

    raw = llm.invoke([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}]).content
    mapped: Dict[str, Any] = {}
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        mapped = json.loads(raw[start:end+1]) if start != -1 and end != -1 else {}
    except Exception:
        mapped = {}

    
    new_fields = dict(current_fields or {})
    for k, v in (mapped or {}).items():
        if k in field_keys:
            new_fields[k] = v

    msg = ToolMessage(
        f"Associated fields updated with: {json.dumps(mapped, ensure_ascii=False)}",
        tool_call_id=tool_call_id,
        name="associate_fields"
    )
    return Command(update={"fields": new_fields, "messages": [msg]})


@tool
def show_fields(
    current_fields: Annotated[GenericCompanyDetailsFields,InjectedState("fields")],
    tool_call_id: Annotated[Optional[str],InjectedToolCallId]="",
)-> Command:
    """Return the current fields as JSON so the agent can talk about them."""

    msg = ToolMessage(
        f"Current fields:{json.dumps(current_fields or {}, ensure_ascii=False, indent = 2)}",
        tool_call_id=tool_call_id,
        name="show_fields"
    )
    return Command(update={"messages":[msg]})

@tool
def show_state(
    state: Annotated[GenericCompanyDetailsState,InjectedState],
    tool_call_id: Annotated[Optional[str],InjectedToolCallId]="",
)-> Command:
    """ Return the  state at any point in time if asked to. Can be used as a progress feedback. """
    fields_dict = state.get("fields") or {}
    all_field_keys = list(GenericCompanyDetailsFields.__annotations__.keys())
    total_fields = len(all_field_keys)
    filled_fields = sum(1 for k in all_field_keys if str(fields_dict.get(k, "")).strip())
    percent = round((filled_fields / total_fields) * 100, 1) if total_fields else 0.0

    # Trim messages for readability (last 5 only), and make them JSON-safe
    msgs = state.get("messages") or []
    recent_msgs = msgs[-5:] if len(msgs) > 5 else msgs
    recent_msgs_safe = [_json_safe(m) for m in recent_msgs]

    snapshot = {
        "status": "ok",
        "progress": {
            "uploaded_excel": bool(state.get("uploaded_excel")),
            "excel_file_path": state.get("excel_file_path"),
            "fields_total": total_fields,
            "fields_filled": filled_fields,
            "percent_complete": percent,
            "remaining_steps": state.get("remaining_steps"),
            "done": state.get("done", False),
        },
        "state": {
            # include everything except the full messages list
            **{k: v for k, v in state.items() if k != "messages"},
            # add a compact, readable version of recent messages
            "messages_tail": recent_msgs_safe,
            "message_count": len(msgs),
        },
    }

    msg = ToolMessage(
        content=_pack(snapshot),
        tool_call_id=tool_call_id,
        name="show_state",
    )
    return Command(update={"messages": [msg]})


@tool
def end_conversation(tool_call_id: Annotated[str, InjectedToolCallId])->Command:
    """ Function used for ending conversation."""
    try:
        msg = ToolMessage(
            f"Ending conversation tool call",
            tool_call_id=tool_call_id,
            name="end_conversation"
        )
        return Command(update={ "done":True,"messages": [msg]})
    except Exception as e:
        msg = ToolMessage(
            content=_pack({"status": "error", "error": f"{e.__class__.__name__}: {e}"}),
            tool_call_id=tool_call_id,
            name="end_conversation",
        )
        return Command(update={"messages": [msg]})


def human_node(state: GenericCompanyDetailsState):
    
    # ask for path
    if not state.get("excel_file_path"):
        payload = interrupt({
            "action": "provide_excel_path",
            "prompt": "Please upload the Excel file or paste an absolute path to it. "
                      "You can also say 'cancel' to chat without uploading.",
            "expected": "Either a string path or {'excel_file_path': '<path>'}",
        })
        if isinstance(payload, str):
            path = payload.strip()
            if path.lower() in {"stop", "cancel", "exit", "quit"}:
                return {"done": True}
        elif isinstance(payload, dict):
            path = payload.get("excel_file_path") or payload.get("path") or ""
        else:
            path = ""

        return {
            "excel_file_path": path,
            "uploaded_excel": bool(path),
            "messages": [{"role": "user", "content": f"{'Use this Excel: ' + path if path else 'Skip upload for now.'}"}],
        }

    # already excel file path case
    payload = interrupt({
        "action": "next_user_message",
        "prompt": "Ask a question or type 'exit' to finish.",
        "expected": "Either a string message or {'message': '<text>'}",
    })
    if isinstance(payload, str):
        text = payload
    elif isinstance(payload, dict):
        text = payload.get("message") or payload.get("text") or ""
    else:
        text = ""
    if text.strip().lower() in {"exit", "quit", "stop"}:
        return {"done": True}
    return {"messages": [{"role": "user", "content": text}]}



REACT_AGENT_PROMPT = (
    "You are a helpful assistant that can read an Excel provided by the human and "
    "fill in the 'fields' of the state whenever possible.\n"
    "- First, if needed, call list_sheets() to inspect the workbook.\n"
    "- Then call extract_fields(sheet_name=...) to load Q→A into state.last_qa.\n"
    "- Next, call associate_fields() to map Q→A to canonical field keys.\n"
    "- Use show_fields() to review current fields and answer questions.\n"
    "- Use show_state() to get the current state and answer questions. Good for when the user wants to know progress. Use the retrieved state fields to generate a human-readable response, not give directly the state attribute as it is. For example : if a state attribute 'favourite_food' = None, after you retrieve the state your response should be something like 'You did not specified yet your favourite food.\n"
    "- Use end_conversation() when the user wants to finish the conversation or after you have the user's approval that field extraction was successfully completed."
    "Keep answers concise unless the human asks for detail."
)

# NOTE: might be a better idea to switch to own iplementation of ReAct agent
react_agent = create_react_agent(
    model = llm,
    tools=[list_sheets, extract_fields, associate_fields, show_fields,end_conversation,show_state],
    prompt = REACT_AGENT_PROMPT,
    state_schema= GenericCompanyDetailsState
)

# mock node for solving state attributes not defined errors 
def init_node(state: GenericCompanyDetailsState):
    return {
        "messages": state.get("messages", []),
        "fields": state.get("fields") or {},
        "qa_dict": state.get("qa_dict") or {},
        "uploaded_excel": state.get("uploaded_excel", False),
        "remaining_steps": state.get("remaining_steps", 10),
    }


# router node
def route_after_agent(state: GenericCompanyDetailsState):
    return "human" if not state.get("done") else END

# build and compile graph
builder = StateGraph(GenericCompanyDetailsState)
builder.add_node("human",human_node)
builder.add_node("agent",react_agent)
builder.add_node("init_node",init_node)

builder.add_edge(START,"init_node")
builder.add_edge("init_node","human")
builder.add_edge("human","agent")
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "human":"human",
        END:END
    }
)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)

def _handoff_node_factory(
        parent_target: Optional[str],
        artifact_key: str = "g_quest",
):
    """
    Factory (closure) function that keeps 'handoff_node' reusable and decoupled.
    Returns a node function that (optionally) jumps back to the parent.
    """
    def g_handoff(state:GenericCompanyDetailsState)->Command:
        """ 
        Handoff function that updates 'artifacts' from the Parent State.  
        It will be compiled as the last node in 'generic_subgraph_builder'.
        """
        artifact = {
            "excel_file_path":state.get("excel_file_path"),
            "uploaded_excel":state.get("uploaded_excel"),
            "fields":state.get("fields"),
            "done":True
            # we can add any custom fields here which are not in either subgraph state or parent state.
            # this 'artifact' dict will be passed to 'artifacts' and the key,value pairs will be stored there.
            # "sample_fields":list(fileds.keys()[:5])
        }

        if parent_target:
            # runtime jump to parent + store artifact under the provided key
            return Command(
                update={"artifacts":{artifact_key:artifact}},
                goto=parent_target,
                graph=Command.PARENT
            )
        # else update and END locally; parent should have a static edge from the subgraph
        return Command(update={"artifacts":{artifact_key:artifact}})
    return g_handoff

def make_generic_questionnaire_subgraph(
        *,
        parent_target: Optional[str] = None,
        artifact_key: str ="g_quest",
        private_memory: bool = False,
):
    """
    Build and compile the Generic Questionnaire subgraph.
    - parent_target = None -> no runtime jump; MUST use a static edge g_quest->post_g_quest
    - parent_target = "post_g_quest" -> jump to Command.PARENT
    - private_memory = True -> the subgraph have its own decoupled memory; else it will share memory with parent graph
    """
    g = StateGraph(GenericCompanyDetailsState)

    g.add_node("g_quest_init_node", init_node)
    g.add_node("g_human", human_node)
    g.add_node("g_agent", react_agent)
    g.add_node("g_handoff", _handoff_node_factory(parent_target, artifact_key))

    g.add_edge(START, "g_quest_init_node")
    g.add_edge("g_quest_init_node", "g_human")
    g.add_edge("g_human", "g_agent")
    g.add_conditional_edges(
        "g_agent",
        route_after_agent,
        {"human": "g_human", END: "g_handoff"},
    )
    g.add_edge("g_handoff", END)

    return g.compile(checkpointer=MemorySaver() if private_memory else None)


# studio 
def _studio(config: RunnableConfig | None = None):
    """
    Used for modular testing of THIS graph in Studio.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    g = StateGraph(GenericCompanyDetailsState)
    g.add_node("init_node",init_node)
    g.add_node("human", human_node)
    g.add_node("agent", react_agent)
    
   
    g.add_edge(START,"init_node")
    g.add_edge("init_node","human")
    g.add_edge("human","agent")
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "human":"human",
            END:END
        }
    )

    
    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)


# --- visualization ---
# from src.utils import render_mermaid_cli

# if __name__ == "__main__":
#     # only run when you execute this file directly (won’t run when Studio imports it)
#     g = app.get_graph(xray=1)               # expand subgraphs
#     mermaid_src = g.draw_mermaid()          # get the Mermaid text
#     try:
#         render_mermaid_cli(mermaid_src, "src/draw/react_g_quest_workflow.png")
#         print("Saved PNG to src/draw/react_g_quest_workflow.png")
#     except FileNotFoundError as e:
#         print(e)
