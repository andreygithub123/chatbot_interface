from react_general_workflow import get_app

def _extract_state(snapshot):
    # v0.1+: StateSnapshot dataclass
    if hasattr(snapshot, "values"):
        return snapshot.values
    # some builds returned a dict
    if isinstance(snapshot, dict):
        return snapshot.get("values", snapshot)
    # older builds returned a tuple: (values, metadata)
    if isinstance(snapshot, tuple):
        return snapshot[0]
    # as a last resort, just return it
    return snapshot

def load_latest_state(thread_id: str, ns: str | None = None):
    app = get_app()
    cfg = {"configurable": {"thread_id": thread_id}}
    if ns:
        cfg["configurable"]["checkpoint_ns"] = ns
    snap = app.get_state(cfg)
    state = _extract_state(snap)
    return state

load_latest_state("aa198db5-d6a5-46ea-b1fe-0852722d1dc6")