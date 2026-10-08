"""Session-owned foreground task status; no threads, shared models or hidden jobs."""
from contextlib import contextmanager
import time

import streamlit as st


@contextmanager
def task(key, label):
    """Catch interruption only after the called backend has unwound its cleanup."""
    tasks = st.session_state.setdefault("studio_tasks", {})
    state = {"label": label, "status": "running", "stage": "Starting", "started": time.monotonic()}
    tasks[key] = state
    try:
        yield state
    except BaseException as exc:
        # Streamlit Stop/Rerun are BaseExceptions. Re-raise them, never suppress
        # cancellation or claim success before the backend finally blocks finish.
        from streamlit.runtime.scriptrunner import StopException, RerunException
        state["status"] = "cancelled" if isinstance(exc, (StopException, RerunException)) else "failed"
        state["stage"] = "Interrupted; backend cleanup finished" if state["status"] == "cancelled" else "Did not complete"
        raise
    else:
        if state["status"] == "running":
            state["status"] = "completed"
            state["stage"] = "Finished"
    finally:
        state["elapsed"] = round(time.monotonic() - state["started"], 1)


def stage(key, message):
    state = st.session_state.get("studio_tasks", {}).get(key)
    if state and state["status"] == "running":
        state["stage"] = str(message)[:240]


def failure(key, message):
    state = st.session_state.get("studio_tasks", {}).get(key)
    if state:
        state.update(status="failed", stage=str(message)[:240])


def render(key):
    state = st.session_state.get("studio_tasks", {}).get(key)
    if not state:
        return
    with st.container(border=True):
        st.write(f"**{state['label']} · {state['status'].upper()}**")
        st.caption(state["stage"])
        if "elapsed" in state:
            st.caption(f"{state['elapsed']} s · this browser session")
        if state["status"] in ("failed", "cancelled"):
            st.info("Your draft and last completed result are retained. Correct the input or local readiness issue, then click the same Run / Prepare / Check action to retry. Nothing restarts automatically.")
        elif state["status"] == "running":
            st.caption("Cancel with top-right Stop. A loading or model-forward phase may finish before cleanup; leaving the page requests interruption, not immediate completion.")
