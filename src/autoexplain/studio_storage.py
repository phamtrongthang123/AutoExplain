"""Explicit, bounded local experiment records. No writes on import or browse."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid

MAX_RECORD_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 50
KINDS = ("gemma", "text", "image", "nla")


def directory():
    base = Path(os.environ.get("XDG_DATA_HOME", ""))
    if not base.is_absolute():
        base = Path.home() / ".local" / "share"
    return base / "autoexplain" / "studio"


@contextmanager
def _db(*, create=False):
    root = directory()
    path = root / "experiments.sqlite3"
    if root.is_symlink() or path.is_symlink():
        raise ValueError("Refusing a symlinked experiment store")
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Create with private permissions before SQLite opens the fixed path.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600) if not path.exists() else None
        if fd is not None:
            os.close(fd)
    elif not path.is_file():
        yield None
        return
    connection = sqlite3.connect(path, timeout=5)
    try:
        if create:
            connection.execute("PRAGMA max_page_count=18432")
            connection.execute("CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, created TEXT NOT NULL, payload TEXT NOT NULL)")
        yield connection
    finally:
        connection.close()


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("Choose a listed experiment")
    return value


def _validate(value):
    if not isinstance(value, dict) or set(value) != {"version", "kind", "record"} or value["version"] != 1:
        raise ValueError("Unsupported experiment schema")
    if value["kind"] not in KINDS or not isinstance(value["record"], dict):
        raise ValueError("Unsupported experiment type")
    # The store contains JSON only, never pickle, Python objects or executable paths.
    def bounded(item, depth=0):
        if depth > 12:
            raise ValueError("Experiment nesting exceeds budget")
        if isinstance(item, dict):
            if len(item) > 128 or any(not isinstance(key, str) or len(key) > 128 for key in item):
                raise ValueError("Experiment fields exceed budget")
            for child in item.values():
                bounded(child, depth + 1)
        elif isinstance(item, list):
            if len(item) > 4096:
                raise ValueError("Experiment array exceeds budget")
            for child in item:
                bounded(child, depth + 1)
        elif isinstance(item, str) and len(item) > 16000:
            raise ValueError("Experiment text exceeds budget")
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError("Experiment must contain JSON values only")
    bounded(value)
    return value


def save(name, kind, record):
    value = _validate({"version": 1, "kind": kind, "record": record})
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    size = len(payload.encode("utf-8"))
    if size > MAX_RECORD_BYTES:
        raise ValueError("Record exceeds 2 MiB; export it instead")
    identity = uuid.uuid4().hex
    with _db(create=True) as db, db:
        db.execute("BEGIN IMMEDIATE")
        count, used = db.execute("SELECT COUNT(*), COALESCE(SUM(length(CAST(payload AS BLOB))), 0) FROM experiments").fetchone()
        if count >= MAX_RECORDS or used + size > MAX_TOTAL_BYTES:
            raise ValueError("Store is full (50 records / 64 MiB). Delete a selected record or export instead")
        db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?)",
                   (identity, name.strip()[:80] or "Untitled experiment", kind,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"), payload))
    return identity


def listing():
    with _db() as db:
        if db is None:
            return []
        return [{"id": row[0], "name": row[1], "kind": row[2], "created": row[3]}
                for row in db.execute("SELECT id, name, kind, created FROM experiments ORDER BY created DESC, id LIMIT 50")]


def open_record(identity):
    with _db() as db:
        row = db.execute("SELECT payload FROM experiments WHERE id=?", (_id(identity),)).fetchone() if db else None
    if row is None or len(row[0].encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("Experiment is missing or exceeds the size limit")
    return _validate(json.loads(row[0], parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite record"))))


def delete(identity):
    with _db() as db:
        if db is not None:
            with db:
                db.execute("DELETE FROM experiments WHERE id=?", (_id(identity),))


def save_control(kind, record, *, key, default_name):
    import streamlit as st
    with st.expander("Save this experiment locally"):
        st.caption("Only Save writes to disk. Includes input text, results, model and settings provenance. Stored unencrypted on this server; shared by users of this OS account/app. Do not save secrets. Original images, credentials and checkpoint paths are excluded. Unsaved use stays transient.")
        name = st.text_input("Experiment name", value=default_name, max_chars=80, key=key + "_name")
        if st.button("Save experiment", key=key + "_save"):
            try:
                save(name, kind, record)
                st.success("Saved locally. Open it from Saved experiments; it survives server restart.")
            except (ValueError, OSError, sqlite3.Error):
                st.error("Could not save: check local storage access and the 2 MiB per-record / 50 records / 64 MiB limits. No automatic overwrite or deletion.")


def render():
    import streamlit as st
    st.title("Saved experiments")
    st.caption("Explicit local records · survive restart · no automatic prompt persistence")
    st.info("Records are unencrypted on this server and shared by users of this OS account/app. Open does not load models or execute inference. Delete removes only the selected app-owned record; it is not a secure disk erase.")
    with st.expander("Storage & privacy details"):
        st.code(str(directory()), language=None)
        st.caption("50 records · 2 MiB each · 64 MiB total JSON. UUID identifiers, parameterized SQLite queries; names never become paths. No checkpoint paths, credentials or original image bytes. Download an export before deleting if you need a backup.")
    try:
        records = listing()
        if not records:
            st.info("No saved experiments yet. Open a recorded pipeline or complete a live run, then choose Save this experiment locally.")
            return
        by_id = {row["id"]: row for row in records}
        identity = st.selectbox("Local experiment", list(by_id),
                                format_func=lambda key: f"{by_id[key]['name']} · {by_id[key]['kind']} · {by_id[key]['created']}")
        left, right = st.columns(2)
        if left.button("Open experiment", type="primary"):
            st.session_state.studio_opened = {"id": identity, **open_record(identity)}
        confirmed = right.checkbox("Confirm deletion of selected experiment", key="delete_confirm_" + identity)
        if right.button("Delete selected experiment", disabled=not confirmed):
            delete(identity)
            if st.session_state.get("studio_opened", {}).get("id") == identity:
                st.session_state.pop("studio_opened", None)
            st.rerun()
        opened = st.session_state.get("studio_opened")
        if not opened:
            return
        st.subheader("Opened experiment · " + by_id.get(opened["id"], {}).get("name", "saved record"))
        record = opened["record"]
        if opened["kind"] == "gemma":
            from .studio_examples import _pipeline
            st.caption(record.get("origin", "recorded evidence"))
            _pipeline(st, record, key="opened_" + opened["id"], allow_save=False)
        elif opened["kind"] == "text":
            from .studio_views import text_result
            text_result(record)
        elif opened["kind"] == "image":
            st.write("Fixed candidate:", record["candidates"][record["target_index"]])
            st.caption("Image bytes were not saved. Original input is identified by SHA-256; signed pixel attribution and scores remain in the numerical record.")
        else:
            st.caption("NLA input, language edits and numerical results; no live tensors or checkpoint paths.")
        with st.expander("Input, results, model & settings provenance"):
            st.json(record)
        st.download_button("Export opened experiment JSON", json.dumps(opened, indent=2, allow_nan=False),
                           file_name="autoexplain-experiment.json", mime="application/json")
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError):
        st.error("This local record/store cannot be opened safely. Check storage access or use an existing export; no model or code was executed.")
