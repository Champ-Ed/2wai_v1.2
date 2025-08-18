import os
import asyncio
import streamlit as st
import logging
import sys
from pathlib import Path
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import uuid
import aiosqlite

# Set up structured logging for Streamlit app
def setup_streamlit_logger(debug: bool = False) -> logging.Logger:
    """Set up a structured logger for Streamlit app."""
    logger = logging.getLogger("StreamlitApp")
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Create formatter with filename, timestamp, and message
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional, creates logs directory)
    try:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / "streamlit_app.log", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # If file logging fails, continue with console only
        pass
    
    return logger

# Initialize logger early
app_logger = setup_streamlit_logger(debug=os.getenv("DEBUG_CONVO") == "1")

# LangSmith tracing (export env from Streamlit secrets) BEFORE importing convo
os.environ.setdefault("LANGCHAIN_API_KEY", st.secrets.get("LANGCHAIN_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "calum-worthy-chatbot")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# NEW: propagate LANGGRAPH_CHECKPOINT_DB from secrets to env BEFORE resolving paths
_db_secret = st.secrets.get("LANGGRAPH_CHECKPOINT_DB", "")
if _db_secret:
    os.environ["LANGGRAPH_CHECKPOINT_DB"] = _db_secret

from convo import OrchestratedConversationalSystem, AgentState

# Define helpers if not already present (avoid duplicate definitions across edits)
if "_resolve_db_path" not in globals():
    def _resolve_db_path() -> str:
        """Resolve the SQLite file path used by LangGraph AsyncSqliteSaver.
        Prefers st.secrets, then env. Ensures parent dir exists and returns absolute path.
        Falls back to /tmp on read-only filesystems (Streamlit Cloud safe).
        """
        # Prefer secrets, fall back to env, then default
        val = st.secrets.get("LANGGRAPH_CHECKPOINT_DB", os.environ.get("LANGGRAPH_CHECKPOINT_DB", "checkpoints.sqlite"))

        # If user provided a connection string or URL, return as-is
        if isinstance(val, str) and "://" in val:
            return val

        # Build absolute path for plain file paths
        p = Path(val)
        if not p.is_absolute():
            # Try CWD first
            p = Path.cwd() / p
        # Try to create parent dir; if it fails, use /tmp/langgraph
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            return str(p)
        except Exception:
            tmp_dir = Path("/tmp/langgraph")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            return str(tmp_dir / (p.name if p.suffix else (str(p.name) + ".sqlite")))


if "_erase_thread_from_db" not in globals():
    async def _erase_thread_from_db(db_path: str, thread_id: str) -> int:
        """Delete all rows for the given thread_id across tables that have such a column.
        Returns number of tables touched.
        """
        touched = 0
        if "://" in db_path and not db_path.startswith("file:"):
            return touched
        try:
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA busy_timeout=3000;")
                async with conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
                    tables = [row[0] async for row in cur]
                await conn.execute("BEGIN")
                for t in tables:
                    async with conn.execute(f"PRAGMA table_info({t})") as cur2:
                        cols = [row[1] async for row in cur2]
                    if "thread_id" in cols:
                        await conn.execute(f"DELETE FROM {t} WHERE thread_id = ?", (thread_id,))
                        touched += 1
                await conn.commit()
        except Exception:
            return touched
        return touched


if "_inspect_db" not in globals():
    async def _inspect_db(db_path: str) -> dict:
        """Return a mapping of table -> row count to verify writes."""
        info = {}
        try:
            async with aiosqlite.connect(db_path) as conn:
                async with conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
                    tables = [row[0] async for row in cur]
                for t in tables:
                    try:
                        async with conn.execute(f"SELECT COUNT(*) FROM {t}") as cur2:
                            row = await cur2.fetchone()
                            info[t] = row[0] if row else 0
                    except Exception:
                        info[t] = "?"
        except Exception as e:
            info["error"] = str(e)
        return info


# URL-based thread id helper
if "_get_or_set_thread_id" not in globals():
    def _get_or_set_thread_id() -> str:
        """Use URL query param `tid` for per-user conversation continuity using st.query_params."""
        try:
            params = dict(st.query_params)
            tid = params.get("tid")
            if isinstance(tid, list):
                tid = tid[0] if tid else None
            if not tid:
                tid = str(uuid.uuid4())
                st.query_params["tid"] = tid
            return tid
        except Exception:
            return str(uuid.uuid4())


# Resolve thread_id once and keep it stable per user via URL
if "thread_id" not in st.session_state:
    st.session_state.thread_id = _get_or_set_thread_id()
else:
    # Keep URL param and session_state in sync if user pasted a new tid
    try:
        _url_tid = dict(st.query_params).get("tid")
        if isinstance(_url_tid, list):
            _url_tid = _url_tid[0] if _url_tid else None
        if _url_tid and _url_tid != st.session_state.thread_id:
            st.session_state.thread_id = _url_tid
    except Exception:
        pass

# Ensure a consistent absolute DB path and share it with backend/env
_db_path = _resolve_db_path()
os.environ["LANGGRAPH_CHECKPOINT_DB"] = _db_path

# Session configuration
session = {
    "api_key": st.secrets.get("OPENAI_API_KEY", ""),
    "model_name": "gpt-4o",
    "base_url": "https://api.openai.com/v1",
    "persona_name": "Calum",
    "avatar_id": "calum",
    "avatar_prompts": {
        "calum": "You are Calum Worthy, a witty activist and actor."
    },
    "temperature": 0.3,
    "debug": True,  # Enable debug to see checkpoint behavior
    "force_sync_flush": False,
    "thread_id": st.session_state.thread_id,
    "checkpoint_db": _db_path,  # pass explicit path to backend
}

# Initialize the embedding model BEFORE constructing the conversation system
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=session.get("api_key")
)

# --- Init session state ---
if "conv" not in st.session_state:
    app_logger.info("Initializing new OrchestratedConversationalSystem")
    st.session_state.conv = OrchestratedConversationalSystem(session=session)
else:
    # Keep the backend in sync with current thread and db if URL changed
    try:
        st.session_state.conv.session.update({
            "thread_id": st.session_state.thread_id,
            "checkpoint_db": _db_path,
        })
        app_logger.debug(f"Updated conversation session with thread_id: {st.session_state.thread_id}")
    except Exception as e:
        app_logger.warning(f"Failed to update conversation session: {e}")
        pass

if "state" not in st.session_state:
    st.session_state.state = AgentState(
        session=session,
        scratchpad=[],
        selected_context="",
        compressed_history="",
        agent_context="",
        response=""
    )

if "messages" not in st.session_state:
    st.session_state.messages = []  # stores dicts: {"role": "user"/"assistant", "content": "..."}

# --- Title ---
st.title("🎭 Calum Worthy - Your Virtual Best Friend")

# --- Render past messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input box ---
if prompt := st.chat_input("Type your message and press Enter..."):
    # 1. Show user message instantly
    app_logger.debug(f"User input received: '{prompt[:50]}...'")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Calum is thinking..."):
            try:
                # Ensure thread_id is synced before processing
                st.session_state.conv.session["thread_id"] = st.session_state.thread_id
                app_logger.debug(f"Processing conversation with thread_id: {st.session_state.thread_id}")
                new_state = asyncio.run(st.session_state.conv.run_turn_fast(prompt, st.session_state.state))
                reply = new_state.get("response", "")
                app_logger.debug(f"Assistant response generated: '{reply[:50]}...'")
                st.markdown(reply)

                # Save state + assistant message
                st.session_state.state.update(new_state)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                app_logger.error(f"Error during conversation processing: {e}")
                st.error(f"An error occurred: {e}")
                # Add fallback response
                reply = "Sorry, I encountered an error. Please try again."
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

# --- End Chat button ---
col1, col2 = st.columns(2)
with col1:
    if st.button("End Chat"):
        try:
            app_logger.info("End Chat button pressed - clearing conversation")
            asyncio.run(st.session_state.conv.store.flush())
            st.session_state.messages.clear()
            st.session_state.state = AgentState(session=session, scratchpad=[])
            # Generate a new thread for a brand-new conversation and update URL
            old_thread_id = st.session_state.thread_id
            st.session_state.thread_id = str(uuid.uuid4())
            app_logger.info(f"New conversation started: {old_thread_id} -> {st.session_state.thread_id}")
            try:
                st.query_params["tid"] = st.session_state.thread_id
            except Exception as e:
                app_logger.warning(f"Failed to update URL query params: {e}")
            # Also sync backend session to new thread immediately
            try:
                st.session_state.conv.session["thread_id"] = st.session_state.thread_id
            except Exception as e:
                app_logger.warning(f"Failed to sync backend thread_id: {e}")
            st.rerun()
        except Exception as e:
            app_logger.error(f"Error ending chat: {e}")
            st.error(f"Error ending chat: {e}")
with col2:
    with st.popover("Admin"):
        db_path = _db_path
        st.caption(f"DB: {db_path}")
        # Show which checkpointer is active and its connection string
        try:
            _cp_info = getattr(st.session_state.get("conv"), "session", {}).get("checkpoint_info")
            if _cp_info:
                st.caption(f"Checkpoint: {_cp_info}")
        except Exception:
            pass
        try:
            import os as _os
            _exists = _os.path.exists(db_path)
            _size = _os.path.getsize(db_path) if _exists else 0
            st.caption(f"Exists: {_exists} | Size: {_size} bytes")
        except Exception:
            pass
        st.caption(f"Thread: {st.session_state.thread_id}")
        if st.button("Inspect DB tables"):
            info = asyncio.run(_inspect_db(db_path))
            if not info:
                st.write("No tables found.")
            else:
                for k, v in info.items():
                    st.caption(f"{k}: {v}")
        if st.button("Erase current thread from DB"):
            touched = asyncio.run(_erase_thread_from_db(db_path, st.session_state.thread_id))
            st.success(f"Erased thread from {touched} table(s).")
            # Clear UI memory but keep same thread_id (starts fresh with same id)
            st.session_state.messages.clear()
            st.session_state.state = AgentState(session=session, scratchpad=[])
        # New: full erase (MemorySaver + disk JSON)
        if st.button("Erase current thread (all)"):
            try:
                st.session_state.conv.erase_thread(st.session_state.thread_id)
            except Exception:
                pass
            st.session_state.messages.clear()
            st.session_state.state = AgentState(session=session, scratchpad=[])
            st.success("Thread erased from memory and disk.")
        
        # Debug: show current scratchpad length
        try:
            current_sp = st.session_state.state.get("scratchpad", [])
            st.caption(f"Current scratchpad length: {len(current_sp) if current_sp else 0}")
            if current_sp and len(current_sp) > 0:
                st.caption(f"Last 3 entries: {current_sp[-3:] if len(current_sp) >= 3 else current_sp}")
        except Exception:
            pass
