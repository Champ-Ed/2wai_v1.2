import streamlit as st

import os
import asyncio
import uuid
import logging
import sys
import json
import time
import hashlib
from typing import TypedDict, List, Optional, Dict, Any, Callable
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
# from dotenv import load_dotenv
import openai
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, ExactMatchFilter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
import traceback
from llama_index.embeddings.openai import OpenAIEmbedding

# Set up structured logging
def setup_logger(name: str, debug: bool = False) -> logging.Logger:
    """Set up a structured logger with file and console output."""
    logger = logging.getLogger(name)
    
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
        file_handler = logging.FileHandler(logs_dir / f"{name}.log", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # If file logging fails, continue with console only
        pass
    
    return logger
from llama_index.core import Settings
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from typing_extensions import Annotated
import operator
import json

# ---------- Helpers restored ----------
def _read_persona_template(path: Optional[str]) -> str:
    """Read persona prompt template from file if available; otherwise return a default.
    Supports placeholders {CONTEXT} and {SUMMARY}. If not present, we will inject under
    the lines starting with 'Context:' and 'Summary:' respectively.
    """
    try:
        p = Path(path or "calum_prompt_v2.1.yaml")
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return (
        "You are Calum Worthy, a witty activist and actor. Answer as Calum.\n"
        "Context:\n{CONTEXT}\n\nSummary:\n{SUMMARY}\n"
    )

class DiskTidStore:
    """Very small JSON store for per-thread scratchpad persistence across sessions."""
    def __init__(self, base_dir: Optional[str] = None, debug: bool = False):
        self.base = Path(base_dir or "thread_checkpoints")
        self.base.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.logger = setup_logger(f"DiskTidStore", debug)

    def _path(self, tid: str) -> Path:
        return self.base / f"{tid}.json"

    def load(self, tid: str) -> Dict[str, Any]:
        try:
            p = self._path(tid)
            if not p.exists():
                return {}
            data = json.loads(p.read_text(encoding="utf-8"))
            if self.debug:
                self.logger.debug(f"Loaded thread {tid}: keys={list(data.keys())}")
            return data if isinstance(data, dict) else {}
        except Exception as e:
            if self.debug:
                self.logger.error(f"Load error for {tid}: {e}")
            return {}

    def save(self, tid: str, data: Dict[str, Any]):
        try:
            p = self._path(tid)
            p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            if self.debug:
                sp = data.get("scratchpad", [])
                self.logger.debug(f"Saved thread {tid}: scratchpad_len={len(sp) if isinstance(sp, list) else 0}")
        except Exception as e:
            if self.debug:
                self.logger.error(f"Save error for {tid}: {e}")

# Deduping reducer for LangGraph channel: removes consecutive duplicates and clamps length
from typing import Iterable

def _reduce_scratchpad(left: List[str], right: List[str]) -> List[str]:
    l = list(left or [])
    r = list(right or [])
    merged = l + r
    if not merged:
        return []
    
    # Step 1: remove consecutive duplicates
    out: List[str] = []
    for item in merged:
        s = str(item)
        if not out or out[-1] != s:
            out.append(s)
    
    # Step 2: aggressive cleanup for corrupted checkpointer data
    # Remove any duplicate patterns that shouldn't exist
    def remove_duplicate_patterns(seq: List[str]) -> List[str]:
        if len(seq) < 10:  # Only clean if we have substantial duplicates
            return seq
            
        # Look for exact 5-turn conversation patterns that repeat
        # Pattern: User -> Calum -> User -> Calum -> User -> (repeat)
        cleaned = []
        seen_5_turn_blocks = set()
        
        i = 0
        while i < len(seq):
            # Try to extract a 5-turn block starting at position i
            if i + 4 < len(seq):
                block = tuple(seq[i:i+5])
                block_str = "|".join(block)
                
                # If we've seen this exact 5-turn block before, skip it
                if block_str in seen_5_turn_blocks:
                    i += 5  # Skip this duplicate block
                    continue
                else:
                    seen_5_turn_blocks.add(block_str)
            
            # Add this item and move to next
            cleaned.append(seq[i])
            i += 1
            
        return cleaned
    
    # Apply aggressive cleanup only if we detect substantial duplication
    if len(out) > 20:
        out = remove_duplicate_patterns(out)
    
    # Step 3: collapse repeated tail blocks (handles AB AB or ABC ABC patterns)
    def collapse_tail(seq: List[str]) -> List[str]:
        changed = True
        while changed:
            changed = False
            n = len(seq)
            for k in (4, 3, 2):
                if n >= 2 * k and seq[n - 2 * k:n - k] == seq[n - k:n]:
                    seq = seq[: n - k]
                    changed = True
                    break
        return seq
    out = collapse_tail(out)
    
    # Step 4: Clamp window
    MAX_LEN = 200
    if len(out) > MAX_LEN:
        out = out[-MAX_LEN:]
    return out


# ---------- Typed states ----------
class AgentState(TypedDict, total=False):
    user_input: str
    session: Dict[str, Any]
    # Use a LangGraph channel with custom reducer to avoid duplicate growth
    scratchpad: Annotated[List[str], _reduce_scratchpad]
    selected_context: str
    compressed_history: str
    agent_context: str
    response: str

# ---------- Embedding helpers ----------
@lru_cache(maxsize=4096)
def cached_openai_embedding(text: str) -> tuple:
    """Synchronous wrapper (cached) for OpenAI embeddings.
       We return a tuple because lists are unhashable for caching; conversion handled by caller."""
    if not text or len(text) > 8192:
        return tuple()
    resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return tuple(resp.data[0].embedding)

async def get_embedding(text: str) -> List[float]:
    # run the cached call in a threadpool so it doesn't block the event loop
    emb_tuple = await asyncio.to_thread(cached_openai_embedding, text)
    return list(emb_tuple)

# ---------- DeepLake store with batching + async wrappers ----------
class DeepLakePDFStore:
    def __init__(self, path: Optional[str] = None, commit_batch: int = 8, debug: bool = False):
        org_id = st.secrets.get("ACTIVELOOP_ORG_ID", "")
        path = path or f"hub://{org_id}/calum_v10"
        self.dataset_path = path
        self.commit_batch = commit_batch
        self.debug = debug or (os.getenv("DEBUG_CONVO") == "1")
        self.logger = setup_logger("DeepLakePDFStore", self.debug)
        # Simple cache for recent queries
        self._query_cache = {}
        self._cache_max_size = 50
        # Per-event-loop locks to avoid cross-loop issues in Streamlit
        self._locks: Dict[int, asyncio.Lock] = {}
        if self.debug:
            self.logger.debug(f"Init store path={path} batch={commit_batch}")

        # Unified read/write store
        self.vector_store = DeepLakeVectorStore(
            dataset_path=path,
            token=st.secrets.get("ACTIVELOOP_TOKEN", ""),
            read_only=False
        )
        # self.vector_store = DeepLakeVectorStore(
        #     dataset_path=path,
        #     read_only=False
        # )
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

    def _loop_lock(self) -> asyncio.Lock:
        """Return an asyncio.Lock bound to the current running loop."""
        loop = asyncio.get_running_loop()
        lid = id(loop)
        lock = self._locks.get(lid)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[lid] = lock
        return lock

    def debug_nodes(self, nodes: List[TextNode]):
        if not self.debug:
            return
        self.logger.debug(f"inserting {len(nodes)} node(s):")
        for n in nodes:
            txt = n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
            self.logger.debug(f"  - id={getattr(n, 'id_', None)} len={len(txt)} meta={getattr(n, 'metadata', {})}")

    async def add_memory(self, agent: str, text: str):
        if not text:
            return
        if self.debug:
            self.logger.debug(f"add_memory queued text='{text[:60]}...'")
        
        # Chunk text
        chunks = self.chunk_text(text)
        nodes = []
        for chunk in chunks:
            node = TextNode(
                id_=str(uuid.uuid4()),
                text=chunk,
                metadata={
                    "agent": agent,
                    "type": "memory",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            nodes.append(node)
        
        # Debug nodes
        self.debug_nodes(nodes)
        
        # Insert nodes via LlamaIndex in a thread to avoid blocking
        try:
            async with self._loop_lock():
                await asyncio.to_thread(self._sync_insert_nodes, nodes)
            if self.debug:
                self.logger.debug("LlamaIndex insert_nodes done (async)")
        except Exception as e:
            if self.debug:
                self.logger.error(f"LlamaIndex insert_nodes error: {e}")

    def _sync_insert_nodes(self, nodes):
        """Synchronous wrapper for node insertion."""
        self.index.insert_nodes(nodes)

    async def flush(self):
        # No-op: using vector store managed commits
        return

    @traceable(name="rag_query")
    async def rag_query(self, query: str, top_k: int = 5, agent_id_filter: Optional[str] = None) -> List[str]:
        if not query:
            return []
            
        # Check cache first
        cache_key = f"{query[:100]}:{top_k}:{agent_id_filter}"
        if cache_key in self._query_cache:
            if self.debug:
                self.logger.debug(f"rag_query cache hit for: {query[:50]}")
            return self._query_cache[cache_key]
            
        if self.debug:
            self.logger.debug(f"rag_query q='{query}' k={top_k} agent_filter={agent_id_filter}")

        try:
            retriever = self.index.as_retriever(similarity_top_k=top_k)

            agent_value = str(agent_id_filter if agent_id_filter is not None else "1")
            retriever.filters = MetadataFilters(filters=[ExactMatchFilter(key="agent", value=agent_value)])

            def _sync_retrieve(q):
                return retriever.retrieve(q)

            # Guard with per-loop lock
            async with self._loop_lock():
                nodes = await asyncio.wait_for(
                    asyncio.to_thread(_sync_retrieve, query),
                    timeout=2.0  # Increased from 0.8 to 2.0 seconds
                )

            results = [node.get_content() for node in nodes]
            
            # Cache the results
            if len(self._query_cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
            self._query_cache[cache_key] = results

            if self.debug:
                self.logger.debug(f"rag_query fetched {len(nodes)} results in time")

            return results
            
        except asyncio.TimeoutError:
            if self.debug:
                self.logger.warning(f"rag_query timed out for query: {query[:50]}")
            return []  # Return empty results on timeout
        except Exception as e:
            # Retry once on transient DeepLake read errors likely due to recent writes
            if "Unable to read sample" in str(e) or "chunks" in str(e):
                if self.debug:
                    print("[DL] rag_query transient read error, retrying once...")
                try:
                    await asyncio.sleep(0.25)
                    retriever = self.index.as_retriever(similarity_top_k=top_k)
                    agent_value = str(agent_id_filter if agent_id_filter is not None else "1")
                    retriever.filters = MetadataFilters(filters=[ExactMatchFilter(key="agent", value=agent_value)])
                    def _sync_retrieve2(q):
                        return retriever.retrieve(q)
                    async with self._loop_lock():
                        nodes = await asyncio.to_thread(_sync_retrieve2, query)
                    results = [node.get_content() for node in nodes]
                    # Cache
                    if len(self._query_cache) >= self._cache_max_size:
                        oldest_key = next(iter(self._query_cache))
                        del self._query_cache[oldest_key]
                    self._query_cache[cache_key] = results
                    if self.debug:
                        print(f"[DL] rag_query retry fetched {len(nodes)} results")
                    return results
                except Exception as e2:
                    if self.debug:
                        print(f"[DL] rag_query retry error: {e2}")
            if self.debug:
                print(f"[DL] rag_query error: {e}")
            return []

    async def debug_raw_query(self, text: str, k: int = 5, agent="1"):
        q_emb = Settings.embed_model.get_text_embedding(text)
        raw = self.vector_store.query(
            VectorStoreQuery(
                query_embedding=q_emb,
                similarity_top_k=k,
                filters=MetadataFilters(filters=[ExactMatchFilter(key="agent", value=str(agent))])
            )
        )
        print("RAW ids:", getattr(raw, "ids", None))
        print("RAW sims:", getattr(raw, "similarities", None))
        print("RAW metas:", getattr(raw, "metadatas", None))

    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        if not text:
            return []
        chunks = []
        words = text.split()
        current_chunk = []
        for word in words:
            if sum(len(w) + 1 for w in current_chunk) + len(word) + 1 <= chunk_size:
                current_chunk.append(word)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

# ---------- Orchestrator (LangGraph) ----------
class OrchestratedConversationalSystem:
    def __init__(self, session: Dict, agent: str = "1"):
        self.session = session
        # Initialize debug early
        self.debug = session.get("debug", False) or (os.getenv("DEBUG_CONVO") == "1")
        self.logger = setup_logger("OrchestratedConversationalSystem", self.debug)
        
        # Debounce mechanism using session state to persist across Streamlit reruns
        self._debounce_ms = 3000  # 3 seconds - much longer to catch Streamlit reruns
        
        # Initialize session state for debounce if needed
        if hasattr(st, 'session_state'):
            if not hasattr(st.session_state, 'last_input_hash'):
                st.session_state.last_input_hash = None
            if not hasattr(st.session_state, 'last_input_time'):
                st.session_state.last_input_time = 0
        
        # Track execution state to prevent Streamlit rerun duplicates
        self._current_execution = None
        self._last_final_state = None
        
        # Initialize persistent checkpointer (AsyncSqliteSaver) per Option A
        db_cfg = self.session.get("checkpoint_db", st.secrets.get("LANGGRAPH_CHECKPOINT_DB", "checkpoints.sqlite"))
        try:
            # Resolve to absolute file path if a plain filename/path was provided
            if "://" in str(db_cfg):
                # Assume it's already a connection string or URL
                self._db_path = str(db_cfg)
                self._conn_str = self._db_path if self._db_path.startswith("sqlite+aiosqlite://") else self._db_path
            else:
                abs_path = str(Path(db_cfg).expanduser().resolve())
                # Ensure parent directory exists
                Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
                # Build aiosqlite conn string
                self._db_path = abs_path
                self._conn_str = "sqlite+aiosqlite:///" + abs_path.replace("\\", "/")
            # Defer opening the async checkpointer until _ensure_graph
            self.checkpointer = None
            self._checkpointer_cm = None
            self.session["checkpoint_info"] = f"sqlite (async): {self._db_path}"
        except Exception as e:
            # Fallback to in-memory if any path/resolve error
            self.checkpointer = MemorySaver()
            self._checkpointer_cm = None
            self.session["checkpoint_info"] = f"memory-saver fallback (init error: {type(e).__name__})"
        
        # Remove disk JSON persistence in Option A
        # self.disk_store = DiskTidStore(session.get("checkpoint_dir", "thread_checkpoints"), debug=self.debug)
        
        self.agent = agent
        self.store = DeepLakePDFStore(commit_batch=8, debug=self.debug)
        # thresholds (configurable via session)
        self.max_turns = session.get("max_turns", 6)
        self.summarize_after_turns = session.get("summarize_after_turns", 6)
        # Load persona template (path can be overridden via session)
        self.persona_template = _read_persona_template(
            session.get("persona_prompt_path", "calum_prompt_v2.1.yaml")
        )
        # Build the graph but defer compilation until first use
        self._graph = None

    async def _ensure_graph(self) -> None:
        """Ensure graph is compiled with checkpointer properly set up."""
        if self._graph is None:
            # Initialize AsyncSqliteSaver if configured and not yet opened
            if getattr(self, "_checkpointer_cm", None) is None and getattr(self, "_conn_str", None):
                try:
                    self._checkpointer_cm = AsyncSqliteSaver.from_conn_string(self._conn_str)
                    await self._checkpointer_cm.__aenter__()
                    self.checkpointer = self._checkpointer_cm
                    if self.debug:
                        print(f"[CHECKPOINT] Opened AsyncSqliteSaver at {getattr(self, '_db_path', self._conn_str)}")
                except Exception as e:
                    # Fallback to memory saver if SQLite cannot be opened
                    if self.debug:
                        print(f"[CHECKPOINT] AsyncSqliteSaver open failed: {e} -> falling back to MemorySaver")
                    self.checkpointer = MemorySaver()
                    self._checkpointer_cm = None
                    self.session["checkpoint_info"] = f"memory-saver fallback (open error: {type(e).__name__})"
            elif self.checkpointer is None:
                # As a last resort use MemorySaver
                self.checkpointer = MemorySaver()
                self.session["checkpoint_info"] = "memory-saver (no sqlite configured)"
            self._graph = self._build_graph()

    @property
    def graph(self):
        """Get the compiled graph."""
        if self._graph is None:
            raise RuntimeError("Graph not ready. Call await _ensure_graph() first.")
        return self._graph

    def _summarize_history_sync(self, history: List[str]) -> str:
        """Sync wrapper for summarization call (used in to_thread)."""
        if not history:
            return ""
        # Small instructive prompt to summarizer LLM
        prompt = (
            f"You are {self.session.get('persona_name','Calum Worthy')}. "
            f"Summarize this conversation in 1-2 witty sentences:\n{' | '.join(history)}"
        )
        try:
            client = openai.OpenAI(
                api_key=self.session["api_key"],
                base_url=self.session.get("base_url", "https://api.openai.com/v1"),
            )
            # Wrap client for LangSmith tracing
            client = wrap_openai(client)
            resp = client.chat.completions.create(
                model=self.session.get("model_name", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""

    async def _summarize_history(self, history: List[str]) -> str:
        return await asyncio.to_thread(self._summarize_history_sync, history)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        async def write_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] write_context enter")
                print(f"[NODE] Incoming state keys: {list(state.keys())}")
                existing_scratchpad = state.get("scratchpad", [])
                print(f"[NODE] Existing scratchpad length: {len(existing_scratchpad)}")
                if existing_scratchpad:
                    print(f"[NODE] Existing scratchpad: {existing_scratchpad}")
            stt = dict(state)
            user_input = stt.get("user_input")
            if not user_input:
                return stt
            uline = f"User: {user_input}"
            last_list = stt.get("scratchpad", []) or []
            # Suppress duplicate if identical to the last, or already present within a MUCH larger recent window
            recent_window = last_list[-20:] if isinstance(last_list, list) else []  # Look back 20 entries instead of 4
            if (isinstance(last_list, list) and last_list and last_list[-1] == uline) or (uline in recent_window):
                if self.debug:
                    print(f"[NODE] write_context: duplicate user line suppressed (recent window of {len(recent_window)})")
                return stt
            # Emit only the delta so the channel appends to prior history
            stt["scratchpad"] = [uline]
            if self.debug:
                print(f"[NODE] Emitted delta to scratchpad: {stt['scratchpad']}")
                print("[NODE] write_context exit")
            return stt

        def select_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] select_context enter")
            st = dict(state)
            user_input = st.get("user_input", "")
            sp = st.get("scratchpad", []) or []
            # Remove consecutive duplicates defensively here as well
            cleaned: List[str] = []
            for item in sp:
                s = str(item)
                if not cleaned or cleaned[-1] != s:
                    cleaned.append(s)
            scratchpad_entries = cleaned[-8:]
            st["selected_context"] = {
                "memories_query": user_input,
                "recent_turns": scratchpad_entries
            }
            if self.debug:
                print("[NODE] select_context exit")
            return st

        async def resolve_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] resolve_context enter")
            st = dict(state)
            sel = st.get("selected_context", {})
            q = sel.get("memories_query", "")
            
            # Fast path: try to get recent memories from cache or use simplified context
            try:
                # Quick memory retrieval with longer timeout to fix RAG issues
                memories_task = asyncio.create_task(self.store.rag_query(q, top_k=3, agent_id_filter="1"))
                memories = await asyncio.wait_for(memories_task, timeout=3.0)  # Increased to 3 seconds
            except asyncio.TimeoutError:
                if self.debug:
                    print("[NODE] Memory query timed out, using fallback")
                memories = []  # Fallback to empty memories for fast response
                # Continue memory query in background
                asyncio.create_task(self._background_memory_query(q))
            except Exception as e:
                if self.debug:
                    print(f"[NODE] Memory query error: {e}")
                memories = []
            
            recent = sel.get("recent_turns", [])
            st["selected_context"] = f"Memories: {' | '.join(memories)}\nRecent: {' | '.join(recent)}"
            st["top_memory"] = memories[0] if memories else ""
            
            # Add user input as memory immediately (but async to avoid blocking)
            asyncio.create_task(self.store.add_memory(self.agent, f"User: {q}"))
            
            if self.debug:
                print("[NODE] resolve_context exit")
            return st

        async def compress_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] compress_context enter")
            st = dict(state)
            if len(st.get("scratchpad", [])) >= self.summarize_after_turns:
                summary = await self._summarize_history(st.get("scratchpad", []))
                if summary:
                    # Add summary immediately (but async to avoid blocking)
                    asyncio.create_task(self.store.add_memory(self.agent, f"Summary: {summary}"))
                    st["compressed_history"] = summary
            if self.debug:
                print("[NODE] compress_context exit")
            return st

        def isolate_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] isolate_context enter/exit")
            stt = dict(state)
            selected_context = stt.get("selected_context", "")
            compressed_history = stt.get("compressed_history", "")
            # Build system prompt from persona template + injected context
            system_prompt = self._build_system_prompt(selected_context, compressed_history)
            # Add debug print for context
            if self.debug:
                print("[DL] LLM system prompt built.")
            stt["agent_context"] = system_prompt
            return stt

        async def llm_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] llm enter")
            stt = dict(state)
            # Otherwise, call LLM with context
            try:
                messages = [
                    {"role": "system", "content": stt.get("agent_context", "")},
                    {"role": "user", "content": stt.get("user_input", "")}
                ]
                def _call_llm():
                    client = openai.OpenAI(
                        api_key=self.session["api_key"],
                        base_url=self.session.get("base_url", "https://api.openai.com/v1"),
                    )
                    client = wrap_openai(client)
                    return client.chat.completions.create(
                        model=self.session.get("model_name", "gpt-4o"),
                        messages=messages,
                        temperature=self.session.get("temperature", 0.3)
                    )
                resp = await asyncio.to_thread(_call_llm)
                answer = resp.choices[0].message.content.strip()
                aline = f"{self.session.get('persona_name','Calum')}: {answer}"
                last_list = stt.get("scratchpad", []) or []
                recent_window = last_list[-20:] if isinstance(last_list, list) else []  # Look back 20 entries instead of 4
                if (isinstance(last_list, list) and last_list and last_list[-1] == aline) or (aline in recent_window):
                    if self.debug:
                        print(f"[NODE] llm: duplicate assistant line suppressed (recent window of {len(recent_window)})")
                    stt["response"] = answer
                    return stt
                # Emit only the assistant line as delta; channel will append
                stt["scratchpad"] = [aline]
                stt["response"] = answer
                if self.debug:
                    print(f"[NODE] Emitted delta to scratchpad: {stt['scratchpad']}")
                asyncio.create_task(self.store.add_memory(self.agent, aline))
            except Exception as e:
                if self.debug:
                    print("[LLM ERROR]", repr(e))
                    traceback.print_exc()
                stt["response"] = "Oops — something went wrong. Try again?"
            if self.debug:
                print("[NODE] llm exit")
            return stt

        # register nodes (mix sync & async nodes; LangGraph will handle)
        graph.add_node("write_context", write_context_node)
        graph.add_node("select_context", select_context_node)
        graph.add_node("resolve_context", resolve_context_node)
        graph.add_node("compress_context", compress_context_node)
        graph.add_node("isolate_context", isolate_context_node)
        graph.add_node("llm", llm_node)

        # edges
        graph.add_edge("write_context", "select_context")
        graph.add_edge("select_context", "resolve_context")
        graph.add_edge("resolve_context", "compress_context")
        graph.add_edge("compress_context", "isolate_context")
        graph.add_edge("isolate_context", "llm")
        graph.add_edge("llm", END)

        graph.set_entry_point("write_context")
        return graph.compile(checkpointer=self.checkpointer)

    def _extract_scratchpad_from_checkpoint(self, checkpoint: Any) -> List[str]:
        """Safely extract scratchpad list from a MemorySaver checkpoint object/dict."""
        try:
            # Many versions return dict-like with key 'values' -> {'channel_values': {...}}
            if isinstance(checkpoint, dict):
                values = checkpoint.get("values", {})
            else:
                values = getattr(checkpoint, "values", {})
                if not isinstance(values, dict):
                    values = {}
            channel_values = values.get("channel_values", {})
            scratchpad = channel_values.get("scratchpad", [])
            return scratchpad if isinstance(scratchpad, list) else []
        except Exception:
            return []

    @traceable(name="conversation_turn")
    async def run_turn(self, user_input: str, state: Optional[AgentState] = None) -> AgentState:
        if self.debug:
            self.logger.debug(f"user_input='{user_input}'")
        
        # Debounce using session state (persists across Streamlit reruns)
        import hashlib
        import time
        input_hash = hashlib.md5(f"{user_input}_{self.session.get('thread_id', '')}".encode()).hexdigest()
        current_time = time.time() * 1000  # ms
        
        # Get debounce state from session state
        last_input_hash = getattr(st.session_state, 'last_input_hash', None) if hasattr(st, 'session_state') else None
        last_input_time = getattr(st.session_state, 'last_input_time', 0) if hasattr(st, 'session_state') else 0
        
        if (last_input_hash == input_hash and 
            current_time - last_input_time < self._debounce_ms):
            if self.debug:
                self.logger.debug(f"Debounced duplicate input within {self._debounce_ms}ms (session state)")
            # Return a dummy state to avoid processing
            return state or AgentState(
                user_input=user_input,
                response="",
                scratchpad=[],
                selected_context="",
                compressed_history="",
                agent_context=""
            )
        
        # Update session state debounce tracking
        if hasattr(st, 'session_state'):
            st.session_state.last_input_hash = input_hash
            st.session_state.last_input_time = current_time
        
        # Ensure graph is ready with async checkpointer
        await self._ensure_graph()
        
        # Thread id controls which conversation is resumed
        thread_id = self.session.get("thread_id", f"agent-{self.agent}")
        config = {"configurable": {"thread_id": thread_id}}
        
        if self.debug:
            self.logger.debug(f"Using thread_id: {thread_id}")
        
        # ALWAYS pass only the new user input to the graph.
        # Let LangGraph's checkpointer handle state restoration automatically.
        # Manual seeding causes double-loading and duplication.
        input_state: AgentState = {"user_input": user_input}
        
        # Only seed from disk on the very first turn when no checkpoint exists at all
        try:
            existing_state = self.checkpointer.get(config)
            has_checkpoint = bool(existing_state)
            
            # CORRUPTION DETECTION: Check for duplicates or excessive length
            if has_checkpoint and existing_state:
                checkpoint_scratchpad = self._extract_scratchpad_from_checkpoint(existing_state)
                
                # Look for duplicate patterns in recent entries
                has_duplicates = False
                if len(checkpoint_scratchpad) >= 4:  # Need at least 4 entries to check for duplicates
                    # Check for exact duplicates in recent entries
                    recent_entries = checkpoint_scratchpad[-8:] if len(checkpoint_scratchpad) >= 8 else checkpoint_scratchpad
                    seen = set()
                    for entry in recent_entries:
                        if entry in seen:
                            has_duplicates = True
                            break
                        seen.add(entry)
                
                if has_duplicates or len(checkpoint_scratchpad) > 15:  # Lower threshold and duplicate detection
                    if self.debug:
                        self.logger.warning(f"CORRUPTION DETECTED: Checkpoint has {len(checkpoint_scratchpad)} entries with duplicates={has_duplicates}, cleaning...")
                    # Force a clean state by erasing the corrupted checkpoint
                    self.erase_thread(thread_id)
                    has_checkpoint = False
                    if self.debug:
                        self.logger.info(f"Erased corrupted checkpoint, starting fresh")
                        
        except Exception:
            has_checkpoint = False
            
        if not has_checkpoint:
            # This is the first turn for this thread - let it start completely fresh
            # Don't seed from disk to avoid double-loading issues
            if self.debug:
                print("[TURN] No checkpoint found, starting fresh conversation")
        elif self.debug:
            print("[TURN] Checkpoint exists, letting LangGraph restore state")
        
        final_state = await self.graph.ainvoke(input_state, config=config)
        
        if self.debug:
            print(f"[TURN] Final state scratchpad length: {len(final_state.get('scratchpad', []))}")
        
        # Removed disk persistence (Option A uses SQLite checkpointer only)
        
        if self.session.get("force_sync_flush"):
            if self.debug:
                print("[TURN] force sync flush")
            await self.store.flush()
        else:
            asyncio.create_task(self.store.flush())
        self._save_checkpoints_to_session()
        return final_state

    @traceable(name="conversation_turn_fast")
    async def run_turn_fast(self, user_input: str, state: Optional[AgentState] = None) -> AgentState:
        """Fast version that returns response immediately and handles memory operations in background."""
        if self.debug:
            print(f"[TURN_FAST] user_input='{user_input}'")
        
        # Debounce using session state (persists across Streamlit reruns)
        import hashlib
        import time
        input_hash = hashlib.md5(f"{user_input}_{self.session.get('thread_id', '')}".encode()).hexdigest()
        current_time = time.time() * 1000  # ms
        
        # Get debounce state from session state
        last_input_hash = getattr(st.session_state, 'last_input_hash', None) if hasattr(st, 'session_state') else None
        last_input_time = getattr(st.session_state, 'last_input_time', 0) if hasattr(st, 'session_state') else 0
        
        if (last_input_hash == input_hash and 
            current_time - last_input_time < self._debounce_ms):
            if self.debug:
                print(f"[TURN_FAST] Debounced duplicate input within {self._debounce_ms}ms (session state)")
            # Return the last completed state if available
            if self._last_final_state:
                return self._last_final_state
            # Return a dummy state to avoid processing
            return state or AgentState(
                user_input=user_input,
                response="",
                scratchpad=[],
                selected_context="",
                compressed_history="",
                agent_context=""
            )
        
        # STREAMLIT RERUN PROTECTION: Use session state to prevent duplicate executions
        if hasattr(st, 'session_state'):
            # Global execution lock to prevent concurrent graph executions
            global_lock_key = f"executing_graph"
            if getattr(st.session_state, global_lock_key, False):
                if self.debug:
                    print(f"[TURN_FAST] GLOBAL EXECUTION LOCK: Another graph execution in progress, waiting...")
                # Return cached result if available
                cached_result_key = f"cached_result_{input_hash}"
                if hasattr(st.session_state, cached_result_key):
                    if self.debug:
                        print(f"[TURN_FAST] Returning cached result from concurrent execution")
                    return getattr(st.session_state, cached_result_key)
                # Otherwise return dummy state
                return state or AgentState(
                    user_input=user_input,
                    response="Processing your request...",
                    scratchpad=[],
                    selected_context="",
                    compressed_history="",
                    agent_context=""
                )
            
            # Set global execution lock
            setattr(st.session_state, global_lock_key, True)
            
            processing_key = f"processing_{input_hash}"
            if getattr(st.session_state, processing_key, False):
                if self.debug:
                    print(f"[TURN_FAST] Streamlit rerun detected, skipping duplicate execution")
                # Return cached result if available
                cached_result_key = f"cached_result_{input_hash}"
                if hasattr(st.session_state, cached_result_key):
                    return getattr(st.session_state, cached_result_key)
                # Otherwise return dummy state
                return state or AgentState(
                    user_input=user_input,
                    response="",
                    scratchpad=[],
                    selected_context="",
                    compressed_history="",
                    agent_context=""
                )
            # Mark as processing
            setattr(st.session_state, processing_key, True)
        
        # Update session state debounce tracking
        if hasattr(st, 'session_state'):
            st.session_state.last_input_hash = input_hash
            st.session_state.last_input_time = current_time
        
        # Ensure graph is ready with async checkpointer
        await self._ensure_graph()
        
        # Thread id controls which conversation is resumed
        thread_id = self.session.get("thread_id", f"agent-{self.agent}")
        config = {"configurable": {"thread_id": thread_id}}
        
        if self.debug:
            print(f"[TURN_FAST] Using thread_id: {thread_id}")
        
        # ALWAYS pass only the new user input to the graph.
        # Let LangGraph's checkpointer handle state restoration automatically.
        input_state: AgentState = {"user_input": user_input}
        
        # EARLY BAILOUT: Check if this exact user input already exists in checkpoint
        try:
            existing_state = self.checkpointer.get(config)
            has_checkpoint = bool(existing_state)
            
            if has_checkpoint and existing_state:
                checkpoint_scratchpad = self._extract_scratchpad_from_checkpoint(existing_state)
                user_line = f"User: {user_input}"
                
                # Check if this exact user line already exists in recent history (even more recent)
                recent_entries = checkpoint_scratchpad[-6:] if checkpoint_scratchpad else []  # Only last 6 entries
                if user_line in recent_entries:
                    if self.debug:
                        print(f"[TURN_FAST] DUPLICATE USER INPUT DETECTED: '{user_line}' already exists in last 6 entries, skipping graph execution")
                    # Find the corresponding response
                    try:
                        user_idx = None
                        for i, entry in enumerate(checkpoint_scratchpad):
                            if entry == user_line:
                                user_idx = i
                        if user_idx is not None and user_idx + 1 < len(checkpoint_scratchpad):
                            existing_response = checkpoint_scratchpad[user_idx + 1]
                            if existing_response.startswith("Calum:"):
                                response_text = existing_response[6:].strip()  # Remove "Calum: " prefix
                                return AgentState(
                                    user_input=user_input,
                                    response=response_text,
                                    scratchpad=checkpoint_scratchpad,
                                    selected_context="",
                                    compressed_history="",
                                    agent_context=""
                                )
                    except:
                        pass
                    # Fallback response
                    return AgentState(
                        user_input=user_input,
                        response="I think I just answered that - let me know if you'd like me to elaborate!",
                        scratchpad=checkpoint_scratchpad,
                        selected_context="",
                        compressed_history="",
                        agent_context=""
                    )
                
                # CORRUPTION DETECTION: Check for duplicates or excessive length
                # Look for duplicate patterns in recent entries
                has_duplicates = False
                if len(checkpoint_scratchpad) >= 4:  # Need at least 4 entries to check for duplicates
                    # Check for exact duplicates in recent entries
                    recent_entries = checkpoint_scratchpad[-8:] if len(checkpoint_scratchpad) >= 8 else checkpoint_scratchpad
                    seen = set()
                    for entry in recent_entries:
                        if entry in seen:
                            has_duplicates = True
                            break
                        seen.add(entry)
                
                if has_duplicates or len(checkpoint_scratchpad) > 15:  # Lower threshold and duplicate detection
                    if self.debug:
                        print(f"[TURN_FAST] CORRUPTION DETECTED: Checkpoint has {len(checkpoint_scratchpad)} entries with duplicates={has_duplicates}, cleaning...")
                    # Force a clean state by erasing the corrupted checkpoint
                    self.erase_thread(thread_id)
                    has_checkpoint = False
                    if self.debug:
                        print(f"[TURN_FAST] Erased corrupted checkpoint, starting fresh")
                        
        except Exception:
            has_checkpoint = False
            
        if not has_checkpoint:
            # This is the first turn for this thread - let it start completely fresh
            # Don't seed from disk to avoid double-loading issues
            if self.debug:
                print("[TURN_FAST] No checkpoint found, starting fresh conversation")
        elif self.debug:
            print("[TURN_FAST] Checkpoint exists, letting LangGraph restore state")
        
        try:
            final_state = await self.graph.ainvoke(input_state, config=config)
            
            if self.debug:
                print(f"[TURN_FAST] Final state scratchpad length: {len(final_state.get('scratchpad', []))}")
            
            # Removed disk persistence (Option A uses SQLite checkpointer only)
            
            # Handle background operations asynchronously (non-blocking)
            asyncio.create_task(self._background_post_turn_operations())
            self._save_checkpoints_to_session()
            
            return final_state
            
        except Exception as e:
            if self.debug:
                print(f"[TURN_FAST] Error during graph execution: {e}")
            # Return error state
            return AgentState(
                user_input=user_input,
                response="Sorry, something went wrong. Please try again.",
                scratchpad=[],
                selected_context="",
                compressed_history="",
                agent_context=""
            )
        finally:
            # Always clear locks in finally block
            if hasattr(st, 'session_state'):
                processing_key = f"processing_{input_hash}"
                cached_result_key = f"cached_result_{input_hash}"
                global_lock_key = f"executing_graph"
                
                setattr(st.session_state, processing_key, False)
                if 'final_state' in locals():
                    setattr(st.session_state, cached_result_key, final_state)
                setattr(st.session_state, global_lock_key, False)  # Clear global execution lock
                
                if self.debug:
                    print(f"[TURN_FAST] Cleared all processing locks (finally)")
            
            # Cache this result for potential reruns
            if 'final_state' in locals():
                self._last_final_state = final_state
            self._current_execution = None  # Clear execution flag

    async def _background_post_turn_operations(self):
        """Handle background operations after response is returned."""
        try:
            if self.debug:
                print("[BACKGROUND] Running post-turn operations...")
            # Just flush memory operations - the actual memory additions happen during graph execution
            await self.store.flush()
            if self.debug:
                print("[BACKGROUND] Post-turn operations completed")
        except Exception as e:
            if self.debug:
                print(f"[BACKGROUND] Error in post-turn operations: {e}")

    async def _background_memory_query(self, query: str):
        """Run memory query in background for future use."""
        try:
            if self.debug:
                print(f"[BACKGROUND] Running delayed memory query for: {query[:50]}...")
            await self.store.rag_query(query, top_k=5, agent_id_filter="1")
            if self.debug:
                print("[BACKGROUND] Delayed memory query completed")
        except Exception as e:
            if self.debug:
                print(f"[BACKGROUND] Delayed memory query error: {e}")

    def _save_checkpoints_to_session(self):
        """Save checkpoints to Streamlit session state for persistence across reloads."""
        try:
            if hasattr(st, 'session_state') and hasattr(self.checkpointer, 'storage'):
                # Make sure we save a copy of the storage dict
                st.session_state.checkpoints = dict(self.checkpointer.storage)
                if self.debug:
                    print(f"[CHECKPOINT] Saved {len(self.checkpointer.storage)} checkpoints to session")
                    print(f"[CHECKPOINT] Saved checkpoint keys: {list(self.checkpointer.storage.keys())}")
                    
                    # Debug: look at the latest checkpoint content
                    thread_id = self.session.get("thread_id", f"agent-{self.agent}")
                    for key, checkpoint in self.checkpointer.storage.items():
                        key_str = str(key)
                        if thread_id in key_str:
                            try:
                                scratchpad = self._extract_scratchpad_from_checkpoint(checkpoint)
                                print(f"[CHECKPOINT] Checkpoint {key_str} scratchpad length: {len(scratchpad) if scratchpad else 0}")
                                if scratchpad:
                                    print(f"[CHECKPOINT] Checkpoint {key_str} scratchpad: {scratchpad}")
                            except Exception as e:
                                print(f"[CHECKPOINT] Error inspecting checkpoint {key_str}: {e}")
                            break
        except Exception as e:
            if self.debug:
                print(f"[CHECKPOINT] Error saving to session: {e}")

    def _persist_thread_to_disk(self, thread_id: str, state: AgentState):
        """Persist minimal thread state to disk keyed by tid."""
        try:
            scratchpad = state.get("scratchpad", []) or []
            # De-duplicate and clamp before saving
            scratchpad = _reduce_scratchpad(scratchpad, [])
            # Trim further to last max_turns to keep files small
            scratchpad = scratchpad[-self.max_turns:]
            compressed = state.get("compressed_history", "") or ""
            data = {"scratchpad": scratchpad, "compressed_history": compressed}
            self.disk_store.save(thread_id, data)
        except Exception as e:
            if self.debug:
                print(f"[DISK] Persist error for {thread_id}: {e}")

    def _load_thread_from_disk(self, thread_id: str) -> Dict[str, Any]:
        data = self.disk_store.load(thread_id)
        sp = data.get("scratchpad", []) if isinstance(data, dict) else []
        if not isinstance(sp, list):
            sp = []
        # De-duplicate and clamp on load
        sp = _reduce_scratchpad(sp, [])
        ch = data.get("compressed_history", "") if isinstance(data, dict) else ""
        if not isinstance(ch, str):
            ch = ""
        if self.debug and sp:
            print(f"[DISK] Seeded scratchpad for {thread_id}: len={len(sp)}")
        return {"scratchpad": sp, "compressed_history": ch}

    def erase_thread(self, thread_id: str):
        """Erase in-memory state for a thread id. For SQLite, use the Admin tools to delete rows."""
        # Remove from MemorySaver storage (fallback scenario only)
        try:
            if hasattr(self.checkpointer, "storage"):
                to_delete = [k for k in list(self.checkpointer.storage.keys()) if thread_id in str(k)]
                for k in to_delete:
                    del self.checkpointer.storage[k]
        except Exception:
            pass
        # Clear from Streamlit session mirror
        try:
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'checkpoints'):
                st.session_state.checkpoints = {
                    k: v for k, v in st.session_state.checkpoints.items() if thread_id not in str(k)
                }
        except Exception:
            pass
        if self.debug:
            print(f"[ERASE] In-memory state for {thread_id} cleared (SQLite rows unchanged)")

    # CLI convenience
    async def run_cli(self):
        print(f"{self.session.get('persona_name','Calum Worthy')} - (type 'exit' to quit)")
        # Ensure graph is ready with checkpointer
        await self._ensure_graph()
        try:
            while True:
                user_input = await asyncio.to_thread(input, "You: ")
                if user_input.strip().lower() == "exit":
                    await self.store.flush()
                    break
                new_state = await self.run_turn(user_input)
                # Force flush after first turn to ensure tensors exist
                await self.store.flush()
                print(f"{self.session.get('persona_name','Calum')}: {new_state.get('response','')}")
                # No manual state tracking needed; checkpointer persists it.
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Clean up resources, especially the checkpointer context manager."""
        if self._checkpointer_cm is not None and self.checkpointer is not None:
            try:
                await self._checkpointer_cm.__aexit__(None, None, None)
            except Exception as e:
                if self.debug:
                    print(f"[CLEANUP] Error closing checkpointer: {e}")

    def _build_system_prompt(self, selected_context: str, compressed_history: str) -> str:
        """Render the persona template with injected context, summary, and current time."""
        tmpl = self.persona_template
        ctx = selected_context or ""
        summ = compressed_history or ""
        now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        # Inject NOW
        if "{NOW}" in tmpl:
            tmpl = tmpl.replace("{NOW}", now_str)
        else:
            # Prepend a Current time line if not provided as a placeholder
            tmpl = f"Current time: {now_str}\n\n" + tmpl
        # Inject CONTEXT
        if "{CONTEXT}" in tmpl:
            tmpl = tmpl.replace("{CONTEXT}", ctx)
        elif "Context:" in tmpl:
            tmpl = tmpl.replace("Context:", f"Context:\n{ctx}\n")
        else:
            tmpl += f"\n\n[Context]\n{ctx}\n"
        # Inject SUMMARY
        if "{SUMMARY}" in tmpl:
            tmpl = tmpl.replace("{SUMMARY}", summ)
        elif "Summary:" in tmpl:
            tmpl = tmpl.replace("Summary:", f"Summary:\n{summ}\n")
        else:
            tmpl += f"\n\n[Summary]\n{summ}\n"
        return tmpl

# ---------- Optional FastAPI adapter (simple) ----------
# app = FastAPI()
# conversational_system = OrchestratedConversationalSystem(session={
#     "api_key": os.getenv("OPENAI_API_KEY"),
#     "model_name": "gpt-4",
#     "base_url": "https://api.openai.com/v1",
#     "persona_name": "Calum",
#     "avatar_id": "calum",
#     "avatar_prompts": {"calum": "You are Calum Worthy, a witty activist and actor."}
# })
#
# @app.post("/chat")
# async def chat_endpoint(payload: Dict[str, str]):
#     user_input = payload.get("user_input", "")
#     if not user_input:
#         raise HTTPException(status_code=400, detail="user_input required")
#     state = AgentState(
#         session=conversational_system.session,
#         scratchpad=[]
#     )
#     new_state = await conversational_system.run_turn(user_input, state)
#     return {"response": new_state.get("response", "")}
#
# if __name__ == "__main__":
#     uvicorn.run("this_module:app", host="0.0.0.0", port=8000, reload=True)

# ---------- Run CLI if invoked directly ----------


if __name__ == "__main__":
    # Set up LangSmith tracing environment variables first
    from dotenv import load_dotenv
    load_dotenv()  # Load from .env file for CLI mode
    
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "calum-worthy-chatbot")
    
    # Only use st.secrets if we're actually in a Streamlit context
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        activeloop_token = st.secrets.get("ACTIVELOOP_TOKEN", os.getenv("ACTIVELOOP_TOKEN", ""))
        activeloop_org = st.secrets.get("ACTIVELOOP_ORG_ID", os.getenv("ACTIVELOOP_ORG_ID", ""))
        langchain_key = st.secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
    except:
        # Fallback to environment variables when not in Streamlit
        api_key = os.getenv("OPENAI_API_KEY", "")
        activeloop_token = os.getenv("ACTIVELOOP_TOKEN", "")
        activeloop_org = os.getenv("ACTIVELOOP_ORG_ID", "")
        langchain_key = os.getenv("LANGCHAIN_API_KEY", "")
    
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("ACTIVELOOP_TOKEN", activeloop_token)
    os.environ.setdefault("ACTIVELOOP_ORG_ID", activeloop_org)
    os.environ.setdefault("LANGCHAIN_API_KEY", langchain_key)

    session = {
        "api_key": api_key,
        "model_name": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "persona_name": "Calum",
        "avatar_id": "calum",
        "avatar_prompts": {"calum": "You are Calum Worthy, a witty activist and actor."},
        "temperature": 0.3,
        "debug": True,               # turn on verbose debug
        "force_sync_flush": False    # set True to wait every turn
    }
    # Initialize the embedding model BEFORE constructing the system
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=session.get("api_key")
    )
    conv = OrchestratedConversationalSystem(session=session)
    try:
        asyncio.run(conv.run_cli())
    except KeyboardInterrupt:
        asyncio.run(conv.store.flush())
        asyncio.run(conv._cleanup())
        raise









