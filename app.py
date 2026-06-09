"""
app.py — Streamlit interface for the Horror Game Unofficial RAG Guide.

Features:
- Premium survival-horror dark-themed styling (custom CSS).
- Conversational chat interface using Streamlit's native chat layout.
- Conversational Memory: Contextual query reformulation and history integration.
- Chunking Strategy Comparison: Dynamic UI toggling between recursive and fixed chunking.
"""

import streamlit as st

# Set page config first
st.set_page_config(
    page_title="Horror Guide RAG Archive",
    page_icon="🔦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from retrieve import retrieve, list_games
from generate import generate_answer


# 1. Warm up/Cache the RAG backend resources by collection
@st.cache_resource
def initialize_backend(collection_name: str) -> bool:
    """Ensure retriever models and database files are loaded and cached.

    Loads the database on demand using `@st.cache_resource` to avoid cold-start 
    latency on page refreshes.

    Args:
        collection_name (str): The name of the ChromaDB collection to initialize.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    from retrieve import _load_resources
    try:
        _load_resources(collection_name)
        return True
    except Exception as e:
        st.error(f"Failed to initialize database '{collection_name}': {e}")
        return False


def run_rag_pipeline(
    query: str,
    chat_history: list[dict],
    game_filter: str | None,
    collection_name: str,
) -> dict:
    """Run the core RAG pipeline: reformulate, retrieve, and generate.

    Args:
        query (str): The raw user query.
        chat_history (list[dict]): A list of past chat turn dictionaries.
        game_filter (str | None): Optional game name prefix filter.
        collection_name (str): Collection name to query.

    Returns:
        dict: A dictionary containing the RAG pipeline outputs:
            - standalone_query (str): The reformulated query.
            - answer (str): The grounded answer string.
            - sources (list[dict]): The chunks cited.
            - raw_chunks (list[dict]): Raw retrieved chunks.
    """
    from generate import reformulate_query, generate_answer
    from retrieve import retrieve

    # 1. Contextual Query Reformulation
    standalone_q = reformulate_query(query, chat_history)

    # 2. Retrieve chunks based on standalone query
    chunks = retrieve(
        standalone_q,
        game_filter=game_filter,
        top_k=5,
        collection_name=collection_name
    )

    # 3. Generate Grounded Answer (incorporating original query, chunks, and history)
    answer, sources = generate_answer(query, chunks, chat_history)

    return {
        "standalone_query": standalone_q,
        "answer": answer,
        "sources": sources,
        "raw_chunks": chunks,
    }


# 2. Inject premium custom CSS for survival horror aesthetic
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:ital,wght@0,400;0,700;1,400;1,700&family=Special+Elite&family=VT323&display=swap');
    
    /* CRT subtle pulsing glow */
    @keyframes pulse {
        0% { opacity: 0.94; }
        50% { opacity: 1; }
        100% { opacity: 0.94; }
    }
    
    /* Vignette and CRT grid overlay combined */
    .stApp::before {
        content: " ";
        display: block;
        position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: 
            linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.3) 50%), 
            linear-gradient(90deg, rgba(255, 0, 0, 0.05), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.05));
        z-index: 999;
        background-size: 100% 4px, 6px 100%;
        pointer-events: none;
    }
    
    /* Background and global text color */
    .stApp {
        background-color: #07080a !important;
        color: #8fa088 !important; /* phosphor green / muted military green */
        font-family: 'Courier Prime', monospace !important;
        animation: pulse 6s infinite;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #0c0e12 !important;
        border-right: 2px solid #8b0000 !important;
    }
    
    [data-testid="stSidebar"] h2 {
        font-family: 'VT323', monospace !important;
        font-size: 2.2rem !important;
        text-shadow: 0 0 8px rgba(255, 51, 51, 0.4);
    }
    
    /* Customize text inside sidebar */
    [data-testid="stSidebar"] .stMarkdown {
        font-family: 'Courier Prime', monospace !important;
        color: #8fa088 !important;
    }
    
    /* Header card styled as a decayed warning console */
    .header-card {
        background-color: #0d0f12;
        border: 2px solid #8b0000;
        border-radius: 4px;
        padding: 30px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 0 20px rgba(139, 0, 0, 0.3);
        position: relative;
    }
    
    .header-title {
        font-family: 'VT323', monospace !important;
        font-size: 3.5rem !important;
        color: #ff3333 !important; /* Crimson Red */
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 0;
        text-shadow: 0 0 12px rgba(255, 51, 51, 0.7);
    }
    
    .header-subtitle {
        font-family: 'VT323', monospace !important;
        font-size: 1.4rem !important;
        color: #ffb000 !important; /* Decayed Amber */
        letter-spacing: 1.5px;
        margin-top: 8px;
        text-shadow: 0 0 6px rgba(255, 176, 0, 0.5);
    }

    /* Custom Chat bubbles or blocks */
    .chat-bubble-user {
        border-left: 3px solid #ffb000 !important;
        background-color: #0d0f12 !important;
        padding: 12px 18px !important;
        border-radius: 4px !important;
        color: #ffb000 !important;
        font-family: 'Courier Prime', monospace !important;
        font-size: 1.05rem;
    }
    .chat-bubble-assistant {
        border-left: 3px solid #ff3333 !important;
        background-color: #0c0e12 !important;
        padding: 15px !important;
        border-radius: 4px !important;
        color: #dcdfdc !important;
        font-family: 'Courier Prime', monospace !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Sources display styling */
    .sources-title {
        font-family: 'VT323', monospace;
        font-size: 1.3rem;
        color: #ffb000;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 15px;
        margin-bottom: 8px;
        border-bottom: 1px solid rgba(255, 176, 0, 0.2);
        padding-bottom: 4px;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #07080a;
    }
    ::-webkit-scrollbar-thumb {
        background: #1b2616;
        border-radius: 2px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #ff3333;
    }
    
    /* Selectbox/Radio styling overrides */
    div[data-baseweb="select"] {
        background-color: #0d0f12 !important;
        border: 1px solid rgba(57, 255, 20, 0.2) !important;
        border-radius: 4px !important;
    }
    div[data-baseweb="select"] div {
        color: #39ff14 !important;
        font-family: 'VT323', monospace !important;
        font-size: 1.1rem !important;
    }
    
    /* Expander styling */
    div[data-testid="stExpander"] {
        background-color: #0d0f12 !important;
        border: 1px solid rgba(255, 51, 51, 0.15) !important;
        border-radius: 4px !important;
        margin-top: 10px;
    }
    
    /* Footer info */
    .footer-info {
        text-align: center;
        margin-top: 50px;
        font-size: 0.85rem;
        color: #495057;
        font-family: 'VT323', monospace;
        letter-spacing: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Sidebar UI - Configuration Control Console
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='font-family: "VT323", monospace; color: #ff3333; letter-spacing: 2px;'>ARCHIVE CONTROL</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Toggle Chunking Strategy Comparison
strategy_label = st.sidebar.radio(
    "Active Chunking Strategy:",
    options=["Recursive Header-Based Chunks", "Fixed-Size Chunks"],
    index=0,
)
collection_name = (
    "horror_guides_recursive" if strategy_label == "Recursive Header-Based Chunks" else "horror_guides_fixed"
)

# Attempt initialization for active collection
backend_ready = initialize_backend(collection_name)

if backend_ready:
    # Load games list dynamically based on chosen collection
    available_games = ["All Games"] + list_games(collection_name)
    selected_game = st.sidebar.selectbox(
        "Focus Archive search to a specific game:",
        options=available_games,
        index=0,
    )
else:
    selected_game = "All Games"
    st.sidebar.warning("SYSTEM OFFLINE: Database unavailable.")

# Sidebar Clear Memory control
st.sidebar.markdown("---")
if st.sidebar.button("Clear Archive Logs"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown(
    """
    ### SYSTEM PROTOCOLS
    
    This terminal enables secure multi-turn retrieval of tactical walkthroughs and collectible archives.
    
    **CORE SPECIFICATIONS:**
    - **VEC.MODEL**: `all-MiniLM-L6-v2`
    - **STORE**: `ChromaDB` (Local Node)
    - **SYNTHESIS**: `llama-3.3-70b` (Groq API)
    - **CHAT REFORM**: `llama-3.1-8b` (Groq API)
    - **CONV.MEMORY**: Active (RRF contextual query reformulation)
    """
)

# 4. Main Panel UI
st.markdown(
    """
    <div class="header-card">
        <h1 class="header-title">🔦 Horror Guide Archive</h1>
        <div class="header-subtitle">Grounded Conversational Walkthrough & Puzzle Intelligence Terminal</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation logs
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    
    with st.chat_message(role):
        if role == "user":
            st.markdown(f'<div class="chat-bubble-user">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-assistant">{content}</div>', unsafe_allow_html=True)
            
            # Show sources for this turn
            sources = msg.get("sources", [])
            if sources:
                with st.expander("Sources Cited"):
                    # Group sources by game title
                    grouped_sources = {}
                    for src in sources:
                        g = src.get("game", "Unknown Game")
                        if g not in grouped_sources:
                            grouped_sources[g] = []
                        doc_key = (src.get("title"), src.get("section_header"))
                        if doc_key not in [(s.get("title"), s.get("section_header")) for s in grouped_sources[g]]:
                            grouped_sources[g].append(src)
                    
                    for game_title, game_srcs in grouped_sources.items():
                        st.markdown(f"**{game_title}**")
                        for s in game_srcs:
                            st.markdown(
                                f"- *{s.get('title', 'Guide')}* — **{s.get('section_header', 'Main')}** `[{s.get('source_file', '')}]`"
                            )
                
                # Show reformulated standalone query if it differs from the original
                standalone = msg.get("standalone_query")
                if standalone and standalone.lower() != msg.get("original_query", "").lower():
                    st.caption(f"Contextual Standalone Query: *\"{standalone}\"*")
                
                # Show raw chunks debug
                raw_chunks = msg.get("raw_chunks", [])
                if raw_chunks:
                    with st.expander("🔍 View Raw Chunks (Developer Debug)"):
                        for idx, chunk in enumerate(raw_chunks, 1):
                            st.markdown(
                                f"**Chunk {idx}** | Game: `{chunk['game']}` | "
                                f"File: `{chunk['source_file']}` (Index: `{chunk['chunk_index']}`) | "
                                f"Distance: `{chunk['distance']}`"
                            )
                            st.code(chunk["text"], language="markdown")
                            st.markdown("---")

if query := st.chat_input("Enter your survival guide question...", key="chat_input"):
    # Render user query bubble
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-bubble-user">{query}</div>', unsafe_allow_html=True)

    # Prepare chat history turns for LLM (strictly text content, no source dicts)
    chat_turns = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("assistant"):
        if not backend_ready:
            st.error("RAG pipeline cannot run because the ChromaDB backend is not initialized.")
        else:
            with st.spinner("Accessing archive files..."):
                try:
                    game_filter = None if selected_game == "All Games" else selected_game
                    
                    # Run RAG Pipeline using the helper function (separation of concerns)
                    result = run_rag_pipeline(
                        query=query,
                        chat_history=chat_turns,
                        game_filter=game_filter,
                        collection_name=collection_name
                    )
                    
                    answer = result["answer"]
                    sources = result["sources"]
                    standalone_q = result["standalone_query"]
                    chunks = result["raw_chunks"]
                    
                    # Render generated answer
                    st.markdown(f'<div class="chat-bubble-assistant">{answer}</div>', unsafe_allow_html=True)
                    
                    # Render sources list
                    if sources:
                        with st.expander("Sources Cited"):
                            grouped_sources = {}
                            for src in sources:
                                g = src.get("game", "Unknown Game")
                                if g not in grouped_sources:
                                     grouped_sources[g] = []
                                doc_key = (src.get("title"), src.get("section_header"))
                                if doc_key not in [(s.get("title"), s.get("section_header")) for s in grouped_sources[g]]:
                                     grouped_sources[g].append(src)
                            
                            for game_title, game_srcs in grouped_sources.items():
                                st.markdown(f"**{game_title}**")
                                for s in game_srcs:
                                    st.markdown(
                                         f"- *{s.get('title', 'Guide')}* — **{s.get('section_header', 'Main')}** `[{s.get('source_file', '')}]`"
                                    )
                        
                        # Render query reformulation caption
                        if standalone_q.lower() != query.lower():
                            st.caption(f"Contextual Standalone Query: *\"{standalone_q}\"*")
                        
                        # Render developer raw chunks debug panel
                        with st.expander("🔍 View Raw Chunks (Developer Debug)"):
                            for idx, chunk in enumerate(chunks, 1):
                                st.markdown(
                                    f"**Chunk {idx}** | Game: `{chunk['game']}` | "
                                    f"File: `{chunk['source_file']}` (Index: `{chunk['chunk_index']}`) | "
                                    f"Distance: `{chunk['distance']}`"
                                )
                                st.code(chunk["text"], language="markdown")
                                st.markdown("---")
                                
                    # Append queries and responses to session state messages
                    st.session_state.messages.append({
                        "role": "user",
                        "content": query,
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "original_query": query,
                        "standalone_query": standalone_q,
                        "raw_chunks": chunks,
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred while running the RAG pipeline: {e}")

# Footer
st.markdown(
    """
    <div class="footer-info">
        SYS.LOC: localhost // ENGINE: sentence-transformers & Groq Llama 3.3 // STATUS: ARMED
    </div>
    """,
    unsafe_allow_html=True,
)
