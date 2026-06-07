"""
app.py — Streamlit interface for the Horror Game Unofficial RAG Guide.

Features:
- Premium survival-horror dark-themed styling (custom CSS).
- Sidebar with game filter dropdown (populated dynamically via list_games()).
- Grounded answer display card.
- Separate "Sources Cited" section grouped by game.
- Collapsible "Raw Retrieved Chunks" expander for developer debugging.
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


# 1. Warm up/Cache the RAG backend resources
@st.cache_resource
def initialize_backend():
    """
    Ensure the retriever's models and databases are initialized once
    and cached across page updates.
    """
    from retrieve import _load_resources
    try:
        _load_resources()
        return True
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")
        return False


# Attempt initialization
backend_ready = initialize_backend()

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
    
    /* Answer Display container */
    .answer-card {
        background-color: #0c0e12 !important;
        border: 2px double #ff3333 !important;
        border-radius: 4px !important;
        padding: 24px;
        margin-top: 25px;
        margin-bottom: 30px;
        box-shadow: 0 0 15px rgba(255, 51, 51, 0.15);
        position: relative;
    }
    
    .answer-card::after {
        content: "TERMINAL READOUT // RECORD SECURE";
        position: absolute;
        top: -12px;
        left: 15px;
        background-color: #07080a;
        padding: 0 8px;
        font-family: 'VT323', monospace;
        font-size: 1rem;
        color: #ff3333;
        letter-spacing: 1px;
    }
    
    .answer-header {
        font-family: 'VT323', monospace;
        font-size: 1.5rem;
        color: #ff3333;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(255, 51, 51, 0.2);
        padding-bottom: 8px;
    }
    
    .answer-body {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #dcdfdc !important;
    }
    
    /* Sources display styling */
    .sources-title {
        font-family: 'VT323', monospace;
        font-size: 1.5rem;
        color: #ffb000;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255, 176, 0, 0.3);
        padding-bottom: 8px;
    }
    
    .game-source-container {
        background-color: #0a0b0e;
        border: 1px dashed rgba(255, 176, 0, 0.3);
        border-left: 4px solid #ffb000;
        border-radius: 2px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4);
    }
    
    .game-source-header {
        font-family: 'VT323', monospace;
        font-size: 1.2rem;
        color: #ffb000;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255, 176, 0, 0.15);
        padding-bottom: 6px;
        text-transform: uppercase;
    }
    
    .source-document-link {
        font-size: 0.95rem;
        color: #8fa088;
        margin-bottom: 8px;
        padding-left: 12px;
        border-left: 2px solid rgba(255, 176, 0, 0.3);
    }
    
    .source-doc-title {
        font-weight: 600;
        color: #dcdfdc;
    }
    
    .source-doc-section {
        color: #39ff14; /* Phosphor Green */
        font-weight: 500;
    }
    
    .source-doc-file {
        font-family: 'VT323', monospace;
        color: #556655;
        font-size: 0.9rem;
        margin-left: 5px;
    }
    
    /* Custom buttons */
    .stButton>button {
        background-color: #8b0000 !important;
        background-image: linear-gradient(180deg, #ff3333 0%, #8b0000 100%) !important;
        color: white !important;
        border: 1px solid #ff5555 !important;
        border-radius: 4px !important;
        padding: 8px 24px !important;
        font-family: 'VT323', monospace !important;
        font-size: 1.3rem !important;
        letter-spacing: 1px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(139, 0, 0, 0.4) !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 15px rgba(255, 51, 51, 0.6) !important;
        border-color: #ff9999 !important;
        color: white !important;
    }
    
    /* Style Streamlit Input fields */
    div[data-baseweb="input"] {
        background-color: #0d0f12 !important;
        border: 1px solid rgba(57, 255, 20, 0.2) !important;
        border-radius: 4px !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #39ff14 !important;
        box-shadow: 0 0 10px rgba(57, 255, 20, 0.4) !important;
    }
    div[data-baseweb="input"] input {
        color: #39ff14 !important;
        font-family: 'Courier Prime', monospace !important;
    }
    div[data-baseweb="input"] input::placeholder {
        color: rgba(57, 255, 20, 0.3) !important;
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
    
    /* Selectbox styling */
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
        border: 1px solid rgba(57, 255, 20, 0.15) !important;
        border-radius: 4px !important;
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

# 3. Sidebar UI
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='font-family: "VT323", monospace; color: #ff3333; letter-spacing: 2px;'>ARCHIVE CONTROL</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

if backend_ready:
    # Load games list dynamically
    available_games = ["All Games"] + list_games()
    selected_game = st.sidebar.selectbox(
        "Focus Archive search to a specific game:",
        options=available_games,
        index=0,
    )
else:
    selected_game = "All Games"
    st.sidebar.warning("SYSTEM OFFLINE: Database unavailable.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### SYSTEM PROTOCOLS
    
    This terminal enables secure retrieval of tactical walkthroughs and collectible archives.
    
    **CORE SPECIFICATIONS:**
    - **VEC.MODEL**: `all-MiniLM-L6-v2`
    - **STORE**: `ChromaDB` (Local Node)
    - **SYNTHESIS**: `llama-3.3-70b` (Groq API)
    - **GROUNDING**: STRICT (Grounded query compliance active)
    """
)

# 4. Main Panel UI
st.markdown(
    """
    <div class="header-card">
        <h1 class="header-title">🔦 Horror Guide Archive</h1>
        <div class="header-subtitle">Grounded Survival & Walkthrough Intelligence Terminal</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search Input
query = st.text_input(
    label="Search query:",
    placeholder="e.g., where is the shotgun in Dead Space?",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 6])
with col1:
    search_button = st.button("Query Archive")

if (search_button or query) and query.strip():
    if not backend_ready:
        st.error("RAG pipeline cannot run because the ChromaDB backend is not initialized.")
    else:
        # Determine actual game filter parameter
        game_filter = None if selected_game == "All Games" else selected_game

        with st.spinner("Searching survival logs..."):
            try:
                # 1. Retrieve
                chunks = retrieve(query, game_filter=game_filter, top_k=5)
                
                # 2. Generate
                answer, sources = generate_answer(query, chunks)
                
                # 3. Render Answer Card
                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="answer-header">
                            <span>🔦</span> Survival Guidance
                        </div>
                        <div class="answer-body">
                            {answer}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 4. Render Grouped Sources Section
                if sources:
                    st.markdown('<div class="sources-title">Sources Cited</div>', unsafe_allow_html=True)
                    
                    # Group sources by game title
                    grouped_sources = {}
                    for src in sources:
                        g = src.get("game", "Unknown Game")
                        if g not in grouped_sources:
                            grouped_sources[g] = []
                        
                        # Deduplicate sources sharing title + section header
                        doc_key = (src.get("title"), src.get("section_header"))
                        if doc_key not in [(s.get("title"), s.get("section_header")) for s in grouped_sources[g]]:
                            grouped_sources[g].append(src)
                    
                    # Display grouped sources
                    for game_title, game_srcs in grouped_sources.items():
                        st.markdown(
                            f"""
                            <div class="game-source-container">
                                <div class="game-source-header">{game_title}</div>
                            """,
                            unsafe_allow_html=True,
                        )
                        for s in game_srcs:
                            st.markdown(
                                f"""
                                <div class="source-document-link">
                                    <span class="source-doc-title">{s.get('title', 'Guide')}</span> — 
                                    <span class="source-doc-section">{s.get('section_header', 'Main')}</span> 
                                    <span class="source-doc-file">[{s.get('source_file', '')}]</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        st.markdown("</div>", unsafe_allow_html=True)

                # 5. Developer Debug Panel
                st.markdown("---")
                with st.expander("🔍 View Raw Chunks (Developer Debug)"):
                    for idx, chunk in enumerate(chunks, 1):
                        st.markdown(
                            f"**Chunk {idx}** | Game: `{chunk['game']}` | "
                            f"File: `{chunk['source_file']}` (Index: `{chunk['chunk_index']}`) | "
                            f"Distance: `{chunk['distance']}`"
                        )
                        st.code(chunk["text"], language="markdown")
                        st.markdown("---")

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
