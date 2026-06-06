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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');
    
    /* Background and global text color */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #12131c;
        border-right: 1px solid rgba(255, 75, 75, 0.15);
    }
    
    /* Header card */
    .header-card {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.05) 0%, rgba(143, 148, 251, 0.03) 100%);
        border: 1px solid rgba(255, 75, 75, 0.15);
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    }
    
    .header-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2.5rem;
        color: #ff4b4b; /* Crimson Red */
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 0;
        text-shadow: 0 0 10px rgba(255, 75, 75, 0.3);
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #8f94fb; /* Lavender Blue */
        letter-spacing: 1.5px;
        margin-top: 8px;
    }
    
    /* Answer Display container */
    .answer-card {
        background-color: #16171f;
        border-left: 5px solid #ff4b4b;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 4px 12px 12px 4px;
        padding: 24px;
        margin-top: 20px;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    
    .answer-header {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.3rem;
        color: #ff4b4b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(255, 75, 75, 0.2);
        padding-bottom: 8px;
    }
    
    .answer-body {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #e5e6eb;
    }
    
    /* Sources display styling */
    .sources-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.3rem;
        color: #8f94fb;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(143, 148, 251, 0.2);
        padding-bottom: 8px;
    }
    
    .game-source-container {
        background-color: #12131a;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .game-source-header {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.1rem;
        color: #4ef083; /* Neon Green */
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(78, 240, 131, 0.15);
        padding-bottom: 6px;
    }
    
    .source-document-link {
        font-size: 0.95rem;
        color: #c5c6c7;
        margin-bottom: 8px;
        padding-left: 12px;
        border-left: 2px solid rgba(143, 148, 251, 0.4);
    }
    
    .source-doc-title {
        font-weight: 600;
        color: #e5e6eb;
    }
    
    .source-doc-section {
        color: #8f94fb;
        font-weight: 500;
    }
    
    .source-doc-file {
        font-family: 'Share Tech Mono', monospace;
        color: #6c757d;
        font-size: 0.8rem;
        margin-left: 5px;
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #ff4b4b 0%, #a82c2c 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 28px !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 1.1rem !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.25) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.45) !important;
        background: linear-gradient(135deg, #ff6b6b 0%, #b83c3c 100%) !important;
    }
    
    /* Footer info */
    .footer-info {
        text-align: center;
        margin-top: 50px;
        font-size: 0.8rem;
        color: #495057;
        font-family: 'Share Tech Mono', monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Sidebar UI
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='font-family: "Share Tech Mono", monospace; color: #ff4b4b; letter-spacing: 1px;'>ARCHIVE CONTROL</h2>
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
    st.sidebar.warning("Database unavailable.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### About the Archive
    This system is a **Retrieval-Augmented Generation (RAG)** pipeline grounded on community survival guides for horror games.
    
    **How it works:**
    1. Your query is embedded using `all-MiniLM-L6-v2`.
    2. The local `ChromaDB` finds the top 5 most relevant passages.
    3. The `llama-3.3-70b-versatile` LLM is prompted via Groq with strict grounding rules to guarantee the response is derived *only* from the guides.
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
