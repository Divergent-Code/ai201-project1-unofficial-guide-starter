"""
retrieve.py — Semantic retrieval from ChromaDB for the Horror Game RAG Guide.

Pipeline stage: User query → Embedding → ChromaDB similarity search → Ranked chunks

Public interface:
    retrieve(query, game_filter, top_k) → list[dict]
    list_games()                         → list[str]

Both are imported by generate.py and app.py. Module-level singletons ensure
the embedding model and ChromaDB collection are loaded only once per process,
avoiding repeated cold-start overhead in the Streamlit app.
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration (must match embed.py exactly)
# ---------------------------------------------------------------------------

COLLECTION_NAME = "horror_guides"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once on first call, reused thereafter.
# Streamlit re-runs the script on every interaction; these persist across
# reruns inside the same process via @st.cache_resource (in app.py).
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
_collection = None
_game_cache: list[str] | None = None  # cached unique game list


def _load_resources() -> None:
    """
    Load the embedding model and ChromaDB collection if not already loaded.
    Raises RuntimeError if the collection doesn't exist (embed.py not yet run).
    """
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found. "
                "Run 'python embed.py' first to build the index."
            ) from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    game_filter: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a given query.

    Game filter logic:
        - If game_filter is None → search the full corpus (all 21 documents).
        - If game_filter is a base game name (e.g. "Alan Wake II (2023)") →
          also match DLC variants (e.g. "Alan Wake II (2023) — DLC Expansions")
          via prefix matching, so the user never has to think about DLC splits.
        - If no game in the collection starts with game_filter → falls back to
          unfiltered search and logs a warning (graceful degradation).

    Args:
        query:       The user's question string.
        game_filter: Optional base game name to restrict retrieval.
        top_k:       Number of results to return (default: 5).

    Returns:
        List of result dicts ordered by relevance (most relevant first):
            text           : str   — full chunk text
            game           : str   — game name from metadata
            title          : str   — document title
            category       : str   — guide category (e.g. "Main Walkthrough")
            section_header : str   — heading that owns this chunk
            source_file    : str   — source .md filename
            chunk_index    : int   — chunk position within the source file
            distance       : float — cosine distance (0 = identical, 2 = opposite)
    """
    _load_resources()

    # Embed the query with the same model used during indexing
    query_embedding = _model.encode([query], show_progress_bar=False).tolist()

    # Build ChromaDB where clause if a game filter is requested
    where_clause = _build_where_clause(game_filter)

    # Query ChromaDB
    query_kwargs: dict = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        query_kwargs["where"] = where_clause

    results = _collection.query(**query_kwargs)

    # Flatten ChromaDB's nested structure into a clean list
    output: list[dict] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        output.append({
            "text": doc,
            "game": meta.get("game", "Unknown"),
            "title": meta.get("title", "Unknown"),
            "category": meta.get("category", "Unknown"),
            "section_header": meta.get("section_header", ""),
            "source_file": meta.get("source_file", ""),
            "chunk_index": int(meta.get("chunk_index", 0)),
            "distance": round(dist, 4),
        })

    return output


def list_games() -> list[str]:
    """
    Return a sorted list of base game names in the collection.

    DLC variants are merged into their parent game name so the dropdown
    in app.py shows one entry per franchise (e.g. "Alan Wake II (2023)"
    covers both the base game and DLC chunks).

    Returns:
        Sorted list of base game name strings.
    """
    _load_resources()
    all_games = _get_all_games()
    # Strip DLC suffixes (anything after " — ") to get base game names
    base_names = sorted({g.split(" \u2014 ")[0] for g in all_games})
    return base_names


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_all_games() -> list[str]:
    """
    Return all unique game name values stored in ChromaDB metadata.
    Result is cached in module-level _game_cache after first call.
    """
    global _game_cache
    if _game_cache is None:
        result = _collection.get(
            limit=_collection.count(),
            include=["metadatas"],
        )
        _game_cache = sorted({
            m["game"] for m in result["metadatas"] if "game" in m
        })
    return _game_cache


def _build_where_clause(game_filter: str | None) -> dict | None:
    """
    Build a ChromaDB 'where' filter dict from a base game name.

    Handles DLC variants by prefix-matching against all known game names.

    Args:
        game_filter: Base game name, or None for unfiltered search.

    Returns:
        ChromaDB where clause dict, or None if no filter should be applied.
    """
    if not game_filter:
        return None

    all_games = _get_all_games()
    matching = [g for g in all_games if g.startswith(game_filter)]

    if not matching:
        print(f"[retrieve] Warning: no game matching '{game_filter}' found. Searching full corpus.")
        return None
    if len(matching) == 1:
        return {"game": {"$eq": matching[0]}}
    # Multiple variants (e.g. base game + DLC) — use $in
    return {"game": {"$in": matching}}


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "where do I find the shotgun"
    game = None  # set to e.g. "Dead Space (2008)" to test filtering

    print(f"Query: '{query}'")
    print(f"Filter: {game or 'None (full corpus)'}\n")

    results = retrieve(query, game_filter=game, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"--- Result {i}  [distance: {r['distance']}] ---")
        print(f"  Game   : {r['game']}")
        print(f"  Title  : {r['title']}")
        print(f"  Header : {r['section_header']}")
        print(f"  Text   : {r['text'][:300]}...")
        print()

    print("Available games in index:")
    for g in list_games():
        print(f"  • {g}")
