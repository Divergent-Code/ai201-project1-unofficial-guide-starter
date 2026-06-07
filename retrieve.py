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

import math
import re
from collections import Counter
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
# BM25 Keyword Search Scorer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Tokenize text by converting to lowercase and extracting words."""
    return re.findall(r"\w+", text.lower())


class BM25Scorer:
    """Pure-Python BM25 scorer for tf-idf style document search."""
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        
        tokenized_docs = [tokenize(doc) for doc in documents]
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avg_doc_len = sum(self.doc_lens) / self.N if self.N > 0 else 1.0
        
        self.doc_term_freqs = [Counter(doc) for doc in tokenized_docs]
        
        doc_freqs = Counter()
        for doc in tokenized_docs:
            doc_freqs.update(set(doc))
        
        self.idfs = {}
        for term, df in doc_freqs.items():
            self.idfs[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_terms: list[str], doc_idx: int) -> float:
        tf_counter = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        score = 0.0
        
        for term in query_terms:
            if term not in self.idfs:
                continue
            tf = tf_counter.get(term, 0)
            if tf == 0:
                continue
            
            idf = self.idfs[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += idf * (numerator / denominator)
            
        return score


# ---------------------------------------------------------------------------
# Module-level singletons — loaded once on first call, reused thereafter.
# Streamlit re-runs the script on every interaction; these persist across
# reruns inside the same process via @st.cache_resource (in app.py).
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
_collections: dict = {}      # collection_name -> Collection
_game_caches: dict = {}      # collection_name -> list[str]
_bm25_scorers: dict = {}     # collection_name -> BM25Scorer
_all_chunks: dict = {}       # collection_name -> list[dict]


def _load_resources(collection_name: str = "horror_guides_recursive") -> None:
    """
    Load the embedding model, ChromaDB collection, BM25 Scorer, and chunks list if not already loaded.
    Raises RuntimeError if the collection doesn't exist (embed.py not yet run).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if collection_name not in _collections:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            _collections[collection_name] = client.get_collection(collection_name)
        except Exception as e:
            raise RuntimeError(
                f"ChromaDB collection '{collection_name}' not found. "
                "Run 'python embed.py' first to build the index."
            ) from e
            
    if collection_name not in _bm25_scorers or collection_name not in _all_chunks:
        col = _collections[collection_name]
        # Fetch all chunks to build lexical index
        all_data = col.get(include=["documents", "metadatas"])
        chunks_list = []
        doc_texts = []
        if all_data and all_data.get("ids"):
            for i in range(len(all_data["ids"])):
                chunk_id = all_data["ids"][i]
                doc = all_data["documents"][i]
                meta = all_data["metadatas"][i]
                chunks_list.append({
                    "id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "bm25_index": i
                })
                doc_texts.append(doc)
            
            _all_chunks[collection_name] = chunks_list
            if doc_texts:
                _bm25_scorers[collection_name] = BM25Scorer(doc_texts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    game_filter: str | None = None,
    top_k: int = 5,
    collection_name: str = "horror_guides_recursive",
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a given query using a hybrid
    vector (semantic) and BM25 (lexical) search merged with Reciprocal Rank Fusion (RRF).

    Game filter logic:
        - If game_filter is None → search the full corpus (all 21 documents).
        - If game_filter is a base game name (e.g. "Alan Wake II (2023)") →
          also match DLC variants (e.g. "Alan Wake II (2023) — DLC Expansions")
          via prefix matching, so the user never has to think about DLC splits.
        - If no game in the collection starts with game_filter → falls back to
          unfiltered search and logs a warning (graceful degradation).

    Args:
        query:           The user's question string.
        game_filter:     Optional base game name to restrict retrieval.
        top_k:           Number of results to return (default: 5).
        collection_name: Name of the ChromaDB collection to retrieve from.

    Returns:
        List of result dicts ordered by RRF relevance (most relevant first):
            text           : str   — full chunk text
            game           : str   — game name from metadata
            title          : str   — document title
            category       : str   — guide category (e.g. "Main Walkthrough")
            section_header : str   — heading that owns this chunk
            source_file    : str   — source .md filename
            chunk_index    : int   — chunk position within the source file
            distance       : float — cosine distance (0 = identical, 2 = opposite) or placeholder
    """
    _load_resources(collection_name)

    bm25_scorer = _bm25_scorers.get(collection_name)
    all_chunks = _all_chunks.get(collection_name)
    collection = _collections[collection_name]

    if not bm25_scorer or not all_chunks:
        # Fallback to pure vector search if lexical index is empty
        return _retrieve_vector_only(query, game_filter, top_k, collection_name)

    # 1. Run Vector Search (get top 50 candidates matching filter)
    query_embedding = _model.encode([query], show_progress_bar=False).tolist()
    where_clause = _build_where_clause(game_filter, collection_name)

    query_kwargs: dict = {
        "query_embeddings": query_embedding,
        "n_results": min(50, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        query_kwargs["where"] = where_clause

    results = collection.query(**query_kwargs)
    
    docs = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []
    dists = results["distances"][0] if results.get("distances") else []
    ids = results["ids"][0] if results.get("ids") else []

    vector_candidates = {}
    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), 1):
        vector_candidates[cid] = {
            "rank": rank,
            "doc": doc,
            "meta": meta,
            "distance": round(dist, 4)
        }

    # 2. Run BM25 Search (get top 50 candidates matching filter)
    matching_games = None
    if game_filter:
        all_games = _get_all_games(collection_name)
        matching_games = [g for g in all_games if g.startswith(game_filter)]

    filtered_chunks = all_chunks
    if matching_games:
        filtered_chunks = [c for c in all_chunks if c["metadata"].get("game") in matching_games]

    query_terms = tokenize(query)
    scored_chunks = []
    for c in filtered_chunks:
        score = bm25_scorer.score(query_terms, c["bm25_index"])
        if score > 0:
            scored_chunks.append((score, c))

    # Sort descending by BM25 score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    keyword_candidates = {}
    for rank, (score, c) in enumerate(scored_chunks[:50], 1):
        keyword_candidates[c["id"]] = {
            "rank": rank,
            "doc": c["text"],
            "meta": c["metadata"]
        }

    # 3. Reciprocal Rank Fusion (RRF) with constant k=60
    union_ids = set(vector_candidates.keys()).union(set(keyword_candidates.keys()))

    rrf_scores = []
    for cid in union_ids:
        v_rank = vector_candidates[cid]["rank"] if cid in vector_candidates else None
        k_rank = keyword_candidates[cid]["rank"] if cid in keyword_candidates else None

        v_score = 1.0 / (60.0 + v_rank) if v_rank is not None else 0.0
        k_score = 1.0 / (60.0 + k_rank) if k_rank is not None else 0.0
        rrf_score = v_score + k_score

        # Retrieve doc details
        if cid in vector_candidates:
            doc = vector_candidates[cid]["doc"]
            meta = vector_candidates[cid]["meta"]
            distance = vector_candidates[cid]["distance"]
        else:
            doc = keyword_candidates[cid]["doc"]
            meta = keyword_candidates[cid]["meta"]
            distance = 0.5  # placeholder for keyword-only retrieval

        rrf_scores.append({
            "rrf_score": rrf_score,
            "id": cid,
            "doc": doc,
            "meta": meta,
            "distance": distance
        })

    # Sort candidates by RRF score descending
    rrf_scores.sort(key=lambda x: x["rrf_score"], reverse=True)

    # 4. Form output
    output = []
    for item in rrf_scores[:top_k]:
        meta = item["meta"]
        output.append({
            "text": item["doc"],
            "game": meta.get("game", "Unknown"),
            "title": meta.get("title", "Unknown"),
            "category": meta.get("category", "Unknown"),
            "section_header": meta.get("section_header", ""),
            "source_file": meta.get("source_file", ""),
            "chunk_index": int(meta.get("chunk_index", 0)),
            "distance": item["distance"],
        })

    return output


def _retrieve_vector_only(
    query: str,
    game_filter: str | None = None,
    top_k: int = 5,
    collection_name: str = "horror_guides_recursive",
) -> list[dict]:
    """Fallback vector-only retrieval function."""
    query_embedding = _model.encode([query], show_progress_bar=False).tolist()
    where_clause = _build_where_clause(game_filter, collection_name)
    collection = _collections[collection_name]
    query_kwargs: dict = {
        "query_embeddings": query_embedding,
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        query_kwargs["where"] = where_clause
    results = collection.query(**query_kwargs)
    output: list[dict] = []
    docs = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []
    dists = results["distances"][0] if results.get("distances") else []
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


def list_games(collection_name: str = "horror_guides_recursive") -> list[str]:
    """
    Return a sorted list of base game names in the collection.

    DLC variants are merged into their parent game name so the dropdown
    in app.py shows one entry per franchise (e.g. "Alan Wake II (2023)"
    covers both the base game and DLC chunks).

    Returns:
        Sorted list of base game name strings.
    """
    _load_resources(collection_name)
    all_games = _get_all_games(collection_name)
    # Strip DLC suffixes (anything after " — ") to get base game names
    base_names = sorted({g.split(" \u2014 ")[0] for g in all_games})
    return base_names


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_all_games(collection_name: str = "horror_guides_recursive") -> list[str]:
    """
    Return all unique game name values stored in ChromaDB metadata.
    Result is cached in game_caches after first call.
    """
    if collection_name not in _game_caches:
        col = _collections[collection_name]
        result = col.get(
            limit=col.count(),
            include=["metadatas"],
        )
        _game_caches[collection_name] = sorted({
            m["game"] for m in result["metadatas"] if "game" in m
        })
    return _game_caches[collection_name]


def _build_where_clause(game_filter: str | None, collection_name: str = "horror_guides_recursive") -> dict | None:
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

    all_games = _get_all_games(collection_name)
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
