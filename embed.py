"""
embed.py — Embed chunks and store in ChromaDB for the Horror Game RAG Guide.

Pipeline stage: Chunked text → Embeddings → ChromaDB vector store
Run once (or whenever documents change) to build/rebuild the index.

Usage:
    python embed.py              # build index (incremental upsert)
    python embed.py --rebuild    # wipe and rebuild collection from scratch
"""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import chunk_all_documents

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLLECTION_NAME = "horror_guides"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
BATCH_SIZE = 100  # number of chunks per upsert call
DOCS_DIR = Path(__file__).parent / "documents"


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def get_collection(client: chromadb.PersistentClient, rebuild: bool = False):
    """
    Get or create the ChromaDB collection.

    Args:
        client:  Connected ChromaDB persistent client.
        rebuild: If True, deletes and recreates the collection before use.

    Returns:
        ChromaDB Collection object.
    """
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass  # Collection may not exist yet — safe to ignore

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )


# ---------------------------------------------------------------------------
# Embedding and upsertion
# ---------------------------------------------------------------------------

def embed_and_store(
    chunks: list[dict],
    model: SentenceTransformer,
    collection,
) -> None:
    """
    Embed all chunks in batches and upsert into the ChromaDB collection.

    IDs are derived from source_file stem + chunk_index and are stable
    across runs, so repeated calls are safe (upsert overwrites duplicates).

    ChromaDB metadata values must be str, int, float, or bool — lists are
    not allowed, so the 'tags' field from sidecar JSON is intentionally omitted.

    Args:
        chunks:     List of chunk dicts from chunk_all_documents().
        model:      Loaded SentenceTransformer model.
        collection: ChromaDB collection to upsert into.
    """
    total = len(chunks)
    print(f"\nEmbedding {total} chunks in batches of {BATCH_SIZE}...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        # Stable, unique IDs: "<file-stem>_<zero-padded-index>"
        ids = [
            f"{Path(c['source_file']).stem}_{c['chunk_index']:05d}"
            for c in batch
        ]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        metadatas = [
            {
                "game": c["game"],
                "title": c["title"],
                "category": c["category"],
                "section_header": c["section_header"],
                "source_file": c["source_file"],
                "chunk_index": int(c["chunk_index"]),
            }
            for c in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        progress = min(batch_start + BATCH_SIZE, total)
        print(f"  Upserted {progress}/{total} chunks...", end="\r")

    print(f"\n  Done. Collection '{COLLECTION_NAME}' now has {collection.count()} vectors.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(rebuild: bool = False) -> None:
    print("=" * 60)
    print("  Horror Game RAG — Embedding Pipeline")
    print("=" * 60)

    # Step 1: Ingest and chunk
    print("\n[1/4] Ingesting and chunking documents...")
    chunks = chunk_all_documents(DOCS_DIR)

    # Step 2: Load embedding model
    print(f"\n[2/4] Loading embedding model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Step 3: Connect to ChromaDB
    print(f"\n[3/4] Connecting to ChromaDB at '{CHROMA_DIR}'...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = get_collection(client, rebuild=rebuild)
    print(f"  Collection '{COLLECTION_NAME}' — {collection.count()} existing vectors.")

    # Step 4: Embed and store
    print("\n[4/4] Embedding and storing chunks...")
    embed_and_store(chunks, model, collection)

    print("\n" + "=" * 60)
    print("  Embedding complete. Run app.py to start the guide.")
    print("=" * 60)


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    if rebuild:
        print("Warning: --rebuild flag detected. Existing collection will be wiped.\n")
    main(rebuild=rebuild)
