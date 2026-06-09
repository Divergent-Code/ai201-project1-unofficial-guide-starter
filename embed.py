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

COLLECTION_RECURSIVE = "horror_guides_recursive"
COLLECTION_FIXED = "horror_guides_fixed"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
BATCH_SIZE = 100  # number of chunks per upsert call
DOCS_DIR = Path(__file__).parent / "documents"


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def get_collection(client: chromadb.PersistentClient, collection_name: str, rebuild: bool = False):
    """Get or create a ChromaDB collection.

    Args:
        client (chromadb.PersistentClient): Connected ChromaDB persistent client.
        collection_name (str): Name of the collection.
        rebuild (bool, optional): If True, deletes and recreates the collection 
            before use. Defaults to False.

    Returns:
        chromadb.Collection: The ChromaDB Collection object.
    """
    if rebuild:
        try:
            client.delete_collection(collection_name)
            print(f"  Deleted existing collection '{collection_name}'.")
        except Exception:
            pass  # Collection may not exist yet — safe to ignore

    return client.get_or_create_collection(
        name=collection_name,
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
    """Embed all chunks in batches and upsert them into ChromaDB.

    IDs are derived from the source file stem and the chunk index (e.g. 
    "file_00001") and are stable across runs, allowing safe repeated execution 
    (upserting will overwrite matching IDs). Note that lists are not supported 
    as metadata values in ChromaDB, so document tags are omitted.

    Args:
        chunks (list[dict]): List of chunk dictionaries containing text and metadata keys.
        model (SentenceTransformer): Loaded SentenceTransformer model.
        collection (chromadb.Collection): ChromaDB collection object to write to.
    """
    total = len(chunks)
    print(f"\nEmbedding {total} chunks in batches of {BATCH_SIZE} into '{collection.name}'...")

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

    print(f"\n  Done. Collection '{collection.name}' now has {collection.count()} vectors.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(rebuild: bool = False) -> None:
    """Execute the full dual-strategy embedding and ingestion pipeline.

    Loads the SentenceTransformer model, establishes a connection to the persistent 
    ChromaDB client, chunks all files in the documents directory (using both 
    recursive and fixed strategies), and stores them in respective collections.

    Args:
        rebuild (bool, optional): If True, existing collections are wiped and 
            re-created from scratch. Defaults to False.
    """
    print("=" * 60)
    print("  Horror Game RAG — Embedding Pipeline (Dual Strategy)")
    print("=" * 60)

    # Step 1: Load embedding model
    print(f"\n[1/5] Loading embedding model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Step 2: Connect to ChromaDB
    print(f"\n[2/5] Connecting to ChromaDB at '{CHROMA_DIR}'...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Step 3: Embed Recursive Chunks
    print("\n[3/5] Processing RECURSIVE (Header-based) chunks...")
    chunks_rec = chunk_all_documents(DOCS_DIR, "recursive")
    col_rec = get_collection(client, COLLECTION_RECURSIVE, rebuild=rebuild)
    embed_and_store(chunks_rec, model, col_rec)

    # Step 4: Embed Fixed Chunks
    print("\n[4/5] Processing FIXED (Sliding window) chunks...")
    chunks_fix = chunk_all_documents(DOCS_DIR, "fixed")
    col_fix = get_collection(client, COLLECTION_FIXED, rebuild=rebuild)
    embed_and_store(chunks_fix, model, col_fix)

    print("\n" + "=" * 60)
    print("  Embedding complete for both strategies.")
    print("=" * 60)


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    if rebuild:
        print("Warning: --rebuild flag detected. Existing collections will be wiped.\n")
    main(rebuild=rebuild)
