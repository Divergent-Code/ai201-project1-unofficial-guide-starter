"""
ingest.py — Document ingestion and recursive chunking for the Horror Game RAG Guide.

Pipeline stage: Document Ingestion → Chunking
Output: List of chunk dicts ready for embedding.

Each chunk dict contains:
    text           : str  — the chunk text (heading re-prepended for sub-chunks)
    game           : str  — e.g. "Amnesia: The Dark Descent (2010)"
    title          : str  — document title from sidecar JSON
    category       : str  — e.g. "Main Walkthrough", "Collectibles and Upgrades Guide"
    section_header : str  — the ### or #### heading that owns this chunk
    source_file    : str  — basename of the source .md file
    chunk_index    : int  — 0-based index within the source document
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_TARGET = 800      # target chunk size in characters
CHUNK_OVERLAP = 150     # overlap between adjacent chunks in characters
MIN_CHUNK_SIZE = 100    # discard chunks smaller than this (headers-only noise)

# Separator hierarchy for recursive splitting (tried in order)
# Level 1: ### heading  Level 2: #### heading  Level 3: blank line (paragraph)
HEADING_PATTERN = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)


# ---------------------------------------------------------------------------
# Sidecar metadata loader
# ---------------------------------------------------------------------------

def load_metadata(md_path: Path) -> dict:
    """
    Load the .json sidecar for a given .md file.
    Falls back to a minimal metadata dict if no sidecar exists.

    Args:
        md_path: Path to the .md document.

    Returns:
        dict with keys: title, associated_game, category, tags, author
    """
    json_path = md_path.with_suffix('.json')
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        meta = raw.get('document_metadata', raw)
        return {
            'title': meta.get('title', md_path.stem),
            'game': meta.get('associated_game', 'Unknown Game'),
            'category': meta.get('category', 'Guide'),
            'author': meta.get('author', 'Unknown'),
            'tags': meta.get('tags', []),
        }
    else:
        # Minimal fallback — derive game name from filename heuristic
        return {
            'title': md_path.stem.replace('_', ' '),
            'game': 'Unknown Game',
            'category': 'Guide',
            'author': 'Unknown',
            'tags': [],
        }


# ---------------------------------------------------------------------------
# Core recursive chunker
# ---------------------------------------------------------------------------

def _split_by_paragraphs(text: str, header: str) -> list[str]:
    """
    Split a block of text at paragraph boundaries (double newlines).
    Re-prepend the originating heading to every sub-chunk so no chunk
    loses its semantic label.

    Args:
        text:   The text content of a single section (heading already stripped).
        header: The ### or #### heading string to prepend to each sub-chunk.

    Returns:
        List of sub-chunk strings, each starting with the heading.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    if not paragraphs:
        return []

    sub_chunks = []
    current = f"{header}\n"

    for para in paragraphs:
        candidate = current + para + "\n\n"
        if len(candidate) <= CHUNK_TARGET or len(current) <= len(f"{header}\n") + 1:
            # Still within budget, or we haven't added anything yet — keep growing
            current = candidate
        else:
            # Flush current sub-chunk (with overlap tail) and start a new one
            if len(current.strip()) >= MIN_CHUNK_SIZE:
                sub_chunks.append(current.strip())
            # Start next sub-chunk: heading + overlap from end of previous chunk
            overlap_text = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
            current = f"{header}\n{overlap_text.strip()}\n\n{para}\n\n"

    if len(current.strip()) >= MIN_CHUNK_SIZE:
        sub_chunks.append(current.strip())

    return sub_chunks


def _extract_sections(md_text: str) -> list[tuple[str, str, int]]:
    """
    Parse a Markdown document and extract sections keyed on ### and #### headings.
    Sections under ## or # headings without a ### child are also captured.

    Returns:
        List of (heading_text, section_body, heading_level) tuples in document order.
    """
    sections = []
    lines = md_text.split('\n')
    current_heading = None
    current_level = 0
    current_body_lines = []

    for line in lines:
        m = HEADING_PATTERN.match(line)
        if m:
            level = len(m.group(1))   # number of # chars
            heading_text = line.strip()

            # Flush previous section
            if current_heading is not None:
                body = '\n'.join(current_body_lines).strip()
                if body:
                    sections.append((current_heading, body, current_level))

            current_heading = heading_text
            current_level = level
            current_body_lines = []
        else:
            current_body_lines.append(line)

    # Flush final section
    if current_heading is not None:
        body = '\n'.join(current_body_lines).strip()
        if body:
            sections.append((current_heading, body, current_level))

    return sections


def chunk_document(md_path: Path, metadata: dict) -> list[dict]:
    """
    Recursively chunk a single Markdown document.

    Strategy (in order):
      1. Split at ### and #### heading boundaries.
      2. For sections within CHUNK_TARGET: emit as a single chunk.
      3. For sections below MIN_CHUNK_SIZE: merge with next section.
      4. For sections exceeding CHUNK_TARGET: split at paragraph breaks,
         re-prepending the heading to every sub-chunk.

    Args:
        md_path:  Path to the .md file.
        metadata: Sidecar metadata dict from load_metadata().

    Returns:
        List of chunk dicts with keys: text, game, title, category,
        section_header, source_file, chunk_index.
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    sections = _extract_sections(md_text)
    raw_chunks: list[str] = []
    pending = ""          # buffer for merging small sections
    pending_header = ""   # heading carried forward during merging

    for heading, body, _level in sections:
        section_text = f"{heading}\n{body}"

        if pending:
            # Try merging pending + this section
            merged = pending + "\n\n" + section_text
            if len(merged) <= CHUNK_TARGET:
                pending = merged
                continue
            else:
                # Flush pending before processing current section
                if len(pending.strip()) >= MIN_CHUNK_SIZE:
                    raw_chunks.append(pending.strip())
                pending = ""
                pending_header = ""

        if len(section_text) <= MIN_CHUNK_SIZE:
            # Too small — hold for merging with next section
            pending = section_text
            pending_header = heading
        elif len(section_text) <= CHUNK_TARGET:
            # Just right — emit as-is
            raw_chunks.append(section_text.strip())
        else:
            # Too large — split at paragraph boundaries
            sub = _split_by_paragraphs(body, heading)
            raw_chunks.extend(sub)

    # Flush any remaining pending content
    if pending and len(pending.strip()) >= MIN_CHUNK_SIZE:
        raw_chunks.append(pending.strip())

    # Build final chunk dicts
    chunks = []
    for idx, text in enumerate(raw_chunks):
        # Extract section header from first line of chunk
        first_line = text.split('\n')[0].strip()
        section_header = first_line if HEADING_PATTERN.match(first_line) else metadata['title']

        chunks.append({
            'text': text,
            'game': metadata['game'],
            'title': metadata['title'],
            'category': metadata['category'],
            'section_header': section_header,
            'source_file': md_path.name,
            'chunk_index': idx,
        })

    return chunks


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------

def chunk_all_documents(docs_dir: str | Path) -> list[dict]:
    """
    Ingest and chunk all .md documents in a directory.

    Expects each .md file to have a matching .json sidecar. Falls back to
    minimal metadata if the sidecar is missing.

    Args:
        docs_dir: Path to the documents directory.

    Returns:
        Flat list of all chunk dicts across all documents, ordered by
        source file then chunk index.
    """
    docs_dir = Path(docs_dir)
    md_files = sorted(docs_dir.glob('*.md'))

    all_chunks = []
    for md_path in md_files:
        if md_path.name == '.gitkeep':
            continue
        metadata = load_metadata(md_path)
        doc_chunks = chunk_document(md_path, metadata)
        all_chunks.extend(doc_chunks)
        print(f"  [{md_path.name}] → {len(doc_chunks)} chunks  (game: {metadata['game']})")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# CLI entry point — run directly to inspect chunk output
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    docs_path = Path(__file__).parent / 'documents'
    print(f"Ingesting documents from: {docs_path}\n")
    chunks = chunk_all_documents(docs_path)

    if '--sample' in sys.argv:
        # Print the first 3 chunks from each unique game for quick inspection
        seen_games: dict[str, int] = {}
        for chunk in chunks:
            game = chunk['game']
            count = seen_games.get(game, 0)
            if count < 3:
                print("=" * 70)
                print(f"FILE   : {chunk['source_file']}  [chunk {chunk['chunk_index']}]")
                print(f"GAME   : {chunk['game']}")
                print(f"HEADER : {chunk['section_header']}")
                print(f"LEN    : {len(chunk['text'])} chars")
                print("-" * 70)
                print(chunk['text'][:400])
                print("...")
                seen_games[game] = count + 1

    if '--stats' in sys.argv:
        sizes = [len(c['text']) for c in chunks]
        print(f"\nChunk size stats:")
        print(f"  Min   : {min(sizes)} chars")
        print(f"  Max   : {max(sizes)} chars")
        print(f"  Mean  : {sum(sizes)//len(sizes)} chars")
        over = [s for s in sizes if s > 1200]
        print(f"  >1200 : {len(over)} chunks (inspect for heading-sparse sections)")
