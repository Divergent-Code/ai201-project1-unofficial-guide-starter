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
CHUNK_HARD_CAP = 1200  # absolute maximum — any chunk above this is force-split

# Separator hierarchy for recursive splitting (tried in order)
# Level 1: ### heading  Level 2: #### heading  Level 3: blank line (paragraph)
HEADING_PATTERN = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)


# ---------------------------------------------------------------------------
# Sidecar metadata loader
# ---------------------------------------------------------------------------

def load_metadata(md_path: Path) -> dict:
    """Load the .json sidecar metadata for a given .md file.

    If no sidecar JSON file exists, falls back to a minimal metadata dictionary
    deriving the title from the file stem.

    Args:
        md_path (Path): Path to the source .md document.

    Returns:
        dict: A dictionary containing the following keys:
            - title (str): The document title.
            - game (str): The associated game name.
            - category (str): The guide category.
            - author (str): The author name.
            - tags (list[str]): List of tags associated with the guide.
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

def _hard_split(text: str, header: str) -> list[str]:
    """Last-resort character-level split for chunks exceeding CHUNK_HARD_CAP.

    Used when a chunk is still too large after paragraph-based splitting (e.g., 
    large tables or continuous paragraphs). Attempts to find a line break (\\n) 
    or sentence boundary ('. ') to split on. Re-prepends the heading to 
    each split continuation piece to maintain contextual breadcrumbs.

    Args:
        text (str): The oversized chunk text (already has the header prepended).
        header (str): The header string to prepend to continuation pieces.

    Returns:
        list[str]: A list of sub-chunk strings, each conforming to CHUNK_HARD_CAP.
    """
    if len(text) <= CHUNK_HARD_CAP:
        return [text]

    pieces = []
    remaining = text
    is_first = True

    while len(remaining) > CHUNK_HARD_CAP:
        split_at = CHUNK_HARD_CAP
        # 1. Try to find the last newline before the cap to preserve table/list row integrity
        boundary_nl = remaining.rfind('\n', 0, CHUNK_HARD_CAP)
        if boundary_nl > CHUNK_HARD_CAP // 2:
            split_at = boundary_nl + 1  # include the newline character
        else:
            # 2. Try to find the last sentence boundary before the cap
            boundary = remaining.rfind('. ', 0, CHUNK_HARD_CAP)
            if boundary > CHUNK_HARD_CAP // 2:
                split_at = boundary + 2  # include the '. '
        piece = remaining[:split_at].strip()
        if piece:
            pieces.append(piece)
        # Start continuation with heading prefix + overlap
        overlap = remaining[max(0, split_at - CHUNK_OVERLAP):split_at].strip()
        remaining = f"{header}\n{overlap}\n{remaining[split_at:].strip()}"
        is_first = False

    if remaining.strip():
        pieces.append(remaining.strip())

    return pieces


def _split_by_paragraphs(text: str, header: str) -> list[str]:
    """Split a section of text at paragraph boundaries (double newlines).

    Applies sliding window overlap logic to sub-chunks. Re-prepends the heading 
    structure to every sub-chunk to ensure they preserve semantic mapping. 
    Strips any leading repeated heading line from the body to avoid duplicate 
    heading entries.

    Args:
        text (str): The body text of the section (heading text already stripped).
        header (str): The full breadcrumb heading prefix to prepend to each sub-chunk.

    Returns:
        list[str]: A list of sub-chunk strings, each beginning with the header.
    """
    # Strip leading repeated heading line (causes double-heading artifacts)
    first_line = text.split('\n')[0].strip()
    if HEADING_PATTERN.match(first_line):
        text = text[len(first_line):].lstrip('\n')

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


def _extract_sections(md_text: str) -> list[tuple[str, str, int, str]]:
    """Parse a Markdown document and extract sections by heading levels.

    Identifies ### and #### headings as section boundaries. Captures sections 
    under # or ## headings if they do not contain ### or #### sub-headings. 
    Maintains an active hierarchy stack to yield breadcrumbs (e.g. "Chapter 1 > Room 2").

    Args:
        md_text (str): The raw text content of the Markdown document.

    Returns:
        list[tuple[str, str, int, str]]: A list of tuples containing:
            - heading_text (str): The full heading line text.
            - section_body (str): The body content under the heading.
            - heading_level (int): The heading level (number of '#' characters).
            - heading_path (str): The breadcrumb string representing the hierarchy.
    """
    sections = []
    lines = md_text.split('\n')
    current_heading = None
    current_level = 0
    current_body_lines = []
    
    # Stack to track active headings: list of (level, clean_heading_text)
    stack = []

    for line in lines:
        m = HEADING_PATTERN.match(line)
        if m:
            level = len(m.group(1))   # number of # chars
            heading_text = line.strip()
            clean_heading = m.group(2).strip()

            # Flush previous section
            if current_heading is not None:
                body = '\n'.join(current_body_lines).strip()
                if body:
                    path_str = " > ".join([h[1] for h in stack])
                    sections.append((current_heading, body, current_level, path_str))

            # Update stack for the new heading (keep only parents with lower level)
            stack = [h for h in stack if h[0] < level]
            stack.append((level, clean_heading))

            current_heading = heading_text
            current_level = level
            current_body_lines = []
        else:
            current_body_lines.append(line)

    # Flush final section
    if current_heading is not None:
        body = '\n'.join(current_body_lines).strip()
        if body:
            path_str = " > ".join([h[1] for h in stack])
            sections.append((current_heading, body, current_level, path_str))

    return sections


def chunk_document(md_path: Path, metadata: dict) -> list[dict]:
    """Recursively chunk a single Markdown document using heading hierarchy.

    Applies the recursive heading-based chunking strategy:
      1. Splits at ### and #### heading levels.
      2. Emits sections falling within CHUNK_TARGET.
      3. Merges tiny sections (under MIN_CHUNK_SIZE) with subsequent sections.
      4. Splits large sections by paragraph and character boundaries, prepending
         the heading path and heading to preserve location mapping in search indices.

    Args:
        md_path (Path): Path to the source .md file.
        metadata (dict): Metadata dictionary returned by load_metadata().

    Returns:
        list[dict]: A list of chunk dictionaries, each containing:
            - text (str): The body text of the chunk (with heading path prepended).
            - game (str): The associated game name.
            - title (str): The document title.
            - category (str): The guide category.
            - section_header (str): The hierarchy path breadcrumb.
            - source_file (str): The basename of the markdown file.
            - chunk_index (int): The 0-based chunk index within the document.
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    sections = _extract_sections(md_text)
    raw_chunks: list[tuple[str, str]] = []  # list of (chunk_text, heading_path)
    pending_body = ""
    pending_heading = ""
    pending_path = ""

    for heading, body, _level, heading_path in sections:
        path_prefix = f"[Section: {heading_path}]" if heading_path else ""
        full_header = f"{path_prefix}\n{heading}" if path_prefix else heading
        section_text = f"{full_header}\n{body}"

        if pending_body:
            # Try merging pending + this section
            merged_body = pending_body + "\n\n" + section_text
            if len(merged_body) <= CHUNK_TARGET:
                pending_body = merged_body
                continue
            else:
                # Flush pending before processing current section
                if len(pending_body.strip()) >= MIN_CHUNK_SIZE:
                    raw_chunks.append((pending_body.strip(), pending_path))
                pending_body = ""
                pending_heading = ""
                pending_path = ""

        if len(section_text) <= MIN_CHUNK_SIZE:
            # Too small — hold for merging with next section
            pending_body = section_text
            pending_heading = heading
            pending_path = heading_path
        elif len(section_text) <= CHUNK_TARGET:
            # Just right — emit as-is
            raw_chunks.append((section_text.strip(), heading_path))
        else:
            # Too large — split at paragraph boundaries, then hard-cap if needed
            sub = _split_by_paragraphs(body, full_header)
            for s in sub:
                for piece in _hard_split(s, full_header):
                    raw_chunks.append((piece, heading_path))

    # Flush any remaining pending content
    if pending_body and len(pending_body.strip()) >= MIN_CHUNK_SIZE:
        raw_chunks.append((pending_body.strip(), pending_path))

    # Build final chunk dicts
    chunks = []
    for idx, (text, path) in enumerate(raw_chunks):
        chunks.append({
            'text': text,
            'game': metadata['game'],
            'title': metadata['title'],
            'category': metadata['category'],
            'section_header': path if path else metadata['title'],
            'source_file': md_path.name,
            'chunk_index': idx,
        })

    return chunks


def chunk_document_fixed(
    md_path: Path,
    metadata: dict,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """Chunk a single document using a fixed-size character sliding window.

    Prepends document title information to every chunk for context and adjusts 
    boundaries to avoid splitting words or sentences when possible.

    Args:
        md_path (Path): Path to the source .md file.
        metadata (dict): Metadata dictionary returned by load_metadata().
        chunk_size (int, optional): Target character size of each chunk. Defaults to 800.
        overlap (int, optional): Character overlap between consecutive chunks. Defaults to 150.

    Returns:
        list[dict]: A list of chunk dictionaries containing standard keys (text, 
        game, title, category, section_header, source_file, chunk_index).
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Normalize newlines
    text = text.replace('\r\n', '\n')

    # Prepend basic context header to every chunk for parity in vector search
    doc_prefix = f"[Document: {metadata['title']}]\n"
    prefix_len = len(doc_prefix)

    slice_size = max(100, chunk_size - prefix_len)
    slice_overlap = max(10, overlap)

    chunks = []
    idx = 0
    start = 0

    while start < len(text):
        end = start + slice_size
        chunk_text = text[start:end]

        # Adjust to newline/word boundaries to avoid splitting mid-word
        if end < len(text):
            search_start = max(start, end - slice_overlap)
            last_space = text.rfind('\n', search_start, end)
            if last_space == -1 or last_space <= search_start:
                last_space = text.rfind('. ', search_start, end)
            if last_space == -1 or last_space <= search_start:
                last_space = text.rfind(' ', search_start, end)

            if last_space != -1 and last_space > search_start:
                end = last_space
                chunk_text = text[start:end]

        chunk_body = chunk_text.strip()
        if len(chunk_body) >= MIN_CHUNK_SIZE:
            full_text = f"{doc_prefix}{chunk_body}"
            chunks.append({
                'text': full_text,
                'game': metadata['game'],
                'title': metadata['title'],
                'category': metadata['category'],
                'section_header': metadata['title'],  # no dynamic headers for fixed size
                'source_file': md_path.name,
                'chunk_index': idx,
            })
            idx += 1

        start = end - slice_overlap
        if start >= len(text) - MIN_CHUNK_SIZE:
            break

    return chunks


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------

def chunk_all_documents(docs_dir: str | Path, strategy: str = "recursive") -> list[dict]:
    """Ingest and chunk all Markdown documents within a directory.

    Loads the sidecar metadata for each file and processes them according to the 
    specified chunking strategy.

    Args:
        docs_dir (str | Path): Path to the documents directory containing .md files.
        strategy (str, optional): The chunking strategy. Choose 'recursive' (default) 
            or 'fixed'.

    Returns:
        list[dict]: A flat list of chunk dictionaries across all ingested documents.
    """
    docs_dir = Path(docs_dir)
    md_files = sorted(docs_dir.glob('*.md'))

    all_chunks = []
    for md_path in md_files:
        if md_path.name == '.gitkeep':
            continue
        metadata = load_metadata(md_path)
        if strategy == "fixed":
            doc_chunks = chunk_document_fixed(md_path, metadata)
        else:
            doc_chunks = chunk_document(md_path, metadata)
        all_chunks.extend(doc_chunks)
        print(f"  [{md_path.name}] → {len(doc_chunks)} chunks  (game: {metadata['game']})")

    print(f"\nTotal chunks ({strategy}): {len(all_chunks)}")
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
