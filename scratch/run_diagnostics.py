import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ingest import chunk_document, load_metadata
from retrieve import retrieve, _retrieve_vector_only, tokenize, _collections, _bm25_scorers, _all_chunks, _load_resources
from generate import generate_answer, reformulate_query

def get_5_sample_chunks():
    print("--- GATHERING 5 SAMPLE CHUNKS ---")
    docs_dir = Path(__file__).parent.parent / "documents"
    md_files = sorted(docs_dir.glob('*.md'))
    if not md_files:
        print("No markdown files found!")
        return
    
    # Let's pick Amnesia complete walkthrough
    amnesia_file = next((f for f in md_files if "Amnesia_The_Dark_Descent_Complete_Walkthrough" in f.name), md_files[0])
    print(f"Using file for samples: {amnesia_file.name}")
    meta = load_metadata(amnesia_file)
    chunks = chunk_document(amnesia_file, meta)
    
    samples = chunks[:5]
    for i, c in enumerate(samples, 1):
        print(f"\n[Chunk {i}]")
        print(json.dumps(c, indent=2))

def run_retrieval_test():
    print("\n--- STANDALONE RETRIEVAL TEST ---")
    query = "What happens if Daniel's sanity drops to zero?"
    print(f"Query: '{query}'")
    results = retrieve(query, game_filter="Amnesia: The Dark Descent (2010)", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"Rank {i} [Distance: {r['distance']}]")
        print(f"  Game: {r['game']}")
        print(f"  Title: {r['title']}")
        print(f"  Header: {r['section_header']}")
        print(f"  Source: {r['source_file']} (Index: {r['chunk_index']})")
        print(f"  Text Snippet: {r['text'][:200]}...")

def run_grounded_generation_tests():
    print("\n--- GROUNDED GENERATION TESTS ---")
    queries = [
        ("Where is the Bloodstained Bracelet found in the Silent Hill 2 Remake, and what do you need to get it?", "Silent Hill 2 Remake (2024)"),
        ("How do you defeat the Leviathan boss in Dead Space?", "Dead Space (2008)"),
        ("Does SOMA have a walkthrough for finding the hidden weapon?", "SOMA (2015)") # Out of scope query
    ]
    
    for q, game in queries:
        print(f"\nQuery: '{q}' (Game: {game})")
        chunks = retrieve(q, game_filter=game, top_k=5)
        print(f"Retrieved {len(chunks)} chunks.")
        try:
            answer, sources = generate_answer(q, chunks)
            print("Answer:")
            print(answer)
            print("Sources cited:")
            for s in sources[:2]:
                print(f"  - {s['title']} | {s['section_header']} ({s['source_file']})")
        except Exception as e:
            print(f"Error during generation: {e}")

def run_hybrid_comparison():
    print("\n--- HYBRID SEARCH COMPARISON ON 3+ QUERIES ---")
    queries = [
        ("What happens if Daniel's sanity drops to zero?", "Amnesia: The Dark Descent (2010)"),
        ("Bloodstained Bracelet in Brookhaven Hospital", "Silent Hill 2 Remake (2024)"),
        ("How do you defeat the Leviathan boss?", "Dead Space (2008)")
    ]
    
    collection_name = "horror_guides_recursive"
    _load_resources(collection_name)
    
    bm25_scorer = _bm25_scorers.get(collection_name)
    all_chunks = _all_chunks.get(collection_name)
    collection = _collections[collection_name]
    
    for idx, (query, game_filter) in enumerate(queries, 1):
        print(f"\nQuery {idx}: '{query}' (Game: {game_filter})")
        
        # 1. Vector Only
        vector_res = _retrieve_vector_only(query, game_filter=game_filter, top_k=5, collection_name=collection_name)
        
        # 2. BM25 Only
        query_terms = tokenize(query)
        matching_games = None
        if game_filter:
            from retrieve import _get_all_games
            all_games = _get_all_games(collection_name)
            matching_games = [g for g in all_games if g.startswith(game_filter)]
        
        filtered = all_chunks
        if matching_games:
            filtered = [c for c in all_chunks if c["metadata"].get("game") in matching_games]
            
        bm25_scored = []
        for c in filtered:
            score = bm25_scorer.score(query_terms, c["bm25_index"])
            if score > 0:
                bm25_scored.append((score, c))
        bm25_scored.sort(key=lambda x: x[0], reverse=True)
        bm25_res = bm25_scored[:5]
        
        # 3. Hybrid (RRF)
        hybrid_res = retrieve(query, game_filter=game_filter, top_k=5, collection_name=collection_name)
        
        print("Vector Only Top 3:")
        for r in vector_res[:3]:
            print(f"  - [{r['distance']}] {r['section_header']} ({r['source_file']}): {r['text'][:100]}...")
            
        print("BM25 Only Top 3:")
        for score, c in bm25_res[:3]:
            print(f"  - [{score:.4f}] {c['metadata'].get('section_header')} ({c['metadata'].get('source_file')}): {c['text'][:100]}...")
            
        print("Hybrid (RRF) Top 3:")
        for r in hybrid_res[:3]:
            print(f"  - [{r['distance']}] {r['section_header']} ({r['source_file']}): {r['text'][:100]}...")

def run_multi_turn_transcript():
    print("\n--- MULTI-TURN CHAT TRANSCRIPT ---")
    history = []
    
    # Turn 1
    t1_q = "Who is Daniel in Amnesia: The Dark Descent?"
    print(f"Turn 1 Query: '{t1_q}'")
    t1_chunks = retrieve(t1_q, game_filter="Amnesia: The Dark Descent (2010)", top_k=5)
    t1_ans, _ = generate_answer(t1_q, t1_chunks, history)
    print(f"Turn 1 Answer: {t1_ans[:150]}...")
    
    history.append({"role": "user", "content": t1_q})
    history.append({"role": "assistant", "content": t1_ans})
    
    # Turn 2
    t2_q = "What happens if his sanity drops to zero?"
    print(f"Turn 2 Query: '{t2_q}'")
    standalone_q = reformulate_query(t2_q, history)
    print(f"Turn 2 Reformulated Standalone Query: '{standalone_q}'")
    t2_chunks = retrieve(standalone_q, game_filter="Amnesia: The Dark Descent (2010)", top_k=5)
    t2_ans, _ = generate_answer(t2_q, t2_chunks, history)
    print(f"Turn 2 Answer: {t2_ans[:150]}...")

if __name__ == "__main__":
    get_5_sample_chunks()
    run_retrieval_test()
    run_grounded_generation_tests()
    run_hybrid_comparison()
    run_multi_turn_transcript()
