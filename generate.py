"""
generate.py — Grounded answer generation using the Groq API.

Pipeline stage: Ranked Chunks + Query → Grounded Prompt → Groq LLM → Grounded Answer

Public interface:
    generate_answer(query, retrieved_chunks) → (answer: str, sources: list[dict])
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"


def generate_answer(query: str, retrieved_chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Generates an answer grounded strictly within the retrieved context passages using Groq.

    Args:
        query:            The user's question.
        retrieved_chunks: A list of retrieved chunk dicts (from retrieve.py).

    Returns:
        tuple (answer, sources)
            - answer: The generated answer string.
            - sources: The list of retrieved chunk dicts used as the context.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. Please add it to your .env file."
        )

    # 1. Format the context passages
    if not retrieved_chunks:
        return "I couldn't find that in the guides I have.", []

    context_passages = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        game = chunk.get("game", "Unknown Game")
        title = chunk.get("title", "Unknown Title")
        header = chunk.get("section_header", "Unknown Section")
        text = chunk.get("text", "")
        
        passage = (
            f"--- Passage {i} ---\n"
            f"Game: {game}\n"
            f"Document: {title}\n"
            f"Section: {header}\n\n"
            f"{text}\n"
        )
        context_passages.append(passage)

    context_str = "\n".join(context_passages)

    # 2. Construct the strict grounding system prompt
    system_prompt = (
        "You are an expert horror game survival guide assistant. Your task is to answer "
        "the user's question based strictly and ONLY on the provided context passages.\n\n"
        "Rules:\n"
        "1. Answer the question using ONLY the facts directly mentioned in the context. "
        "Do not speculate, extrapolate, or assume anything.\n"
        "2. Do NOT use any external or pre-existing knowledge about the games, story, or mechanics. "
        "If a detail is not explicitly written in the passages, treat it as entirely unknown.\n"
        "3. If the context passages do not contain enough information to answer the question, "
        "respond with exactly this phrase and nothing else:\n"
        "\"I couldn't find that in the guides I have.\"\n"
        "4. Keep your answer clear, direct, and factual. Cite specific details from the passages."
    )

    user_prompt = f"Context passages:\n{context_str}\n\nUser Question: {query}"

    # 3. Call the Groq API
    client = Groq(api_key=GROQ_API_KEY)
    
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=DEFAULT_MODEL,
            temperature=0.0,  # 0.0 temperature for maximum deterministic grounding
            max_tokens=1024,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        # Fallback to a different model if there's a model-specific failure/rate limit
        print(f"[generate] Warning: Primary model failed ({e}). Trying fallback model...", file=sys.stderr)
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=FALLBACK_MODEL,
                temperature=0.0,
                max_tokens=1024,
            )
            answer = completion.choices[0].message.content
        except Exception as fallback_err:
            raise RuntimeError(
                f"Both Groq models failed. Please check your API key and connection. Error: {fallback_err}"
            ) from fallback_err

    # Strip any extra whitespace
    answer = answer.strip()

    return answer, retrieved_chunks


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from retrieve import retrieve

    # Grab query and game filter from command line arguments
    args = sys.argv[1:]
    if not args:
        print("Usage: python generate.py \"<question>\" [\"<game_filter>\"]")
        print("Using default test query...")
        query = "where is the shotgun in Dead Space"
        game_filter = "Dead Space (2008)"
    else:
        query = args[0]
        game_filter = args[1] if len(args) > 1 else None

    print(f"Query: '{query}'")
    print(f"Game filter: '{game_filter}'")
    print("Retrieving chunks...")
    
    try:
        chunks = retrieve(query, game_filter=game_filter, top_k=5)
        print(f"Retrieved {len(chunks)} chunks.")
        
        print("\nGenerating answer...")
        answer, sources = generate_answer(query, chunks)
        
        print("\n=== GENERATED ANSWER ===")
        print(answer)
        print("========================")
        
        print("\nSources Cited:")
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s['game']} -> {s['title']} | Section: {s['section_header']} (dist: {s['distance']})")
            
    except Exception as err:
        print(f"\nError occurred: {err}", file=sys.stderr)
