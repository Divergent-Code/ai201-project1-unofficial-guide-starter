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


_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    """Get or create the cached Groq client instance.

    Reuses the client across calls to avoid cold-start latency.
    """
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. Please add it to your .env file."
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def reformulate_query(query: str, chat_history: list[dict], client: Groq | None = None) -> str:
    """Rephrase user's follow-up question into a standalone query.

    Uses a fast LLM (`llama-3.1-8b-instant`) to rewrite contextual queries 
    (which may contain pronouns/references) into self-contained search terms. 
    If there is no chat history, returns the query as-is.

    Args:
        query (str): The latest user message or query.
        chat_history (list[dict]): A list of dictionaries representing past chat turns, 
            where each dict contains 'role' and 'content' keys.
        client (Groq, optional): Pre-instantiated Groq client. Defaults to None.

    Returns:
        str: The standalone query string to be used for retrieval.
    """
    if not chat_history:
        return query

    if client is None:
        if not GROQ_API_KEY:
            return query
        try:
            client = get_groq_client()
        except ValueError:
            return query

    # Format history for the model
    formatted_history = []
    for turn in chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        formatted_history.append(f"{role.upper()}: {content}")
        
    history_str = "\n".join(formatted_history)
    
    system_prompt = (
        "You are an AI assistant that helps formulate search queries for a gaming RAG system.\n"
        "Your task is to analyze the conversation history and the user's latest follow-up question, "
        "and rephrase the question into a standalone query that can be understood without any history.\n\n"
        "Rules:\n"
        "1. Do NOT answer the question. Only output the rephrased standalone query.\n"
        "2. Keep the query concise and factual, focusing on key terms (e.g. game title, specific items, puzzles, or locations mentioned earlier).\n"
        "3. If the user's question is already a standalone query and does not need any context from the history, return it exactly as-is.\n"
        "4. Output ONLY the rephrased query string and nothing else."
    )
    
    user_prompt = (
        f"Conversation History:\n{history_str}\n\n"
        f"Latest User Question: {query}\n\n"
        f"Standalone Query:"
    )
    
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.1-8b-instant",  # fast model for query reformulation
            temperature=0.0,
            max_tokens=256,
        )
        reformulated = completion.choices[0].message.content.strip()
        # Clean any surrounding quotes
        if (reformulated.startswith('"') and reformulated.endswith('"')) or \
           (reformulated.startswith("'") and reformulated.endswith("'")):
            reformulated = reformulated[1:-1].strip()
        return reformulated
    except Exception as e:
        print(f"[generate] Warning: Query reformulation failed ({e}). Using raw query.", file=sys.stderr)
        return query


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
    client: Groq | None = None,
) -> tuple[str, list[dict]]:
    """Generate an answer grounded strictly in the retrieved context passages.

    Applies strict system-prompt grounding constraints using the primary 
    synthesis model `llama-3.3-70b-versatile`, with a fallback to `llama-3.1-8b-instant` 
    if needed. Temperature is set to 0.0 to prevent hallucination.

    Args:
        query (str): The user's question.
        retrieved_chunks (list[dict]): List of retrieved chunk dictionaries.
        chat_history (list[dict], optional): List of conversation turn dicts. 
            Defaults to None.
        client (Groq, optional): Pre-instantiated Groq client. Defaults to None.

    Raises:
        ValueError: If GROQ_API_KEY is not defined in the environment.
        RuntimeError: If both primary and fallback API models fail to execute.

    Returns:
        tuple[str, list[dict]]: A tuple containing:
            - answer (str): The strictly grounded answer string.
            - sources (list[dict]): The list of retrieved chunks used as references.
    """
    if client is None:
        client = get_groq_client()

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
        "4. Keep your answer clear, direct, and factual. Cite specific details from the passages.\n"
        "5. You are engaged in a conversation. Maintain continuity with the chat history if relevant, "
        "but do NOT violate the grounding rule."
    )

    messages = [{"role": "system", "content": system_prompt}]
    
    # 3. Add chat history for contextual generation
    if chat_history:
        for turn in chat_history:
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_prompt = f"Context passages:\n{context_str}\n\nUser Question: {query}"
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            messages=messages,
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
                messages=messages,
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
