# The Unofficial Guide — Project 1

---

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->

Horror Game Guides — Community Walkthroughs and Collectible References.

This domain covers unofficial, community-authored walkthrough guides and collectible/trophy references for a curated set of horror video games. The knowledge collected is specifically the kind that does not appear in official game manuals: step-by-step puzzle solutions, exact item locations, enemy patrol patterns, survival strategy, alternate-path routing, and collectible coordinates described in player-friendly language.

This knowledge is valuable and hard to find through official channels because game publishers no longer distribute walkthroughs. Official documentation (if it exists) covers mechanics but never guides players room-by-room through the game. Community walkthroughs live across hundreds of individual pages, wikis, and GameFAQs threads that are difficult to query holistically. Spoiler-sensitive players need targeted answers ("where is the safe key in Chapter 3") without reading an entire walkthrough wiki page. Players who are stuck at specific moments need concise, grounded answers that cannot be hallucinated — every item location, every puzzle solution, must come directly from the source text.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Alan Wake II Complete Walkthrough | GameFAQs community FAQ | https://gamefaqs.gamespot.com/pc/344538-alan-wake-ii/faqs/81253 |
| 2 | Alan Wake II Collectibles and Trophies Guide | GameFAQs community FAQ | https://gamefaqs.gamespot.com/pc/344538-alan-wake-ii/faqs/81253 |
| 3 | Alan Wake II DLC Walkthrough | GameFAQs community FAQ | https://gamefaqs.gamespot.com/pc/344538-alan-wake-ii/faqs/81253 |
| 4 | Alien: Isolation Complete Walkthrough | GameFAQs community FAQ | https://gamefaqs.gamespot.com/ps4/751154-alien-isolation/faqs/70566 |
| 5 | Alien: Isolation Blueprint and Collectibles Guide | GameFAQs community FAQ | https://gamefaqs.gamespot.com/ps4/751154-alien-isolation/faqs/78206 |
| 6 | Amnesia: The Dark Descent Complete Walkthrough | GameFAQs community FAQ | https://gamefaqs.gamespot.com/pc/978772-amnesia-the-dark-descent/faqs/60857 |
| 7 | Amnesia: The Dark Descent Collectibles Guide | GameFAQs community FAQ | https://gamefaqs.gamespot.com/pc/978772-amnesia-the-dark-descent/faqs/60857 |
| 8 | Darkwood Guide and Walkthrough | GameFAQs community FAQ | https://gamefaqs.gamespot.com/ps4/263121-darkwood/faqs/81899 |
| 9 | Darkwood Achievements Guide | Steam community guide | https://steamcommunity.com/sharedfiles/filedetails/?id=1122007626 |
| 10 | Dead Space Complete Walkthrough | GameFAQs community FAQ | https://gamefaqs.gamespot.com/xbox360/943338-dead-space/faqs/54497 |
| 11 | Dead Space Collectibles and Upgrade Guide | GameFAQs community FAQ | https://gamefaqs.gamespot.com/xbox360/943338-dead-space/faqs/54497 |
| 12 | Outlast 2 Complete Walkthrough | GameFAQs community FAQ | https://gamefaqs.gamespot.com/pc/182702-outlast-2/faqs/75183 |
| 13 | Outlast 2 Collectibles and Documents Guide | GameFAQs community FAQ | https://gamefaqs.gamespot.com/pc/182702-outlast-2/faqs/75183 |
| 14 | Resident Evil 4 Remake Complete Walkthrough | Scribd community document | https://www.scribd.com/document/820391023/RESIDENT-EVIL-4-REMAKE-WALKTHROUGH |
| 15 | Resident Evil 4 Remake Collectibles and Trophy Roadmap | Scribd community document | https://www.scribd.com/document/820391023/RESIDENT-EVIL-4-REMAKE-WALKTHROUGH |
| 16 | Resident Evil 4 Remake: Separate Ways DLC Walkthrough | Scribd community document | https://www.scribd.com/document/820391023/RESIDENT-EVIL-4-REMAKE-WALKTHROUGH |
| 17 | Silent Hill 2 Remake Items and Collectibles Guide | GameFAQs community FAQ | https://gamefaqs.gamespot.com/ps5/383909-silent-hill-2/faqs/81517 |
| 18 | Silent Hill 2 Remake Puzzles and Trophies Guide | GameFAQs community FAQ | https://gamefaqs.gamespot.com/ps5/383909-silent-hill-2/faqs/81517 |
| 19 | Silent Hill 2 Remake Complete Walkthrough | Community walkthrough (markdown) | documents/Silent_Hill_2_Remake_Complete_Walkthrough.md |
| 20 | SOMA Complete Walkthrough | TrueAchievements community walkthrough | https://www.trueachievements.com/game/SOMA/walkthrough |
| 21 | The Evil Within Complete Walkthrough | GameFAQs community FAQ | https://gamefaqs.gamespot.com/ps4/711441-the-evil-within/faqs/70314 |
| 22 | The Evil Within Collectibles and Upgrade Guide | GameFAQs community FAQ | https://gamefaqs.gamespot.com/ps4/711441-the-evil-within/faqs/70314 |

---

## Chunking Strategy

**Chunk size:** ~800 characters (approximately 150–200 tokens)

**Overlap:** 150 characters (~30 tokens)

**Why these choices fit your documents:**

The documents in this corpus have a consistent hierarchical Markdown structure: a top-level `#` title, `##` part/act headers, `###` area or objective headers, and `####` granular sub-section headers. The chunker splits at `###` and `####` heading boundaries first. When a section is shorter than 800 characters, it is held in a merge buffer and combined with the following section. Sections longer than 800 characters are split at paragraph breaks (blank lines), never mid-sentence. Any section that still exceeds the 1,200-character hard cap after paragraph splitting is force-split at the nearest sentence boundary.

To prevent loss of contextual mapping, the full heading hierarchy path (e.g., `[Section: Chapter 5 > Objective: Find the Key > Room 302]`) is prepended to every chunk's text. This injects high-level keywords — such as a chapter name or game title — directly into the chunk body so they are indexed for both semantic and keyword search.

The 800-character target was chosen because most individual sub-sections in these documents (a single room's item list, a single boss phase, a single puzzle solution) fall in the 300–900 character range. Using `top_k=5` retrieved chunks at ~800 characters each keeps the total retrieved context under ~800 tokens.

The 150-character overlap prevents information at chunk boundaries (e.g., a puzzle solution that begins at the end of one chunk) from being lost between adjacent chunks.

**Final chunk count:** 2,678 chunks across 22 documents

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`

This model was chosen because it runs entirely locally with no API cost, has a 384-dimensional output space, and achieves strong semantic similarity on short factual queries like the ones this system handles. For a project with a local corpus and no inference budget, it is the most practical choice.

**Production tradeoff reflection:**

If deploying this system for real users with no cost constraint, the primary tradeoffs to weigh would be:

- **Context length**: `all-MiniLM-L6-v2` has a 256-token input limit. Chunks longer than this are silently truncated during embedding, which can degrade retrieval quality for dense walkthrough sections. A model like `all-mpnet-base-v2` (768-token limit) or `text-embedding-3-small` (8,192-token limit via OpenAI API) would handle longer chunks more faithfully.
- **Domain accuracy**: General-purpose embedding models may underperform on highly domain-specific text (game-specific item names, enemy names, puzzle terminology). Fine-tuned or instruction-tuned models such as `e5-large-v2` or `bge-large-en-v1.5` typically outperform MiniLM on factual retrieval tasks.
- **Latency**: Locally-run models add embedding inference time per query. For sub-200ms response targets, API-based embeddings with server-side caching may be preferable.
- **Multilingual support**: Not needed for this English-only corpus, but `multilingual-e5-large` would be required for a multi-language version.
- **Cost**: For high-query-volume production systems, the free local embedding approach becomes very attractive vs. pay-per-token API models.

---

## Grounded Generation

**System prompt grounding instruction:**

The system prompt passed to `llama-3.3-70b-versatile` via the Groq API explicitly forbids the model from using any knowledge outside the retrieved passages:

> *"You are an expert horror game survival guide assistant. Your task is to answer the user's question based strictly and ONLY on the provided context passages.*
> *Rules:*
> *1. Answer the question using ONLY the facts directly mentioned in the context. Do not speculate, extrapolate, or assume anything.*
> *2. Do NOT use any external or pre-existing knowledge about the games, story, or mechanics. If a detail is not explicitly written in the passages, treat it as entirely unknown.*
> *3. If the context passages do not contain enough information to answer the question, respond with exactly this phrase and nothing else: "I couldn't find that in the guides I have."*
> *4. Keep your answer clear, direct, and factual. Cite specific details from the passages."*

Temperature is set to `0.0` to maximize deterministic, fact-bound output and minimize creative drift.

**How source attribution is surfaced in the response:**

Each retrieved chunk is injected into the user-turn prompt as a numbered passage with explicit metadata headers (`Game:`, `Document:`, `Section:`) prepended to the text. The Streamlit UI then displays a deduplicated "Sources Cited" section below every answer, grouped by game title, showing the document name, section header, and source filename for each chunk used.

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What happens if Daniel's sanity drops to zero in Amnesia: The Dark Descent? | Daniel becomes temporarily incapacitated — stumbles, falls, briefly unresponsive. Does not die from zero sanity alone but is extremely vulnerable during recovery. | Daniel becomes temporarily incapacitated at critical sanity loss; he stumbles and falls and is briefly unresponsive. | Relevant | Accurate |
| 2 | Where is the Bloodstained Bracelet found in the Silent Hill 2 Remake, and what do you need to get it? | Pool (Pool Wall — Eye) on 2F of Brookhaven Hospital. Requires combining Bent Needle + Medical Tube first; Pool Hatch must be accessible. | Found between the dive boards at the pool area where there is an eye with leakage; retrieved the Bracelet Puzzle section as a top source (distance 0.28). | Relevant | Partially Accurate |
| 3 | How do you defeat the Leviathan boss in Dead Space? | Phase 1: Shoot glowing yellow spots at tentacle bases. Phase 2: Shoot orange core in mouth, use Kinesis to redirect explosive balls back in. Phase 3: Both attacks combined. | Correctly described three phases: Phase 1 side walkways + yellow glowing spots on tentacles; Phase 2 ADS cannon usage; Phase 3 combined. | Relevant | Accurate |
| 4 | In Resident Evil 4 Remake, what is the Blue Medallion challenge and how many are there? | Blue Medallions are hanging targets found throughout the game that must be shot to complete Merchant requests. Number varies by region/chapter. | Described Blue Medallions as targets to destroy for a Merchant Request; number and exact locations partially covered. | Relevant | Partially Accurate |
| 5 | What is the WAU in SOMA and why is it a threat? | WAU (Warden Unit Automation) is PATHOS-II's facility management AI that went rogue, forcibly integrating humans with machinery to "preserve" them. Avoidance rather than combat is the strategy. | "I couldn't find that in the guides I have." — complete retrieval failure; all 5 retrieved chunks had distances above 0.65. | Off-target | Inaccurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:**

*"What is the WAU in SOMA and why is it a threat?"* (with game filter: SOMA (2015))

**What the system returned:**

> "I couldn't find that in the guides I have."

All 5 retrieved chunks had cosine distances above 0.65 (with the top result at 0.783), indicating that the retriever found no meaningfully relevant passage. The system correctly refused to hallucinate an answer but was unable to find the information at all.

**Root cause (tied to a specific pipeline stage):**

The failure originates at the **Chunking** stage and compounds into a **Retrieval** failure. The SOMA walkthrough is structured as a strict location-by-location guide (Upsilon Facility → Phi Site → Lambda → Tau → etc.). The WAU is a lore concept — a rogue facility AI — whose description is distributed across contextual asides embedded within walkthrough prose rather than collected under a dedicated heading such as `### What is the WAU?`. Because the chunker splits on Markdown headings, the WAU's description ends up fragmented across multiple walkthrough-step chunks. No single chunk has sufficient semantic density around the phrase "WAU" or "Warden Unit Automation" for the embedding model to surface it as a top result for a conceptual lore query. The very high cosine distances confirm near-random retrieval — nothing in the index closely matched the question's semantic space.

**What you would change to fix it:**

Two targeted fixes would address this:

1. **Source-side preprocessing**: Add a short dedicated "Lore and Key Concepts" section to the SOMA walkthrough document that consolidates WAU, SOMA, and key entity explanations under a single `### Lore` heading, making them chunk-able and retrievable as a unit.
2. **Chunk metadata enrichment**: During ingestion, detect named entities (e.g., "WAU", "PATHOS-II") in the document and store them as searchable metadata tags. A keyword pre-filter on the metadata tag "WAU" before vector search would surface the relevant chunks regardless of embedding distance.

---

## Spec Reflection

**One way the spec helped you during implementation:**

The Chunking Strategy section of `planning.md` — particularly the decision to prepend the full heading hierarchy path to every chunk — turned out to be the single most impactful design choice. During implementation, early test queries on ambiguous terms like "safe" or "collectibles" returned results from the wrong game because the chunk text alone carried no game-level context. Adding the `[Section: Game > Chapter > Area]` prefix directly into the chunk body made those keywords indexable for both semantic and keyword search, and the correct-game results jumped to the top immediately. Having that decision committed to the spec before coding meant it was implemented correctly on the first pass rather than discovered as a patch.

**One way your implementation diverged from the spec, and why:**

The spec described a TF-based keyword scorer for the lexical component of hybrid search. During implementation, this was upgraded to a full **BM25 scorer** (with IDF weighting, term frequency normalization, and document length normalization). The plain TF scorer failed on long documents like the Darkwood walkthrough (342KB) because common terms like "the" and "area" dominated scores regardless of query relevance. BM25's IDF term weights and length normalization corrected this, producing meaningful keyword scores even across the corpus's highly variable document lengths. The Reciprocal Rank Fusion merge logic remained unchanged from the spec.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* The full Chunking Strategy and Retrieval Approach sections from `planning.md`, the document directory structure, and a sample `.md` file with its `.json` sidecar — asking for `ingest.py` with a `chunk_documents()` function implementing recursive header-based chunking at `###`/`####` boundaries with 800-char target and 150-char overlap, and `retrieve.py` with hybrid BM25 + vector search merged via Reciprocal Rank Fusion.
- *What it produced:* Complete `ingest.py` with `_extract_sections()`, `_split_by_paragraphs()`, `_hard_split()`, and `chunk_document()` functions; complete `retrieve.py` with `BM25Scorer`, `_load_resources()`, `retrieve()`, and `list_games()` implementing the full hybrid RRF pipeline.
- *What I changed or overrode:* The spec described a plain TF keyword scorer; the AI independently upgraded it to full BM25 with IDF and document-length normalization after recognizing that the corpus's extreme size variation (10KB to 342KB) would cause TF-only scoring to be dominated by document length. This was accepted as an improvement. The heading hierarchy breadcrumb format (`[Section: A > B > C]`) was also refined from the spec's simpler description to ensure double-heading artifacts were stripped during paragraph sub-splitting.

**Instance 2**

- *What I gave the AI:* The Architecture diagram from `planning.md`, the Evaluation Plan questions, and the output schema from `retrieve.py` — asking for `generate.py` with a `generate_answer()` function building a grounded prompt and calling the Groq API, and `app.py` as a Streamlit interface with a survival-horror dark aesthetic, sidebar game filter dropdown, answer display card, and grouped sources section.
- *What it produced:* Complete `generate.py` with strict grounding system prompt, primary/fallback model logic (`llama-3.3-70b-versatile` → `mixtral-8x7b-32768`), and temperature `0.0` for deterministic output; complete `app.py` with custom CSS survival-horror theme (crimson red, Share Tech Mono font, dark background), `@st.cache_resource` backend initialization, game filter dropdown, answer card, deduplicated grouped sources display, and a developer debug expander.
- *What I changed or overrode:* The system prompt wording was reviewed and the explicit fallback phrase ("I couldn't find that in the guides I have.") was kept exactly as written — this was intentional so that the Q5 SOMA failure would produce a clean, unambiguous "no answer" rather than a hedged hallucination. The Streamlit layout column ratio (`[1, 6]` for button vs. input) was also adjusted from the initial equal-width layout to better fit the horror terminal aesthetic.
