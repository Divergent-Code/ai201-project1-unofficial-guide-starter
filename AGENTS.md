# AGENTS.md — AI201 Project 1: The Unofficial Guide (Starter)

## Project Overview

This is a **starter template** for CodePath's AI201 course, Project 1: "The Unofficial Guide." It is intentionally minimal — students are expected to implement the full retrieval-augmented generation (RAG) pipeline themselves.

The goal of the project is to build a grounded Q&A system over a curated document collection. Students choose a domain, gather unofficial or hard-to-find knowledge (e.g., Reddit threads, forum posts, reviews), and build a pipeline that chunks, embeds, retrieves, and generates answers constrained to those documents.

The repository currently contains:
- **Template documents** (`README.md`, `planning.md`) with sections to be filled in by the student as they design and implement their system.
- **Dependency manifest** (`requirements.txt`) specifying the core libraries for the RAG pipeline.
- **Environment template** (`.env.example`) for API key management.
- **Empty documents directory** (`documents/`) where source materials should be placed.

There is **no application code yet** — this is a blank slate.

---

## Technology Stack

| Layer | Library / Tool | Purpose |
|-------|---------------|---------|
| Language | Python 3.x | Implementation language |
| Embeddings | `sentence-transformers==3.4.1` | Local embedding model (e.g., all-MiniLM-L6-v2) |
| Vector Store | `chromadb>=0.6.0` | Local vector database for chunk storage and similarity search |
| LLM Client | `groq==0.15.0` | API client for Groq-hosted LLMs (Llama, Mixtral, etc.) |
| Configuration | `python-dotenv==1.0.1` | Load `GROQ_API_KEY` from `.env` file |
| UI (optional) | `gradio>=6.9.0` or `streamlit>=1.40.0` | Query interface for Milestone 5 (commented out in requirements) |
| PDF Parsing (optional) | `pdfplumber==0.11.4` | If documents include PDFs (commented out in requirements) |

No build tools (e.g., `pyproject.toml`, `setup.py`, `package.json`) are present. This is a simple pip-based project.

---

## Project Structure

```
.
├── documents/           # Source documents for the RAG corpus (empty except .gitkeep)
├── .env.example         # Template for GROQ_API_KEY
├── .gitignore           # Excludes .env, Python caches, venvs, ChromaDB local dirs
├── README.md            # Project report template (to be filled by student)
├── planning.md          # Design spec template (to be filled before coding)
└── requirements.txt     # Python dependencies
```

### Key Files

- **`planning.md`** — The student writes their design spec here *before* implementing. It covers domain choice, document sources, chunking strategy, retrieval approach, evaluation plan, anticipated challenges, architecture diagram, and AI tool usage plan.
- **`README.md`** — The final submission report. It mirrors `planning.md` but is updated with actual implementation details: final chunk count, embedding model choice, grounding mechanism, evaluation results, failure case analysis, reflection, and AI usage disclosure.
- **`documents/`** — All raw source material (txt, md, pdf, etc.) should live here. The ingestion script will read from this directory.

---

## Build and Run Commands

Since this is a starter with no source code yet, there is no executable entrypoint. The standard workflow for a student implementing this project is:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your Groq API key from https://console.groq.com

# 4. Place source documents
# Copy or download documents into the documents/ directory

# 5. Implement and run the pipeline
# (Students typically create scripts such as ingest.py, embed.py, retrieve.py, app.py)
```

No `pytest`, `tox`, `make`, or CI configuration is present. Testing is manual and guided by the 5 evaluation questions defined in `planning.md`.

---

## Code Style Guidelines

There are no enforced linters or formatters in the template. Students are expected to follow standard Python conventions:

- **PEP 8** for naming and layout.
- **Type hints** are encouraged but not required.
- **Docstrings** for functions, especially those handling chunking, embedding, and retrieval.
- **Environment variables** for secrets only; never hardcode API keys.
- **Modular design** is strongly suggested: separate ingestion, chunking, embedding, retrieval, and generation into distinct functions or modules so the architecture diagram maps cleanly to code.

---

## Testing Instructions

Formal automated tests are not part of this template. Evaluation is done via the **5 test questions** defined in `planning.md` and reported in `README.md`:

1. Write 5 specific, testable questions about your domain.
2. Record the expected answer for each.
3. Run each question through your system.
4. Score **retrieval quality** (Relevant / Partially relevant / Off-target) and **response accuracy** (Accurate / Partially accurate / Inaccurate).
5. Document at least one failure case with a root-cause analysis tied to a specific pipeline stage.

This manual evaluation is a required section of the final `README.md`.

---

## Security Considerations

- **API Keys**: The `.env` file is listed in `.gitignore`. Never commit real credentials. The only required secret is `GROQ_API_KEY`.
- **Local Data**: ChromaDB local persistence directories (`chroma_db/`, `chroma/`) are also gitignored. These may contain embedded text fragments from source documents.
- **Document Sources**: Students are expected to scrape or download publicly available text (Reddit, forums, etc.). Respect robots.txt and terms of service. Do not ingest private or copyrighted material without permission.
- **Grounding**: The project explicitly requires grounding the LLM to retrieved documents to prevent hallucination. The system prompt must instruct the model to answer only from the provided context and to surface source attribution.

---

## Notes for AI Agents

- If you are asked to implement this project, read `planning.md` first. It contains the student's design decisions and should drive your implementation.
- Check `documents/` to see what source material is available before writing ingestion logic.
- Assume the student may uncomment optional dependencies (gradio, streamlit, pdfplumber) based on their Milestone 5 interface choice or document types.
- When generating code, keep the architecture diagram from `planning.md` in mind: Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation.
