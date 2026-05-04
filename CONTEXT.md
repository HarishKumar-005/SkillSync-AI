# SkillSync AI — Week 1 CONTEXT.md

## 0) Purpose of this file
This document is the source of truth for **Week 1** of the SkillSync AI project. Any coding agent, including Antigravity, should read this file first before writing code.

The goal of Week 1 is **not** to finish the entire project. The goal is to deliver a **working foundation**:

1. upload documents,
2. parse and clean text,
3. split content into chunks,
4. store chunks in a knowledge base,
5. build a basic chat interface,
6. answer simple questions from the uploaded document.

Week 1 must be reliable, understandable, and demo-ready.

---

## 1) Project identity
**Project name:** SkillSync AI

**One-line description:**
SkillSync AI is a Resume-to-Job Description Match Copilot that analyzes uploaded documents, builds a searchable knowledge base, and answers questions from those documents using a retrieval-based chat workflow.

**Primary long-term vision:**
- Resume vs JD matching
- skill-gap analysis
- AI explanations
- resume improvement suggestions
- progress tracking
- exportable reports

**Week 1 scope:**
- document ingestion
- text extraction
- chunking
- knowledge base creation
- basic retrieval/Q&A chat

---

## 2) Week 1 goal in plain language
The app should let a user upload a resume or related document, then ask simple questions about it.

Example questions:
- What skills are mentioned?
- What is the candidate’s experience summary?
- Which tools and technologies are listed?
- What sections are present in the document?

The app does **not** need advanced matching, scoring, or rewriting in Week 1. Those are later phases.

---

## 3) Week 1 deliverables
The Week 1 build must include:

### Functional deliverables
- File upload interface
- Document parser
- Text cleaner
- Chunking logic
- Vector/knowledge-base storage
- Retrieval pipeline
- Basic chat UI
- Baseline question-answer flow

### Testing deliverables
- Verify document loading works
- Verify chunking works
- Verify retrieval works
- Verify answers come from the uploaded content
- Verify the app handles empty or invalid uploads gracefully

---

## 4) What must be built first
The correct order is:

1. **Upload file**
2. **Extract text**
3. **Clean text**
4. **Chunk text**
5. **Create embeddings**
6. **Store in Chroma**
7. **Ask questions**
8. **Retrieve relevant chunks**
9. **Generate a grounded answer**

Do not start with fancy UI, memory, agents, or report generation before the ingestion pipeline works.

---

## 5) Recommended Week 1 tech stack
This stack is chosen for speed, simplicity, and low cost.

### Frontend
- **Streamlit** for the UI

### Backend / app logic
- **Python**
- Optional: **FastAPI** later, but Week 1 can be Streamlit-first if needed

### Parsing libraries
- `pypdf` or `PyPDF2` for PDF text extraction
- `python-docx` for DOCX extraction

### Text processing
- Python standard library
- Optional: `re` for cleaning

### Chunking
- Simple recursive or fixed-size chunking
- Overlap supported

### Embeddings
- Local embeddings via `sentence-transformers`
- Recommended model: `all-MiniLM-L6-v2`

### Knowledge base / vector store
- **ChromaDB**

### Optional LLM for Q&A generation
- A free or low-cost model later, but Week 1 can start with retrieval-first answers

---

## 6) Why this architecture
The Week 1 system should be a **small RAG pipeline**:

User uploads file → parser extracts text → chunker splits text → embeddings are created → Chroma stores vectors → user asks a question → relevant chunks are retrieved → answer is generated

This architecture is preferred because:
- it is easy to debug,
- it is easy to demo,
- it is cheap,
- it is scalable later,
- it keeps the AI grounded in the document.

---

## 7) ChromaDB usage for Week 1
ChromaDB is the vector store used to save document chunks as embeddings.

### What Chroma stores
For each chunk, store:
- chunk text
- source file name
- chunk index
- optional metadata like page number

### What Chroma does
When a user asks a question, the app converts the question into an embedding, compares it with stored embeddings, and returns the most similar chunks.

### Week 1 approach
Use **local Chroma** first.

### Important deployment note
For Week 1 and demo purposes, treat Chroma as **rebuild-on-upload** storage unless persistent hosting is explicitly added later.

That means:
- upload document
- build Chroma index from that document
- answer questions within that session

This is simpler and much safer for deployment.

---

## 8) Scope boundaries
### In scope for Week 1
- PDF upload
- DOCX upload if time allows
- text extraction
- cleaning
- chunking
- Chroma storage
- simple question answering
- source chunk display

### Out of scope for Week 1
- resume scoring
- JD comparison logic
- skill-gap dashboard
- resume rewriting
- user accounts
- history tracking
- reports/PDF export
- multi-document memory
- agentic workflows
- advanced orchestration

---

## 9) Suggested app behavior
### Upload stage
User uploads a document.

### Parsing stage
The system extracts raw text.

### Cleaning stage
The system removes extra whitespace, repeated blank lines, and obvious noise.

### Chunking stage
The text is split into chunks that are small enough for retrieval.

### Knowledge base stage
Chunks are embedded and stored in Chroma.

### Q&A stage
User asks a question.
The app retrieves the top relevant chunks and generates an answer grounded in those chunks.

### Output stage
The app displays:
- answer
- top source chunks
- status messages
- error messages if any

---

## 10) Recommended chunking rules
Use chunking that is easy to explain and debug.

### Good default values
- chunk size: 500–1000 characters or an equivalent token-safe size
- overlap: 50–150 characters

### Chunking goals
- keep semantic meaning intact
- avoid chunks that are too large
- avoid too many tiny chunks
- preserve section boundaries when possible

### Good chunking examples
- section-based chunking
- paragraph-based chunking
- recursive fallback chunking

---

## 11) Retrieval and answer strategy
Week 1 should favor **grounded retrieval** over pure generation.

### Retrieval flow
1. convert query to embedding
2. search Chroma for top-k chunks
3. pass those chunks into a small answer prompt or response builder
4. return a concise answer

### Answer style
- short
- factual
- document-grounded
- no hallucinated content

### Important rule
If the answer is not in the document, the system should say so clearly.

---

## 12) UI requirements for Week 1
The UI should be minimal and clean.

### Required UI elements
- title/header: SkillSync AI
- upload file button
- document status message
- chunk count or document loaded indicator
- question input box
- answer display section
- source snippet display section

### Nice-to-have UI elements
- progress/status text
- clear/reset button
- expandable source chunks

### Avoid in Week 1
- heavy dashboards
- charts
- animations that distract from functionality
- complex styling that delays progress

---

## 13) Error handling requirements
The app must handle common failures gracefully.

### Expected error cases
- empty upload
- unsupported file type
- unreadable PDF
- missing extracted text
- no chunks generated
- query asked before loading a document
- retrieval returns no relevant match

### Required behavior
- show a human-readable message
- do not crash
- keep the app usable after error

---

## 14) Logging and debug visibility
Week 1 should include visible debug signals, such as:
- number of extracted characters
- number of chunks created
- whether embeddings were generated
- whether Chroma indexing succeeded
- top retrieved chunk count

This helps with testing and viva explanation.

---

## 15) Suggested file structure
A clean Week 1 structure may look like this:

```text
skillsync-ai/
├── app.py
├── requirements.txt
├── .env
├── README.md
├── data/
│   └── uploads/
├── src/
│   ├── parser.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   └── qa.py
└── chroma_db/
```

This is a suggested structure, not a strict requirement.

---

## 16) Recommended dependencies
Keep dependencies minimal in Week 1.

### Core
- streamlit
- chromadb
- sentence-transformers
- pypdf
- python-docx
 NOTE: (Skip this i have already installed these packages)
### Optional
- fastapi
- uvicorn
- langchain
- openai or google-generativeai only if you actually use an LLM in Week 1

Avoid unnecessary packages.

---

## 17) Testing plan for Week 1
Testing must focus on the ingestion and retrieval pipeline.

### Test 1: Document loading
- Upload a valid PDF
- Confirm text is extracted
- Confirm no crash

### Test 2: Chunking
- Confirm chunks are created
- Confirm chunk count is reasonable
- Confirm chunk contents are meaningful

### Test 3: Storage
- Confirm chunks are inserted into Chroma
- Confirm metadata is preserved

### Test 4: Retrieval
- Ask a question that is answerable from the file
- Confirm relevant chunk(s) are retrieved

### Test 5: Baseline Q&A
- Confirm the app produces a response
- Confirm response is grounded in the uploaded document

### Test 6: Negative cases
- blank file
- unsupported format
- question before upload
- question unrelated to document

---

## 18) Acceptance criteria for Week 1
Week 1 is complete only if all of the following are true:

- a document can be uploaded successfully
- the system extracts text
- the text is chunked properly
- chunks are stored in the vector database
- the user can ask a question
- the app answers using retrieved context
- the app shows meaningful errors when things fail
- the app is stable enough for a live demo

---

## 19) Quality standards
The implementation should be:
- simple
- readable
- modular
- testable
- explainable
- demo-ready

Prefer clarity over complexity.

---

## 20) Week 1 non-goals
Do not spend time on:
- advanced prompting
- chain-of-thought style orchestration
- memory across sessions
- multi-agent systems
- resume scoring logic
- deployment persistence optimization
- production security hardening beyond basics

Those are future steps.

---

## 21) Best-practice design principles
- Keep the pipeline deterministic where possible
- Use AI only where language generation is needed
- Ground responses in retrieved document chunks
- Save time by building one vertical slice end-to-end
- Make the app easy to explain in viva
- Keep dependencies minimal

---

## 22) Week 1 implementation mindset
Think in this order:

**Can the app ingest a document?**

**Can it extract usable text?**

**Can it split the document into meaningful chunks?**

**Can those chunks be searched semantically?**

**Can the app answer a basic question from the uploaded content?**

If the answer to all five is yes, Week 1 is successful.

---

## 23) Notes for future weeks
This Week 1 base should later support:
- JD analysis
- resume matching
- scoring
- missing skill extraction
- AI explanations
- rewriting suggestions
- history and reporting

Do not design Week 1 in a way that blocks these later features.

---

