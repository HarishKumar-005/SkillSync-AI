# SkillSync AI

**Resume-to-Job Description Match Copilot** — Analyze uploaded documents, build a searchable knowledge base, and answer questions from those documents using a retrieval-based chat workflow.

## Week 1 — Document Q&A Foundation

### Features
- 📄 **Document Upload**: PDF and DOCX file support
- 🔍 **Text Extraction**: Automatic parsing with pypdf and python-docx
- 🧩 **Smart Chunking**: Recursive text splitting with paragraph/sentence awareness
- 🧠 **Semantic Search**: Vector embeddings via sentence-transformers (all-MiniLM-L6-v2)
- 💾 **Knowledge Base**: ChromaDB vector storage with cosine similarity
- 💬 **Chat Interface**: Ask questions and get document-grounded answers
- 📊 **Debug Visibility**: Character counts, chunk counts, indexing status

### Architecture
```
Upload → Parse → Clean → Chunk → Embed → Store → Query → Retrieve → Answer 
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment (`.venv` already set up)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Run Tests
```bash
python -m pytest tests/test_pipeline.py -v
```

---

## How to Use

1. **Upload** a PDF or DOCX document using the sidebar
2. **Wait** for the processing pipeline to complete (you'll see status indicators)
3. **Ask** questions about the document in the chat input
4. **View** answers with source chunks and confidence scores

### Example Questions
- *What skills are mentioned?*
- *What is the candidate's experience summary?*
- *Which tools and technologies are listed?*
- *What certifications does the candidate have?*

---

## Project Structure
```
SkillSync AI/
├── app.py                  # Streamlit main app
├── requirements.txt        # Dependencies
├── .env                    # Environment variables (future use)
├── README.md               # This file
├── CONTEXT.md              # Project source of truth
├── data/uploads/           # Temporary upload storage
├── src/
│   ├── __init__.py         # Package init
│   ├── parser.py           # PDF/DOCX text extraction
│   ├── cleaner.py          # Text normalization
│   ├── chunker.py          # Recursive chunking with overlap
│   ├── embeddings.py       # Sentence-transformers wrapper
│   ├── vectorstore.py      # ChromaDB operations
│   ├── retriever.py        # Semantic search pipeline
│   └── qa.py               # Answer formulation
└── tests/
    └── test_pipeline.py    # Automated pipeline tests
```

---

## Tech Stack
| Component | Technology |
|-----------|-----------|
| UI | Streamlit 1.57.0 |
| Vector DB | ChromaDB 1.5.8 |
| Embeddings | sentence-transformers 5.4.1 (all-MiniLM-L6-v2) |
| PDF Parsing | pypdf 6.10.2 |
| DOCX Parsing | python-docx 1.2.0 |

---

## Known Limitations (Week 1)
- Single document per session (no multi-document memory)
- No external LLM — answers are retrieval-based excerpts, not generated prose
- Scanned/image-only PDFs cannot be parsed (no OCR)
- No persistent storage — knowledge base is rebuilt on each upload
- No user accounts or session history

---

## Future Weeks
- **Week 2**: JD analysis, resume-JD matching, skill-gap identification
- **Week 3**: AI explanations, resume improvement suggestions, exportable reports
