"""
SkillSync AI — Week 1 Streamlit Application

A Resume-to-Job Description Match Copilot that analyzes uploaded documents,
builds a searchable knowledge base, and answers questions from those documents
using a retrieval-based chat workflow.

Week 1 scope: Document upload, parsing, chunking, ChromaDB storage, and basic Q&A.
"""

import os
import tempfile
import logging
import streamlit as st

from src.parser import parse_file
from src.cleaner import clean_text
from src.chunker import chunk_text
from src.embeddings import get_embedding_model, embed_texts
from src.vectorstore import get_chroma_client, create_collection, add_chunks
from src.retriever import retrieve
from src.qa import answer_question, _detect_target_section

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SkillSync AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Styles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* Header styling */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .app-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .app-header p {
        color: #8892b0;
        font-size: 1rem;
    }

    /* Status cards */
    .status-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .status-card .label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-card .value {
        color: #e2e8f0;
        font-size: 1.3rem;
        font-weight: 700;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* Chat styling */
    .source-chunk {
        background: #1e293b;
        border-left: 3px solid #667eea;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #cbd5e1;
    }

    /* Success/info badges */
    .badge-success {
        display: inline-block;
        background: #065f46;
        color: #6ee7b7;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-info {
        display: inline-block;
        background: #1e3a5f;
        color: #7dd3fc;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = None
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "doc_info" not in st.session_state:
    st.session_state.doc_info = {}


# ---------------------------------------------------------------------------
# Helper: Process uploaded document
# ---------------------------------------------------------------------------
def process_document(uploaded_file) -> tuple[bool, str]:
    """
    Full pipeline: parse → clean → chunk → embed → store.

    Returns:
        Tuple of (success: bool, message: str).
    """
    # Save uploaded file to a temp location
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        # Step 1: Parse
        raw_text, error = parse_file(tmp_path)
        if error:
            return False, f"❌ Parsing failed: {error}"

        # Step 2: Clean
        cleaned = clean_text(raw_text)
        if not cleaned:
            return False, "❌ No usable text after cleaning. The file may be empty or image-based."

        # Step 3: Chunk
        chunks = chunk_text(cleaned, chunk_size=800, overlap=150)
        if not chunks:
            return False, "❌ Chunking produced no chunks. The file content may be too short."

        # Step 4: Embed
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts)
        if not embeddings:
            return False, "❌ Embedding generation failed."

        # Step 5: Store in ChromaDB
        client = get_chroma_client()
        collection = create_collection(client, "skillsync_docs")
        added = add_chunks(collection, chunks, embeddings, uploaded_file.name)

        if added == 0:
            return False, "❌ Failed to store chunks in the knowledge base."

        # Save to session state
        st.session_state.chroma_client = client
        st.session_state.collection = collection
        st.session_state.document_loaded = True
        st.session_state.doc_info = {
            "filename": uploaded_file.name,
            "raw_chars": len(raw_text),
            "cleaned_chars": len(cleaned),
            "num_chunks": len(chunks),
            "num_indexed": added,
        }

        return True, f"✅ Document processed! {added} chunks indexed."

    except Exception as e:
        logger.exception("Document processing failed")
        return False, f"❌ Unexpected error: {str(e)}"
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 SkillSync AI")
    st.markdown("*Week 1 — Document Q&A*")
    st.divider()

    # File upload
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a resume or document",
        type=["pdf", "docx"],
        help="Supported formats: PDF, DOCX",
        key="file_uploader",
    )

    if uploaded_file is not None:
        # Check if this is a new file (different from what's already loaded)
        current_file = st.session_state.doc_info.get("filename", "")
        if current_file != uploaded_file.name or not st.session_state.document_loaded:
            with st.spinner("🔄 Processing document..."):
                success, message = process_document(uploaded_file)
            if success:
                st.success(message)
            else:
                st.error(message)

    st.divider()

    # Document status
    st.markdown("### 📊 Document Status")
    if st.session_state.document_loaded:
        info = st.session_state.doc_info
        st.markdown(f"""
        <div class="status-card">
            <div class="label">Loaded File</div>
            <div class="value" style="font-size: 0.95rem;">📎 {info.get('filename', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 Characters", f"{info.get('cleaned_chars', 0):,}")
        with col2:
            st.metric("🧩 Chunks", info.get('num_chunks', 0))

        st.markdown(f'<span class="badge-success">✓ Knowledge base ready</span>', unsafe_allow_html=True)
    else:
        st.info("No document loaded yet. Upload a PDF or DOCX to get started.")

    st.divider()

    # Reset button
    if st.button("🗑️ Clear & Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.collection = None
        st.session_state.chroma_client = None
        st.session_state.document_loaded = False
        st.session_state.doc_info = {}
        st.rerun()


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>🧠 SkillSync AI</h1>
    <p>Upload a document and ask questions about its content</p>
</div>
""", unsafe_allow_html=True)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🧠"):
        st.markdown(message["content"])

        # Show sources if present (for assistant messages)
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 View source chunks ({len(message['sources'])} found)", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    score_pct = f"{source.get('score', 0):.0%}"
                    st.markdown(
                        f'<div class="source-chunk">'
                        f'<strong>Chunk {source.get("chunk_index", i)}</strong> '
                        f'<span class="badge-info">Relevance: {score_pct}</span><br><br>'
                        f'{source.get("text", "")[:400]}{"..." if len(source.get("text", "")) > 400 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your document...", key="chat_input"):
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant", avatar="🧠"):
        if not st.session_state.document_loaded:
            response_text = "⚠️ No document has been uploaded yet. Please upload a PDF or DOCX file using the sidebar, then try again."
            st.markdown(response_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": [],
            })
        else:
            with st.spinner("🔍 Searching document..."):
                target_section = _detect_target_section(prompt)
                
                results = []
                if target_section:
                    # Stage 1: Targeted retrieval
                    results = retrieve(
                        collection=st.session_state.collection,
                        query=prompt,
                        top_k=5,
                        where_filter={"section": target_section}
                    )
                
                # Stage 2: Fallback to global retrieval if targeted yields nothing
                if not results:
                    results = retrieve(
                        collection=st.session_state.collection,
                        query=prompt,
                        top_k=5,
                    )

                # Generate answer
                response = answer_question(prompt, results)

            # --- Section 1: FINAL ANSWER ---
            st.markdown("**📌 Answer:**")
            st.markdown(response["answer"])

            # Display confidence + query type badges
            confidence = response["confidence"]
            query_type = response.get("query_type", "detail")
            conf_colors = {
                "high": ("🟢", "#065f46", "#6ee7b7"),
                "medium": ("🟡", "#78350f", "#fbbf24"),
                "low": ("🟠", "#7c2d12", "#fb923c"),
                "none": ("🔴", "#7f1d1d", "#fca5a5"),
            }
            icon, bg, fg = conf_colors.get(confidence, conf_colors["none"])
            st.markdown(
                f'<span style="background:{bg}; color:{fg}; padding:0.2rem 0.6rem; '
                f'border-radius:9999px; font-size:0.75rem; font-weight:600; margin-right:0.5rem;">'
                f'{icon} Confidence: {confidence}</span>'
                f'<span style="background:#1e3a5f; color:#7dd3fc; padding:0.2rem 0.6rem; '
                f'border-radius:9999px; font-size:0.75rem; font-weight:600;">'
                f'Query: {query_type}</span>',
                unsafe_allow_html=True,
            )

            # --- Section 2: SUPPORTING EVIDENCE ---
            if response["sources"]:
                with st.expander(f"📚 Supporting Evidence ({len(response['sources'])} chunks)", expanded=False):
                    for i, source in enumerate(response["sources"], 1):
                        score_pct = f"{source.get('score', 0):.0%}"
                        section_label = source.get("section", "GENERAL")
                        st.markdown(
                            f'<div class="source-chunk">'
                            f'<strong>Chunk {source.get("chunk_index", i)}</strong> '
                            f'<span class="badge-info">Relevance: {score_pct}</span> '
                            f'<span style="background:#312e81; color:#a5b4fc; padding:0.15rem 0.5rem; '
                            f'border-radius:9999px; font-size:0.7rem; font-weight:600;">'
                            f'📂 {section_label}</span><br><br>'
                            f'{source.get("text", "")[:400]}{"..." if len(source.get("text", "")) > 400 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"],
            })

# ---------------------------------------------------------------------------
# Footer (when no messages)
# ---------------------------------------------------------------------------
if not st.session_state.messages and not st.session_state.document_loaded:
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Step 1: Upload**
        Upload a resume or document (PDF/DOCX) using the sidebar.
        """)
    with col2:
        st.markdown("""
        **Step 2: Wait**
        The system will extract, chunk, and index your document automatically.
        """)
    with col3:
        st.markdown("""
        **Step 3: Ask**
        Ask questions about the document content in the chat below.
        """)

    st.markdown("---")
    st.markdown("#### 💡 Example Questions")
    example_cols = st.columns(2)
    with example_cols[0]:
        st.markdown("- *What skills are mentioned?*")
        st.markdown("- *What is the candidate's experience summary?*")
    with example_cols[1]:
        st.markdown("- *Which tools and technologies are listed?*")
        st.markdown("- *What sections are present in the document?*")
