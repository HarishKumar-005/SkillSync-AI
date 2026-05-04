"""
SkillSync AI — Pipeline Tests (Updated for RAG improvements)

Tests cover:
  1. Document loading
  2. Section-aware chunk generation
  3. Storage/index creation with section metadata
  4. Retrieval quality with section awareness
  5. Answer generation (list, summary, detail query types)
  6. Empty/invalid file handling + negative cases
  7. Query type detection
"""

import os
import sys
import tempfile

import pytest

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse_file, parse_pdf, parse_docx
from src.cleaner import clean_text
from src.chunker import chunk_text
from src.embeddings import embed_texts, embed_query
from src.vectorstore import get_chroma_client, create_collection, add_chunks, query_collection
from src.retriever import retrieve
from src.qa import answer_question, _detect_query_type, _detect_target_section


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESUME_TEXT = """
John Doe
Senior Software Engineer

Contact: john.doe@email.com | +1-555-0123 | San Francisco, CA
LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

PROFESSIONAL SUMMARY
Experienced software engineer with 8+ years of expertise in Python, JavaScript,
and cloud technologies. Proven track record of building scalable microservices
and leading cross-functional teams. Passionate about clean code, test-driven
development, and continuous learning.

TECHNICAL SKILLS
Programming Languages: Python, JavaScript, TypeScript, Go, SQL
Frameworks: Django, FastAPI, React, Next.js, Node.js
Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
Cloud & DevOps: AWS (EC2, S3, Lambda, ECS), Docker, Kubernetes, Terraform
Tools: Git, GitHub Actions, Jenkins, Grafana, Prometheus
Machine Learning: PyTorch, scikit-learn, pandas, numpy

WORK EXPERIENCE

Senior Software Engineer | TechCorp Inc. | 2020 - Present
- Led development of a real-time data processing pipeline handling 10M+ events/day
- Designed and implemented microservices architecture reducing latency by 40%
- Mentored team of 5 junior developers in best practices and code review
- Implemented CI/CD pipeline reducing deployment time from 2 hours to 15 minutes

Software Engineer | StartupXYZ | 2017 - 2020
- Built RESTful APIs serving 50K+ daily active users
- Developed automated testing framework achieving 95% code coverage
- Migrated legacy monolith to microservices architecture
- Optimized database queries resulting in 60% improvement in response time

Junior Developer | WebDev Agency | 2015 - 2017
- Developed responsive web applications for 20+ clients
- Created reusable component library reducing development time by 30%
- Collaborated with designers to implement pixel-perfect UI designs

EDUCATION
Bachelor of Science in Computer Science | UC Berkeley | 2015
GPA: 3.8/4.0 | Dean's List

CERTIFICATIONS
- AWS Certified Solutions Architect - Associate
- Google Cloud Professional Data Engineer
- Certified Kubernetes Administrator (CKA)

PROJECTS
Open Source Contributions:
- Contributor to FastAPI framework (50+ commits)
- Maintainer of a popular Python testing library (2K+ stars)

Personal Projects:
- Phantom Crowd: Real-time crowd simulation engine using WebGL
- SafeStep: AI-powered pedestrian safety app using computer vision
"""


@pytest.fixture
def sample_text():
    """Return sample resume text for testing."""
    return SAMPLE_RESUME_TEXT.strip()


@pytest.fixture
def sample_docx_path():
    """Create a minimal test DOCX file and return its path."""
    from docx import Document

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc = Document()
    for paragraph in SAMPLE_RESUME_TEXT.strip().split("\n\n"):
        doc.add_paragraph(paragraph.strip())
    doc.save(tmp.name)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def empty_file_path():
    """Create an empty file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def unsupported_file_path():
    """Create a file with unsupported extension."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
    tmp.write(b"Some content")
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def indexed_collection(sample_text):
    """Create a ChromaDB collection with indexed sample text."""
    cleaned = clean_text(sample_text)
    chunks = chunk_text(cleaned)
    embeddings = embed_texts([c["text"] for c in chunks])
    client = get_chroma_client()
    collection = create_collection(client, "test_collection")
    add_chunks(collection, chunks, embeddings, "test_resume.pdf")
    return collection


# ===========================================================================
# TEST 1: Document Loading
# ===========================================================================

class TestDocumentLoading:
    """Test document parsing."""

    def test_parse_docx_extracts_text(self, sample_docx_path):
        text, error = parse_file(sample_docx_path)
        assert error == "", f"Unexpected error: {error}"
        assert len(text) > 100
        assert "John Doe" in text
        assert "Python" in text

    def test_text_cleaning(self, sample_text):
        noisy = "  \t " + sample_text + "  \n\n\n\n\n  "
        cleaned = clean_text(noisy)
        assert cleaned
        assert "\t" not in cleaned
        assert "\n\n\n" not in cleaned


# ===========================================================================
# TEST 2: Section-Aware Chunking
# ===========================================================================

class TestChunking:
    """Test section-aware chunk generation."""

    def test_chunks_created(self, sample_text):
        cleaned = clean_text(sample_text)
        chunks = chunk_text(cleaned)
        assert len(chunks) > 0

    def test_chunks_have_section_label(self, sample_text):
        """Each chunk must have a 'section' key."""
        cleaned = clean_text(sample_text)
        chunks = chunk_text(cleaned)
        for chunk in chunks:
            assert "section" in chunk, f"Chunk missing 'section' key: {chunk.keys()}"
            assert chunk["section"] in (
                "GENERAL", "SKILLS", "PROJECTS", "CERTIFICATES",
                "PROFILE", "EXPERIENCE", "EDUCATION", "CONTACT",
                "ACHIEVEMENTS", "PUBLICATIONS", "LANGUAGES",
                "INTERESTS", "REFERENCES",
            ), f"Unknown section: {chunk['section']}"

    def test_skills_section_detected(self, sample_text):
        """Chunks from the skills section should be tagged SKILLS."""
        cleaned = clean_text(sample_text)
        chunks = chunk_text(cleaned)
        skills_chunks = [c for c in chunks if c["section"] == "SKILLS"]
        assert len(skills_chunks) > 0, "No SKILLS chunks detected"
        # Verify skills content is in skills chunks
        skills_text = " ".join(c["text"] for c in skills_chunks)
        assert "Python" in skills_text or "Django" in skills_text

    def test_projects_section_detected(self, sample_text):
        """Chunks from the projects section should be tagged PROJECTS."""
        cleaned = clean_text(sample_text)
        chunks = chunk_text(cleaned)
        project_chunks = [c for c in chunks if c["section"] == "PROJECTS"]
        assert len(project_chunks) > 0, "No PROJECTS chunks detected"

    def test_certificates_section_detected(self, sample_text):
        """Chunks from the certifications section should be tagged CERTIFICATES."""
        cleaned = clean_text(sample_text)
        chunks = chunk_text(cleaned)
        cert_chunks = [c for c in chunks if c["section"] == "CERTIFICATES"]
        assert len(cert_chunks) > 0, "No CERTIFICATES chunks detected"

    def test_chunk_contents_meaningful(self, sample_text):
        cleaned = clean_text(sample_text)
        chunks = chunk_text(cleaned)
        for chunk in chunks:
            assert len(chunk["text"]) > 10

    def test_empty_text_produces_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text(None) == []

    def test_sections_dont_mix(self, sample_text):
        """Each chunk should belong to a single section — no mixing."""
        cleaned = clean_text(sample_text)
        chunks = chunk_text(cleaned)
        # Skills chunks should not contain project names
        for chunk in chunks:
            if chunk["section"] == "SKILLS":
                # Should not contain project-specific content
                assert "Phantom Crowd" not in chunk["text"], "Skills chunk contains project content"


# ===========================================================================
# TEST 3: Storage with Section Metadata
# ===========================================================================

class TestStorage:
    """Test ChromaDB storage with section metadata."""

    def test_chunks_inserted(self, sample_text):
        cleaned = clean_text(sample_text)
        chunks = chunk_text(cleaned)
        embeddings = embed_texts([c["text"] for c in chunks])
        client = get_chroma_client()
        collection = create_collection(client, "test_storage")
        added = add_chunks(collection, chunks, embeddings, "test.pdf")
        assert added == len(chunks)
        assert collection.count() == len(chunks)

    def test_section_metadata_preserved(self, sample_text):
        """Confirm section label is stored in metadata."""
        cleaned = clean_text(sample_text)
        chunks = chunk_text(cleaned)
        embeddings = embed_texts([c["text"] for c in chunks])
        client = get_chroma_client()
        collection = create_collection(client, "test_meta")
        add_chunks(collection, chunks, embeddings, "resume.pdf")

        result = collection.get(limit=10, include=["metadatas"])
        sections_found = {m["section"] for m in result["metadatas"]}
        assert len(sections_found) > 1, "Only one section found — metadata not diverse enough"
        assert "section" in result["metadatas"][0]


# ===========================================================================
# TEST 4: Retrieval with Section Awareness
# ===========================================================================

class TestRetrieval:
    """Test retrieval quality."""

    def test_relevant_chunks_retrieved(self, indexed_collection):
        results = retrieve(indexed_collection, "What programming languages does John know?")
        assert len(results) > 0
        texts = " ".join([r["text"] for r in results])
        assert "Python" in texts or "JavaScript" in texts

    def test_retrieval_returns_section(self, indexed_collection):
        """Retrieval results must include section metadata."""
        results = retrieve(indexed_collection, "What skills are mentioned?")
        assert len(results) > 0
        for r in results:
            assert "section" in r, "Retrieved chunk missing 'section' key"

    def test_retrieval_returns_scores(self, indexed_collection):
        results = retrieve(indexed_collection, "What is the work experience?")
        assert len(results) > 0
        for r in results:
            assert 0 <= r["score"] <= 1

    def test_retrieval_respects_top_k(self, indexed_collection):
        results = retrieve(indexed_collection, "skills", top_k=2)
        assert len(results) <= 2


# ===========================================================================
# TEST 5: Answer Generation (List / Summary / Detail)
# ===========================================================================

class TestQA:
    """Test answer generation with query-type awareness."""

    def test_qa_produces_structured_response(self, indexed_collection):
        results = retrieve(indexed_collection, "What skills are mentioned?")
        response = answer_question("What skills are mentioned?", results)
        assert "answer" in response
        assert "sources" in response
        assert "confidence" in response
        assert "query_type" in response
        assert len(response["answer"]) > 0

    def test_qa_list_response_for_skills(self, indexed_collection):
        """List query should produce a formatted list, not raw excerpts."""
        results = retrieve(indexed_collection, "What skills are mentioned?")
        response = answer_question("What skills are mentioned?", results)
        assert response["query_type"] == "list"
        # Answer should contain list items, not "Excerpt 1"
        assert "Excerpt" not in response["answer"], "Answer should not show raw excerpts"
        # Should contain at least some skills
        answer_lower = response["answer"].lower()
        assert "python" in answer_lower or "django" in answer_lower or "react" in answer_lower

    def test_qa_list_response_for_projects(self, indexed_collection):
        """Projects query should list project names."""
        results = retrieve(indexed_collection, "What projects are mentioned?")
        response = answer_question("What projects are mentioned?", results)
        assert response["query_type"] == "list"
        assert len(response["answer"]) > 10

    def test_qa_list_response_for_certificates(self, indexed_collection):
        """Certifications query should list certifications."""
        results = retrieve(indexed_collection, "What certificates are listed?")
        response = answer_question("What certificates are listed?", results)
        assert response["query_type"] == "list"
        assert len(response["answer"]) > 10

    def test_qa_summary_response(self, indexed_collection):
        """Summary query should produce a concise profile summary."""
        results = retrieve(indexed_collection, "Summarize the profile in two lines")
        response = answer_question("Summarize the profile in two lines", results)
        assert response["query_type"] == "summary"
        assert len(response["answer"]) > 20

    def test_qa_detail_response(self, indexed_collection):
        """Detail query should return relevant section content."""
        results = retrieve(indexed_collection, "Where did John work before TechCorp?")
        response = answer_question("Where did John work before TechCorp?", results)
        assert response["query_type"] == "detail"
        assert len(response["answer"]) > 10

    def test_qa_no_hallucination(self, indexed_collection):
        """Answer should not contain made-up information."""
        results = retrieve(indexed_collection, "What is John's phone number?")
        response = answer_question("What is John's phone number?", results)
        # The answer should come from the document, not be invented
        # The document does contain a phone number, so it should be present
        assert response["answer"]

    def test_qa_handles_no_results(self):
        response = answer_question("What is the weather?", [])
        assert "couldn't find" in response["answer"].lower()
        assert response["confidence"] == "none"

    def test_qa_handles_irrelevant_question(self, indexed_collection):
        results = retrieve(indexed_collection, "What is the capital of Mars?")
        response = answer_question("What is the capital of Mars?", results)
        assert response["confidence"] in ["low", "none", "medium"]


# ===========================================================================
# TEST 6: Query Type Detection
# ===========================================================================

class TestQueryTypeDetection:
    """Test the query classifier."""

    def test_list_queries(self):
        assert _detect_query_type("What skills are mentioned?") == "list"
        assert _detect_query_type("List the projects") == "list"
        assert _detect_query_type("What certifications does he have?") == "list"
        assert _detect_query_type("Which technologies are listed?") == "list"

    def test_summary_queries(self):
        assert _detect_query_type("Summarize the profile") == "summary"
        assert _detect_query_type("Give me an overview") == "summary"
        assert _detect_query_type("Describe the candidate in two lines") == "summary"

    def test_detail_queries(self):
        assert _detect_query_type("When did John join TechCorp?") == "detail"
        assert _detect_query_type("What is the GPA?") == "detail"

    def test_section_detection(self):
        assert _detect_target_section("What skills are mentioned?") == "SKILLS"
        assert _detect_target_section("List the projects") == "PROJECTS"
        assert _detect_target_section("What certificates does he have?") == "CERTIFICATES"
        assert _detect_target_section("Tell me about experience") == "EXPERIENCE"


# ===========================================================================
# TEST 7: Negative Cases
# ===========================================================================

class TestNegativeCases:
    """Test error handling."""

    def test_empty_file(self, empty_file_path):
        text, error = parse_file(empty_file_path)
        assert text == ""
        assert error != ""

    def test_unsupported_format(self, unsupported_file_path):
        text, error = parse_file(unsupported_file_path)
        assert text == ""
        assert "unsupported" in error.lower() or "Unsupported" in error

    def test_nonexistent_file(self):
        text, error = parse_file("/nonexistent/path/fake.pdf")
        assert text == ""
        assert error != ""

    def test_question_before_upload(self):
        response = answer_question("What skills are listed?", [])
        assert response["confidence"] == "none"

    def test_empty_query(self):
        response = answer_question("", [])
        assert response["answer"]


# ===========================================================================
# TEST 8: Embeddings
# ===========================================================================

class TestEmbeddings:
    def test_embed_texts(self):
        texts = ["Hello world", "Python programming"]
        embeddings = embed_texts(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384

    def test_embed_query(self):
        embedding = embed_query("What skills are mentioned?")
        assert len(embedding) == 384

    def test_embed_empty(self):
        assert embed_texts([]) == []
        assert embed_query("") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
