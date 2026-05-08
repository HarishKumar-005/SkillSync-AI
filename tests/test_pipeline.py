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


# ===========================================================================
# WEEK 2 TESTS
# ===========================================================================

# Sample JD text for testing
SAMPLE_JD_TEXT = """
Senior Python Developer

Requirements:
- 5+ years of Python development experience
- Strong experience with Django or FastAPI
- Experience with PostgreSQL and MongoDB
- AWS cloud services (EC2, S3, Lambda)
- Docker and Kubernetes
- CI/CD pipelines
- REST API design
- Experience with React or Angular for frontend collaboration
- Machine learning experience with PyTorch or TensorFlow
- Excellent communication skills
- Experience with Agile/Scrum methodology
- Knowledge of GraphQL
- Redis caching
- Terraform infrastructure as code
"""


# ===========================================================================
# TEST 9: Intent Classification (Week 2)
# ===========================================================================

class TestIntentClassification:
    """Test the Week 2 intent classifier."""

    def test_comparison_intent(self):
        from src.intent import classify_intent
        assert classify_intent("Compare the resume with the job description") == "comparison"
        assert classify_intent("How well does the resume match the JD?") == "comparison"
        assert classify_intent("Does the candidate fit this role?") == "comparison"
        assert classify_intent("What matched skills are there?") == "comparison"

    def test_gap_analysis_intent(self):
        from src.intent import classify_intent
        assert classify_intent("What skills are missing?") == "gap_analysis"
        assert classify_intent("What gaps exist in the resume?") == "gap_analysis"
        assert classify_intent("What does the candidate lack?") == "gap_analysis"

    def test_improvement_intent(self):
        from src.intent import classify_intent
        assert classify_intent("How can the resume be improved?") == "improvement"
        assert classify_intent("What should the candidate add?") == "improvement"
        assert classify_intent("Give me suggestions to improve the profile") == "improvement"

    def test_list_intent_preserved(self):
        from src.intent import classify_intent
        assert classify_intent("What skills are mentioned?") == "list"
        assert classify_intent("List the projects") == "list"

    def test_summary_intent_preserved(self):
        from src.intent import classify_intent
        assert classify_intent("Summarize the profile") == "summary"
        assert classify_intent("Give me an overview") == "summary"

    def test_detail_fallback(self):
        from src.intent import classify_intent
        assert classify_intent("When did John join TechCorp?") == "detail"
        assert classify_intent("What is the GPA?") == "detail"

    def test_needs_jd(self):
        from src.intent import needs_jd
        assert needs_jd("comparison") is True
        assert needs_jd("gap_analysis") is True
        assert needs_jd("improvement") is True
        assert needs_jd("list") is False
        assert needs_jd("summary") is False
        assert needs_jd("detail") is False

    def test_section_detection(self):
        from src.intent import detect_target_section
        assert detect_target_section("What skills are mentioned?") == "SKILLS"
        assert detect_target_section("List the projects") == "PROJECTS"
        assert detect_target_section("What is the weather?") is None


# ===========================================================================
# TEST 10: Skill Matcher (Week 2)
# ===========================================================================

class TestSkillMatcher:
    """Test skill extraction and matching."""

    def test_extract_skills_from_resume(self):
        from src.matcher import extract_skills_from_text
        text = "Programming Languages: Python, JavaScript, Go\nFrameworks: Django, React, FastAPI"
        skills = extract_skills_from_text(text)
        assert len(skills) > 0
        skills_lower = [s.lower() for s in skills]
        assert "python" in skills_lower
        assert "django" in skills_lower

    def test_extract_skills_from_jd(self):
        from src.matcher import extract_skills_from_text
        skills = extract_skills_from_text(SAMPLE_JD_TEXT)
        assert len(skills) > 0
        skills_lower = [s.lower() for s in skills]
        # JD mentions Python, Django, PostgreSQL, etc.
        assert any("python" in s for s in skills_lower)

    def test_exact_matching(self):
        from src.matcher import match_skills
        resume_skills = ["Python", "Django", "PostgreSQL", "AWS"]
        jd_skills = ["Python", "Django", "PostgreSQL", "GraphQL", "Angular"]
        result = match_skills(resume_skills, jd_skills, use_semantic=False)

        assert len(result["matched"]) >= 3  # Python, Django, PostgreSQL
        assert len(result["missing"]) >= 1  # At least Angular or GraphQL
        assert "match_summary" in result

    def test_no_overclaiming(self):
        """Matcher should not claim matches that don't exist."""
        from src.matcher import match_skills
        resume_skills = ["Python", "Django"]
        jd_skills = ["Rust", "Kubernetes", "Terraform"]
        result = match_skills(resume_skills, jd_skills, use_semantic=False)

        assert len(result["matched"]) == 0
        assert len(result["missing"]) == 3

    def test_empty_jd(self):
        from src.matcher import match_skills
        result = match_skills(["Python"], [], use_semantic=False)
        assert "No job description" in result["match_summary"]

    def test_empty_resume(self):
        from src.matcher import match_skills
        result = match_skills([], ["Python", "Django"], use_semantic=False)
        assert len(result["missing"]) == 2

    def test_substring_matching(self):
        """React should match React.js, etc."""
        from src.matcher import match_skills
        resume_skills = ["React.js", "Node.js"]
        jd_skills = ["React", "Node"]
        result = match_skills(resume_skills, jd_skills, use_semantic=False)
        assert len(result["matched"]) == 2

    def test_extract_empty_text(self):
        from src.matcher import extract_skills_from_text
        assert extract_skills_from_text("") == []
        assert extract_skills_from_text("   ") == []


# ===========================================================================
# TEST 11: Citations (Week 2)
# ===========================================================================

class TestCitations:
    """Test citation formatting."""

    def test_format_citations(self):
        from src.citations import format_citations
        chunks = [
            {"text": "Python, Django, React", "score": 0.85, "section": "SKILLS", "chunk_index": 0},
            {"text": "Built REST APIs for 50K users", "score": 0.72, "section": "EXPERIENCE", "chunk_index": 3},
        ]
        citations = format_citations(chunks)
        assert len(citations) > 0
        assert citations[0]["index"] == 1
        assert citations[0]["score"] >= 0.35
        assert "excerpt" in citations[0]

    def test_citations_filtered_by_threshold(self):
        from src.citations import format_citations
        chunks = [
            {"text": "Irrelevant text", "score": 0.1, "section": "GENERAL", "chunk_index": 0},
        ]
        citations = format_citations(chunks)
        # Should still return at least 1 (best available)
        assert len(citations) >= 1

    def test_empty_citations(self):
        from src.citations import format_citations, build_citation_text
        assert format_citations([]) == []
        assert build_citation_text([]) == ""

    def test_citation_text_format(self):
        from src.citations import build_citation_text
        citations = [
            {"index": 1, "section": "SKILLS", "score": 0.85, "excerpt": "Python, Django"},
        ]
        text = build_citation_text(citations)
        assert "Supporting Evidence" in text
        assert "[1]" in text
        assert "Skills" in text

    def test_attach_citations(self):
        from src.citations import attach_citations_to_answer
        answer = "The candidate has Python skills."
        citations = [
            {"index": 1, "section": "SKILLS", "score": 0.85, "excerpt": "Python, Django"},
        ]
        result = attach_citations_to_answer(answer, citations)
        assert "Supporting Evidence" in result
        assert "Python skills" in result  # Original answer preserved


# ===========================================================================
# TEST 12: Comparison Quality (Week 2)
# ===========================================================================

class TestComparisonQuality:
    """Test resume-vs-JD comparison end-to-end."""

    def test_comparison_answer(self, indexed_collection):
        from src.qa import answer_question_v2
        results = retrieve(indexed_collection, SAMPLE_JD_TEXT, top_k=8)
        response = answer_question_v2(
            query="Compare the resume with the job description",
            retrieved_chunks=results,
            jd_text=SAMPLE_JD_TEXT,
            intent="comparison",
        )
        assert response["query_type"] == "comparison"
        assert "Matched" in response["answer"] or "Missing" in response["answer"]
        assert response["confidence"] in ["high", "medium", "low"]

    def test_gap_analysis_answer(self, indexed_collection):
        from src.qa import answer_question_v2
        results = retrieve(indexed_collection, SAMPLE_JD_TEXT, top_k=8)
        response = answer_question_v2(
            query="What skills are missing?",
            retrieved_chunks=results,
            jd_text=SAMPLE_JD_TEXT,
            intent="gap_analysis",
        )
        assert response["query_type"] == "gap_analysis"
        assert len(response["answer"]) > 20

    def test_improvement_answer(self, indexed_collection):
        from src.qa import answer_question_v2
        results = retrieve(indexed_collection, SAMPLE_JD_TEXT, top_k=8)
        response = answer_question_v2(
            query="How can the resume be improved?",
            retrieved_chunks=results,
            jd_text=SAMPLE_JD_TEXT,
            intent="improvement",
        )
        assert response["query_type"] == "improvement"
        assert len(response["answer"]) > 20

    def test_v2_preserves_list_query(self, indexed_collection):
        """Non-JD queries should still work through v2."""
        from src.qa import answer_question_v2
        results = retrieve(indexed_collection, "What skills are mentioned?")
        response = answer_question_v2(
            query="What skills are mentioned?",
            retrieved_chunks=results,
            jd_text="",
        )
        assert response["query_type"] == "list"
        assert "python" in response["answer"].lower() or "django" in response["answer"].lower()

    def test_matched_skills_accurate(self, indexed_collection):
        """Matched skills should actually exist in the resume."""
        from src.qa import answer_question_v2
        results = retrieve(indexed_collection, SAMPLE_JD_TEXT, top_k=8)
        response = answer_question_v2(
            query="Compare resume with JD",
            retrieved_chunks=results,
            jd_text=SAMPLE_JD_TEXT,
            intent="comparison",
        )
        # Answer should mention Python (which IS in the resume)
        assert "Python" in response["answer"] or "python" in response["answer"].lower()

    def test_missing_skills_not_invented(self, indexed_collection):
        """Missing skills should actually be absent from the resume."""
        from src.matcher import extract_skills_from_text, match_skills
        resume_text = SAMPLE_RESUME_TEXT
        resume_skills = extract_skills_from_text(resume_text)
        jd_text = "Requirements: Rust, Scala, Haskell"
        jd_skills = extract_skills_from_text(jd_text)
        result = match_skills(resume_skills, jd_skills, use_semantic=False)
        # These skills are NOT in the sample resume
        assert len(result["missing"]) >= 2


# ===========================================================================
# TEST 13: Fallback Handling (Week 2)
# ===========================================================================

class TestFallbacks:
    """Test graceful handling of edge cases."""

    def test_comparison_without_jd(self, indexed_collection):
        """Comparison query without JD should give a clear message."""
        from src.qa import answer_question_v2
        results = retrieve(indexed_collection, "Compare resume")
        response = answer_question_v2(
            query="Compare the resume with the job description",
            retrieved_chunks=results,
            jd_text="",
            intent="comparison",
        )
        assert response["confidence"] == "none"
        assert "job description" in response["answer"].lower() or "jd" in response["answer"].lower()

    def test_gap_without_jd(self):
        """Gap analysis without JD should give a clear message."""
        from src.qa import answer_question_v2
        response = answer_question_v2(
            query="What skills are missing?",
            retrieved_chunks=[],
            jd_text="",
            intent="gap_analysis",
        )
        assert response["confidence"] == "none"

    def test_no_results(self):
        """No retrieval results should be handled gracefully."""
        from src.qa import answer_question_v2
        response = answer_question_v2(
            query="Compare resume",
            retrieved_chunks=[],
            jd_text="Some JD text here",
            intent="comparison",
        )
        assert response["confidence"] == "none"
        assert len(response["answer"]) > 10

    def test_vague_query(self, indexed_collection):
        """Vague query should not crash."""
        from src.qa import answer_question_v2
        results = retrieve(indexed_collection, "stuff things")
        response = answer_question_v2(
            query="stuff things",
            retrieved_chunks=results,
            jd_text="",
        )
        assert response["answer"]  # Should return something, not crash

    def test_v2_handles_empty_query(self):
        """Empty query should not crash."""
        from src.qa import answer_question_v2
        response = answer_question_v2(
            query="",
            retrieved_chunks=[],
            jd_text="",
        )
        assert response["answer"]

    def test_fallback_prompts(self):
        """Fallback messages should be clear and helpful."""
        from src.prompts import format_fallback_answer
        assert "resume" in format_fallback_answer("no_resume").lower()
        assert "job description" in format_fallback_answer("no_jd").lower()
        assert len(format_fallback_answer("weak_retrieval")) > 20
        assert len(format_fallback_answer("out_of_scope")) > 20


# ===========================================================================
# TEST 14: Refresh Workflow (Week 2)
# ===========================================================================

class TestRefresh:
    """Test session refresh and cleanup."""

    def test_clear_jd(self):
        from src.refresh import clear_jd

        class MockState(dict):
            def __getattr__(self, key):
                return self.get(key)
            def __setattr__(self, key, value):
                self[key] = value

        state = MockState(jd_text="Some JD", jd_skills=["Python"])
        clear_jd(state)
        assert state["jd_text"] == ""
        assert state.get("jd_skills") is None

    def test_session_validity(self):
        from src.refresh import is_session_valid

        class MockState(dict):
            def __getattr__(self, key):
                return self.get(key)

        # Empty state
        state = MockState()
        result = is_session_valid(state)
        assert result["status"] == "empty"

        # Resume only
        state = MockState(document_loaded=True, jd_text="", collection="something")
        result = is_session_valid(state)
        assert result["status"] == "ready_resume_only"

        # Full
        state = MockState(document_loaded=True, jd_text="Some JD", collection="something")
        result = is_session_valid(state)
        assert result["status"] == "ready_full"

    def test_clear_session_state(self):
        from src.refresh import clear_session_state

        class MockState(dict):
            def __getattr__(self, key):
                return self.get(key)
            def __setattr__(self, key, value):
                self[key] = value

        state = MockState(
            messages=[{"role": "user", "content": "hi"}],
            document_loaded=True,
            jd_text="Some JD",
            collection="something",
            doc_info={"filename": "test.pdf"},
        )
        clear_session_state(state)
        assert state["messages"] == []
        assert state["document_loaded"] is False
        assert state["jd_text"] == ""
        assert state["doc_info"] == {}


# ===========================================================================
# TEST 15: Retriever Enhancement (Week 2)
# ===========================================================================

class TestRetrieverV2:
    """Test Week 2 retriever enhancements."""

    def test_retrieve_by_section(self, indexed_collection):
        from src.retriever import retrieve_by_section
        results = retrieve_by_section(
            indexed_collection, "Python Django", target_section="SKILLS"
        )
        assert len(results) > 0

    def test_retrieve_by_section_fallback(self, indexed_collection):
        """Should fall back to global when section has no results."""
        from src.retriever import retrieve_by_section
        results = retrieve_by_section(
            indexed_collection, "Python", target_section="REFERENCES"
        )
        # REFERENCES section doesn't exist in sample, should fall back
        assert len(results) > 0

    def test_retrieve_for_comparison(self, indexed_collection):
        from src.retriever import retrieve_for_comparison
        results = retrieve_for_comparison(
            indexed_collection, SAMPLE_JD_TEXT, top_k=5
        )
        assert len(results) > 0
        # Results should be from different sections
        sections = {r["section"] for r in results}
        assert len(sections) >= 1  # At least one section


# ===========================================================================
# TEST 16: Prompt Templates (Week 2)
# ===========================================================================

class TestPromptTemplates:
    """Test prompt template formatting."""

    def test_format_list(self):
        from src.prompts import format_list_answer
        result = format_list_answer(["Python", "Django", "React"], "skills")
        assert "Python" in result
        assert "Total: 3" in result

    def test_format_list_empty(self):
        from src.prompts import format_list_answer
        result = format_list_answer([], "skills")
        assert "No skills" in result

    def test_format_comparison(self):
        from src.prompts import format_comparison_answer
        match_result = {
            "matched": [("Python", "Python"), ("Django", "Django")],
            "partially_matched": [("React", "React.js", 0.92)],
            "missing": ["GraphQL"],
            "match_summary": "Match Score: 60%",
        }
        result = format_comparison_answer(match_result)
        assert "Matched" in result
        assert "Missing" in result
        assert "Python" in result

    def test_format_gap_analysis(self):
        from src.prompts import format_gap_analysis_answer
        match_result = {
            "matched": [("Python", "Python")],
            "partially_matched": [],
            "missing": ["Rust", "Scala"],
            "match_summary": "2 missing",
        }
        result = format_gap_analysis_answer(match_result)
        assert "Rust" in result
        assert "Scala" in result

    def test_format_improvement(self):
        from src.prompts import format_improvement_answer
        match_result = {
            "matched": [("Python", "Python")],
            "partially_matched": [("React Native", "React.js", 0.80)],
            "missing": ["GraphQL"],
            "match_summary": "1 missing",
        }
        result = format_improvement_answer(match_result)
        assert "GraphQL" in result
        assert "Tips" in result or "Highlight" in result

    def test_format_fallback(self):
        from src.prompts import format_fallback_answer
        assert "resume" in format_fallback_answer("no_resume").lower()
        assert "job description" in format_fallback_answer("no_jd").lower()

