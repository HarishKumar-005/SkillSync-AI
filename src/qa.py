"""
Question-Answer Module — Synthesizes grounded answers from retrieved chunks.

Approach:
  1. Detect query type (list, summary, detail, yes/no)
  2. Extract structured information from relevant chunks
  3. Generate a clean FINAL ANSWER (not raw excerpts)
  4. Return supporting evidence separately

No external LLM — all answer synthesis is done by intelligent text extraction.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Similarity threshold below which results are considered irrelevant
RELEVANCE_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Query type detection
# ---------------------------------------------------------------------------

# Patterns for each query type
_LIST_PATTERNS = [
    r'(?i)\bwhat\s+(?:are|were)\s+(?:the\s+)?(?:skills?|tools?|technologies?|tech\s+stack)',
    r'(?i)\bwhat\s+(?:skills?|tools?|technologies?|projects?|certific)',
    r'(?i)\blist\s+(?:the\s+)?(?:skills?|projects?|certific|tools?|technologies?)',
    r'(?i)\bwhich\s+(?:skills?|tools?|technologies?|projects?|certific|languages?)',
    r'(?i)\bwhat\s+(?:projects?|certific)',
    r'(?i)\b(?:skills?|projects?|certific|tools?|technologies?)\s+(?:are|were)\s+(?:mentioned|listed|included)',
    r'(?i)\bname\s+(?:the\s+)?(?:skills?|projects?|certific)',
]

_SUMMARY_PATTERNS = [
    r'(?i)\bsummar(?:y|ize|ise)\b',
    r'(?i)\boverview\b',
    r'(?i)\btell\s+me\s+about\b',
    r'(?i)\bdescribe\b',
    r'(?i)\bwho\s+is\b',
    r'(?i)\bwhat\s+(?:is|does)\s+(?:the|this)\s+(?:document|resume|profile|candidate)',
    r'(?i)\bbrief(?:ly)?\b',
    r'(?i)\bin\s+(?:two|2|few|short)\s+(?:lines?|sentences?|words?)',
]

_SECTION_KEYWORDS = {
    "skills": "SKILLS",
    "skill": "SKILLS",
    "tools": "SKILLS",
    "technologies": "SKILLS",
    "tech stack": "SKILLS",
    "programming": "SKILLS",
    "languages": "SKILLS",
    "projects": "PROJECTS",
    "project": "PROJECTS",
    "certificates": "CERTIFICATES",
    "certifications": "CERTIFICATES",
    "certificate": "CERTIFICATES",
    "certification": "CERTIFICATES",
    "experience": "EXPERIENCE",
    "work": "EXPERIENCE",
    "education": "EDUCATION",
    "profile": "PROFILE",
    "summary": "PROFILE",
    "contact": "CONTACT",
    "achievements": "ACHIEVEMENTS",
    "awards": "ACHIEVEMENTS",
}


def _detect_query_type(query: str) -> str:
    """
    Classify query into: 'list', 'summary', or 'detail'.
    """
    for pattern in _LIST_PATTERNS:
        if re.search(pattern, query):
            return "list"
    for pattern in _SUMMARY_PATTERNS:
        if re.search(pattern, query):
            return "summary"
    return "detail"


def _detect_target_section(query: str) -> str | None:
    """
    Detect which document section the query is asking about.

    Returns:
        Section label string (e.g., 'SKILLS') or None.
    """
    query_lower = query.lower()
    for keyword, section in _SECTION_KEYWORDS.items():
        if keyword in query_lower:
            return section
    return None


# ---------------------------------------------------------------------------
# Information extraction from chunks
# ---------------------------------------------------------------------------

def _extract_list_items(text: str) -> list[str]:
    """
    Extract individual items from a chunk of text.
    Handles comma-separated lists, bullet points, and line-separated items.
    """
    items = []

    # Split by common list separators
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove bullet-point prefixes
        line = re.sub(r'^[\-\•\*\▪\►\➤\→\‣\⁃]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)

        # If line contains a colon (like "Languages: Python, Java, C++"), split after colon
        if ':' in line:
            parts = line.split(':', 1)
            label = parts[0].strip()
            values = parts[1].strip()
            if values:
                # Split comma-separated values
                sub_items = [v.strip() for v in re.split(r'[,;|]', values) if v.strip()]
                if sub_items:
                    items.extend(sub_items)
                    continue

        # If line has commas, split by commas
        if ',' in line:
            sub_items = [v.strip() for v in line.split(',') if v.strip()]
            # Only treat as list if items are reasonably short (< 80 chars each)
            if all(len(item) < 80 for item in sub_items):
                items.extend(sub_items)
                continue

        # Otherwise add the whole line as one item
        if len(line) < 200 and line:
            items.append(line)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in items:
        item_clean = item.strip().rstrip('.')
        if item_clean and item_clean.lower() not in seen:
            seen.add(item_clean.lower())
            unique.append(item_clean)

    return unique


def _build_list_answer(query: str, relevant_chunks: list[dict], target_section: str | None) -> str:
    """
    Build a clean list answer by extracting items from relevant chunks.
    """
    # Prioritize chunks from the target section
    if target_section:
        section_chunks = [c for c in relevant_chunks if c.get("section") == target_section]
        other_chunks = [c for c in relevant_chunks if c.get("section") != target_section]
        ordered = section_chunks + other_chunks
    else:
        ordered = relevant_chunks

    # Extract items from the most relevant chunks
    all_items = []
    for chunk in ordered[:3]:
        items = _extract_list_items(chunk["text"])
        all_items.extend(items)

    # Deduplicate
    seen = set()
    unique_items = []
    for item in all_items:
        key = item.lower().strip()
        if key not in seen and len(item) > 1:
            seen.add(key)
            unique_items.append(item)

    if not unique_items:
        # Fall back to presenting the chunk text directly
        return _build_detail_answer(query, relevant_chunks)

    # Determine the subject from the query
    subject = "items"
    query_lower = query.lower()
    if "skill" in query_lower or "tool" in query_lower or "technolog" in query_lower:
        subject = "skills/technologies"
    elif "project" in query_lower:
        subject = "projects"
    elif "certif" in query_lower:
        subject = "certifications"
    elif "experience" in query_lower or "work" in query_lower:
        subject = "work experience entries"
    elif "education" in query_lower:
        subject = "education entries"

    # Format the answer
    answer_lines = [f"The document mentions the following **{subject}**:\n"]
    for item in unique_items:
        answer_lines.append(f"- {item}")

    answer_lines.append(f"\n*Total: {len(unique_items)} {subject} found.*")
    return "\n".join(answer_lines)


def _build_summary_answer(query: str, relevant_chunks: list[dict]) -> str:
    """
    Build a concise summary from the most relevant chunks.
    """
    # Prioritize PROFILE/GENERAL chunks for summaries
    profile_chunks = [c for c in relevant_chunks if c.get("section") in ("PROFILE", "GENERAL")]
    experience_chunks = [c for c in relevant_chunks if c.get("section") == "EXPERIENCE"]
    skills_chunks = [c for c in relevant_chunks if c.get("section") == "SKILLS"]

    # Collect key text snippets
    summary_parts = []

    # From profile
    for chunk in profile_chunks[:1]:
        text = chunk["text"].strip()
        # Take the first 2-3 meaningful sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        meaningful = [s for s in sentences if len(s) > 20]
        if meaningful:
            summary_parts.append(' '.join(meaningful[:3]))

    # From experience (brief)
    for chunk in experience_chunks[:1]:
        text = chunk["text"].strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        meaningful = [s for s in sentences if len(s) > 20]
        if meaningful:
            summary_parts.append(meaningful[0])

    # From skills (brief)
    for chunk in skills_chunks[:1]:
        items = _extract_list_items(chunk["text"])
        if items:
            top_items = items[:6]
            summary_parts.append(f"Key skills include: {', '.join(top_items)}.")

    if summary_parts:
        return "**Profile Summary:**\n\n" + ' '.join(summary_parts)

    # Fallback: just use top chunks
    return _build_detail_answer(query, relevant_chunks)


def _build_detail_answer(query: str, relevant_chunks: list[dict]) -> str:
    """
    Build a detailed answer from the most relevant chunks.
    """
    if not relevant_chunks:
        return "I couldn't find specific information about that in the document."

    # Combine the top relevant chunk texts into a coherent response
    top_chunk = relevant_chunks[0]
    text = top_chunk["text"].strip()

    # Truncate if very long
    if len(text) > 600:
        # Try to end at a sentence boundary
        sentences = re.split(r'(?<=[.!?])\s+', text[:600])
        if len(sentences) > 1:
            text = ' '.join(sentences[:-1])
        else:
            text = text[:600] + "..."

    section = top_chunk.get("section", "GENERAL")
    section_display = section.replace("_", " ").title()

    return f"**From the {section_display} section:**\n\n{text}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate a structured answer from retrieved document chunks.

    The response has two parts:
      1. A synthesized FINAL ANSWER (not raw excerpts)
      2. Supporting source chunks for transparency

    Args:
        query: The user's question.
        retrieved_chunks: List of chunk dicts from the retriever.

    Returns:
        Dict with keys:
          - answer: The synthesized final answer text
          - sources: List of source chunks used
          - confidence: 'high', 'medium', 'low', or 'none'
          - num_sources: Number of relevant sources found
          - query_type: Detected query type ('list', 'summary', 'detail')
    """
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find any relevant information. Please make sure a document has been uploaded and try rephrasing your question.",
            "sources": [],
            "confidence": "none",
            "num_sources": 0,
            "query_type": "unknown",
        }

    # Detect query type and target section
    query_type = _detect_query_type(query)
    target_section = _detect_target_section(query)

    # Filter chunks by relevance threshold
    relevant_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk.get("score", 0) >= RELEVANCE_THRESHOLD
    ]

    # If a target section is detected, boost chunks from that section
    if target_section and relevant_chunks:
        section_matches = [c for c in relevant_chunks if c.get("section") == target_section]
        other_matches = [c for c in relevant_chunks if c.get("section") != target_section]
        relevant_chunks = section_matches + other_matches

    if not relevant_chunks:
        return {
            "answer": "I couldn't find information closely related to your question in the uploaded document. Try asking something more specific about the document's content.",
            "sources": [retrieved_chunks[0]] if retrieved_chunks else [],
            "confidence": "low",
            "num_sources": 0,
            "query_type": query_type,
        }

    # Determine confidence based on top score
    top_score = relevant_chunks[0]["score"]
    if top_score >= 0.7:
        confidence = "high"
    elif top_score >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"

    # Generate the answer based on query type
    if query_type == "list":
        answer = _build_list_answer(query, relevant_chunks, target_section)
    elif query_type == "summary":
        answer = _build_summary_answer(query, relevant_chunks)
    else:
        answer = _build_detail_answer(query, relevant_chunks)

    logger.info(
        "Generated %s-confidence %s answer from %d relevant chunks (top score: %.4f)",
        confidence, query_type, len(relevant_chunks), top_score,
    )

    return {
        "answer": answer,
        "sources": relevant_chunks[:5],
        "confidence": confidence,
        "num_sources": len(relevant_chunks),
        "query_type": query_type,
    }
