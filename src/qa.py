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
    r'(?i)\b(?:what|list|name|show|which)\b.*?\b(?:skills?|tools?|technologies?|projects?|certificat(?:es?|ions?)|languages?|education|experience|achievements?|awards?)\b',
    r'(?i)\b(?:skills?|projects?|certificat(?:es?|ions?)|tools?|technologies?)\s+(?:are|were)\b',
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
    "languages": "LANGUAGES",
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
    "interests": "INTERESTS",
    "hobbies": "INTERESTS",
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

def _extract_list_items(text: str, target_section: str | None = None) -> list[str]:
    """
    Extract individual items from a chunk of text.
    Handles comma-separated lists, bullet points, and line-separated items.
    Aware of the target section (e.g., skips splitting by comma for PROJECTS).
    """
    items = []
    # Common resume headers to ignore if they sneak into the chunk
    ignore_headers = {"projects", "technical skills", "skills", "certifications", "certificates", 
                      "education", "work experience", "experience", "profile", "summary", 
                      "languages", "interests", "hobbies", "professional skills", "soft skills", 
                      "hard skills", "tools", "frameworks", "databases", "cloud & big data", "tools & ides"}

    raw_lines = text.split('\n')
    
    # Pre-processing: split inline bullets
    split_lines = []
    for line in raw_lines:
        line = line.strip()
        bullets = re.findall(r'[\•\*\▪\►\➤\→\‣\⁃]', line)
        if len(bullets) > 1:
            parts = re.split(r'[\•\*\▪\►\➤\→\‣\⁃]', line)
            split_lines.extend([p.strip() for p in parts if p.strip()])
        else:
            split_lines.append(line)

    # Education block merging
    processed_lines = []
    if target_section == "EDUCATION":
        i = 0
        while i < len(split_lines):
            line = split_lines[i]
            if not line:
                i += 1
                continue
            
            if line.upper() == "EDUCATION":
                processed_lines.append(line)
                i += 1
                continue

            has_bullet_curr = bool(re.match(r'^[\-\•\*\▪\►\➤\→\‣\⁃]|\d+[\.\)]', line))
            if not has_bullet_curr and i + 1 < len(split_lines):
                next_line = split_lines[i+1]
                if next_line and next_line.upper() != "EDUCATION":
                    has_bullet_next = bool(re.match(r'^[\-\•\*\▪\►\➤\→\‣\⁃]|\d+[\.\)]', next_line))
                    if not has_bullet_next and len(line) < 100 and len(next_line) < 100:
                        line = f"{line} - {next_line}"
                        i += 1
            processed_lines.append(line)
            i += 1
    else:
        processed_lines = split_lines

    items = []
    for line in processed_lines:
        line = line.strip()
        if not line:
            continue

        # Ignore obvious section headers
        if len(line) < 40 and line.lower() in ignore_headers:
            continue
            
        if target_section and line.upper() == target_section:
            continue
            
        # Ignore contact-like lines (emails, urls) if we are not in CONTACT
        if target_section != "CONTACT" and ("@" in line or "github.com" in line.lower() or "linkedin.com" in line.lower()):
            continue

        # Remove bullet-point prefixes
        clean_line = re.sub(r'^[\-\•\*\▪\►\➤\→\‣\⁃]\s*', '', line)
        clean_line = re.sub(r'^\d+[\.\)]\s*', '', clean_line)

        is_comma_split_section = target_section in ("SKILLS", "LANGUAGES")

        # For skills/languages, if line contains a colon, split after colon
        if target_section in ("SKILLS", "LANGUAGES") and ':' in clean_line:
            parts = clean_line.split(':', 1)
            values = parts[1].strip()
            if values:
                # Split comma-separated values
                sub_items = [v.strip() for v in re.split(r'[,;|]', values) if v.strip()]
                if sub_items:
                    items.extend(sub_items)
                    continue

        # For comma-split sections, if line has commas, split by commas
        if is_comma_split_section and ',' in clean_line:
            sub_items = [v.strip() for v in clean_line.split(',') if v.strip()]
            if all(len(item) < 80 for item in sub_items):
                items.extend(sub_items)
                continue

        # For projects/certificates/etc, we treat the whole line as one item
        if len(clean_line) > 2 and len(clean_line) < 150:
            if target_section in ("PROJECTS", "CERTIFICATES", "EDUCATION", "EXPERIENCE", "ACHIEVEMENTS"):
                
                # Project Noise Reduction
                if target_section == "PROJECTS":
                    cl_lower = clean_line.lower()
                    if cl_lower.startswith("tech stack") or cl_lower.startswith("technologies") or cl_lower.startswith("tools"):
                        continue

                has_bullet = bool(re.match(r'^[\-\•\*\▪\►\➤\→\‣\⁃]|\d+[\.\)]', line))
                has_separator = bool(re.search(r'\s+[\-\–\—\|]\s+', clean_line))
                is_short_title = len(clean_line) < 80 and not clean_line.endswith('.')
                
                if has_bullet or has_separator or is_short_title:
                    items.append(clean_line)
            else:
                items.append(clean_line)

    # Normalize: Split combined items like "AWS & Hadoop" -> "AWS", "Hadoop"
    final_items = []
    for item in items:
        if target_section in ("SKILLS", "LANGUAGES"):
            if '&' in item or ' and ' in item.lower():
                parts = re.split(r'\s+&\s+|\s+and\s+', item, flags=re.IGNORECASE)
                final_items.extend([p.strip() for p in parts if p.strip()])
            else:
                final_items.append(item)
        else:
            final_items.append(item)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in final_items:
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
        if not section_chunks:
            return f"I couldn't find a dedicated {target_section.replace('_', ' ').title()} section in the document."
        ordered = section_chunks
    else:
        ordered = relevant_chunks

    # Extract items from the most relevant chunks
    all_items = []
    for chunk in ordered[:3]:
        items = _extract_list_items(chunk["text"], target_section=target_section)
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
    if target_section == "PROJECTS" or "project" in query_lower:
        subject = "projects"
    elif target_section == "CERTIFICATES" or "certif" in query_lower:
        subject = "certifications"
    elif target_section == "SKILLS" or "skill" in query_lower or "tool" in query_lower or "technolog" in query_lower:
        subject = "skills/technologies"
    elif target_section == "LANGUAGES" or "language" in query_lower:
        subject = "languages"
    elif target_section == "EXPERIENCE" or "experience" in query_lower or "work" in query_lower:
        subject = "work experience entries"
    elif target_section == "EDUCATION" or "education" in query_lower:
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
        items = _extract_list_items(chunk["text"], target_section="SKILLS")
        if items:
            top_items = items[:6]
            summary_parts.append(f"Key skills include: {', '.join(top_items)}.")

    if summary_parts:
        summary_text = ' '.join(summary_parts)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary_text) if s.strip()]
        
        query_lower = query.lower()
        if "one line" in query_lower or "1 line" in query_lower or "single line" in query_lower:
            summary_text = sentences[0] if sentences else summary_text
        elif "two lines" in query_lower or "2 lines" in query_lower or "short" in query_lower:
            summary_text = " ".join(sentences[:min(2, len(sentences))]) if sentences else summary_text
            
        return "**Profile Summary:**\n\n" + summary_text.strip()

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

    # Append Confidence Scoring visually
    if "I couldn't find" not in answer:
        answer += f"\n\n*Confidence: {confidence.upper()}*"

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


# ---------------------------------------------------------------------------
# Week 2: Enhanced Q&A with intent routing, citations, and skill-gap analysis
# ---------------------------------------------------------------------------

def answer_question_v2(
    query: str,
    retrieved_chunks: list[dict],
    jd_text: str = "",
    intent: str | None = None,
) -> dict:
    """
    Week 2 enhanced answer generation with intent routing, citations, and
    skill-gap analysis.

    Falls back to Week 1 answer_question() for basic queries without JD.

    Args:
        query: The user's question.
        retrieved_chunks: List of chunk dicts from the retriever.
        jd_text: Job description text (empty if not provided).
        intent: Pre-classified intent. If None, will be classified.

    Returns:
        Dict with keys:
          - answer: The synthesized final answer text
          - sources: List of source chunks used
          - confidence: 'high', 'medium', 'low', or 'none'
          - num_sources: Number of relevant sources found
          - query_type: Detected intent/query type
          - citations: List of citation dicts
    """
    from src.intent import classify_intent, detect_target_section, needs_jd
    from src.citations import format_citations, attach_citations_to_answer
    from src.matcher import extract_skills_from_text, match_skills
    from src.prompts import (
        format_comparison_answer, format_gap_analysis_answer,
        format_improvement_answer, format_fallback_answer,
    )

    # Classify intent if not provided
    if intent is None:
        intent = classify_intent(query)

    # Check if JD is needed but missing
    if needs_jd(intent) and not jd_text.strip():
        return {
            "answer": format_fallback_answer("no_jd"),
            "sources": [],
            "confidence": "none",
            "num_sources": 0,
            "query_type": intent,
            "citations": [],
        }

    # For non-JD queries, delegate to Week 1 logic with citations added
    if not needs_jd(intent):
        base_response = answer_question(query, retrieved_chunks)

        # Add citations to Week 1 responses
        citations = format_citations(base_response.get("sources", []))
        if citations and base_response["confidence"] != "none":
            base_response["answer"] = attach_citations_to_answer(
                base_response["answer"], citations
            )
        base_response["citations"] = citations
        return base_response

    # --- JD-based intents: comparison, gap_analysis, improvement ---

    if not retrieved_chunks:
        return {
            "answer": format_fallback_answer("weak_retrieval"),
            "sources": [],
            "confidence": "none",
            "num_sources": 0,
            "query_type": intent,
            "citations": [],
        }

    # Filter by relevance threshold
    relevant_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk.get("score", 0) >= RELEVANCE_THRESHOLD
    ]

    if not relevant_chunks:
        relevant_chunks = retrieved_chunks[:3]  # Use top 3 even if below threshold

    # Extract skills from resume chunks and JD
    resume_text = "\n".join(c["text"] for c in relevant_chunks)
    resume_skills = extract_skills_from_text(resume_text)
    jd_skills = extract_skills_from_text(jd_text)

    # Perform skill matching
    match_result = match_skills(resume_skills, jd_skills)

    # Generate answer based on intent
    if intent == "comparison":
        answer = format_comparison_answer(match_result)
    elif intent == "gap_analysis":
        answer = format_gap_analysis_answer(match_result)
    elif intent == "improvement":
        answer = format_improvement_answer(match_result)
    else:
        answer = format_comparison_answer(match_result)

    # Determine confidence from match quality
    matched_count = len(match_result.get("matched", []))
    partial_count = len(match_result.get("partially_matched", []))
    total_jd = len(jd_skills) if jd_skills else 1

    match_ratio = (matched_count + partial_count * 0.5) / total_jd
    if match_ratio >= 0.6:
        confidence = "high"
    elif match_ratio >= 0.3:
        confidence = "medium"
    else:
        confidence = "low"

    # Format citations
    citations = format_citations(relevant_chunks)
    answer = attach_citations_to_answer(answer, citations)

    # Append confidence
    answer += f"\n\n*Confidence: {confidence.upper()}*"

    logger.info(
        "Generated %s-confidence %s answer: %d matched, %d partial, %d missing skills",
        confidence, intent, matched_count, partial_count, len(match_result.get("missing", [])),
    )

    return {
        "answer": answer,
        "sources": relevant_chunks[:5],
        "confidence": confidence,
        "num_sources": len(relevant_chunks),
        "query_type": intent,
        "citations": citations,
    }

