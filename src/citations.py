"""
Citation Engine — Formats evidence references for grounded answers.

Provides compact [Source N] references that point to the most relevant
document chunks supporting each answer.
"""

import logging

logger = logging.getLogger(__name__)

# Maximum number of citations to include per answer
MAX_CITATIONS = 4
# Minimum relevance score to include as citation
CITATION_THRESHOLD = 0.35


def format_citations(relevant_chunks: list[dict], max_citations: int = MAX_CITATIONS) -> list[dict]:
    """
    Select and format the best supporting evidence chunks as citations.

    Args:
        relevant_chunks: List of chunk dicts with 'text', 'score', 'section' keys.
        max_citations: Maximum number of citations to include.

    Returns:
        List of citation dicts with keys:
          - index: Citation number (1-based)
          - section: Document section the evidence comes from
          - score: Relevance score
          - excerpt: Short text excerpt (max 200 chars)
          - full_text: Full chunk text
    """
    if not relevant_chunks:
        return []

    # Filter by threshold and take top N
    qualified = [
        c for c in relevant_chunks
        if c.get("score", 0) >= CITATION_THRESHOLD
    ]

    if not qualified:
        # If nothing meets threshold, take the best one anyway
        qualified = relevant_chunks[:1]

    # Deduplicate by section — prefer highest score per section
    seen_sections = {}
    for chunk in qualified:
        section = chunk.get("section", "GENERAL")
        if section not in seen_sections or chunk.get("score", 0) > seen_sections[section].get("score", 0):
            seen_sections[section] = chunk

    # Re-rank: section-diverse citations first, then by score
    diverse_chunks = sorted(seen_sections.values(), key=lambda c: c.get("score", 0), reverse=True)

    # If we need more, add remaining chunks by score
    remaining = [c for c in qualified if c not in diverse_chunks]
    all_candidates = diverse_chunks + remaining

    citations = []
    for i, chunk in enumerate(all_candidates[:max_citations], 1):
        text = chunk.get("text", "")
        # Create a compact excerpt
        excerpt = _make_excerpt(text, max_length=200)

        citations.append({
            "index": i,
            "section": chunk.get("section", "GENERAL"),
            "score": chunk.get("score", 0),
            "excerpt": excerpt,
            "full_text": text,
            "chunk_index": chunk.get("chunk_index", -1),
        })

    logger.info("Formatted %d citations from %d candidate chunks.", len(citations), len(relevant_chunks))
    return citations


def build_citation_text(citations: list[dict]) -> str:
    """
    Build a formatted citation block for display.

    Args:
        citations: List of citation dicts from format_citations().

    Returns:
        Formatted citation text string.
    """
    if not citations:
        return ""

    lines = ["\n\n---\n📋 **Supporting Evidence:**\n"]
    for cite in citations:
        score_pct = f"{cite['score']:.0%}"
        section = cite['section'].replace('_', ' ').title()
        lines.append(
            f"**[{cite['index']}]** _{section}_ (Relevance: {score_pct})\n"
            f"> {cite['excerpt']}\n"
        )

    return "\n".join(lines)


def attach_citations_to_answer(answer: str, citations: list[dict]) -> str:
    """
    Merge citation references into an answer text.

    Args:
        answer: The generated answer text.
        citations: List of citation dicts.

    Returns:
        Answer text with citation block appended.
    """
    if not citations:
        return answer

    citation_block = build_citation_text(citations)
    return answer + citation_block


def _make_excerpt(text: str, max_length: int = 200) -> str:
    """
    Create a compact excerpt from a text chunk.

    Takes the first meaningful sentences up to max_length characters.
    """
    if not text:
        return ""

    # Clean up whitespace
    clean = " ".join(text.split())

    if len(clean) <= max_length:
        return clean

    # Try to cut at a sentence boundary
    truncated = clean[:max_length]
    last_period = truncated.rfind(".")
    last_comma = truncated.rfind(",")

    if last_period > max_length * 0.5:
        return truncated[:last_period + 1]
    elif last_comma > max_length * 0.6:
        return truncated[:last_comma] + "..."
    else:
        # Cut at last space
        last_space = truncated.rfind(" ")
        if last_space > 0:
            return truncated[:last_space] + "..."
        return truncated + "..."
