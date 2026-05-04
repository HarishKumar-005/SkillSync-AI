"""
Text Chunker — Section-aware splitting with overlap for embedding.

Strategy:
  1. Detect document sections (SKILLS, PROJECTS, CERTIFICATES, etc.)
  2. Split text by sections first to preserve semantic boundaries
  3. Within each section, apply recursive splitting if too long
  4. Tag every chunk with its section label

Default config: 800 char chunks, 150 char overlap.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Common resume/document section headings (case-insensitive)
SECTION_PATTERNS = [
    # Pattern, Label
    (r'(?i)\b(?:technical\s+)?skills?\b', 'SKILLS'),
    (r'(?i)\b(?:core\s+)?competenc(?:ies|e)\b', 'SKILLS'),
    (r'(?i)\btechnolog(?:ies|y)\b', 'SKILLS'),
    (r'(?i)\bprojects?\b', 'PROJECTS'),
    (r'(?i)\bcertificat(?:ions?|es?)\b', 'CERTIFICATES'),
    (r'(?i)\bprofessional\s+summary\b', 'PROFILE'),
    (r'(?i)\b(?:career\s+)?(?:summary|objective|profile|about)\b', 'PROFILE'),
    (r'(?i)\b(?:work\s+)?experience\b', 'EXPERIENCE'),
    (r'(?i)\bemployment\s+history\b', 'EXPERIENCE'),
    (r'(?i)\beducation(?:al)?\s*(?:background|qualifications?)?\b', 'EDUCATION'),
    (r'(?i)\bachievements?\b', 'ACHIEVEMENTS'),
    (r'(?i)\bawards?\b', 'ACHIEVEMENTS'),
    (r'(?i)\bpublications?\b', 'PUBLICATIONS'),
    (r'(?i)\bcontact\s*(?:info(?:rmation)?)?\b', 'CONTACT'),
    (r'(?i)\blanguages?\b', 'LANGUAGES'),
    (r'(?i)\binterests?\b', 'INTERESTS'),
    (r'(?i)\bhobbies\b', 'INTERESTS'),
    (r'(?i)\breferences?\b', 'REFERENCES'),
]


def _detect_section(line: str) -> Optional[str]:
    """
    Check if a line is a section heading.

    A section heading is typically:
      - A short line (< 60 chars)
      - No colon (which indicates a content line like "Languages: Python...")
      - No bullet prefix or comma-heavy content
      - Matches a known section pattern

    Returns:
        Section label string or None.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None

    # Content lines are NOT headings — skip lines with colons, bullets, commas, pipes
    if ':' in stripped:
        return None
    if stripped.startswith(('-', '•', '*', '▪', '►', '➤', '→')):
        return None
    if stripped.count(',') >= 2:
        return None
    if '|' in stripped:
        return None

    # Lines that are mostly uppercase or very short are likely headings
    for pattern, label in SECTION_PATTERNS:
        if re.search(pattern, stripped):
            return label

    return None


def _split_into_sections(text: str) -> list[dict]:
    """
    Split text into sections based on detected headings.

    Returns:
        List of dicts with keys: section, text, start_char.
        If no sections detected, returns the entire text as one 'GENERAL' section.
    """
    lines = text.split('\n')
    sections = []
    current_section = "GENERAL"
    current_lines = []
    current_start = 0
    char_pos = 0

    for line in lines:
        detected = _detect_section(line)
        if detected and current_lines:
            # Save the previous section
            section_text = '\n'.join(current_lines).strip()
            if section_text:
                sections.append({
                    "section": current_section,
                    "text": section_text,
                    "start_char": current_start,
                })
            current_section = detected
            current_lines = [line]
            current_start = char_pos
        else:
            if detected:
                current_section = detected
                current_start = char_pos
            current_lines.append(line)
        char_pos += len(line) + 1  # +1 for newline

    # Don't forget the last section
    section_text = '\n'.join(current_lines).strip()
    if section_text:
        sections.append({
            "section": current_section,
            "text": section_text,
            "start_char": current_start,
        })

    if not sections:
        sections.append({
            "section": "GENERAL",
            "text": text.strip(),
            "start_char": 0,
        })

    logger.info(
        "Detected %d sections: %s",
        len(sections),
        [s["section"] for s in sections],
    )
    return sections


# ---------------------------------------------------------------------------
# Recursive sub-chunking (for sections that are too long)
# ---------------------------------------------------------------------------

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_text(text: str, separator: str) -> list[str]:
    """Split text by a separator, keeping the separator at the end of each part."""
    if separator == " ":
        return text.split(separator)

    parts = text.split(separator)
    result = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            result.append(part + separator)
        else:
            result.append(part)
    return [p for p in result if p.strip()]


def _recursive_chunk(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """
    Recursively split text into pieces no larger than chunk_size.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        chunks = []
        for i in range(0, len(text), chunk_size):
            piece = text[i:i + chunk_size]
            if piece.strip():
                chunks.append(piece)
        return chunks

    current_sep = separators[0]
    remaining_seps = separators[1:]
    parts = _split_text(text, current_sep)

    chunks = []
    current_chunk = ""

    for part in parts:
        if len(current_chunk) + len(part) <= chunk_size:
            current_chunk += part
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            if len(part) > chunk_size:
                sub_chunks = _recursive_chunk(part, chunk_size, remaining_seps)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = part

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
    separators: Optional[list[str]] = None,
) -> list[dict]:
    """
    Split text into section-aware, overlapping chunks.

    Steps:
      1. Detect sections (SKILLS, PROJECTS, CERTIFICATES, etc.)
      2. Within each section, split into sub-chunks if the section is too long
      3. Add overlap between consecutive chunks within the same section
      4. Tag each chunk with its section label

    Args:
        text: Cleaned text to chunk.
        chunk_size: Maximum characters per chunk (default 800).
        overlap: Number of characters to overlap between chunks (default 150).
        separators: List of separators to try, in order. Defaults to SEPARATORS.

    Returns:
        List of chunk dicts with keys: text, index, section, start_char, end_char.
        Empty list if text is empty.
    """
    if not text or not text.strip():
        logger.warning("No text provided for chunking.")
        return []

    if separators is None:
        separators = SEPARATORS

    # Step 1: Split into sections
    sections = _split_into_sections(text)

    # Step 2: Sub-chunk each section and build final chunk list
    all_chunks = []
    global_index = 0

    for section_info in sections:
        section_text = section_info["text"]
        section_label = section_info["section"]
        section_start = section_info["start_char"]

        # Get raw sub-chunks for this section
        raw_sub_chunks = _recursive_chunk(section_text, chunk_size, separators)

        if not raw_sub_chunks:
            continue

        # Add overlap between consecutive sub-chunks within the same section
        for i, chunk_content in enumerate(raw_sub_chunks):
            if i > 0 and overlap > 0:
                prev = raw_sub_chunks[i - 1]
                overlap_text = prev[-overlap:] if len(prev) >= overlap else prev
                space_idx = overlap_text.find(' ')
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]
                final_text = overlap_text + " " + chunk_content
            else:
                final_text = chunk_content

            # Find position in original text
            search_start = max(0, section_start - overlap)
            start_pos = text.find(chunk_content[:50], search_start)
            if start_pos == -1:
                start_pos = section_start

            all_chunks.append({
                "text": final_text.strip(),
                "index": global_index,
                "section": section_label,
                "start_char": start_pos,
                "end_char": start_pos + len(chunk_content),
            })
            global_index += 1
            section_start = start_pos + len(chunk_content)

    logger.info(
        "Created %d section-aware chunks from %d characters (chunk_size=%d, overlap=%d)",
        len(all_chunks), len(text), chunk_size, overlap,
    )

    return all_chunks
