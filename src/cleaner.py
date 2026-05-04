"""
Text Cleaner — Normalizes extracted text for consistent chunking.

Operations:
  - Remove null bytes and control characters
  - Collapse multiple whitespace into single spaces within lines
  - Collapse multiple blank lines into at most two newlines
  - Strip leading/trailing whitespace
"""

import re
import logging

logger = logging.getLogger(__name__)


def clean_text(raw_text: str) -> str:
    """
    Clean and normalize extracted document text.

    Args:
        raw_text: Raw text from the document parser.

    Returns:
        Cleaned text. Empty string if input is empty.
    """
    if not raw_text or not raw_text.strip():
        logger.warning("Received empty text for cleaning.")
        return ""

    text = raw_text

    # Remove null bytes and non-printable control characters (keep newlines, tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Replace tabs with spaces
    text = text.replace('\t', ' ')

    # Collapse multiple spaces (within lines) into single space
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Collapse 3+ consecutive newlines into exactly 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Strip overall leading/trailing whitespace
    text = text.strip()

    logger.info(
        "Cleaned text: %d chars (from %d raw chars, %.1f%% reduction)",
        len(text), len(raw_text),
        (1 - len(text) / max(len(raw_text), 1)) * 100,
    )

    return text
