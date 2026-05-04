"""
Document Parser — Extracts raw text from PDF and DOCX files.

Supported formats:
  - PDF  (via pypdf)
  - DOCX (via python-docx)
"""

import os
import logging
from pathlib import Path

from pypdf import PdfReader
from docx import Document

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text as a single string. Empty string if extraction fails.
    """
    try:
        reader = PdfReader(file_path)
        if len(reader.pages) == 0:
            logger.warning("PDF has no pages: %s", file_path)
            return ""

        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text)
            else:
                logger.debug("Page %d has no extractable text.", i + 1)

        full_text = "\n\n".join(pages_text)
        logger.info(
            "Extracted %d characters from %d pages in %s",
            len(full_text), len(reader.pages), os.path.basename(file_path),
        )
        return full_text

    except Exception as e:
        logger.error("Failed to parse PDF '%s': %s", file_path, e)
        return ""


def parse_docx(file_path: str) -> str:
    """
    Extract text from a DOCX file.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Extracted text as a single string. Empty string if extraction fails.
    """
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        if not paragraphs:
            logger.warning("DOCX has no text content: %s", file_path)
            return ""

        full_text = "\n\n".join(paragraphs)
        logger.info(
            "Extracted %d characters from %d paragraphs in %s",
            len(full_text), len(paragraphs), os.path.basename(file_path),
        )
        return full_text

    except Exception as e:
        logger.error("Failed to parse DOCX '%s': %s", file_path, e)
        return ""


def parse_file(file_path: str) -> tuple[str, str]:
    """
    Dispatch to the correct parser based on file extension.

    Args:
        file_path: Path to the document file.

    Returns:
        Tuple of (extracted_text, error_message).
        If successful, error_message is empty.
        If failed, extracted_text is empty and error_message describes the issue.
    """
    path = Path(file_path)

    if not path.exists():
        return "", f"File not found: {file_path}"

    if path.stat().st_size == 0:
        return "", "The uploaded file is empty."

    extension = path.suffix.lower()

    if extension == ".pdf":
        text = parse_pdf(file_path)
    elif extension == ".docx":
        text = parse_docx(file_path)
    else:
        return "", f"Unsupported file format: '{extension}'. Please upload a PDF or DOCX file."

    if not text.strip():
        return "", "No readable text could be extracted from this file. The file may be scanned/image-based or corrupted."

    return text, ""
