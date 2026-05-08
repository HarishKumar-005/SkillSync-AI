"""
Refresh Workflow — Clean knowledge base rebuild for session switching.

Handles clearing stale state and rebuilding when:
  - A new resume is uploaded
  - A new JD is entered
  - User explicitly clicks refresh
"""

import logging

logger = logging.getLogger(__name__)


def clear_session_state(session_state) -> None:
    """
    Clear all document-related session state for a fresh start.

    Args:
        session_state: Streamlit session state object.
    """
    keys_to_clear = [
        "collection",
        "chroma_client",
        "document_loaded",
        "doc_info",
        "jd_text",
        "jd_skills",
        "resume_skills",
        "messages",
    ]

    for key in keys_to_clear:
        if key in session_state:
            if key == "messages":
                session_state[key] = []
            elif key == "doc_info":
                session_state[key] = {}
            elif key == "document_loaded":
                session_state[key] = False
            elif key == "jd_text":
                session_state[key] = ""
            else:
                session_state[key] = None

    logger.info("Session state cleared for fresh start.")


def clear_knowledge_base(session_state) -> bool:
    """
    Wipe the ChromaDB collection and reset document state.

    Args:
        session_state: Streamlit session state object.

    Returns:
        True if successful, False otherwise.
    """
    try:
        client = session_state.get("chroma_client")
        if client:
            try:
                client.delete_collection("skillsync_docs")
                logger.info("Deleted ChromaDB collection 'skillsync_docs'.")
            except Exception:
                pass  # Collection may not exist

        session_state.collection = None
        session_state.chroma_client = None
        session_state.document_loaded = False
        session_state.doc_info = {}

        # Clear cached skills
        if "resume_skills" in session_state:
            session_state.resume_skills = None
        if "jd_skills" in session_state:
            session_state.jd_skills = None

        logger.info("Knowledge base cleared.")
        return True

    except Exception as e:
        logger.error("Failed to clear knowledge base: %s", e)
        return False


def clear_jd(session_state) -> None:
    """
    Clear only the job description state (keep resume loaded).

    Args:
        session_state: Streamlit session state object.
    """
    if "jd_text" in session_state:
        session_state.jd_text = ""
    if "jd_skills" in session_state:
        session_state.jd_skills = None

    logger.info("JD state cleared.")


def is_session_valid(session_state) -> dict:
    """
    Check the current session state validity.

    Returns:
        Dict with keys:
          - has_resume: bool
          - has_jd: bool
          - has_collection: bool
          - status: str description
    """
    has_resume = bool(session_state.get("document_loaded", False))
    has_jd = bool(session_state.get("jd_text", "").strip())
    has_collection = session_state.get("collection") is not None

    if has_resume and has_jd:
        status = "ready_full"
    elif has_resume:
        status = "ready_resume_only"
    elif has_jd:
        status = "jd_only_no_resume"
    else:
        status = "empty"

    return {
        "has_resume": has_resume,
        "has_jd": has_jd,
        "has_collection": has_collection,
        "status": status,
    }
