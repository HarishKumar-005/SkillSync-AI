"""
Embeddings Module — Wraps sentence-transformers for text encoding.

Uses the 'all-MiniLM-L6-v2' model (384-dimensional embeddings).
Model is loaded once and cached at module level for efficiency.
"""

import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Module-level cache for the embedding model
_model: SentenceTransformer | None = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> SentenceTransformer:
    """
    Load and return the embedding model, caching it for reuse.

    Returns:
        A SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded. Dimension: %d", _model.get_embedding_dimension())
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Encode a list of text strings into embedding vectors.

    Args:
        texts: List of text strings to encode.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    if not texts:
        logger.warning("No texts provided for embedding.")
        return []

    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    logger.info("Embedded %d texts into %d-dimensional vectors.", len(texts), embeddings.shape[1])
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Encode a single query string into an embedding vector.

    Args:
        query: The query text to encode.

    Returns:
        Embedding vector as a list of floats.
    """
    if not query or not query.strip():
        logger.warning("Empty query provided for embedding.")
        return []

    model = get_embedding_model()
    embedding = model.encode(query, show_progress_bar=False, normalize_embeddings=True)
    return embedding.tolist()
