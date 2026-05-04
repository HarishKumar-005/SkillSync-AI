"""
Retriever Module — Orchestrates semantic search over stored document chunks.

Encodes the user query, searches ChromaDB, and returns ranked results
with text, similarity score, and source metadata.
"""

import logging
from src.embeddings import embed_query
from src.vectorstore import query_collection

logger = logging.getLogger(__name__)


def retrieve(
    collection,
    query: str,
    top_k: int = 5,
    where_filter: dict | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a given query.

    Args:
        collection: ChromaDB collection to search.
        query: User's question string.
        top_k: Number of top results to return. Defaults to 5.
        where_filter: Optional filter dict to restrict search (e.g., {"section": "PROJECTS"}).

    Returns:
        List of result dicts, each containing:
          - text: The chunk text
          - score: Similarity score (lower distance = more similar for cosine)
          - source: Source filename
          - chunk_index: Position of chunk in original document
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to retriever.")
        return []

    # Encode the query
    query_embedding = embed_query(query)
    if not query_embedding:
        logger.error("Failed to embed query.")
        return []

    # Search ChromaDB
    results = query_collection(collection, query_embedding, n_results=top_k, where_filter=where_filter)

    # Unpack results into a clean format
    retrieved = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i] if results["distances"] else 0.0
            # ChromaDB with cosine returns distance (0 = identical, 2 = opposite)
            # Convert to a similarity score: 1 - (distance / 2)
            similarity = 1.0 - (distance / 2.0)

            retrieved.append({
                "text": results["documents"][0][i],
                "score": round(similarity, 4),
                "distance": round(distance, 4),
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "chunk_index": results["metadatas"][0][i].get("chunk_index", -1),
                "section": results["metadatas"][0][i].get("section", "GENERAL"),
            })

    logger.info(
        "Retrieved %d chunks for query: '%s...'",
        len(retrieved), query[:50],
    )
    return retrieved
