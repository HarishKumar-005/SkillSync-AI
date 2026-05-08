"""
Retriever Module — Orchestrates semantic search over stored document chunks.

Encodes the user query, searches ChromaDB, and returns ranked results
with text, similarity score, and source metadata.

Week 2 additions:
  - retrieve_by_section(): Section-targeted retrieval with fallback
  - retrieve_for_comparison(): Pulls resume chunks relevant to JD requirements
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


# ---------------------------------------------------------------------------
# Week 2: Enhanced retrieval functions
# ---------------------------------------------------------------------------

def retrieve_by_section(
    collection,
    query: str,
    target_section: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Section-targeted retrieval with automatic fallback to global search.

    Tries the target section first. If it yields no results or the target
    section is None, falls back to global retrieval.

    Args:
        collection: ChromaDB collection to search.
        query: User's question string.
        target_section: Section to target (e.g., 'SKILLS'). None for global.
        top_k: Number of top results to return.

    Returns:
        List of result dicts (same format as retrieve()).
    """
    results = []

    # Stage 1: Targeted retrieval
    if target_section:
        results = retrieve(
            collection=collection,
            query=query,
            top_k=top_k,
            where_filter={"section": target_section},
        )

    # Stage 2: Fallback to global if targeted yields nothing
    if not results:
        results = retrieve(
            collection=collection,
            query=query,
            top_k=top_k,
        )

    return results


def retrieve_for_comparison(
    collection,
    jd_text: str,
    top_k: int = 8,
) -> list[dict]:
    """
    Retrieve resume chunks most relevant to a job description.

    Uses the JD text as the query to find the best matching resume content.
    Prioritizes SKILLS and EXPERIENCE sections.

    Args:
        collection: ChromaDB collection containing resume chunks.
        jd_text: The job description text.
        top_k: Number of chunks to retrieve.

    Returns:
        List of result dicts from the resume, ranked by relevance to JD.
    """
    if not jd_text or not jd_text.strip():
        logger.warning("Empty JD text for comparison retrieval.")
        return []

    # Retrieve skills-focused chunks
    skills_results = retrieve(
        collection=collection,
        query=jd_text,
        top_k=top_k // 2,
        where_filter={"section": "SKILLS"},
    )

    # Retrieve experience-focused chunks
    exp_results = retrieve(
        collection=collection,
        query=jd_text,
        top_k=top_k // 2,
        where_filter={"section": "EXPERIENCE"},
    )

    # Retrieve global (catches PROJECTS, CERTIFICATES, etc.)
    global_results = retrieve(
        collection=collection,
        query=jd_text,
        top_k=top_k // 2,
    )

    # Merge and deduplicate by chunk_index
    seen = set()
    merged = []
    for chunk in skills_results + exp_results + global_results:
        key = chunk.get("chunk_index", id(chunk))
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    # Sort by score descending
    merged.sort(key=lambda c: c.get("score", 0), reverse=True)

    logger.info(
        "Comparison retrieval: %d unique chunks from %d skills + %d exp + %d global",
        len(merged), len(skills_results), len(exp_results), len(global_results),
    )

    return merged[:top_k]
