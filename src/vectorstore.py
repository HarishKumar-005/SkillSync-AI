"""
Vector Store Module — Manages ChromaDB collection lifecycle.

Uses an ephemeral (in-memory) ChromaDB client per the CONTEXT.md requirement
of "rebuild-on-upload" storage for Week 1.
"""

import logging
import chromadb

logger = logging.getLogger(__name__)


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Create and return an ephemeral (in-memory) ChromaDB client.

    Returns:
        A ChromaDB client instance.
    """
    client = chromadb.Client()
    logger.info("Created ephemeral ChromaDB client.")
    return client


def create_collection(client: chromadb.ClientAPI, name: str = "skillsync_docs") -> chromadb.Collection:
    """
    Create or reset a ChromaDB collection.

    If a collection with the given name already exists, it is deleted first
    to ensure a clean rebuild on each upload.

    Args:
        client: ChromaDB client instance.
        name: Collection name. Defaults to 'skillsync_docs'.

    Returns:
        A fresh ChromaDB collection.
    """
    # Delete existing collection if it exists (rebuild-on-upload)
    try:
        client.delete_collection(name)
        logger.debug("Deleted existing collection '%s'.", name)
    except Exception:
        pass  # Collection didn't exist, which is fine

    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Created ChromaDB collection '%s' with cosine similarity.", name)
    return collection


def add_chunks(
    collection: chromadb.Collection,
    chunks: list[dict],
    embeddings: list[list[float]],
    source_filename: str,
) -> int:
    """
    Insert document chunks with embeddings and metadata into a ChromaDB collection.

    Args:
        collection: Target ChromaDB collection.
        chunks: List of chunk dicts (must have 'text' and 'index' keys).
        embeddings: Corresponding embedding vectors.
        source_filename: Name of the source document.

    Returns:
        Number of chunks successfully added.
    """
    if not chunks or not embeddings:
        logger.warning("No chunks or embeddings to add.")
        return 0

    if len(chunks) != len(embeddings):
        logger.error(
            "Mismatch: %d chunks vs %d embeddings.", len(chunks), len(embeddings)
        )
        return 0

    ids = [f"{source_filename}_chunk_{chunk['index']}" for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source": source_filename,
            "chunk_index": chunk["index"],
            "section": chunk.get("section", "GENERAL"),
            "start_char": chunk.get("start_char", 0),
            "end_char": chunk.get("end_char", 0),
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    count = collection.count()
    logger.info(
        "Added %d chunks from '%s' to collection. Total documents: %d",
        len(chunks), source_filename, count,
    )
    return len(chunks)


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 5,
    where_filter: dict | None = None,
) -> dict:
    """
    Query the ChromaDB collection for the most similar chunks.

    Args:
        collection: ChromaDB collection to search.
        query_embedding: Embedding vector of the query.
        n_results: Number of results to return. Defaults to 5.
        where_filter: Optional metadata filter dict (e.g., {"section": "PROJECTS"}).

    Returns:
        ChromaDB query result dict with keys: ids, documents, metadatas, distances.
    """
    if not query_embedding:
        logger.warning("Empty query embedding provided.")
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Clamp n_results to collection size
    total = collection.count()
    if total == 0:
        logger.warning("Collection is empty.")
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    actual_n = min(n_results, total)
    
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": actual_n,
        "include": ["documents", "metadatas", "distances"],
    }
    
    if where_filter:
        # Check if any documents match the filter before querying, 
        # as ChromaDB might throw an error if no documents match and we ask for n_results.
        # Actually, chroma handles empty filter results gracefully by returning empty lists.
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    logger.info("Query returned %d results.", len(results["ids"][0]))
    return results
