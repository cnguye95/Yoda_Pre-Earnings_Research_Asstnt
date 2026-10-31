"""ChromaDB wrapper for storing and querying SEC filing chunks by embedding.

ChromaStore holds a persistent Chroma collection ("filings_openai") where
every chunk from every processed filing is stored as a 1536-dim OpenAI vector.

Documents are keyed by "{accession_number}_{chunk_index}" so upserting the
same filing twice is idempotent — the second call overwrites the first.

Critical correctness invariant: every query filters by accession_number via
Chroma's `where=` clause. Without this filter, a query for one company's
filing could return chunks from a completely different company.

The data/chroma/ directory is created by Chroma on first use and is already
gitignored (added in Phase 0).
"""

import hashlib
import threading

import chromadb
import chromadb.config

from yoda import config
from yoda.ingest.chunker import Chunk
from yoda.retrieval.embeddings import embed_texts

# Where the database lives, and the one collection Yoda reads (vectors from
# text-embedding-3-small). Any other collection is a leftover of an earlier
# experiment and is dropped by prune_vector_store().
_CHROMA_DIR = config.DATA_DIR / "chroma"
_COLLECTION = "filings_openai"

# Every connection to the database uses the same settings (Chroma refuses
# two in one process with different ones), with Chroma's anonymous usage
# reporting off, so the database never contacts anyone.
_SETTINGS = chromadb.config.Settings(anonymized_telemetry=False)


# ---------------------------------------------------------------------------
# Chunk-list fingerprint
# ---------------------------------------------------------------------------

def _fingerprint(chunks: list[Chunk]) -> str:
    """Return a short code that changes whenever any chunk's text changes.

    It is a SHA-256 hash (a standard way to turn any amount of text into a
    fixed-length code) of every chunk's text in order, separated by a byte
    that never appears in text so "ab"+"c" and "a"+"bc" give different codes.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# ChromaStore class
# ---------------------------------------------------------------------------

class ChromaStore:
    """Persistent Chroma vector store for SEC filing chunks.

    A class is used here because it holds the Chroma client and collection
    objects across multiple upsert/query calls within a single process. Using
    a class avoids reopening the on-disk database on every call. It also
    adds up the embedding calls it made and the tokens OpenAI charged for
    them (*embedding_calls*, *embedding_tokens*), so the report that used it
    can count that cost; a lock keeps parallel workers' counts intact.
    """

    def __init__(self) -> None:
        """Open (or create) the on-disk database and its filings collection."""
        self.embedding_calls, self.embedding_tokens = 0, 0
        self._usage_lock = threading.Lock()
        # Chroma creates the data/chroma/ directory if it doesn't exist.
        self._client = chromadb.PersistentClient(path=str(_CHROMA_DIR), settings=_SETTINGS)

        # Get or create the collection. Cosine distance is the standard metric
        # for unit-vector embeddings from text-embedding-3-small.
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        accession_number: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks and their embeddings in the collection.

        If a chunk with the same ID already exists (same accession + index),
        its document, embedding, and metadata are overwritten. This makes
        repeated calls for the same filing safe and idempotent.

        Parameters
        ----------
        accession_number : str
            The SEC accession number for the filing (e.g. "0000320193-25-000123").
            Used as part of every chunk's ID and stored in metadata for filtering.
        chunks : list[Chunk]
            The Chunk objects produced by chunk_filing().
        embeddings : list[list[float]]
            Parallel list of vectors, one per chunk. Must be the same length as *chunks*.

        Raises ValueError if *chunks* and *embeddings* differ in length,
        because pairing a vector with the wrong chunk would corrupt searches.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings for {accession_number}")

        # Every chunk records a fingerprint of the whole chunk list, so a later
        # run can tell whether the stored chunks still match the current text.
        fingerprint = _fingerprint(chunks)

        # Build the parallel lists Chroma's upsert expects. Each chunk's ID
        # ("{accession}_{index}") is unique within the whole collection.
        ids = [f"{accession_number}_{chunk.chunk_index}" for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "accession":   accession_number,
                "section":     chunk.section,
                "chunk_index": chunk.chunk_index,
                "char_start":  chunk.char_start,
                "char_end":    chunk.char_end,
                "document":    chunk.document,
                "fingerprint": fingerprint,
            }
            for chunk in chunks
        ]

        # Upsert all chunks at once.
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* and add the call and its tokens to this store's totals."""
        embedded = embed_texts(texts)
        with self._usage_lock:
            self.embedding_calls += 1
            self.embedding_tokens += embedded.tokens
        return embedded.vectors

    def count(self, accession_number: str) -> int:
        """Return how many chunks are stored for *accession_number*."""
        # include=[] fetches only the IDs, so no documents or vectors are loaded.
        result = self._collection.get(where={"accession": accession_number}, include=[])
        return len(result["ids"])

    def index_if_missing(self, accession_number: str, chunks: list[Chunk]) -> bool:
        """Embed and store *chunks* unless exactly these chunks are already stored.

        Returns True if it embedded (costing OpenAI tokens), False if it skipped.
        "Exactly these" means the same number of chunks AND the same
        fingerprint of their text. Comparing counts catches a filing left
        half-stored by a crashed run; comparing fingerprints catches chunks
        whose text changed (for example after a chunker fix) while the count
        stayed the same.
        """
        # Skip when the stored chunks are this exact list of chunks.
        stored = self._collection.get(where={"accession": accession_number}, include=["metadatas"])
        fingerprint = _fingerprint(chunks)
        if len(stored["ids"]) == len(chunks) and all(
            meta.get("fingerprint") == fingerprint for meta in stored["metadatas"]
        ):
            return False

        # Otherwise clear whatever is stored for this filing first. Upsert only
        # overwrites matching IDs, so if the chunker now makes FEWER chunks,
        # the old extra ones would linger, keep the counts unequal, and force
        # a re-embed on every run.
        self._collection.delete(where={"accession": accession_number})
        embeddings = self._embed([c.text for c in chunks])
        self.upsert(accession_number, chunks, embeddings)
        return True

    def query(
        self,
        accession_number: str,
        query_text: str,
        k: int = 5,
    ) -> list[Chunk]:
        """Return the *k* chunks most semantically similar to *query_text*.

        Results are filtered to *accession_number* only — chunks from other
        filings are never returned, even if they score higher. This is the
        critical correctness invariant that prevents cross-filing pollution.

        Parameters
        ----------
        accession_number : str
            Only chunks from this filing are considered.
        query_text : str
            The search query. It is embedded with the same model used at
            upsert time so the vector spaces match.
        k : int
            Number of results to return (default 5).
        """
        # Embed the query text so it lives in the same vector space as the stored chunks.
        query_vector = self._embed([query_text])[0]

        # Query Chroma with the where= filter locked to this accession.
        # Without the filter, results from other filings could appear.
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where={"accession": accession_number},
        )
        return self._to_chunks(results)

    def query_any(self, accession_numbers: list[str], query_text: str, k: int = 5) -> list[Chunk]:
        """Return the *k* chunks most similar to *query_text* across several filings.

        Used for a company's 8-Ks: one search over all of them returns the
        best matches wherever they are, instead of k from every 8-K. The
        accession filter still applies — only the listed filings are searched.
        Returns [] when *accession_numbers* is empty.
        """
        if not accession_numbers:
            return []
        query_vector = self._embed([query_text])[0]
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where={"accession": {"$in": list(accession_numbers)}},
        )
        return self._to_chunks(results)

    @staticmethod
    def _to_chunks(results: dict) -> list[Chunk]:
        """Turn a raw Chroma query response into Chunk objects, most similar first."""
        # Reconstruct Chunk objects from the raw Chroma response.
        # results["documents"][0] and results["metadatas"][0] are parallel
        # lists of the k nearest neighbours in order of similarity.
        chunks: list[Chunk] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Chunks stored before the "document" label existed read back as ""
        # (a 10-Q/10-K chunk), so they don't need re-embedding.
        for doc, meta in zip(documents, metadatas):
            chunks.append(
                Chunk(
                    text=doc,
                    section=meta["section"],
                    chunk_index=meta["chunk_index"],
                    char_start=meta["char_start"],
                    char_end=meta["char_end"],
                    document=meta.get("document", ""),
                )
            )

        return chunks


# ---------------------------------------------------------------------------
# Pruning — the store would otherwise only grow
# ---------------------------------------------------------------------------

def prune_vector_store(keep: set[str]) -> tuple[int, list[str]]:
    """Delete what Yoda no longer reads; return (chunks deleted, collections dropped).

    *keep* holds every document key still read (accession numbers of cached
    filings and 8-Ks, and cached transcripts' keys). Chunks of any other
    document (an older 10-Q a newer one replaced, an old 8-K) are deleted,
    and so is every collection other than _COLLECTION. Local and free: no
    API call. A document deleted here is embedded again if it is ever read
    again.
    """
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR), settings=_SETTINGS)

    # Leftover collections from earlier experiments.
    dropped = sorted(collection.name for collection in client.list_collections() if collection.name != _COLLECTION)
    for name in dropped:
        client.delete_collection(name)

    # Chunks of documents no longer read.
    collection = client.get_or_create_collection(name=_COLLECTION, metadata={"hnsw:space": "cosine"})
    stored = [meta["accession"] for meta in collection.get(include=["metadatas"])["metadatas"]]
    stale = sorted(set(stored) - keep)
    if stale:
        collection.delete(where={"accession": {"$in": stale}})
    return sum(accession in stale for accession in stored), dropped
