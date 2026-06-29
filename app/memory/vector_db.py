"""ChromaDB semantic memory — the agent's long-term "lessons learned" store.

Per design §5, historical lessons and adjusted rules are stored in a vector DB
so the agent can reference past trades in future decision cycles. We persist a
short natural-language "lesson" per closed/decided trade plus structured
metadata (symbol, outcome, P&L), and let future runs query semantically
("what happened last time I sold calls on a high-IV utility?").

Testability: the embedding model (sentence-transformers) and ChromaDB are heavy,
so both are LAZY and the underlying collection is injectable. Tests pass a tiny
in-memory fake collection; production uses a persistent Chroma collection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger("vector-db")


class TradeMemory:
    def __init__(
        self,
        *,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_backend: Optional[str] = None,
        embedding_function: Any = None,   # override Chroma's embedder (tests/light use)
        collection: Any = None,           # inject a ready collection-like object (tests)
    ) -> None:
        self._persist_dir = Path(persist_dir or settings.chroma_dir)
        self._collection_name = collection_name or settings.chroma_collection
        self._embedding_model = embedding_model or settings.embedding_model
        self._embedding_backend = embedding_backend or settings.embedding_backend
        self._embedding_function = embedding_function
        self._collection = collection     # if injected, we never touch chromadb

    def _pick_embedding_backend(self) -> str:
        """Decide the embedding backend: 'sentence_transformers' or 'default'
        (ChromaDB's built-in ONNX MiniLM, which needs no extra package).

        'auto' (the default) prefers sentence-transformers when it's installed and
        otherwise falls back to ONNX — so lessons store out of the box without the
        heavy torch dependency. Same underlying MiniLM-L6-v2 model either way.
        """
        if self._embedding_backend != "auto":
            return self._embedding_backend
        try:
            import sentence_transformers  # noqa: F401
            return "sentence_transformers"
        except ImportError:
            return "default"

    def _build_embedding_function(self) -> Any:
        from chromadb.utils import embedding_functions
        backend = self._pick_embedding_backend()
        if backend == "sentence_transformers":
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model)
        logger.info("Using ChromaDB default ONNX embeddings "
                    "(install sentence-transformers to use it instead).")
        return embedding_functions.DefaultEmbeddingFunction()

    # ── lazy collection resolution ────────────────────────────────────
    def _coll(self) -> Any:
        if self._collection is not None:
            return self._collection
        import chromadb  # lazy, heavy
        client = chromadb.PersistentClient(path=str(self._persist_dir))
        ef = self._embedding_function or self._build_embedding_function()
        self._collection = client.get_or_create_collection(
            name=self._collection_name, embedding_function=ef)
        logger.info("Chroma collection '%s' ready at %s", self._collection_name, self._persist_dir)
        return self._collection

    # ── write ─────────────────────────────────────────────────────────
    def add_lesson(self, lesson_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """Store one trade lesson. Metadata values must be str/int/float/bool
        (Chroma constraint), so we coerce/drop anything else."""
        clean_meta = {
            k: v for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool)) and v is not None
        }
        self._coll().add(ids=[lesson_id], documents=[text], metadatas=[clean_meta])
        logger.info("Stored trade lesson %s (%d chars)", lesson_id, len(text))

    # ── read ──────────────────────────────────────────────────────────
    def query(self, text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search for relevant past lessons. Returns a flat list of
        {id, document, metadata, distance}."""
        res = self._coll().query(query_texts=[text], n_results=n_results)
        return self._flatten(res)

    @staticmethod
    def _flatten(res: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chroma returns parallel lists-of-lists (one row per query). We issue a
        single query, so unwrap row 0 into a list of dicts."""
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out = []
        for i, _id in enumerate(ids):
            out.append({
                "id": _id,
                "document": docs[i] if i < len(docs) else None,
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            })
        return out

    def count(self) -> int:
        return self._coll().count()
