# retrieve_logic.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re

# --- Robust imports to support different LangChain versions ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings  # very old versions

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


@dataclass
class RetrievalConfig:
    vector_dir: str = "../vector"  # Directory where FAISS index is stored
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieve a larger candidate pool before filtering and re-ranking
    candidate_k: int = 30

    # Final number of documents returned to LLM
    final_k: int = 5

    # Use Max Marginal Relevance to reduce redundancy
    use_mmr: bool = True
    mmr_fetch_k: int = 30
    mmr_lambda: float = 0.5

    # Optional similarity threshold (distance-based; lower = more similar)
    max_distance_threshold: Optional[float] = None

    # Metadata field names (adjust if your KB uses different keys)
    grade_key: str = "grade"
    subject_key: str = "subject"
    source_key: str = "source"


class RetrievalEngine:
    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.embed_model)

        # Only use allow_dangerous_deserialization=True
        # if you trust the local FAISS index
        self.vs = FAISS.load_local(
            cfg.vector_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    # ---------------- Query preprocessing ----------------
    def _normalize_query(self, q: str) -> str:
        """
        Perform lightweight query normalization.
        """
        q = q.strip()
        q = re.sub(r"\s+", " ", q)
        return q

    # ---------------- Metadata filtering ----------------
    def _metadata_match(self, doc: Document, grade: Optional[int], subject: Optional[str]) -> bool:
        """
        Filter documents based on grade and subject metadata.
        """
        md = doc.metadata or {}

        if grade is not None:
            if str(md.get(self.cfg.grade_key, "")).strip() != str(grade):
                return False

        if subject:
            if str(md.get(self.cfg.subject_key, "")).strip().lower() != str(subject).strip().lower():
                return False

        return True

    # ---------------- Deduplication ----------------
    def _dedup(self, docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Remove duplicate chunks based on source + page (+ chunk_id).
        """
        seen = set()
        out = []

        for d, score in docs:
            md = d.metadata or {}
            key = (
                md.get(self.cfg.source_key, ""),
                md.get("page", md.get("page_number", "")),
                md.get("chunk_id", md.get("chunk", "")),
            )
            if key in seen:
                continue

            seen.add(key)
            out.append((d, score))

        return out

    # ---------------- Main retrieval ----------------
    def retrieve(
        self,
        question: str,
        grade: Optional[int] = None,
        subject: Optional[str] = None,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Unified retrieval interface:
        retrieve(question, grade, subject, k) -> List[Document]
        """
        question = self._normalize_query(question)
        final_k = k or self.cfg.final_k

        # Step 1: Retrieve a large candidate set
        candidate_docs = None

        if self.cfg.use_mmr:
            # 1) Newer versions: returns List[Tuple[Document, float]]
            if hasattr(self.vs, "max_marginal_relevance_search_with_score"):
                candidate_docs = self.vs.max_marginal_relevance_search_with_score(
                    question,
                    k=min(self.cfg.candidate_k, self.cfg.mmr_fetch_k),
                    fetch_k=self.cfg.mmr_fetch_k,
                    lambda_mult=self.cfg.mmr_lambda,
                )

            # 2) Common versions: returns List[Document] (no scores)
            elif hasattr(self.vs, "max_marginal_relevance_search"):
                docs = self.vs.max_marginal_relevance_search(
                    question,
                    k=min(self.cfg.candidate_k, self.cfg.mmr_fetch_k),
                    fetch_k=self.cfg.mmr_fetch_k,
                    lambda_mult=self.cfg.mmr_lambda,
                )
                # Add dummy scores so downstream code stays consistent
                candidate_docs = [(d, 0.0) for d in docs]

        # 3) Fallback: similarity search with score
        if candidate_docs is None:
            candidate_docs = self.vs.similarity_search_with_score(
                question,
                k=self.cfg.candidate_k
            )

        # Step 2: Metadata filtering
        filtered = [
            (d, s) for d, s in candidate_docs
            if self._metadata_match(d, grade, subject)
        ]

        # Step 3: Deduplication
        filtered = self._dedup(filtered)

        # Step 4: Optional similarity threshold filtering
        if self.cfg.max_distance_threshold is not None:
            filtered = [
                (d, s) for d, s in filtered
                if s <= self.cfg.max_distance_threshold
            ]

        # Step 5: Sort by similarity (ascending distance)
        filtered.sort(key=lambda x: x[1])

        # Return top-k documents
        return [d for d, _ in filtered[:final_k]]


# -------- Global singleton interface for external use --------
_ENGINE: Optional[RetrievalEngine] = None

def retrieve(question: str, grade: Optional[int] = None, subject: Optional[str] = None, k: Optional[int] = None):
    """
    External interface used by LLM/Agent.
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RetrievalEngine(RetrievalConfig())

    return _ENGINE.retrieve(question, grade, subject, k)