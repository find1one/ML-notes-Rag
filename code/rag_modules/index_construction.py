"""
Vector index construction and loading.
"""

import logging
import os
import hashlib
import json
from pathlib import Path
from typing import Any, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """Build, save, and load the FAISS vector index."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        index_save_path: str = "./vector_index_ml_notes",
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore: Optional[FAISS] = None
        self.manifest_path = Path(self.index_save_path) / "manifest.json"
        self.setup_embeddings()

    def setup_embeddings(self) -> None:
        logger.info("Initializing embedding model: %s", self.model_name)

        model_kwargs = {"device": "cpu"}
        if os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1":
            model_kwargs["local_files_only"] = True

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )

    def build_vector_index(self, chunks: List[Document], skipped_files: Optional[List[dict]] = None, evaluation: Optional[dict] = None) -> FAISS:
        if not chunks:
            raise ValueError("Document chunks cannot be empty.")

        logger.info("Building FAISS index with %s chunks", len(chunks))
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
        self._write_manifest(chunks, skipped_files=skipped_files, evaluation=evaluation)
        return self.vectorstore

    def add_documents(self, new_chunks: List[Document]) -> None:
        if not self.vectorstore:
            raise ValueError("Please build or load the vector index first.")

        logger.info("Adding %s new chunks to FAISS index", len(new_chunks))
        self.vectorstore.add_documents(new_chunks)

    def save_index(self) -> None:
        if not self.vectorstore:
            raise ValueError("Please build or load the vector index first.")

        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_save_path)
        logger.info("Saved FAISS index to: %s", self.index_save_path)

    def load_index(self, chunks: Optional[List[Document]] = None, require_evaluated: bool = False) -> Optional[FAISS]:
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info("Index path does not exist: %s", self.index_save_path)
            return None

        if chunks is not None and not self._manifest_matches(chunks, require_evaluated=require_evaluated):
            logger.warning("Saved FAISS index manifest does not match the current corpus or model")
            return None

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded FAISS index from: %s", self.index_save_path)
            return self.vectorstore
        except Exception as exc:
            logger.warning("Failed to load FAISS index: %s", exc)
            return None

    def _manifest(
        self,
        chunks: List[Document],
        skipped_files: Optional[List[dict]] = None,
        evaluation: Optional[dict] = None,
        built_at: Optional[str] = None,
    ) -> dict:
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk.metadata.get("chunk_id", "").encode("utf-8"))
            digest.update(chunk.page_content.encode("utf-8"))
        topics = sorted({str(chunk.metadata.get("topic", "Unknown")) for chunk in chunks})
        return {
            "embedding_model": self.model_name,
            "chunk_count": len(chunks),
            "corpus_fingerprint": digest.hexdigest(),
            "indexed_topics": topics,
            "skipped_files": skipped_files or [],
            "evaluation": evaluation or {"status": "not_run"},
            "built_at": built_at,
            "schema_version": 2,
        }

    def _write_manifest(
        self,
        chunks: List[Document],
        skipped_files: Optional[List[dict]] = None,
        evaluation: Optional[dict] = None,
    ) -> None:
        from datetime import datetime, timezone

        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        manifest = self._manifest(
            chunks,
            skipped_files=skipped_files,
            evaluation=evaluation,
            built_at=datetime.now(timezone.utc).isoformat(),
        )
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _manifest_matches(self, chunks: List[Document], require_evaluated: bool = False) -> bool:
        try:
            saved = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        expected = self._manifest(
            chunks,
            skipped_files=saved.get("skipped_files", []),
            evaluation=saved.get("evaluation", {"status": "not_run"}),
            built_at=saved.get("built_at"),
        )
        if saved != expected:
            return False
        if require_evaluated and saved.get("evaluation", {}).get("status") != "passed":
            return False
        return True

    def write_manifest_for_existing_index(
        self,
        chunks: List[Document],
        skipped_files: Optional[List[dict]] = None,
        evaluation: Optional[dict[str, Any]] = None,
    ) -> None:
        self._write_manifest(chunks, skipped_files=skipped_files, evaluation=evaluation)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("Please build or load the vector index first.")

        return self.vectorstore.similarity_search(query, k=k)
