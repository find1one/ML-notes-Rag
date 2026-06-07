"""
Vector index construction and loading.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

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

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        if not chunks:
            raise ValueError("Document chunks cannot be empty.")

        logger.info("Building FAISS index with %s chunks", len(chunks))
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
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

    def load_index(self) -> Optional[FAISS]:
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info("Index path does not exist: %s", self.index_save_path)
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

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("Please build or load the vector index first.")

        return self.vectorstore.similarity_search(query, k=k)
