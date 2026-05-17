"""
RAG system configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RAGConfig:
    """Configuration for the machine-learning notes RAG system."""

    data_path: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "data" / "ML-Notes-in-Markdown-master")
    )
    index_save_path: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "code" / "vector_index_ml_notes")
    )

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "kimi-k2.6"

    top_k: int = 4

    temperature: float = 1.0
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_path": self.data_path,
            "index_save_path": self.index_save_path,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


DEFAULT_CONFIG = RAGConfig()
