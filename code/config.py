"""Runtime configuration for the ML Notes RAG service."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the repository-level environment file for every entry point (uvicorn,
# CLI scripts, tests, and Streamlit).  IDE-specific env-file settings are not
# applied when the API is started from a regular shell.
load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass
class RAGConfig:
    """Configuration shared by CLI, API, evaluation, and Streamlit."""

    data_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "ML-Notes-in-Markdown-master"))
    index_save_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "code" / "vector_index_ml_notes"))
    embedding_model: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL_PATH", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "kimi-k2.6"))
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 4))
    candidate_k: int = field(default_factory=lambda: _env_int("RAG_CANDIDATE_K", 80))
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 1200))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 150))
    # Kimi K2.6 currently only accepts temperature=0.6 via Moonshot's API.
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.6))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 800))
    first_token_wait_seconds: int = field(default_factory=lambda: _env_int("FIRST_TOKEN_WAIT_SECONDS", 10))
    first_token_timeout_seconds: int = field(default_factory=lambda: _env_int("FIRST_TOKEN_TIMEOUT_SECONDS", 30))
    stream_idle_timeout_seconds: int = field(default_factory=lambda: _env_int("STREAM_IDLE_TIMEOUT_SECONDS", 20))
    orphan_request_timeout_seconds: int = field(default_factory=lambda: _env_int("ORPHAN_REQUEST_TIMEOUT_SECONDS", 120))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))
    cache_ttl_seconds: int = field(default_factory=lambda: _env_int("EXACT_CACHE_TTL_SECONDS", 86400))
    database_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", "mysql+pymysql://rag_user:rag_password@127.0.0.1:3306/rag_app"
    ))
    database_connect_timeout: int = field(default_factory=lambda: _env_int("DATABASE_CONNECT_TIMEOUT", 5))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


DEFAULT_CONFIG = RAGConfig()
