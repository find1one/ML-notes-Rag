"""
Interactive RAG system for the machine-learning Markdown notes.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
)

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MLNotesRAGSystem:
    """RAG system for ML notes stored as Markdown files."""

    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"Data path does not exist: {self.config.data_path}")

    def initialize_system(self) -> None:
        print("Initializing ML notes RAG system...")

        self.data_module = DataPreparationModule(
            self.config.data_path,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
        )
        if os.getenv("MOONSHOT_API_KEY"):
            self.generation_module = GenerationIntegrationModule(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            logger.warning("MOONSHOT_API_KEY is not set; generation endpoints will remain unavailable")

        print("System initialized.")

    def build_knowledge_base(self) -> None:
        print("\nBuilding knowledge base...")

        print("Loading Markdown notes...")
        self.data_module.load_documents()
        print("Chunking Markdown notes...")
        chunks = self.data_module.chunk_documents()

        vectorstore = self.index_module.load_index(chunks)

        if vectorstore is None:
            print("No saved ML-notes index found. Building a new FAISS index...")
            vectorstore = self.index_module.build_vector_index(chunks)
            self.index_module.save_index()
        else:
            print("Loaded saved ML-notes FAISS index.")

        print("Initializing hybrid retriever...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        stats = self.data_module.get_statistics()
        print("\nKnowledge base statistics:")
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Chunks: {stats['total_chunks']}")
        print(f"   Topics: {list(stats['topics'].keys())}")
        print(f"   Average chunk size: {stats['avg_chunk_size']:.1f}")
        print("Knowledge base is ready.")

    def load_knowledge_base(self) -> None:
        print("\nLoading verified knowledge base...")

        print("Loading Markdown notes...")
        self.data_module.load_documents()
        print("Chunking Markdown notes...")
        chunks = self.data_module.chunk_documents()

        vectorstore = self.index_module.load_index(chunks, require_evaluated=True)
        if vectorstore is None:
            raise ValueError(
                "Verified FAISS index is unavailable. Run `python code/build_index.py --publish` before starting the API."
            )

        print("Initializing hybrid retriever...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        stats = self.data_module.get_statistics()
        print("\nKnowledge base statistics:")
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Chunks: {stats['total_chunks']}")
        print(f"   Topics: {list(stats['topics'].keys())}")
        print(f"   Skipped files: {len(stats.get('skipped_files', []))}")
        print("Verified knowledge base is ready.")

    def ask_question(self, question: str, stream: bool = False):
        if not self.retrieval_module:
            raise ValueError("Please build the knowledge base first.")
        if not self.generation_module:
            raise ValueError("Please set the MOONSHOT_API_KEY environment variable.")

        print(f"\nQuestion: {question}")
        # Import locally to avoid the module-level main <-> rag_service import cycle.
        from rag_service import RAGService

        service = RAGService(self)
        prepared = service.prepare(question)
        print(f"Query type: {prepared.route_type}")
        print(f"Retrieval query: {prepared.retrieval_query}")

        if prepared.topic:
            print(f"Applying metadata filters: {{'topic': '{prepared.topic}'}}")
        fallback = "当前笔记中没有找到足够依据。你可以换一个更具体的机器学习主题或算法名。"
        if not prepared.docs:
            return iter((fallback,)) if stream else fallback

        chunk_info = []
        for chunk in prepared.docs:
            title = chunk.metadata.get("title", "Untitled")
            section = chunk.metadata.get("section_path", "Unknown section")
            chunk_info.append(f"{title} / {section}")
        print(f"Retrieved {len(prepared.docs)} chunks: {', '.join(chunk_info)}")

        if prepared.gate_decision != "passed":
            return iter((fallback,)) if stream else fallback
        if stream:
            return service.stream_answer(prepared)
        return service.generate_answer(prepared)

    def _extract_filters_from_query(self, query: str) -> dict:
        filters = {}
        topic = DataPreparationModule.match_topic_from_query(query)
        if topic:
            filters["topic"] = topic
        return filters

    def search_by_topic(self, topic: str, query: str = "") -> List[str]:
        if not self.retrieval_module:
            raise ValueError("Please build the knowledge base first.")

        search_query = query if query else topic
        filters = {"topic": topic}
        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=10)

        titles = []
        for doc in docs:
            title = doc.metadata.get("title", "Untitled")
            if title not in titles:
                titles.append(title)
        return titles

    def run_interactive(self) -> None:
        print("=" * 60)
        print("Machine Learning Notes RAG - Interactive Q&A")
        print("=" * 60)
        print("中文提问即可；系统会检索英文 Markdown 笔记并用中文回答。")

        self.initialize_system()
        self.build_knowledge_base()

        print("\n输入 exit、quit 或空行结束。")

        while True:
            try:
                user_input = input("\n你的问题: ").strip()
                if user_input.lower() in ["quit", "exit", ""]:
                    break

                stream_choice = input("是否流式输出? (y/n, 默认 y): ").strip().lower()
                use_stream = stream_choice != "n"

                if use_stream:
                    chunks = iter(self.ask_question(user_input, stream=True))
                    first_chunk = next(chunks)
                    print("\n回答:")
                    print(first_chunk, end="", flush=True)
                    for chunk in chunks:
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    answer = self.ask_question(user_input, stream=False)
                    print("\n回答:")
                    print(f"{answer}\n")
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"处理问题时出错: {exc}")

        print("\n已退出 ML notes RAG 系统。")


RecipeRAGSystem = MLNotesRAGSystem


def main() -> None:
    try:
        rag_system = MLNotesRAGSystem()
        rag_system.run_interactive()
    except Exception as exc:
        logger.error("System error: %s", exc)
        print(f"系统错误: {exc}")


if __name__ == "__main__":
    main()
