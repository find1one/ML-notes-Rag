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

        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("Please set the MOONSHOT_API_KEY environment variable.")

    def initialize_system(self) -> None:
        print("Initializing ML notes RAG system...")

        self.data_module = DataPreparationModule(self.config.data_path)
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
        )
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        print("System initialized.")

    def build_knowledge_base(self) -> None:
        print("\nBuilding knowledge base...")

        vectorstore = self.index_module.load_index()

        print("Loading Markdown notes...")
        self.data_module.load_documents()
        print("Chunking Markdown notes...")
        chunks = self.data_module.chunk_documents()

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

    def ask_question(self, question: str, stream: bool = False):
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("Please build the knowledge base first.")

        print(f"\nQuestion: {question}")

        route_type = self.generation_module.query_router(question)
        print(f"Query type: {route_type}")

        filters = self._extract_filters_from_query(question)
        if route_type == "list" and filters.get("topic"):
            topic_docs = self.data_module.filter_documents_by_topic(filters["topic"])
            if topic_docs:
                print(f"Listing documents for topic: {filters['topic']}")
                return self.generation_module.generate_list_answer(question, topic_docs)

        if route_type == "list":
            rewritten_query = question
        else:
            print("Rewriting query for English ML-note retrieval...")
            rewritten_query = self.generation_module.query_rewrite(question)

        if filters:
            print(f"Applying metadata filters: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query,
                filters,
                top_k=self.config.top_k,
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query,
                top_k=self.config.top_k,
            )

        if not relevant_chunks:
            return "当前笔记中没有找到足够依据。你可以换一个更具体的机器学习主题或算法名。"

        chunk_info = []
        for chunk in relevant_chunks:
            title = chunk.metadata.get("title", "Untitled")
            section = chunk.metadata.get("section_path", "Unknown section")
            chunk_info.append(f"{title} / {section}")
        print(f"Retrieved {len(relevant_chunks)} chunks: {', '.join(chunk_info)}")

        if route_type == "list":
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            return self.generation_module.generate_list_answer(question, relevant_docs)

        if route_type == "detail":
            if stream:
                return self.generation_module.generate_step_by_step_answer_stream(question, relevant_chunks)
            return self.generation_module.generate_step_by_step_answer(question, relevant_chunks)

        if stream:
            return self.generation_module.generate_basic_answer_stream(question, relevant_chunks)
        return self.generation_module.generate_basic_answer(question, relevant_chunks)

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

                print("\n回答:")
                if use_stream:
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    answer = self.ask_question(user_input, stream=False)
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
