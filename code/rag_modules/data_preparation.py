"""
Data preparation for the machine-learning Markdown notes.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """Load Markdown notes, enrich metadata, and split them into retrievable chunks."""

    TOPIC_ALIASES = {
        "prerequisites": ["prerequisite", "基础", "先修", "预备知识", "数学基础"],
        "regression": ["regression", "回归", "线性回归", "多项式回归", "svr"],
        "classification": ["classification", "分类", "逻辑回归", "knn", "svm", "朴素贝叶斯", "决策树", "随机森林"],
        "clustering": ["clustering", "聚类", "k-means", "层次聚类", "gaussian mixture", "gmm"],
        "association rule learning": ["association", "关联规则", "apriori", "eclat"],
        "reinforcement learning": ["reinforcement", "强化学习"],
        "natural language processing": ["nlp", "自然语言处理", "文本处理"],
        "deep learning": ["deep learning", "深度学习", "神经网络", "activation"],
        "dimensionality reduction": ["dimensionality", "降维", "pca"],
        "recommendation engines": ["recommendation", "推荐系统", "推荐引擎"],
        "model selection and boosting": ["model selection", "boosting", "模型选择", "提升"],
        "time series": ["time series", "时间序列"],
        "constraint satisfaction problems": ["constraint", "约束满足", "csp"],
        "appendix": ["appendix", "附录", "programming", "python", "r"],
    }

    def __init__(self, data_path: str, chunk_size: int = 1200, chunk_overlap: int = 150):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}
        self.skipped_files: List[Dict[str, Any]] = []

    # 使用langchain封装好的Document对象，保存正文和元数据，方便下一步的chunk
    def load_documents(self) -> List[Document]:
        logger.info("Loading Markdown notes from %s", self.data_path)
        
        # 返回路径的绝对路径并检查其是否存在
        data_root = Path(self.data_path).resolve()
        if not data_root.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        documents: List[Document] = []

        # 遍历所有 md格式文件(rglob(*.md))，读取内容，获取相对路径，检查内容长度，生成父id
        for md_file in sorted(data_root.rglob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                relative_path = md_file.resolve().relative_to(data_root).as_posix()
                body = content.strip() # 返回去掉首尾空白字符的字符串
                if len(body) < 100: #字符串长度小于一百不参考
                    self.skipped_files.append({
                        "path": relative_path,
                        "reason": "empty_or_too_short",
                        "size": len(content),
                    })
                    continue
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()#16进制hash制

                # langchain中的Document对象，包含page_content和metadata两个属性
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "relative_path": relative_path,
                        "parent_id": parent_id,
                        "doc_type": "parent",
                    },
                )
                self._enhance_metadata(doc)
                documents.append(doc)
            except Exception as exc:
                logger.warning("Failed to read Markdown file %s: %s", md_file, exc)

        self.documents = documents
        logger.info("Loaded %s Markdown documents", len(documents))
        return documents

    # 根据文档内容和文件名增强元数据，包括标题、主题、章节和节路径
    def _enhance_metadata(self, doc: Document) -> None:
        source = Path(doc.metadata.get("source", ""))
        relative_path = doc.metadata.get("relative_path", source.name)
        parts = Path(relative_path).parts #相对路径拆成多个parts Path("Regression/Linear Regression.md").parts -- ("Regression", "Linear Regression.md")

        chapter = parts[0] if len(parts) > 1 else "Root"
        topic = self._topic_from_chapter(chapter) # 获得topic，详情见_topic_from_chapter方法
        # 先尝试从 Markdown 正文中提取一级标题；如果正文中没有标题，就使用文件名生成标题。
        title = self._extract_title(doc.page_content) or self._title_from_filename(source.stem)

        doc.metadata.update(
            {
                "title": title,
                "topic": topic,
                "chapter": chapter,
                "section_path": title,
            }
        )

    def chunk_documents(self) -> List[Document]:
        if not self.documents:
            raise ValueError("Please load documents before chunking.")

        logger.info("Splitting Markdown notes by headings and chunk length")
        chunks = self._markdown_header_split()

        for index, chunk in enumerate(chunks):
            chunk.metadata["batch_index"] = index
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = chunks
        logger.info("Generated %s chunks", len(chunks))
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False,
        )
        length_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_chunks: List[Document] = []

        for doc in self.documents:
            try:
                header_chunks = markdown_splitter.split_text(doc.page_content)
            except Exception as exc:
                logger.warning("Markdown split failed for %s: %s", doc.metadata.get("source"), exc)
                header_chunks = [Document(page_content=doc.page_content, metadata={})]

            for header_index, header_chunk in enumerate(header_chunks):
                merged_metadata = dict(doc.metadata)
                merged_metadata.update(header_chunk.metadata)
                section_path = self._build_section_path(merged_metadata)
                merged_metadata["section_path"] = section_path

                if len(header_chunk.page_content) > 1400:
                    sub_chunks = length_splitter.split_text(header_chunk.page_content)
                else:
                    sub_chunks = [header_chunk.page_content]

                for sub_index, text in enumerate(sub_chunks):
                    chunk_id_seed = "|".join(
                        [
                            merged_metadata.get("relative_path", ""),
                            section_path,
                            str(header_index),
                            str(sub_index),
                            hashlib.md5(text.encode("utf-8")).hexdigest(),
                        ]
                    )
                    child_id = hashlib.md5(chunk_id_seed.encode("utf-8")).hexdigest()
                    metadata = dict(merged_metadata)
                    metadata.update(
                        {
                            "chunk_id": child_id,
                            "doc_type": "child",
                            "chunk_index": len(all_chunks),
                        }
                    )
                    self.parent_child_map[child_id] = metadata["parent_id"]
                    all_chunks.append(Document(page_content=text, metadata=metadata))

        return all_chunks

    def filter_documents_by_topic(self, topic: str) -> List[Document]:
        normalized = topic.lower()
        return [
            doc
            for doc in self.documents
            if doc.metadata.get("topic", "").lower() == normalized
            or doc.metadata.get("chapter", "").lower() == normalized
        ]

    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {}

        topics: Dict[str, int] = {}
        chapters: Dict[str, int] = {}
        for doc in self.documents:
            topic = doc.metadata.get("topic", "Unknown")
            chapter = doc.metadata.get("chapter", "Unknown")
            topics[topic] = topics.get(topic, 0) + 1
            chapters[chapter] = chapters.get(chapter, 0) + 1

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "topics": topics,
            "chapters": chapters,
            "avg_chunk_size": (
                sum(chunk.metadata.get("chunk_size", 0) for chunk in self.chunks) / len(self.chunks)
                if self.chunks
                else 0
            ),
            "skipped_files": self.skipped_files,
        }

    def export_metadata(self, output_path: str) -> None:
        import json

        metadata_list = [
            {
                "source": doc.metadata.get("source"),
                "relative_path": doc.metadata.get("relative_path"),
                "title": doc.metadata.get("title"),
                "topic": doc.metadata.get("topic"),
                "chapter": doc.metadata.get("chapter"),
                "content_length": len(doc.page_content),
            }
            for doc in self.documents
        ]

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(metadata_list, file, ensure_ascii=False, indent=2)

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        parent_relevance: Dict[str, int] = {}
        parent_docs_map: Dict[str, Document] = {}

        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if not parent_id:
                continue
            parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1
            if parent_id not in parent_docs_map:
                for doc in self.documents:
                    if doc.metadata.get("parent_id") == parent_id:
                        parent_docs_map[parent_id] = doc
                        break

        sorted_parent_ids = sorted(parent_relevance, key=parent_relevance.get, reverse=True)
        return [parent_docs_map[parent_id] for parent_id in sorted_parent_ids if parent_id in parent_docs_map]

    @classmethod
    def get_supported_topics(cls) -> List[str]:
        return list(cls.TOPIC_ALIASES.keys())

    @classmethod
    def match_topic_from_query(cls, query: str) -> Optional[str]:
        # 检查用户问题中是否包含已知主题或其别名，如果匹配到，则通过_display_topic返回标准化的主题名称,如果没匹配到返回None
        normalized = query.lower()
        for topic, aliases in cls.TOPIC_ALIASES.items():
            if topic in normalized or any(alias.lower() in normalized for alias in aliases):
                return cls._display_topic(topic)
        return None

    @staticmethod
    def _extract_title(content: str) -> Optional[str]:
        for line in content.splitlines():
            match = re.match(r"^\s*#\s+(.+?)\s*$", line) #匹配以#开头的标题行，提取标题文本
            if match:
                return match.group(1).strip() #取出第一个括号 (.+?) 捕获到的内容，也就是标题文字。
        return None

    @staticmethod
    def _title_from_filename(stem: str) -> str:
        title = re.sub(r"^\d+[-_]", "", stem)
        title = title.replace("-", " ").replace("_", " ")
        return title.strip() or stem

    # 从章节中提取主题并更改格式
    @classmethod
    def _topic_from_chapter(cls, chapter: str) -> str:
        raw = re.sub(r"^\d+[-_]", "", chapter) #\d+一个或多个数字，[-_]一个"-"或"_"，^表示开头，替换为空字符串。必须是01_之类的情况
        raw = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw) #在小写字母后面紧跟大写字母的地方插入空格，"DeepLearning" -> "Deep Learning"
        raw = raw.replace("-", " ").replace("_", " ").strip() #横杠和下划线都替换成空格，并且去掉首位空格
        return cls._display_topic(raw.lower()) #全变小写，再传给_display_topic。

    # 如果在known找到，返回标准化的主题名称，否则返回首字母大写的主题名称
    @staticmethod
    def _display_topic(topic: str) -> str:
        known = {
            "prerequisites": "Prerequisites",
            "regression": "Regression",
            "classification": "Classification",
            "clustering": "Clustering",
            "association rule learning": "Association Rule Learning",
            "reinforcement learning": "Reinforcement Learning",
            "natural language processing": "Natural Language Processing",
            "deep learning": "Deep Learning",
            "dimensionality reduction": "Dimensionality Reduction",
            "recommendation engines": "Recommendation Engines",
            "model selection and boosting": "Model Selection and Boosting",
            "time series": "Time Series",
            "constraint satisfaction problems": "Constraint Satisfaction Problems",
            "appendix": "Appendix",
        }
        return known.get(topic, topic.title())

    @staticmethod
    def _build_section_path(metadata: Dict[str, Any]) -> str:
        headings = [
            metadata.get("h1") or metadata.get("title"),
            metadata.get("h2"),
            metadata.get("h3"),
            metadata.get("h4"),
        ]
        return " > ".join(str(item).strip() for item in headings if item)
