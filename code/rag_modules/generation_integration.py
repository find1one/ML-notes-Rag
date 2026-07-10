"""
LLM integration for machine-learning notes Q&A.
"""

import logging
import os
from typing import List

from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """Route, rewrite, and answer questions using retrieved ML-note context."""

    def __init__(self, model_name: str = "kimi-k2.6", temperature: float = 0.6, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = self._normalize_temperature(model_name, temperature)
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    @staticmethod
    def _normalize_temperature(model_name: str, temperature: float) -> float:
        """Apply model-specific parameter constraints before making a request."""
        if model_name.lower().startswith("kimi-k2") and temperature != 0.6:
            logger.warning(
                "%s only accepts temperature=0.6; overriding configured value %s",
                model_name,
                temperature,
            )
            return 0.6
        return temperature

    def setup_llm(self) -> None:
        logger.info("Initializing LLM: %s", self.model_name)

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("Please set the MOONSHOT_API_KEY environment variable.")

        llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            moonshot_api_key=api_key,
        )
        extra_body = self._model_extra_body(self.model_name)
        # MoonshotChat's legacy adapter drops model_kwargs, while Runnable.bind
        # forwards extra_body to both invoke() and stream().
        self.llm = llm.bind(extra_body=extra_body) if extra_body else llm

    @staticmethod
    def _model_extra_body(model_name: str) -> dict:
        """Disable long-form thinking for short, retrieval-grounded answers."""
        if model_name.lower() in {"kimi-k2.5", "kimi-k2.6"}:
            return {"thinking": {"type": "disabled"}}
        return {}

    @staticmethod
    def _nonempty_chunks(chunks):
        """Yield visible answer text and fail loudly when a stream is empty."""
        emitted = False
        for chunk in chunks:
            if not chunk:
                continue
            emitted = True
            yield chunk
        if not emitted:
            raise RuntimeError(
                "Moonshot returned no answer content. Check the model's thinking mode and max_tokens."
            )

    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        context = self._build_context(context_docs)
        prompt = self._basic_prompt()
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        context = self._build_context(context_docs, max_length=3200)
        prompt = self._detail_prompt()
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        context = self._build_context(context_docs)
        prompt = self._basic_prompt()
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        yield from self._nonempty_chunks(chain.stream(query))

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        context = self._build_context(context_docs, max_length=3200)
        prompt = self._detail_prompt()
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        yield from self._nonempty_chunks(chain.stream(query))

    def query_rewrite(self, query: str) -> str:
        prompt = PromptTemplate(
            template="""
你是机器学习笔记检索查询改写器。用户可能用中文提问，但知识库是英文 Markdown 笔记。

请把问题改写成更适合检索英文机器学习笔记的短查询：
- 保留原意。
- 补充关键英文术语。
- 如果用户问题已经包含清晰英文术语，可以直接返回原问题。
- 不要回答问题，只输出最终检索查询。

示例：
中文：线性回归怎么做变量选择
输出：linear regression variable selection feature selection p-value backward elimination

中文：K-means 聚类的步骤
输出：K-means clustering algorithm steps

原始问题：{query}
最终检索查询：""",
            input_variables=["query"],
        )
        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        rewritten = chain.invoke(query).strip()
        logger.info("Query rewritten: %r -> %r", query, rewritten)
        return rewritten or query

    def query_router(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
请把用户的问题分类为以下三类之一，只返回 list、detail 或 general。

list：用户想要主题、算法、文档或知识点列表。
detail：用户想要解释某个概念、算法步骤、公式、假设、优缺点或使用场景。
general：其他一般问题。

用户问题：{query}
分类结果：""")
        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        result = chain.invoke(query).strip().lower()
        return result if result in {"list", "detail", "general"} else "general"

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "没有在机器学习笔记中找到相关条目。"

        seen = set()
        items = []
        for doc in context_docs:
            title = doc.metadata.get("title", "Untitled")
            topic = doc.metadata.get("topic", "Unknown")
            key = (topic, title)
            if key not in seen:
                seen.add(key)
                items.append(f"{title}（{topic}）")

        return "根据当前笔记，相关条目包括：\n" + "\n".join(
            f"{index}. {item}" for index, item in enumerate(items, 1)
        )

    def _build_context(self, docs: List[Document], max_length: int = 2400) -> str:
        if not docs:
            return "没有检索到相关机器学习笔记。"

        context_parts = []
        current_length = 0

        for index, doc in enumerate(docs, 1):
            metadata_info = (
                f"[S{index}] "
                f"title={doc.metadata.get('title', 'Untitled')} | "
                f"topic={doc.metadata.get('topic', 'Unknown')} | "
                f"section={doc.metadata.get('section_path', 'Unknown')} | "
                f"path={doc.metadata.get('relative_path', doc.metadata.get('source', ''))}"
            )
            doc_text = f"{metadata_info}\n{doc.page_content.strip()}\n"
            if current_length + len(doc_text) > max_length:
                remaining = max_length - current_length
                if remaining > 300:
                    context_parts.append(doc_text[:remaining].rstrip() + "\n[content truncated]")
                break
            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n" + ("=" * 50 + "\n").join(context_parts)

    @staticmethod
    def _basic_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""
你是一个严谨的机器学习学习助手。请只根据给定笔记上下文回答用户问题。

要求：
- 用中文回答。
- 如果上下文不足，明确说“当前笔记中没有找到足够依据”，不要编造。
- 回答控制在 4 句话以内。
- 关键结论必须带来源编号，例如 [S1]。
- 优先引用笔记中的术语、标题或章节。
- 对英文术语保留英文，并给出必要中文解释。

用户问题：{question}

相关笔记：
{context}

回答：""")

    @staticmethod
    def _detail_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""
你是一个严谨的机器学习学习助手。请基于给定 Markdown 笔记，用中文解释用户问题。

回答结构按实际内容选择，不要硬凑；通常可以包含：
1. 核心概念
2. 关键公式或步骤
3. 适用前提/假设
4. 注意点或局限
5. 笔记来源标题

限制：
- 只能使用上下文中有依据的信息。
- 如果资料不足，直接说明缺失点。
- 回答尽量简短，优先列 3-5 个要点。
- 每个关键结论必须带来源编号，例如 [S1]。
- 不要把外部知识伪装成笔记内容。

用户问题：{question}

相关笔记：
{context}

回答：""")
