"""
Streamlit UI for the ML notes RAG system.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG
from rag_modules import (
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
)


st.set_page_config(page_title="ML Notes RAG", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def load_retrieval_stack() -> Tuple[DataPreparationModule, RetrievalOptimizationModule]:
    data_module = DataPreparationModule(DEFAULT_CONFIG.data_path)
    data_module.load_documents()
    chunks = data_module.chunk_documents()

    index_module = IndexConstructionModule(
        DEFAULT_CONFIG.embedding_model,
        DEFAULT_CONFIG.index_save_path,
    )
    vectorstore = index_module.load_index(chunks)
    if vectorstore is None:
        vectorstore = index_module.build_vector_index(chunks)
        index_module.save_index()

    retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
    return data_module, retrieval_module


@st.cache_resource(show_spinner=False)
def load_generation_module() -> GenerationIntegrationModule:
    return GenerationIntegrationModule(
        model_name=DEFAULT_CONFIG.llm_model,
        temperature=DEFAULT_CONFIG.temperature,
        max_tokens=DEFAULT_CONFIG.max_tokens,
    )


def route_query(question: str) -> str:
    list_keywords = ["哪些", "有什么", "列表", "方法", "算法", "目录", "内容"]
    if any(keyword in question for keyword in list_keywords):
        return "list"
    return "detail"


def rewrite_query_fallback(question: str) -> str:
    replacements: Dict[str, str] = {
        "线性回归": "linear regression",
        "变量选择": "variable selection feature selection p-value backward elimination",
        "分类": "classification",
        "聚类": "clustering",
        "强化学习": "reinforcement learning",
        "深度学习": "deep learning",
        "时间序列": "time series",
        "状态空间": "state space models",
        "决策树": "decision tree",
        "随机森林": "random forest",
        "朴素贝叶斯": "naive bayes",
        "降维": "dimensionality reduction",
        "主成分": "principal component analysis PCA",
    }
    rewritten = question
    for source, target in replacements.items():
        rewritten = rewritten.replace(source, f"{source} {target}")
    return rewritten


def retrieve_chunks(
    question: str,
    retrieval_module: RetrievalOptimizationModule,
    use_llm_rewrite: bool,
    top_k: int,
) -> Tuple[str, List]:
    topic = DataPreparationModule.match_topic_from_query(question)

    if use_llm_rewrite:
        generator = load_generation_module()
        retrieval_query = generator.query_rewrite(question)
    else:
        retrieval_query = rewrite_query_fallback(question)

    filters = {"topic": topic} if topic else {}
    if filters:
        docs = retrieval_module.metadata_filtered_search(retrieval_query, filters, top_k=top_k)
    else:
        docs = retrieval_module.hybrid_search(retrieval_query, top_k=top_k)
    return retrieval_query, docs


def render_sources(docs: List) -> None:
    if not docs:
        st.info("没有检索到相关 chunk。")
        return

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        title = metadata.get("title", "Untitled")
        topic = metadata.get("topic", "Unknown")
        section = metadata.get("section_path", "Unknown")
        path = metadata.get("relative_path", metadata.get("source", ""))
        with st.expander(f"{index}. {title} · {section}", expanded=index == 1):
            st.caption(f"Topic: {topic} | Source: {path}")
            st.markdown(doc.page_content[:1600])


def generate_answer(question: str, docs: List, route_type: str) -> str:
    generator = load_generation_module()
    if route_type == "detail":
        return generator.generate_step_by_step_answer(question, docs)
    return generator.generate_basic_answer(question, docs)


def format_retrieval_fallback_answer(question: str, docs: List) -> str:
    if not docs:
        return "没有检索到足够相关的笔记内容，暂时无法生成回答。"

    lines = [
        "LLM 没有返回可用回答。下面是基于检索结果整理的笔记片段，供你先判断召回是否正确。",
        "",
        f"问题：{question}",
        "",
        "检索依据：",
    ]

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        title = metadata.get("title", "Untitled")
        topic = metadata.get("topic", "Unknown")
        section = metadata.get("section_path", "Unknown")
        path = metadata.get("relative_path", metadata.get("source", ""))
        excerpt = " ".join(doc.page_content.strip().split())
        if len(excerpt) > 500:
            excerpt = excerpt[:500].rstrip() + "..."
        lines.extend(
            [
                "",
                f"{index}. **{title}**（{topic}）",
                f"   - Section: `{section}`",
                f"   - Source: `{path}`",
                f"   - Excerpt: {excerpt}",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    st.title("Machine Learning Notes RAG")
    st.caption("中文提问，检索英文机器学习 Markdown 笔记，并显示回答来源。")

    api_key_available = bool(os.getenv("MOONSHOT_API_KEY"))

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top-k chunks", min_value=1, max_value=8, value=DEFAULT_CONFIG.top_k)
        use_llm_answer = st.toggle("Generate answer with LLM", value=api_key_available)
        use_llm_rewrite = st.toggle("Use LLM query rewrite", value=False)
        st.caption(f"MOONSHOT_API_KEY: {'set' if api_key_available else 'not set'}")
        st.divider()
        st.write("Data path")
        st.code(DEFAULT_CONFIG.data_path)
        st.write("Index path")
        st.code(DEFAULT_CONFIG.index_save_path)

    data_module, retrieval_module = load_retrieval_stack()
    stats = data_module.get_statistics()

    metric_cols = st.columns(3)
    metric_cols[0].metric("Documents", stats.get("total_documents", 0))
    metric_cols[1].metric("Chunks", stats.get("total_chunks", 0))
    metric_cols[2].metric("Topics", len(stats.get("topics", {})))

    default_question = "线性回归怎么做变量选择？"
    question = st.text_input("Question", value=default_question)
    run = st.button("Search", type="primary")

    if not run and "last_question" not in st.session_state:
        st.stop()

    if run:
        st.session_state["last_question"] = question
        st.session_state.pop("last_answer", None)
        st.session_state.pop("last_error", None)

    question = st.session_state.get("last_question", question).strip()
    if not question:
        st.warning("请输入问题。")
        st.stop()

    route_type = route_query(question)
    topic = DataPreparationModule.match_topic_from_query(question)

    st.subheader("Retrieval")
    st.write(f"Query type: `{route_type}`")
    st.write(f"Detected topic: `{topic or 'None'}`")

    if route_type == "list" and topic:
        docs = data_module.filter_documents_by_topic(topic)
        st.write(f"Listing documents for topic `{topic}`.")
        answer = format_doc_list(docs)
        st.subheader("Answer")
        st.markdown(answer)
        st.subheader("Documents")
        render_parent_docs(docs)
        st.stop()

    with st.spinner("Retrieving chunks..."):
        retrieval_query, docs = retrieve_chunks(question, retrieval_module, use_llm_rewrite, top_k)

    st.write("Retrieval query:")
    st.code(retrieval_query)

    st.subheader("Answer")
    if use_llm_answer:
        if not api_key_available:
            st.error("MOONSHOT_API_KEY is not set. Turn off LLM answer generation or set the key.")
        elif not docs:
            st.warning("没有检索到相关内容，因此没有调用 LLM 生成回答。")
        else:
            try:
                with st.spinner("Generating answer with Moonshot/Kimi..."):
                    answer = generate_answer(question, docs, route_type).strip()
                if answer:
                    st.session_state["last_answer"] = answer
                    st.markdown(answer)
                else:
                    st.warning("LLM 返回了空回答。请检查模型配置、API 状态，或先查看下方检索来源。")
                    st.markdown(format_retrieval_fallback_answer(question, docs))
            except Exception as exc:
                st.session_state["last_error"] = str(exc)
                st.error("LLM answer generation failed.")
                st.exception(exc)
                st.markdown(format_retrieval_fallback_answer(question, docs))
    else:
        st.info("LLM answer generation is disabled. Review retrieved sources below.")

    st.subheader("Retrieved Sources")
    render_sources(docs)


def format_doc_list(docs: List) -> str:
    if not docs:
        return "没有找到对应主题的文档。"
    lines = ["根据当前笔记，相关条目包括："]
    seen = set()
    index = 1
    for doc in docs:
        title = doc.metadata.get("title", "Untitled")
        topic = doc.metadata.get("topic", "Unknown")
        key = (topic, title)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{index}. {title}（{topic}）")
        index += 1
    return "\n".join(lines)


def render_parent_docs(docs: List) -> None:
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        title = metadata.get("title", "Untitled")
        topic = metadata.get("topic", "Unknown")
        path = metadata.get("relative_path", metadata.get("source", ""))
        with st.expander(f"{index}. {title}", expanded=index <= 3):
            st.caption(f"Topic: {topic} | Source: {path}")
            st.markdown(doc.page_content[:1200])


if __name__ == "__main__":
    main()
