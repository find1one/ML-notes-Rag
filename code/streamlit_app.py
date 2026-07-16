"""
Streamlit UI for the ML notes RAG system.
"""

import os
import sys
from pathlib import Path
from typing import List

import streamlit as st

sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG
from main import MLNotesRAGSystem
from rag_service import RAGPrepared, RAGService


st.set_page_config(page_title="ML Notes RAG", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def load_rag_service() -> RAGService:
    rag_system = MLNotesRAGSystem(DEFAULT_CONFIG)
    rag_system.initialize_system()
    rag_system.load_knowledge_base()
    return RAGService(rag_system)


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
        use_llm_answer = st.toggle("Generate answer with LLM", value=api_key_available)
        st.caption(f"MOONSHOT_API_KEY: {'set' if api_key_available else 'not set'}")
        st.divider()
        st.write("Data path")
        st.code(DEFAULT_CONFIG.data_path)
        st.write("Index path")
        st.code(DEFAULT_CONFIG.index_save_path)

    service = load_rag_service()
    data_module = service.rag_system.data_module
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

    try:
        with st.spinner("Retrieving notes..."):
            prepared: RAGPrepared = service.prepare(question)
    except Exception as exc:
        st.error("RAG preparation failed.")
        st.exception(exc)
        st.stop()

    st.subheader("Retrieval")
    st.write(f"Query type: `{prepared.route_type}`")
    st.write(f"Detected topic: `{prepared.topic or 'None'}`")
    st.write("Retrieval query:")
    st.code(prepared.retrieval_query)

    docs = prepared.docs

    st.subheader("Answer")
    if prepared.gate_decision != "passed":
        st.warning("当前笔记中没有找到足够依据。")
        if prepared.rejection_reason:
            st.caption(f"Evidence gate: {prepared.rejection_reason}")
    elif use_llm_answer:
        if not api_key_available:
            st.error("MOONSHOT_API_KEY is not set. Turn off LLM answer generation or set the key.")
        elif not docs:
            st.warning("没有检索到相关内容，因此没有调用 LLM 生成回答。")
        else:
            try:
                with st.spinner("Generating answer with Moonshot/Kimi..."):
                    answer = service.generate_answer(prepared).strip()
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


if __name__ == "__main__":
    main()
