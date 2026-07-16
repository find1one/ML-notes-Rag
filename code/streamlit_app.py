"""Streamlit client for the ML Notes RAG FastAPI service."""

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.append(str(Path(__file__).parent))

from api_client import APIClientError, RAGAPIClient, apply_chat_event, new_chat_result


st.set_page_config(page_title="ML Notes RAG", page_icon="🔎", layout="wide")


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("没有检索到相关来源。")
        return

    for index, source in enumerate(sources, start=1):
        title = source.get("title") or "Untitled"
        topic = source.get("topic") or "Unknown"
        section = source.get("section") or "Unknown"
        path = source.get("path") or ""
        with st.expander(f"{index}. {title} · {section}", expanded=index == 1):
            st.caption(f"Topic: {topic} | Source: {path}")
            if source.get("score") is not None:
                st.caption(f"Score: {source['score']}")
            if source.get("excerpt"):
                st.markdown(source["excerpt"])


def render_result(result: dict[str, Any]) -> None:
    terminal_event = result.get("terminal_event")
    cached = result.get("cached", False)

    st.subheader("Answer")
    if result.get("answer"):
        st.markdown(result["answer"])
    elif terminal_event == "rejected":
        st.warning("当前笔记中没有找到足够依据。")
    elif result.get("client_error"):
        st.error(result["client_error"])

    if terminal_event == "done":
        st.success("已从 Redis 缓存返回。" if cached else "已完成检索和生成。")
    elif terminal_event == "rejected":
        st.warning(f"请求被证据门控拒绝：{result.get('reason') or '证据不足'}")
    elif terminal_event == "degraded":
        st.error(f"后端降级结束：{result.get('reason') or '未知原因'}")
    elif terminal_event == "cancelled":
        st.warning("请求已取消。")

    debug_values = {
        "Trace ID": result.get("trace_id"),
        "Query ID": result.get("query_id"),
        "Terminal event": terminal_event,
        "Cache hit": cached,
        "Route type": result.get("route_type"),
        "Gate decision": result.get("gate_decision"),
    }
    with st.expander("Request details", expanded=False):
        for label, value in debug_values.items():
            st.write(f"{label}: `{value}`")
        if result.get("retrieval_query"):
            st.write("Retrieval query:")
            st.code(result["retrieval_query"])

    st.subheader("Retrieved Sources")
    render_sources(result.get("sources") or [])


def render_feedback(client: RAGAPIClient, result: dict[str, Any]) -> None:
    if result.get("terminal_event") != "done":
        return
    st.subheader("Feedback")
    if result.get("feedback_submitted"):
        st.success("反馈已提交，感谢你的评价。")
        return
    if not result.get("query_id"):
        st.info("本次回答没有可用的 query_id，数据库可能未就绪，暂时无法提交反馈。")
        return

    with st.form("feedback_form"):
        rating_label = st.radio("这个回答有帮助吗？", ["有帮助", "没帮助"], horizontal=True)
        comment = st.text_area("补充说明（可选）", max_chars=2000)
        submitted = st.form_submit_button("提交反馈")
    if submitted:
        rating = "helpful" if rating_label == "有帮助" else "not_helpful"
        try:
            client.submit_feedback(result["query_id"], rating, comment)
        except APIClientError as exc:
            st.error(str(exc))
        else:
            result["feedback_submitted"] = True
            st.session_state["chat_result"] = result
            st.rerun()


def main() -> None:
    st.title("Machine Learning Notes RAG")
    st.caption("Streamlit 通过 FastAPI 检索机器学习笔记、生成回答并复用 Redis 缓存。")

    with st.sidebar:
        st.header("Backend")
        api_url = st.text_input(
            "FastAPI URL",
            value=os.getenv("RAG_API_URL", "http://127.0.0.1:8000"),
        ).strip()
        cache_label = st.radio("Cache mode", ["优先使用缓存", "强制重新生成"])
        debug = st.toggle("Debug logging", value=False)

    client = RAGAPIClient(api_url)
    readiness = None
    readiness_error = None
    try:
        readiness = client.get_readiness()
    except APIClientError as exc:
        readiness_error = str(exc)

    if readiness_error:
        st.error(readiness_error)
        st.caption("请先启动 FastAPI：`cd code && uvicorn api:app --reload`。")
    elif not readiness.get("rag_ready", readiness.get("ready", False)):
        st.error("FastAPI 已连接，但 RAG 服务尚未就绪。请检查已发布索引和 MOONSHOT_API_KEY。")
    else:
        database_state = "ready" if readiness.get("database_ready") else "unavailable"
        st.success(f"FastAPI ready · Database {database_state}")
        if not readiness.get("database_ready"):
            st.caption("问答仍可使用，但本次请求可能没有 query_id，反馈功能将不可用。")

    rag_ready = bool(readiness and readiness.get("rag_ready", readiness.get("ready", False)))
    with st.form("question_form"):
        question = st.text_input("Question", value="线性回归怎么做变量选择？")
        submitted = st.form_submit_button("Ask", type="primary", disabled=not rag_ready)

    if submitted:
        question = question.strip()
        if not question:
            st.warning("请输入问题。")
        else:
            result = new_chat_result(question)
            answer_placeholder = st.empty()
            status_placeholder = st.empty()
            cache_mode = "default" if cache_label == "优先使用缓存" else "fresh"
            try:
                with st.spinner("FastAPI 正在处理..."):
                    for event in client.stream_chat(question, cache_mode=cache_mode, debug=debug):
                        apply_chat_event(result, event)
                        if event.event == "token":
                            answer_placeholder.markdown(result["answer"] + "▌")
                        elif event.event == "waiting_for_first_token":
                            status_placeholder.info("正在等待模型返回首个 token...")
            except APIClientError as exc:
                result["client_error"] = str(exc)
                result["terminal_event"] = result["terminal_event"] or "client_error"
            finally:
                answer_placeholder.empty()
                status_placeholder.empty()
                st.session_state["chat_result"] = result

    result = st.session_state.get("chat_result")
    if result:
        render_result(result)
        render_feedback(client, result)


if __name__ == "__main__":
    main()
