import logging
import time 

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from main import MLNotesRAGSystem
from rag_logger import log_rag_query

app = FastAPI()
logger = logging.getLogger(__name__)
MAX_EMPTY_ANSWER_RETRIES = 2

rag_system = MLNotesRAGSystem()
rag_system.initialize_system()
rag_system.build_knowledge_base()

class ChatRequest(BaseModel):
    question: str
class ChatResponse(BaseModel):
    answer: str

class Source(BaseModel):
    title: str
    topic: str
    section: str
    path: str

class ChatDebug(ChatResponse):
    route_type: str
    retrieval_query: Optional[str]
    sources: list[Source]


def _is_blank(text: str) -> bool:
    return not text or not text.strip()


def _raise_empty_answer() -> None:
    raise HTTPException(status_code=502, detail="LLM returned an empty response")


def _generate_answer(question: str, route_type: str, retrieved_docs: list) -> str:
    if route_type == "list":
        parent_docs = rag_system.data_module.get_parent_documents(retrieved_docs)
        return rag_system.generation_module.generate_list_answer(question, parent_docs)
    if route_type == "detail":
        return rag_system.generation_module.generate_step_by_step_answer(question, retrieved_docs)
    return rag_system.generation_module.generate_basic_answer(question, retrieved_docs)

## save the metadata of retrieved documents in the log
def _format_top_sources(retrieved_docs: list) -> list[dict]:
    sources = []
    for doc in retrieved_docs:
        metadata = doc.metadata
        sources.append({
            "title": metadata.get("title", "Untitled"),
            "topic": metadata.get("topic", "Unknown topic"),
            "section": metadata.get("section_path", "Unknown section"),
            "path": metadata.get("relative_path", metadata.get("source", ""))
        })
    return sources

@app.get("/health")# 健康检查端点
def health_check():
    return {"status": "ok"}


@app.get("/ready")# 就绪检查端点
def readiness_check():
    return {
        "ready": bool(
            rag_system
            and rag_system.retrieval_module
            and rag_system.generation_module
        )
    }

@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = ""
    for attempt in range(MAX_EMPTY_ANSWER_RETRIES + 1):
        answer = rag_system.ask_question(question, stream=False)
        if not _is_blank(answer):# not empty, break the loop and return the answer
            break
        logger.warning("LLM returned an empty answer for /chat; retry=%s", attempt + 1)

    if _is_blank(answer):
        _raise_empty_answer()
    return ChatResponse(answer=answer)

@app.post("/chat/debug")
def chat_debug(request: ChatRequest) -> ChatDebug:
    start_time = time.perf_counter()
    question = request.question.strip()
    retrieval_query = None
    retrieved_docs = []
    retrieval_metrics = None
    stage = "validate"
    try:
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        stage = "route"
        route_type = rag_system.generation_module.query_router(question)
        filters = rag_system._extract_filters_from_query(question)
        
        stage = "rewrite"
        # 根据路由类型决定检索查询
        if route_type == "list":
            retrieval_query = question
        else:
            retrieval_query = rag_system.generation_module.query_rewrite(question)
        
        stage = "retrieval"
        # 根据是否有过滤条件选择检索方法
        if filters:
            retrieved_docs = rag_system.retrieval_module.metadata_filtered_search(
                retrieval_query,
                filters,
                top_k=rag_system.config.top_k,
            )
        else:
            retrieved_docs = rag_system.retrieval_module.hybrid_search(
                retrieval_query,
                top_k=rag_system.config.top_k,
            )
        retrieval_metrics = rag_system.retrieval_module.last_metrics.copy()

        if not retrieved_docs:
            return ChatDebug(
                answer="No relevant documents found",
                route_type=route_type,
                retrieval_query=retrieval_query,
                sources=[]
            )
        stage = "generation"
        answer = ""
        for attempt in range(MAX_EMPTY_ANSWER_RETRIES + 1):
            answer = _generate_answer(question, route_type, retrieved_docs)
            if not _is_blank(answer):
                break
            logger.warning("LLM returned an empty answer for /chat/debug; retry=%s", attempt + 1)

        if _is_blank(answer):
            _raise_empty_answer()
        
        sources = []
        for doc in retrieved_docs:
            metadata = doc.metadata
            sources.append(Source(
                title=metadata.get("title", "Untitled"),
                topic=metadata.get("topic", "Unknown topic"),
                section=metadata.get("section_path", "Unknown section"),
                path=metadata.get("relative_path", metadata.get("source", ""))
            ))
        stage = "response"
        log_rag_query(
            question=question,
            retrieval_query=retrieval_query,
            top_sources=_format_top_sources(retrieved_docs),
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            stage=stage,
            error=None,
            retrieval_metrics=retrieval_metrics,
        )
        return ChatDebug(
            answer=answer,
            route_type=route_type,
            retrieval_query=retrieval_query,
            sources=sources
        )
    
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_rag_query(
            question=question,
            retrieval_query=retrieval_query,
            top_sources=_format_top_sources(retrieved_docs),
            latency_ms=latency_ms,
            stage=stage,
            error=str(exc),
            retrieval_metrics=retrieval_metrics
        )
        raise

@app.post("/query")
def query(query:ChatRequest)->