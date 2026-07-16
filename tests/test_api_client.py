import json

import httpx
import pytest

from api_client import (
    APIClientError,
    RAGAPIClient,
    SSEEvent,
    SSEProtocolError,
    apply_chat_event,
    new_chat_result,
    parse_sse,
)


def _sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def test_parse_sse_handles_arbitrary_chunks_and_multiple_events():
    text = _sse("accepted", {"trace_id": "abc"}) + _sse("token", {"text": "回答"})
    chunks = [text[:7], text[7:31], text[31:49], text[49:]]
    events = list(parse_sse(chunks))
    assert [(event.event, event.data) for event in events] == [
        ("accepted", {"trace_id": "abc"}),
        ("token", {"text": "回答"}),
    ]


def test_parse_sse_handles_crlf_split_between_chunks():
    events = list(parse_sse(["event: done\r", "\ndata: {\"query_id\": 1}\r", "\n\r\n"]))
    assert events == [SSEEvent("done", {"query_id": 1})]


def test_parse_sse_rejects_invalid_json():
    with pytest.raises(SSEProtocolError, match="Invalid JSON"):
        list(parse_sse(["event: done\ndata: not-json\n\n"]))


def test_stream_chat_sends_cache_and_debug_options():
    received = {}

    def handler(request):
        received.update(json.loads(request.content))
        body = _sse("retrieval_started", {"cached": False}) + _sse("done", {"query_id": 7, "cached": False})
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = RAGAPIClient("http://test", transport=httpx.MockTransport(handler))
    events = list(client.stream_chat("问题", cache_mode="fresh", debug=True))
    assert received == {"question": "问题", "cache_mode": "fresh", "debug": True}
    assert [event.event for event in events] == ["retrieval_started", "done"]


def test_readiness_and_feedback_requests():
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.url.path == "/ready":
            return httpx.Response(200, json={"ready": True, "rag_ready": True, "database_ready": True})
        return httpx.Response(200, json={"ok": True})

    client = RAGAPIClient("http://test/", transport=httpx.MockTransport(handler))
    assert client.get_readiness()["ready"] is True
    client.submit_feedback(3, "helpful", "清楚")
    assert requests[-1] == (
        "POST",
        "/feedback",
        {"query_id": 3, "rating": "helpful", "comment": "清楚"},
    )


def test_http_error_uses_fastapi_detail():
    transport = httpx.MockTransport(lambda request: httpx.Response(503, json={"detail": "not ready"}))
    client = RAGAPIClient("http://test", transport=transport)
    with pytest.raises(APIClientError, match="not ready"):
        list(client.stream_chat("问题"))


def test_stream_requires_terminal_event():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=_sse("token", {"text": "partial"}))
    )
    client = RAGAPIClient("http://test", transport=transport)
    with pytest.raises(SSEProtocolError, match="without a terminal event"):
        list(client.stream_chat("问题"))


def test_chat_result_accumulates_tokens_and_terminal_metadata():
    result = new_chat_result("问题")
    apply_chat_event(result, SSEEvent("retrieval_started", {"cached": True}))
    apply_chat_event(result, SSEEvent("token", {"text": "缓存"}))
    apply_chat_event(result, SSEEvent("token", {"text": "回答"}))
    apply_chat_event(result, SSEEvent("done", {"query_id": 4, "cached": True}))
    assert result["answer"] == "缓存回答"
    assert result["cached"] is True
    assert result["query_id"] == 4
    assert result["terminal_event"] == "done"


def test_chat_result_keeps_sources_on_degraded_event():
    result = new_chat_result("问题")
    apply_chat_event(result, SSEEvent("retrieval_done", {"sources": [{"title": "A"}]}))
    apply_chat_event(result, SSEEvent("degraded", {"reason": "timeout", "sources": []}))
    assert result["sources"] == [{"title": "A"}]
    assert result["reason"] == "timeout"
