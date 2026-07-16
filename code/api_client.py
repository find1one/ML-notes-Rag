"""HTTP/SSE client used by the Streamlit frontend."""

from dataclasses import dataclass
import json
from typing import Any, Iterable, Iterator, Literal, Optional

import httpx


class APIClientError(RuntimeError):
    """Raised when the backend cannot provide a valid response."""


class SSEProtocolError(APIClientError):
    """Raised when the backend emits malformed SSE data."""


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: dict[str, Any]


def new_chat_result(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "answer": "",
        "sources": [],
        "trace_id": None,
        "query_id": None,
        "cached": False,
        "route_type": None,
        "retrieval_query": None,
        "gate_decision": None,
        "terminal_event": None,
        "reason": None,
        "client_error": None,
        "feedback_submitted": False,
    }


def apply_chat_event(result: dict[str, Any], event: SSEEvent) -> None:
    """Apply one backend event to a Streamlit-friendly result object."""
    payload = event.data
    if event.event == "accepted":
        result["trace_id"] = payload.get("trace_id")
    elif event.event == "retrieval_started":
        result["cached"] = bool(payload.get("cached", False))
    elif event.event == "retrieval_done":
        result["sources"] = payload.get("sources") or result["sources"]
        result["cached"] = bool(payload.get("cached", result["cached"]))
        result["route_type"] = payload.get("route_type")
        result["retrieval_query"] = payload.get("retrieval_query")
        result["gate_decision"] = payload.get("gate_decision")
    elif event.event == "token":
        result["answer"] += str(payload.get("text", ""))
    elif event.event == "done":
        result["query_id"] = payload.get("query_id")
        result["cached"] = bool(payload.get("cached", result["cached"]))
        result["terminal_event"] = "done"
    elif event.event in {"rejected", "degraded", "cancelled"}:
        result["terminal_event"] = event.event
        result["reason"] = payload.get("reason") or payload.get("error")
        result["sources"] = payload.get("sources") or result["sources"]


def parse_sse(chunks: Iterable[str]) -> Iterator[SSEEvent]:
    """Parse SSE frames from arbitrary text chunks."""
    buffer = ""
    pending_carriage_return = False
    for chunk in chunks:
        if pending_carriage_return:
            chunk = "\r" + chunk
            pending_carriage_return = False
        if chunk.endswith("\r"):
            chunk = chunk[:-1]
            pending_carriage_return = True
        buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            event = _parse_sse_frame(raw_event)
            if event is not None:
                yield event

    if pending_carriage_return:
        buffer += "\n"
    if buffer.strip():
        event = _parse_sse_frame(buffer)
        if event is not None:
            yield event


def _parse_sse_frame(raw_event: str) -> Optional[SSEEvent]:
    event_name = "message"
    data_lines: list[str] = []
    for line in raw_event.splitlines():
        if not line or line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)

    if not data_lines:
        return None
    try:
        payload = json.loads("\n".join(data_lines))
    except json.JSONDecodeError as exc:
        raise SSEProtocolError(f"Invalid JSON in SSE event '{event_name}'") from exc
    if not isinstance(payload, dict):
        raise SSEProtocolError(f"SSE event '{event_name}' data must be a JSON object")
    return SSEEvent(event=event_name, data=payload)


class RAGAPIClient:
    def __init__(
        self,
        base_url: str,
        timeout: Optional[httpx.Timeout] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout or httpx.Timeout(150.0, connect=5.0)
        self.transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self.transport,
        )

    def get_readiness(self) -> dict[str, Any]:
        try:
            with self._client() as client:
                response = client.get("/ready")
                self._raise_for_status(response)
                payload = response.json()
        except httpx.HTTPError as exc:
            raise APIClientError(f"Cannot connect to FastAPI at {self.base_url}: {exc}") from exc
        except ValueError as exc:
            raise APIClientError("FastAPI /ready returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise APIClientError("FastAPI /ready returned an invalid payload")
        return payload

    def stream_chat(
        self,
        question: str,
        cache_mode: Literal["default", "fresh"] = "default",
        debug: bool = False,
    ) -> Iterator[SSEEvent]:
        payload = {"question": question, "cache_mode": cache_mode, "debug": debug}
        try:
            with self._client() as client:
                with client.stream("POST", "/v1/chat/stream", json=payload) as response:
                    self._raise_for_status(response)
                    terminal_seen = False
                    for event in parse_sse(response.iter_text()):
                        terminal_seen = terminal_seen or event.event in {"done", "rejected", "degraded", "cancelled"}
                        yield event
                    if not terminal_seen:
                        raise SSEProtocolError("FastAPI stream ended without a terminal event")
        except APIClientError:
            raise
        except httpx.HTTPError as exc:
            raise APIClientError(f"FastAPI stream failed: {exc}") from exc

    def submit_feedback(
        self,
        query_id: int,
        rating: Literal["helpful", "not_helpful"],
        comment: Optional[str] = None,
    ) -> None:
        payload = {"query_id": query_id, "rating": rating, "comment": comment or None}
        try:
            with self._client() as client:
                response = client.post("/feedback", json=payload)
                self._raise_for_status(response)
        except APIClientError:
            raise
        except httpx.HTTPError as exc:
            raise APIClientError(f"Feedback request failed: {exc}") from exc

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = None
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail")
        except ValueError:
            pass
        message = detail or response.text[:300] or response.reason_phrase
        raise APIClientError(f"FastAPI returned HTTP {response.status_code}: {message}")
