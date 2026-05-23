"""MCP server that exposes a self-hosted Mem0 REST API as MCP tools."""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Any, Callable, Dict, Optional, TypeVar

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("mem0_mcp_server")

T = TypeVar("T")

try:
    from smithery.decorators import smithery
except ImportError:  # pragma: no cover

    class _SmitheryFallback:
        @staticmethod
        def server(*args, **kwargs):  # type: ignore[misc]
            def decorator(func: Callable[..., T]) -> Callable[..., T]:  # type: ignore[type-var]
                return func

            return decorator

    smithery = _SmitheryFallback()  # type: ignore[assignment]

try:
    from .schemas import ConfigSchema
except ImportError:  # pragma: no cover
    from schemas import ConfigSchema  # type: ignore[no-redef]

BASE_URL = os.getenv("MEM0_HOST", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("MEM0_API_KEY", "")
DEFAULT_USER_ID = os.getenv("MEM0_DEFAULT_USER_ID", "mem0-mcp")
TIMEOUT = 60.0

HEADERS = {"X-API-Key": API_KEY}


def _call(method: str, path: str, **kwargs) -> str:
    try:
        r = httpx.request(method, f"{BASE_URL}{path}", headers=HEADERS, timeout=TIMEOUT, **kwargs)
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False)
    except httpx.HTTPStatusError as exc:
        logger.error("mem0 %s %s → %s", method, path, exc.response.status_code)
        return json.dumps({"error": str(exc), "status": exc.response.status_code}, ensure_ascii=False)
    except Exception as exc:
        logger.error("mem0 request failed: %s", exc)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@smithery.server(config_schema=ConfigSchema)
def create_server() -> FastMCP:
    server = FastMCP(
        "mem0",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8081")),
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @server.tool(description="Store a new preference, fact, or conversation snippet.")
    def add_memory(
        text: Annotated[str, Field(description="Plain sentence summarizing what to store.")],
        messages: Annotated[
            Optional[list[Dict[str, str]]],
            Field(default=None, description="Structured conversation history with `role`/`content`."),
        ] = None,
        user_id: Annotated[Optional[str], Field(default=None, description="User scope for this write.")] = None,
        agent_id: Annotated[Optional[str], Field(default=None, description="Optional agent identifier.")] = None,
        run_id: Annotated[Optional[str], Field(default=None, description="Optional run identifier.")] = None,
        metadata: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Arbitrary metadata JSON.")] = None,
        ctx: Context | None = None,
    ) -> str:
        conversation = messages or [{"role": "user", "content": text}]
        payload: Dict[str, Any] = {"messages": conversation}
        payload["user_id"] = user_id or (DEFAULT_USER_ID if not (agent_id or run_id) else None)
        if agent_id:
            payload["agent_id"] = agent_id
        if run_id:
            payload["run_id"] = run_id
        if metadata:
            payload["metadata"] = metadata
        payload = {k: v for k, v in payload.items() if v is not None}
        return _call("POST", "/memories", json=payload)

    @server.tool(
        description="Run a semantic search over existing memories. user_id is injected automatically."
    )
    def search_memories(
        query: Annotated[str, Field(description="Natural language description of what to find.")],
        filters: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Additional filter clauses.")] = None,
        limit: Annotated[Optional[int], Field(default=None, description="Maximum results to return.")] = None,
        ctx: Context | None = None,
    ) -> str:
        effective_filters = filters or {"user_id": DEFAULT_USER_ID}
        payload: Dict[str, Any] = {"query": query, "filters": effective_filters}
        if limit is not None:
            payload["limit"] = limit
        return _call("POST", "/search", json=payload)

    @server.tool(description="List memories for a user/agent/run.")
    def get_memories(
        user_id: Annotated[Optional[str], Field(default=None, description="User scope; defaults to server user.")] = None,
        agent_id: Annotated[Optional[str], Field(default=None, description="Optional agent scope.")] = None,
        run_id: Annotated[Optional[str], Field(default=None, description="Optional run scope.")] = None,
        ctx: Context | None = None,
    ) -> str:
        params: Dict[str, str] = {}
        params["user_id"] = user_id or DEFAULT_USER_ID
        if agent_id:
            params["agent_id"] = agent_id
        if run_id:
            params["run_id"] = run_id
        return _call("GET", "/memories", params=params)

    @server.tool(description="Fetch a single memory by its memory_id.")
    def get_memory(
        memory_id: Annotated[str, Field(description="Exact memory_id to fetch.")],
        ctx: Context | None = None,
    ) -> str:
        return _call("GET", f"/memories/{memory_id}")

    @server.tool(description="Overwrite an existing memory's text.")
    def update_memory(
        memory_id: Annotated[str, Field(description="Exact memory_id to overwrite.")],
        text: Annotated[str, Field(description="Replacement text for the memory.")],
        ctx: Context | None = None,
    ) -> str:
        return _call("PUT", f"/memories/{memory_id}", json={"text": text})

    @server.tool(description="Delete one memory by its memory_id.")
    def delete_memory(
        memory_id: Annotated[str, Field(description="Exact memory_id to delete.")],
        ctx: Context | None = None,
    ) -> str:
        return _call("DELETE", f"/memories/{memory_id}")

    @server.tool(description="Delete all memories for a user/agent/run.")
    def delete_all_memories(
        user_id: Annotated[Optional[str], Field(default=None, description="User scope; defaults to server user.")] = None,
        agent_id: Annotated[Optional[str], Field(default=None, description="Optional agent scope.")] = None,
        run_id: Annotated[Optional[str], Field(default=None, description="Optional run scope.")] = None,
        ctx: Context | None = None,
    ) -> str:
        params: Dict[str, str] = {}
        params["user_id"] = user_id or DEFAULT_USER_ID
        if agent_id:
            params["agent_id"] = agent_id
        if run_id:
            params["run_id"] = run_id
        return _call("DELETE", "/memories", params=params)

    @server.tool(description="List which users/agents/runs currently hold memories.")
    def list_entities(ctx: Context | None = None) -> str:
        return _call("GET", "/entities")

    @server.tool(description="Delete a user/agent/run entity and cascade-delete its memories.")
    def delete_entities(
        user_id: Annotated[Optional[str], Field(default=None, description="Delete this user.")] = None,
        agent_id: Annotated[Optional[str], Field(default=None, description="Delete this agent.")] = None,
        run_id: Annotated[Optional[str], Field(default=None, description="Delete this run.")] = None,
        ctx: Context | None = None,
    ) -> str:
        if user_id:
            return _call("DELETE", f"/entities/user/{user_id}")
        if agent_id:
            return _call("DELETE", f"/entities/agent/{agent_id}")
        if run_id:
            return _call("DELETE", f"/entities/run/{run_id}")
        return json.dumps({"error": "scope_missing", "detail": "Provide user_id, agent_id, or run_id."})

    @server.prompt()
    def memory_assistant() -> str:
        """Get help with memory operations and best practices."""
        return f"""Mem0 MCP server — self-hosted at {BASE_URL}, default user: {DEFAULT_USER_ID}.

Tools: add_memory, search_memories, get_memories, get_memory, update_memory,
       delete_memory, delete_all_memories, list_entities, delete_entities

user_id defaults to '{DEFAULT_USER_ID}' when not specified."""

    return server


def main() -> None:
    server = create_server()
    logger.info("Starting Mem0 MCP server (host=%s, user=%s)", BASE_URL, DEFAULT_USER_ID)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
