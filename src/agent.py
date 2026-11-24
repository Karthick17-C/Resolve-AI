import asyncio
import json
import os
import httpx
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool, create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from pydantic import create_model

from src.embeddings import LocalBGEEmbeddings

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("RESOLVE_API_KEY", "sk-resolve-secret-key-98765")
MODEL_PATH = os.getenv("MODEL_PATH", "./resolve-final-model")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_local_db")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "resolve_knowledge_base")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")


class DirectMCPHttpClient:
    """Async-safe JSON-RPC 2.0 client for Streamable HTTP MCP servers."""

    def __init__(self, url: str = MCP_SERVER_URL) -> None:
        self.url = url
        self._lock = asyncio.Lock()
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    async def _post_rpc(self, payload: dict, expect_response: bool = True) -> dict | None:
        async with self._lock:
            async with httpx.AsyncClient(timeout=15.0) as client:
                async with client.stream("POST", self.url, headers=self.headers, json=payload) as resp:
                    if "mcp-session-id" in resp.headers:
                        self.headers["mcp-session-id"] = resp.headers["mcp-session-id"]
                    if not expect_response:
                        return None
                    content_type = resp.headers.get("content-type", "")
                    if "text/event-stream" in content_type:
                        async for line in resp.aiter_lines():
                            if line.startswith("data:"):
                                return json.loads(line[5:].strip())
                    raw = await resp.aread()
                    return json.loads(raw.decode())

    async def initialize(self) -> None:
        await self._post_rpc({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "resolve-ai-agent", "version": "1.0.0"},
            },
        })
        await self._post_rpc(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            expect_response=False,
        )

    async def list_tools(self) -> list[dict]:
        data = await self._post_rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        return data["result"]["tools"]

    async def call_tool(self, name: str, arguments: dict) -> str:
        data = await self._post_rpc({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        content = data.get("result", {}).get("content", [])
        return "\n".join(c.get("text", "") for c in content if c.get("type") == "text")


def build_langchain_tools_from_mcp(mcp_client: DirectMCPHttpClient, tools_meta: list[dict]) -> list:
    lc_tools = []
    for tool in tools_meta:
        props = tool.get("inputSchema", {}).get("properties", {})
        required = set(tool.get("inputSchema", {}).get("required", []))
        fields = {
            k: (str, ... if k in required else None)
            for k in props
        }
        ArgsSchema = create_model(f"{tool['name']}_Schema", **fields)

        def make_caller(t_name: str = tool["name"], cli: DirectMCPHttpClient = mcp_client):
            async def _call_mcp(**kwargs) -> str:
                return await cli.call_tool(t_name, kwargs)
            return _call_mcp

        lc_tools.append(
            StructuredTool.from_function(
                coroutine=make_caller(),
                name=tool["name"],
                description=tool.get("description", tool["name"]),
                args_schema=ArgsSchema,
            )
        )
    return lc_tools


async def query_resolve_ai(user_query: str) -> str:
    llm = ChatOpenAI(
        model=MODEL_PATH,
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
        temperature=0.1,
        max_tokens=256,
        request_timeout=45,
        model_kwargs={"parallel_tool_calls": False},
    )

    embeddings = LocalBGEEmbeddings()
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        path=QDRANT_PATH,
    )
    rag_tool = create_retriever_tool(
        vector_store.as_retriever(search_kwargs={"k": 2}),
        name="search_company_docs",
        description="Searches Resolve Telecom FAQs, router troubleshooting steps, and return/billing policies.",
    )

    try:
        mcp_client = DirectMCPHttpClient(MCP_SERVER_URL)
        await mcp_client.initialize()
        tools_meta = await mcp_client.list_tools()
        mcp_tools = build_langchain_tools_from_mcp(mcp_client, tools_meta)

        system_prompt = (
            "You are Resolve AI, a concise customer support assistant. "
            "Use 'search_company_docs' for router or policy questions. "
            "For call bookings, call 'check_availability' first, then call 'book_customer_call'. "
            "Keep your final reply brief and clear."
        )
        agent = create_agent(
            model=llm,
            tools=[rag_tool] + mcp_tools,
            system_prompt=system_prompt,
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_query}]},
            config={"recursion_limit": 10},
        )
        return result["messages"][-1].content
    finally:
        vector_store.client.close()


if __name__ == "__main__":
    sample_query = (
        "My FiberBox X1 router has a blinking red light. What steps should I try first? "
        "Also, book a technician call for Alice (alice@example.com) on 2026-11-25 15:00 "
        "for red light optical signal loss."
    )
    response = asyncio.run(query_resolve_ai(sample_query))
    print(response)