"""Launch the ClauseWatch MCP server over stdio and exercise it like an MCP client would.

    python scripts/mcp_smoke_test.py            # list tools + retrieval-only tools (no LLM calls)
    python scripts/mcp_smoke_test.py --llm      # also call query_contracts (uses the LLM)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent


def _show(result) -> None:
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        print("tool error:", " ".join(getattr(c, "text", "") for c in result.content))
        return
    payload = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if payload is None:
        payload = [getattr(c, "text", str(c)) for c in result.content]
    print(json.dumps(payload, indent=1, default=str)[:1500])


async def main(use_llm: bool) -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "clausewatch.mcp_server"], cwd=str(ROOT))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])

        contracts = await session.call_tool("list_contracts", {})
        _show(contracts)

        print("\n--- search_clauses")
        _show(await session.call_tool("search_clauses", {
            "query": "limitation of liability aggregate cap",
            "contract_ids": ["papajohnsinternationalinc-20190617-8-k-ex-10-1-1-d6d30a"],
            "clause_types": ["liability"],
            "top_n": 2,
        }))

        print("\n--- error handling (unknown contract id)")
        _show(await session.call_tool("flag_risks", {"contract_id": "does-not-exist"}))

        if use_llm:
            print("\n--- query_contracts")
            _show(await session.call_tool("query_contracts", {
                "question": "Which law governs the Kitov Pharma manufacturing agreement?"
            }))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    asyncio.run(main(ap.parse_args().llm))
