"""
NCR Real Estate MCP Server
==========================
Exposes the full NCR Real Estate REST API as an MCP (Model Context Protocol)
server so that LLM clients (Claude Desktop, Cursor, chat.py, etc.) can call
prediction and analytics tools directly.

Architecture
------------
* Uses FastMCP v3 ``FastMCP.from_fastapi(app)`` to auto-generate MCP tools
  from every FastAPI endpoint in ``api/main.py`` (ASGI transport — no separate
  HTTP server required).
* Models are loaded at import time (``_parallel_startup()`` runs on import),
  so the first request may be slow but subsequent calls are fast.

Usage
-----
STDIO (Claude Desktop / chat.py):
    python mcp_server.py

HTTP (MCP Inspector / browser testing):
    python mcp_server.py --transport http --port 9000

Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "ncr-real-estate": {
          "command": "python",
          "args": ["C:/path/to/real_estate/mcp_server.py"],
          "cwd":  "C:/path/to/real_estate"
        }
      }
    }
"""

import argparse
import sys
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Import FastAPI app (triggers model loading at module level) ────────────────
# NOTE: _parallel_startup() runs when api.main is imported, loading all models,
# circle rates, and road segments into memory. This may take 10-30 seconds on
# first import.
from api.main import app  # noqa: E402  (must come after sys.path setup)

# ── Import FastMCP ─────────────────────────────────────────────────────────────
try:
    import fastmcp
except ImportError:
    print(
        "fastmcp is not installed. Run:\n"
        "  pip install fastmcp\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Build MCP server from the FastAPI app ─────────────────────────────────────
# FastMCP.from_fastapi uses httpx.ASGITransport so no HTTP server is needed.
# All FastAPI endpoints are automatically exposed as MCP tools, resources, and
# prompts based on their OpenAPI schema.
mcp = fastmcp.FastMCP.from_fastapi(
    app,
    name="NCR Real Estate AI",
)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCR Real Estate MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode: stdio (default) or http",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port for HTTP transport (default: 9000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for HTTP transport (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    if args.transport == "http":
        import uvicorn

        http_app = mcp.http_app()
        print(
            f"NCR Real Estate MCP server running on http://{args.host}:{args.port}/mcp/",
            file=sys.stderr,
        )
        uvicorn.run(http_app, host=args.host, port=args.port)
    else:
        # STDIO — used by Claude Desktop, Cursor, chat.py
        mcp.run()
