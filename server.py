"""MCP server entry point. Streamable HTTP, BYO credentials via headers."""

from __future__ import annotations

import logging

from src.config import settings

log = logging.getLogger("antiplagiat-mcp")


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    log.info(
        "antiplagiat-mcp starting on %s:%d (Stage 0 skeleton — no tools registered yet)",
        settings.host,
        settings.port,
    )
    # TODO Stage 3: wire up the official `mcp` Python SDK with Streamable HTTP
    # transport, register `analyze_text` / `analyze_file` / `explain_highlights`
    # tools, and inject `RequestCredentials` per request.


if __name__ == "__main__":
    main()
