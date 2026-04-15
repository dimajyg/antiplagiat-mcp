"""MCP server entry point.

Stage 0: minimal FastAPI app exposing `/healthz` and a stub `/mcp` so the
systemd unit, Caddy reverse proxy and TLS chain can be verified end-to-end
before the real MCP transport is wired up in Stage 3.
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

from src import __version__
from src.config import settings
from src.language import detect

log = logging.getLogger("antiplagiat-mcp")

app = FastAPI(title="antiplagiat-mcp", version=__version__)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": __version__, "stage": "0-skeleton"}


@app.get("/mcp")
@app.post("/mcp")
def mcp_stub() -> dict:
    return {
        "status": "not_implemented",
        "message": "MCP transport is wired up in Stage 3 — see PLAN.md",
    }


@app.post("/debug/detect-language")
def debug_detect(payload: dict) -> dict:
    text = payload.get("text", "")
    return {"language": detect(text)}


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    log.info("antiplagiat-mcp v%s starting on %s:%d", __version__, settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
