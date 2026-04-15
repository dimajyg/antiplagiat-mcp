"""MCP server entry point.

Stage 0: minimal FastAPI app exposing `/healthz` and a stub `/mcp` so the
systemd unit, Caddy reverse proxy and TLS chain can be verified end-to-end
before the real MCP transport is wired up in Stage 3.
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI, Header

from src import __version__
from src.config import RequestCredentials, settings
from src.language import detect
from src.pipeline import Pipeline

log = logging.getLogger("antiplagiat-mcp")

app = FastAPI(title="antiplagiat-mcp", version=__version__)
pipeline = Pipeline()


@app.get("/")
def root() -> dict:
    return {
        "service": "antiplagiat-mcp",
        "version": __version__,
        "repo": "https://github.com/dimajyg/antiplagiat-mcp",
    }


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


@app.post("/debug/analyze")
async def debug_analyze(
    payload: dict,
    x_openrouter_key: str | None = Header(default=None),
    x_serper_key: str | None = Header(default=None),
    x_sapling_key: str | None = Header(default=None),
) -> dict:
    text = payload.get("text", "")
    creds = RequestCredentials.from_headers(
        {
            "X-OpenRouter-Key": x_openrouter_key or "",
            "X-Serper-Key": x_serper_key or "",
            "X-Sapling-Key": x_sapling_key or "",
        },
        settings,
    )
    result = await pipeline.analyze(
        text=text,
        creds=creds,
        mode=payload.get("mode", "fast"),
        check_ai=payload.get("check_ai", True),
        check_plagiarism=payload.get("check_plagiarism", True),
    )
    return result.to_mcp()


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    log.info("antiplagiat-mcp v%s starting on %s:%d", __version__, settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
