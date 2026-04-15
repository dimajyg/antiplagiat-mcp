"""MCP server entry point.

Stack:
* FastMCP exposes the `analyze_text` tool over Streamable HTTP at /mcp/.
* FastAPI hosts /healthz, /, and a debug /debug/analyze used during smoke tests.
* A single FastAPI middleware extracts per-request credentials from headers
  into a contextvar that FastMCP tools read via `current_credentials()`.
* Caddy on the VPS terminates TLS on :8443 and reverse-proxies to :8765.
"""

from __future__ import annotations

import contextlib
import logging

import uvicorn
from fastapi import FastAPI, Header, Request

from src import __version__
from src.config import RequestCredentials, settings
from src.language import detect
from src.mcp_app import (
    current_credentials,
    mcp,
    pipeline,
    reset_credentials,
    set_credentials,
    streamable_http_asgi,
)

log = logging.getLogger("antiplagiat-mcp")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # FastMCP's streamable HTTP transport needs its session manager running
    # for the lifetime of the app. Without this the /mcp/ subapp errors out.
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="antiplagiat-mcp", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def credentials_middleware(request: Request, call_next):
    creds = RequestCredentials.from_headers(dict(request.headers), settings)
    token = set_credentials(creds)
    try:
        return await call_next(request)
    finally:
        reset_credentials(token)


@app.get("/")
def root() -> dict:
    return {
        "service": "antiplagiat-mcp",
        "version": __version__,
        "repo": "https://github.com/dimajyg/antiplagiat-mcp",
        "mcp_endpoint": "/mcp/",
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/debug/detect-language")
def debug_detect(payload: dict) -> dict:
    return {"language": detect(payload.get("text", ""))}


@app.post("/debug/analyze")
async def debug_analyze(
    payload: dict,
    x_openrouter_key: str | None = Header(default=None),
    x_serper_key: str | None = Header(default=None),
    x_sapling_key: str | None = Header(default=None),
) -> dict:
    # The middleware already populated the contextvar from the same headers,
    # so we just delegate. The explicit Header() params above are kept for
    # OpenAPI documentation only.
    creds = current_credentials()
    result = await pipeline.analyze(
        text=payload.get("text", ""),
        creds=creds,
        mode=payload.get("mode", "fast"),
        check_ai=payload.get("check_ai", True),
        check_plagiarism=payload.get("check_plagiarism", True),
    )
    return result.to_mcp()


# Mount the FastMCP Streamable HTTP transport. Trailing slash matters —
# the MCP client should connect to https://host:8443/mcp/ .
app.mount("/mcp", streamable_http_asgi())


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    log.info("antiplagiat-mcp v%s starting on %s:%d", __version__, settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
