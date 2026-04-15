"""FastMCP server: registers the analyze_text tool and exposes a Streamable HTTP ASGI app.

The trick to passing per-request credentials into a stateless MCP tool is a
`contextvars.ContextVar` set by a Starlette/FastAPI middleware before each
request reaches the MCP handler. Tools read it via `current_credentials()`.
"""

from __future__ import annotations

import contextvars
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .config import RequestCredentials, settings
from .pipeline import Pipeline

_creds_var: contextvars.ContextVar[RequestCredentials | None] = contextvars.ContextVar(
    "antiplagiat_creds", default=None
)


def set_credentials(creds: RequestCredentials) -> contextvars.Token:
    return _creds_var.set(creds)


def reset_credentials(token: contextvars.Token) -> None:
    _creds_var.reset(token)


def current_credentials() -> RequestCredentials:
    c = _creds_var.get()
    if c is None:
        return RequestCredentials.from_headers({}, settings)
    return c


# Single shared pipeline (lazy model loading happens on first call).
pipeline = Pipeline()

mcp = FastMCP(
    name="antiplagiat-mcp",
    instructions=(
        "Score arbitrary text for AI-generation probability and plagiarism "
        "(Russian primary, English supported). The server is stateless and "
        "expects API keys in request headers: X-OpenRouter-Key (embeddings), "
        "X-Serper-Key (web search for plagiarism), X-Sapling-Key (optional "
        "deep AI detection). Without keys the tool still runs the local "
        "perplexity + burstiness signals and returns a 'skipped' plagiarism "
        "section."
    ),
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
)


@mcp.tool()
async def analyze_text(
    text: Annotated[str, Field(description="The text to analyse, in any language.")],
    mode: Annotated[
        str, Field(description="'fast' (local only) or 'deep' (also calls external APIs).")
    ] = "fast",
    check_ai: Annotated[bool, Field(description="Run AI-generation detection.")] = True,
    check_plagiarism: Annotated[
        bool, Field(description="Run web-based plagiarism detection.")
    ] = True,
) -> dict:
    """Analyse a fragment for AI-generation probability and plagiarism.

    Returns a structured report with raw signals (perplexity, burstiness,
    suspicious sentences, matched sources) so the caller can reason about
    the result rather than trusting a single score.
    """
    creds = current_credentials()
    result = await pipeline.analyze(
        text=text,
        creds=creds,
        mode=mode,  # type: ignore[arg-type]
        check_ai=check_ai,
        check_plagiarism=check_plagiarism,
    )
    return result.to_mcp()


def streamable_http_asgi():
    """Return the Streamable HTTP ASGI app for mounting under FastAPI."""
    return mcp.streamable_http_app()
