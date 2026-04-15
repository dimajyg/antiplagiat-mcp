# antiplagiat-mcp

MCP server that scores arbitrary text for **plagiarism** and **AI generation**, with **Russian** as the primary language and English as a secondary one.

The server is **stateless with respect to credentials** — it stores no API keys per user. Anyone can plug it into their Claude Code (or any MCP client) and bring their own keys via request headers.

Public endpoint (no SLA, runs on a personal VPS, may go down):

> `https://antiplagiat-mcp.tailf03eb8.ts.net/mcp/`

## Plug it into Claude Code

Add this to your `.claude/mcp.json` (or whichever MCP config your client uses):

```json
{
  "mcpServers": {
    "antiplagiat": {
      "url": "https://antiplagiat-mcp.tailf03eb8.ts.net/mcp/",
      "headers": {
        "X-OpenRouter-Key": "sk-or-v1-...",
        "X-Serper-Key": "",
        "X-Sapling-Key": ""
      }
    }
  }
}
```

| Header              | Purpose                                                 | Required? |
|---------------------|---------------------------------------------------------|-----------|
| `X-OpenRouter-Key`  | Embeddings via OpenRouter (`text-embedding-3-small`)    | Optional, falls back to a local multilingual model |
| `X-Serper-Key`      | Web search for plagiarism source matching (Serper.dev)  | Required for the plagiarism layer; without it, that section returns `skipped_reason` |
| `X-Sapling-Key`     | Optional deeper AI detector for `mode="deep"`           | Optional |

The server reads these headers per request and never persists them.

## Tools exposed

### `analyze_text(text, mode, check_ai, check_plagiarism)`

Returns a structured report:

```json
{
  "hash": "<sha256 of input>",
  "language": "ru" | "en" | "other",
  "ai": {
    "language": "ru",
    "ai_probability": 0.75,
    "confidence": "medium",
    "perplexity": 9.7,
    "perplexity_baseline": [10.0, 30.0],
    "burstiness": 2.8,
    "sentence_count": 4,
    "avg_sentence_length": 13.5,
    "suspicious_sentences": [
      { "text": "...", "perplexity": 6.4, "char_start": 0, "char_end": 84 }
    ],
    "notes": ["perplexity 9.7 far below baseline 10–30", "burstiness 2.8 below threshold 3.0"]
  },
  "plagiarism": {
    "match_percentage": 0.0,
    "matches": [],
    "shingles_searched": 0,
    "sources_fetched": 0,
    "skipped_reason": "no Serper API key in request headers"
  },
  "summary": "AI probability: 75% (medium) | Plagiarism: skipped"
}
```

## How the scores are computed

The local layer is built from three signals that are language-agnostic in spirit and cheap on CPU:

1. **Perplexity** of the text under a small reference LM (`ai-forever/rugpt3small_based_on_gpt2` for Russian, `distilgpt2` for English). LLM-generated prose tends to be more predictable, i.e. lower PPL relative to a baseline. Sliding window across the input.
2. **Burstiness** — std-dev of sentence lengths in tokens. Human writing is bursty (short, long, short); LLM writing is rhythmically flat. This is doing more of the work than perplexity in our calibration.
3. **Per-sentence breakdown** — every sentence is scored individually so the response can highlight the most predictable fragments.

These are blended into `ai_probability` with a transparent heuristic. Every raw signal is surfaced so you can override the verdict.

> ⚠️ **Honest disclaimer.** We initially tried `yaya36095/xlm-roberta-text-detector` as a multilingual classifier; on real samples it returned ~100% AI for every input including obviously human Russian and English text. There's no similarly-sized open-source detector calibrated for Russian at the time of writing. So the local layer leans on statistical signals, not classifiers. For high-stakes use, pass an `X-Sapling-Key` and request `mode="deep"`.

The plagiarism layer is the standard shingle pipeline: characteristic 7-word shingles → Serper.dev quoted search → trafilatura content extraction → exact-substring matches plus paraphrase similarity (cosine on embeddings). It only runs when the client passes a Serper key.

## Calibration snapshot

From `python scripts/calibrate.py` against the live endpoint, on a tiny labelled set:

| label | lang | source  |  ppl | burst |  aiP | conf   |
|-------|------|---------|-----:|------:|-----:|--------|
| human | ru   | diary   | 38.4 |   3.3 | 0.45 | medium |
| human | ru   | blog    | 24.3 |   6.8 | 0.50 | medium |
| ai    | ru   | chatgpt |  9.7 |   2.8 | 0.75 | medium |
| ai    | ru   | chatgpt | 26.2 |   1.5 | 0.60 | medium |
| human | en   | blog    | 51.5 |   6.1 | 0.45 | medium |
| ai    | en   | chatgpt | 39.6 |   5.7 | 0.50 | medium |

The Russian split looks like the system intends — humans 0.45–0.50, AI 0.60–0.75 — driven mostly by burstiness. English separation is much weaker and we recommend `mode="deep"` (Sapling) for English-heavy use.

This is **not** a benchmark. It's a sanity check you can rerun after any threshold change to see whether the detector still pulls in the right direction.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
cp .env.example .env  # fill in your own keys for local testing
python scripts/download_models.py
python server.py
```

Run the tests:

```bash
pytest -q          # 12 tests, will skip the heavy model ones if /models is empty
```

## Deployment

The public endpoint runs on a personal VPS, fronted by **Tailscale Funnel** so the origin IP is never exposed:

* `/opt/antiplagiat-mcp/` (venv, models, real `.env` chmod 600)
* `systemd` unit `antiplagiat-mcp.service` runs uvicorn on `127.0.0.1:8765`
* `tailscaled` Funnel exposes the local port at `https://antiplagiat-mcp.tailf03eb8.ts.net/`
* TLS cert is provisioned automatically by Tailscale (Let's Encrypt under the hood)
* Public traffic enters via Tailscale's edge network — only `185.40.234.0/24` IPs touch the internet, the VPS is invisible to port scanners

See [`deploy/`](./deploy/) for the systemd unit and the Tailscale serve config.

## Security

* Real secrets only ever live at `/opt/antiplagiat-mcp/.env` on the server (`chmod 600`)
* `.env` patterns are in `.gitignore`, `pre-commit` runs `gitleaks` on every commit, GitHub push protection is enabled
* Client keys arrive in HTTP headers, get wrapped in an immutable `RequestCredentials` object, never logged or persisted
* TLS is real Let's Encrypt, no self-signed shenanigans

## Repository layout

```
antiplagiat-mcp/
├── server.py               FastAPI + mounted FastMCP entry point
├── src/
│   ├── config.py           ServerSettings + RequestCredentials
│   ├── language.py         RU/EN routing
│   ├── embeddings.py       OpenRouter primary, multilingual-e5-small fallback
│   ├── pipeline.py         Orchestration: language → AI → plagiarism → blend
│   ├── mcp_app.py          FastMCP instance and the analyze_text tool
│   ├── cache.py            SQLite cache (anonymous, content-hash keyed)
│   └── detectors/
│       ├── ai_local.py     Perplexity + burstiness AI detector
│       ├── plagiarism.py   Shingle → Serper → fetch → similarity pipeline
│       └── external.py     Optional Sapling / GPTZero adapters (Stage 6)
├── scripts/
│   ├── download_models.py  Pulls the 4 HF models on first deploy
│   └── calibrate.py        Re-runs the labelled corpus against the live server
├── deploy/
│   ├── tailscale-serve.md  Tailscale Funnel setup notes
│   └── antiplagiat-mcp.service
├── tests/                  Unit tests + integration tests skipped without models
└── PLAN.md                 Stages and design notes
```
