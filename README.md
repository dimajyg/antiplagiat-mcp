# antiplagiat-mcp

MCP server that checks text for **plagiarism** and **AI generation**, in Russian and English.

Built to be plugged into Claude Code (or any MCP client) and used by anyone with their own API keys — the server stores no credentials.

## Status

🚧 Work in progress. See [`PLAN.md`](./PLAN.md) for the roadmap.

## Architecture

Two-layer pipeline:

1. **Local layer (free)** — runs on the server's CPU, no per-request cost
   - Multilingual AI-detector (`xlm-roberta`)
   - Perplexity (`ruGPT3-small` / `distilgpt2`)
   - Burstiness, sentence-level statistics
   - Sentence shingling for plagiarism candidates

2. **Network layer (BYO key)** — uses the *client's* API keys passed via headers
   - Embeddings via OpenRouter (`openai/text-embedding-3-small`)
   - Web search via Serper.dev for plagiarism source matching
   - Optional Sapling / GPTZero for deeper AI detection

## How clients pass their keys

The server is **stateless** with respect to credentials. Add this to your `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "antiplagiat": {
      "url": "https://46-232-250-248.sslip.io/mcp",
      "headers": {
        "X-OpenRouter-Key": "sk-or-v1-...",
        "X-Serper-Key": "...",
        "X-Sapling-Key": ""
      }
    }
  }
}
```

Each request reuses these headers; nothing is persisted server-side.

## Tools exposed

- `analyze_text(text, mode, check_ai, check_plagiarism)` — full analysis
- `analyze_file(path, ...)` — same, for a file
- `explain_highlights(analysis_id)` — drill into suspicious fragments

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
cp .env.example .env  # fill in your own keys for local testing
python scripts/download_models.py
python server.py
```

## Deployment

See [`deploy/`](./deploy/) for the systemd unit and Caddy config used on the public server.

## Security

- Never commit secrets — `.gitignore` blocks `.env*`, `pre-commit` runs `gitleaks`, GitHub push protection is enabled
- Real `.env` only ever lives at `/opt/antiplagiat-mcp/.env` on the server, `chmod 600`
- Client keys are read from request headers in memory and never logged
