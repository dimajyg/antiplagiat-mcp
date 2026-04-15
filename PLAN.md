# Development plan

## Goals

- MCP server callable from Claude Code that scores any text on:
  - **AI generation probability** (LLM-written or not)
  - **Plagiarism** (matches against the open web + paraphrase similarity)
- **Russian as primary language**, English as second
- **Multi-tenant by design** — clients bring their own API keys via request headers; the server holds no secrets per user
- Cheap to run: local models on CPU + bring-your-own paid APIs only when the client wants them

## Resource budget

VPS: 4 vCPU / 8 GB RAM, no GPU. Headroom ~6 GB after existing services.

| Component            | RAM   | Latency (CPU, ~2K chars) |
|----------------------|-------|--------------------------|
| xlm-roberta detector | ~500MB| 1–2s                     |
| ruGPT3-small (PPL)   | ~500MB| 0.5–1s                   |
| distilgpt2 (PPL en)  | ~330MB| 0.3–1s                   |
| multilingual-e5-small| ~450MB| <0.1s                    |
| **Total**            | ~2 GB | 2–5s                     |

## Stack

| Concern        | Choice                                                  |
|----------------|---------------------------------------------------------|
| Language       | Python 3.13                                             |
| MCP framework  | Official `mcp` SDK                                      |
| Transport      | Streamable HTTP behind Caddy (TLS via Let's Encrypt)    |
| Public hostname| `46-232-250-248.sslip.io` (no domain purchase needed)   |
| Embeddings     | OpenRouter `/v1/embeddings`, fallback to local e5-small |
| Web search     | Serper.dev (per-request via client header)              |
| Local AI det.  | `yaya36095/xlm-roberta-text-detector` (ONNX-quantized)  |
| Perplexity (ru)| `ai-forever/rugpt3small_based_on_gpt2`                  |
| Perplexity (en)| `distilgpt2`                                            |
| Sentence split | `razdel` (RU) + `nltk` (EN)                             |
| Cache          | SQLite (anonymous, keyed by content hash)               |
| Process mgmt   | systemd                                                 |

## Stages

### Stage 0 — Bootstrap (this PR)
- [x] Repo skeleton, `.gitignore`, `.env.example`
- [x] `pre-commit` + `gitleaks` to block secret leaks
- [x] `pyproject.toml` with pinned deps
- [x] README and PLAN
- [ ] Public GitHub repo created and pushed
- [ ] VPS directory `/opt/antiplagiat-mcp/` provisioned, `venv` created, deps installed
- [ ] Models downloaded and ONNX-quantized
- [ ] Real `.env` only on the VPS, `chmod 600`

### Stage 1 — Local AI detector
- [ ] `language.py` — auto-detect ru/en
- [ ] `detectors/ai_local.py`:
  - [ ] xlm-roberta classifier → `ai_probability`
  - [ ] Perplexity (router by language) → `perplexity`
  - [ ] Burstiness → `std` of sentence lengths
  - [ ] Per-sentence breakdown
- [ ] Tests on a small RU/EN corpus

### Stage 2 — Plagiarism pipeline
- [ ] `detectors/plagiarism.py`:
  - [ ] `razdel` sentence splitter
  - [ ] N-gram shingling, characteristic-shingle selection
  - [ ] Serper.dev parallel search
  - [ ] Trafilatura fetcher for candidate URLs
  - [ ] Exact + paraphrase (embedding) matching
- [ ] `embeddings.py` with OpenRouter primary, local e5-small fallback

### Stage 3 — MCP server
- [ ] `server.py` exposing `analyze_text`, `analyze_file`, `explain_highlights`
- [ ] Streamable HTTP transport
- [ ] Header-based credential middleware (per-request, not stored)
- [ ] SQLite cache (content-hash keyed, no PII)

### Stage 4 — Deploy
- [ ] Caddy reverse proxy with `46-232-250-248.sslip.io` + Let's Encrypt
- [ ] systemd unit `antiplagiat-mcp.service`
- [ ] Health endpoint + structured logging
- [ ] Smoke test from a remote MCP client

### Stage 5 — Quality and calibration
- [ ] Build a tiny labelled corpus (50 RU human, 50 RU AI, 30 EN human, 30 EN AI, 20 plagiarism cases)
- [ ] Compute precision/recall/F1, false-positive rate
- [ ] Pick thresholds, store in `config.yaml`
- [ ] Document accuracy in README

### Stage 6 — Polish (later)
- [ ] PDF/DOCX support
- [ ] Optional self-corpus checking (user uploads own documents)
- [ ] Web UI for ad-hoc checks
- [ ] Per-model "AI signature" detection (GPT-4 vs Claude vs Llama style)

## Security checklist

- [x] `.env*` ignored, `.env.example` is the only env file in git
- [x] `pre-commit` + `gitleaks`
- [ ] Enable GitHub Push Protection in repo settings
- [ ] Server reads client keys from headers only, never writes them anywhere
- [ ] Logs scrub anything matching `sk-*` / `Authorization`
- [ ] Caddy enforces HTTPS, HTTP redirected
